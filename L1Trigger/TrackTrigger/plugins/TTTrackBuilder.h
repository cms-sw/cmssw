/*! \class   TTTrackBuilder
 *  \brief   Plugin to load the Track finding algorithm and produce the
 *           collection of Tracks that goes in the event content.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#ifndef L1TK_TRACK_BUILDER_H
#define L1TK_TRACK_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTTrackAlgorithmRecord.h"
//#include "classNameFinder.h"

#include <memory>
#include <map>
#include <vector>

template<  typename T  >
class TTTrackBuilder : public edm::EDProducer
{
  public:
    /// Constructor
    explicit TTTrackBuilder( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~TTTrackBuilder();

  private:
    /// Tracking algorithm
    const StackedTrackerGeometry           *theStackedTracker;
    edm::ESHandle< TTTrackAlgorithm< T > > theTrackFindingAlgoHandle;
    edm::InputTag                          TTStubsInputTag;

    /// Other stuff
    bool enterAssociativeMemoriesWorkflow;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

}; /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Constructors
template< typename T >
TTTrackBuilder< T >::TTTrackBuilder( const edm::ParameterSet& iConfig )
{
  produces< std::vector< TTTrack< T > > >( "Seeds" );
  produces< std::vector< TTTrack< T > > >( "NoDup" );
  TTStubsInputTag = iConfig.getParameter< edm::InputTag >( "TTStubsBricks" );
  enterAssociativeMemoriesWorkflow = iConfig.getParameter< bool >( "AssociativeMemories" );
}

/// Destructor
template< typename T >
TTTrackBuilder< T >::~TTTrackBuilder() {}

/// Begin run
template< typename T >
void TTTrackBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry references
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTracker = StackedTrackerGeomHandle.product();

  /// Get the tracking algorithm 
  iSetup.get< TTTrackAlgorithmRecord >().get( theTrackFindingAlgoHandle );
  /// Print some information when loaded
  std::cout  << std::endl;
  std::cout  << "TTTrackBuilder< " << templateNameFinder< T >() << " > loaded modules:"
             << "\n\tTTTrackAlgorithm:\t" << theTrackFindingAlgoHandle->AlgorithmName()
             << std::endl;
  std::cout  << std::endl;
}

/// End run
template< typename T >
void TTTrackBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ) {}

/// Implement the producer
template< typename T >
void TTTrackBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  /// The temporary collection is used to store tracks
  /// before removal of duplicates
  std::vector< TTTrack< T > > tempTrackCollection;
  tempTrackCollection.clear();
  std::auto_ptr< std::vector< TTTrack< T > > > TTTracksSeedsForOutput( new std::vector< TTTrack< T > > );
  std::auto_ptr< std::vector< TTTrack< T > > > TTTracksForOutput( new std::vector< TTTrack< T > > );
  std::auto_ptr< std::vector< TTTrack< T > > > TTTracksForOutputPurged( new std::vector< TTTrack< T > > );

  /// Get the Stubs already stored away
  edm::Handle< std::vector< TTStub< T > > > TTStubHandle;
  iEvent.getByLabel( TTStubsInputTag, TTStubHandle );

  if ( enterAssociativeMemoriesWorkflow )
  {
    /// Enter AM 
    std::cerr << "TEST: AM workflow" << std::endl;
    theTrackFindingAlgoHandle->PatternFinding();
    theTrackFindingAlgoHandle->PatternRecognition();

  } /// End AM workflow
  else
  {
    /// Tracklet-based approach

    /// Create the Seeds and map the Stubs per Sector/Wedge
    std::vector< TTTrack< T > > theseSeeds;
    std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< T > > > > *stubSectorWedgeMap;
    stubSectorWedgeMap = new std::map< std::pair< unsigned int, unsigned int >, std::vector< edm::Ptr< TTStub< T > > > >();
    theTrackFindingAlgoHandle->CreateSeeds( theseSeeds, stubSectorWedgeMap, TTStubHandle );

    /// Get the number of sectors
    unsigned int nSectors = theTrackFindingAlgoHandle->ReturnNumberOfSectors();
    unsigned int nWedges = theTrackFindingAlgoHandle->ReturnNumberOfWedges();

    /// Here all the seeds are available and all the stubs are stored
    /// in a sector-wise map: loop over seeds, find the sector, attach stubs
    /// Store the seeds menawhile ...
    for ( unsigned int it = 0; it < theseSeeds.size(); it++ )
    {
      /// Get the seed and immediately store the seed as it is being modified later!
      TTTrack< T > curSeed = theseSeeds.at(it);
      TTTracksSeedsForOutput->push_back( curSeed );

      /// Find the sector and the stubs to be attached
      unsigned int curSector0 = curSeed.getSector() + nSectors; /// This is to use the %nSectors later
      unsigned int curWedge0 = curSeed.getWedge();

      /// Loop over the sector and its two neighbors
      for ( unsigned int iSector = 0; iSector < 2; iSector++ )
      {
        for ( unsigned int iWedge = 0; iWedge < 2; iWedge++)
        {
          /// Find the correct sector index
          unsigned int curSector = ( curSector0 + iSector -1 )%nSectors;
          int curWedge = curWedge0 + iWedge - 1;
          if ( curWedge < 0 || curWedge >= (int)nWedges )
            continue;

          std::pair< unsigned int, unsigned int > sectorWedge = std::make_pair( curSector, (unsigned int)curWedge );

          /// Skip sector if empty
          if ( stubSectorWedgeMap->find( sectorWedge ) == stubSectorWedgeMap->end() )
            continue;

          std::vector< edm::Ptr< TTStub< T > > > stubsToAttach = stubSectorWedgeMap->find( sectorWedge )->second;

          /// Loop over the stubs in the Sector
          for ( unsigned int iv = 0; iv < stubsToAttach.size(); iv++ )
          {
            /// Here we have same-sector-different-SL seed and stubs
            theTrackFindingAlgoHandle->AttachStubToSeed( curSeed, stubsToAttach.at(iv) );
          } /// End of nested loop over stubs in the Sector
        }
      } /// End of loop over the sector and its two neighbors

      /// Here the seed is completed with all its matched stubs
      /// The seed is now a track and it is time to fit it
      theTrackFindingAlgoHandle->FitTrack( curSeed );

      /// Refit tracks if needed
      if ( curSeed.getLargestResIdx() > -1 )
      {
        if ( curSeed.getStubPtrs().size() > 3 && curSeed.getChi2() > 100.0 )
        {
          std::vector< edm::Ptr< TTStub< T > > > theseStubs = curSeed.getStubPtrs();
          theseStubs.erase( theseStubs.begin()+curSeed.getLargestResIdx() );
          curSeed.setStubPtrs( theseStubs );
          curSeed.setLargestResIdx( -1 );
          theTrackFindingAlgoHandle->FitTrack( curSeed );
        }
      }

      /// Store the fitted track in the output
      TTTracksForOutput->push_back( curSeed );

    } /// End of loop over seeds

  } /// End of non-AM

  /// Remove duplicates
  std::vector< bool > toBeDeleted;
  for ( unsigned int i = 0; i < TTTracksForOutput->size(); i++ )
  {
    toBeDeleted.push_back( false );
  }

  for ( unsigned int i = 0; i < TTTracksForOutput->size(); i++ )
  {
    /// This check is necessary as the bool may be reset in a previous iteration
    if ( toBeDeleted.at(i) )
      continue;

    /// Check if the track has min 3 stubs
    if ( TTTracksForOutput->at(i).getStubPtrs().size() < 3 )
      continue;

    /// Count the number of PS stubs
    unsigned int nPSi = 0;
    for ( unsigned int is = 0; is < TTTracksForOutput->at(i).getStubPtrs().size(); is++ )
    {
      StackedTrackerDetId stDetId( TTTracksForOutput->at(i).getStubPtrs().at(is)->getDetId() );
      bool isPS = theStackedTracker->isPSModule( stDetId );
      if ( isPS )
        nPSi++;
    }

    bool hasBL1i = TTTracksForOutput->at(i).hasStubInBarrel(1);

    /// Nested loop to compare tracks with each other
    for ( unsigned int j = i+1 ; j < TTTracksForOutput->size(); j++ )
    {
      /// This check is necessary as the bool may be reset in a previous iteration
      if ( toBeDeleted.at(j) )
        continue;

      /// Check if they are the same track
      if ( TTTracksForOutput->at(i).isTheSameAs( TTTracksForOutput->at(j) ) )
      {
        /// Check if the track has min 3 stubs
        if ( TTTracksForOutput->at(j).getStubPtrs().size() < 3 )
          continue;

        /// Count the number of PS stubs
        unsigned int nPSj = 0;
        for ( unsigned int js = 0; js < TTTracksForOutput->at(j).getStubPtrs().size(); js++ )
        {
          StackedTrackerDetId stDetId( TTTracksForOutput->at(j).getStubPtrs().at(js)->getDetId() );
          bool isPS = theStackedTracker->isPSModule( stDetId );
          if ( isPS )
            nPSj++;
        }

        /// Choose the one with the largest number of PS stubs
        if ( nPSi > nPSj )
        {
          toBeDeleted[j] = true;
          continue;
        }
        else if ( nPSi < nPSj )
        {
          toBeDeleted[i] = true;
          continue;
        }
        /// Here we are if the two tracks have the same number of PS stubs

        /// Check which one has a stub in Barrel L1
        bool hasBL1j = TTTracksForOutput->at(j).hasStubInBarrel(1);

        if ( hasBL1i || hasBL1j )
        {
          if ( !hasBL1i )
          {
            toBeDeleted[i] = true;
            continue;
          }
          if ( !hasBL1j )
          {
            toBeDeleted[j] = true;
            continue;
          }
        }
        /// We get here only if both have BL1 or both have not

        /// Choose the one with the largest number of PS stubs
        if ( nPSi > nPSj )
        {
          toBeDeleted[j] = true;
          continue;
        }
        else if ( nPSi < nPSj )
        {
          toBeDeleted[i] = true;
          continue;
        }
        /// Here we are if the two tracks have the same number of PS stubs

        /// Compare Chi2
        if ( TTTracksForOutput->at(i).getChi2Red() > TTTracksForOutput->at(j).getChi2Red() )
        {
          toBeDeleted[i] = true;
        }
        else
        {
          toBeDeleted[j] = true;
        }
        continue;
      }
    }

    if ( toBeDeleted.at(i) ) continue; /// Is it really necessary?
  }

  /// Store only the non-deleted tracks
  for ( unsigned int i = 0; i < TTTracksForOutput->size(); i++ )
  {
    if ( toBeDeleted.at(i) )
      continue;

    TTTracksForOutputPurged->push_back( TTTracksForOutput->at(i) );
  }

  /// Put in the event content
  iEvent.put( TTTracksSeedsForOutput, "Seeds" );
  iEvent.put( TTTracksForOutputPurged, "NoDup" );
}

#endif

