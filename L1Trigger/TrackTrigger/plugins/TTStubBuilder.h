/*! \class TTStubBuilder
* \brief Plugin to load the Stub finding algorithm and produce the
* collection of Stubs that goes in the event content.
* \details After moving from SimDataFormats to DataFormats,
* the template structure of the class was maintained
* in order to accomodate any types other than PixelDigis
* in case there is such a need in the future.
*
* \author Andrew W. Rose
* \author Nicola Pozzobon
* \author Ivan Reid
* \date 2013, Jul 18
*
*/

#ifndef L1_TRACK_TRIGGER_STUB_BUILDER_H
#define L1_TRACK_TRIGGER_STUB_BUILDER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include <memory>
#include <map>
#include <vector>

template< typename T >
class TTStubBuilder : public edm::EDProducer
{
  public:
    /// Constructor
    explicit TTStubBuilder( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~TTStubBuilder();

  private:
    /// Data members
    const StackedTrackerGeometry *theStackedTracker;
    edm::ESHandle< TTStubAlgorithm< T > > theStubFindingAlgoHandle;
    edm::InputTag TTClustersInputTag;

    /// Mandatory methods
    virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

    /// Sorting method for stubs
    /// NOTE: this must be static!
    static bool SortStubBendPairs( const std::pair< unsigned int, double >& left, const std::pair< unsigned int, double >& right );

}; /// Close class

/*! \brief Implementation of methods
* \details Here, in the header file, the methods which do not depend
* on the specific type <T> that can fit the template.
* Other methods, with type-specific features, are implemented
* in the source file.
*/

/// Constructors
template< typename T >
TTStubBuilder< T >::TTStubBuilder( const edm::ParameterSet& iConfig )
{
  TTClustersInputTag = iConfig.getParameter< edm::InputTag >( "TTClusters" );
  produces< edmNew::DetSetVector< TTCluster< T > > >( "ClusterAccepted" );
  produces< edmNew::DetSetVector< TTStub< T > > >( "StubAccepted" );
  produces< edmNew::DetSetVector< TTStub< T > > >( "StubRejected" );
}

/// Destructor
template< typename T >
TTStubBuilder< T >::~TTStubBuilder(){}

/// Begin run
template< typename T >
void TTStubBuilder< T >::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry references
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTracker = StackedTrackerGeomHandle.product();

  /// Get the stub finding algorithm
  iSetup.get< TTStubAlgorithmRecord >().get( theStubFindingAlgoHandle );

  /// Print some information when loaded
  std::cout << std::endl;
  std::cout << "TTStubBuilder< " << templateNameFinder< T >() << " > loaded modules:"
            << "\n\tTTStubAlgorithm:\t" << theStubFindingAlgoHandle->AlgorithmName()
            << std::endl;
  std::cout << std::endl;
}

/// End run
template< typename T >
void TTStubBuilder< T >::endRun( const edm::Run& run, const edm::EventSetup& iSetup ){}

/// Implement the producer
template< typename T >
void TTStubBuilder< T >::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output
  std::auto_ptr< edmNew::DetSetVector< TTCluster< T > > > TTClusterDSVForOutput( new edmNew::DetSetVector< TTCluster< T > > );
  std::auto_ptr< edmNew::DetSetVector< TTStub< T > > > TTStubDSVForOutputTemp( new edmNew::DetSetVector< TTStub< T > > );
  std::auto_ptr< edmNew::DetSetVector< TTStub< T > > > TTStubDSVForOutputAccepted( new edmNew::DetSetVector< TTStub< T > > );
  std::auto_ptr< edmNew::DetSetVector< TTStub< T > > > TTStubDSVForOutputRejected( new edmNew::DetSetVector< TTStub< T > > );

  /// Get the Clusters already stored away
  edm::Handle< edmNew::DetSetVector< TTCluster< T > > > TTClusterHandle;
  iEvent.getByLabel( TTClustersInputTag, TTClusterHandle );

  /// Get the maximum number of stubs per ROC
  /// (CBC3-style)
  unsigned maxStubs = theStackedTracker->getCBC3MaxStubs();

  /// Loop over the detector elements
  StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;
  for ( StackedTrackerIterator = theStackedTracker->stacks().begin();
        StackedTrackerIterator != theStackedTracker->stacks().end();
        ++StackedTrackerIterator )
  {
    StackedTrackerDetUnit* Unit = *StackedTrackerIterator;
    StackedTrackerDetId Id = Unit->Id();
    assert(Unit == theStackedTracker->idToStack(Id));
    
    /// Get the DetIds of each sensor
    DetId id0 = Unit->stackMember(0);
    DetId id1 = Unit->stackMember(1);

    /// Check that everything is ok in the maps
    if ( theStackedTracker->findPairedDetector( id0 ) != id1 ||
         theStackedTracker->findPairedDetector( id1 ) != id0 )
    {
      std::cerr << "A L E R T! error in detector association within Pt module (detector-to-detector)" << std::endl;
      continue;
    }

    if ( theStackedTracker->findStackFromDetector( id0 ) != Id ||
         theStackedTracker->findStackFromDetector( id1 ) != Id )
    {
      std::cerr << "A L E R T! error in detector association within Pt module (detector-to-module)" << std::endl;
      continue;
    }

    /// Go on only if both detectors have Clusters
    if ( TTClusterHandle->find( id0 ) == TTClusterHandle->end() ||
         TTClusterHandle->find( id1 ) == TTClusterHandle->end() )
      continue;

    /// Get the DetSets of the Clusters
    edmNew::DetSet< TTCluster< T > > innerClusters = (*TTClusterHandle)[ id0 ];
    edmNew::DetSet< TTCluster< T > > outerClusters = (*TTClusterHandle)[ id1 ];

    typename edmNew::DetSet< TTCluster< T > >::iterator innerClusterIter, outerClusterIter;

    /// If there are Clusters in both sensors
    /// you can try and make a Stub
    /// This is ~redundant
    if ( innerClusters.size() == 0 || outerClusters.size() == 0 )
      continue;

    /// Create the vectors of objects to be passed to the FastFillers
    std::vector< TTCluster< T > > *tempInner = new std::vector< TTCluster< T > >();
    std::vector< TTCluster< T > > *tempOuter = new std::vector< TTCluster< T > >();
    std::vector< TTStub< T > > *tempOutput = new std::vector< TTStub< T > >();
    std::vector< TTStub< T > > *tempRejected = new std::vector< TTStub< T > >();
    tempInner->clear();
    tempOuter->clear();
    tempOutput->clear();
    tempRejected->clear();

    /// Get chip size information
    const GeomDetUnit* det0 = theStackedTracker->idToDetUnit( Id, 0 );
    const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
    const int chipSize = 2 * top0->rowsperroc(); /// Need to find ASIC size in half-strip units
    std::map< int, std::vector< TTStub< T > > > moduleStubs; /// Temporary storage for stubs before max check

    /// Loop over pairs of Clusters
    for ( innerClusterIter = innerClusters.begin();
          innerClusterIter != innerClusters.end();
          ++innerClusterIter )
    {
      for ( outerClusterIter = outerClusters.begin();
            outerClusterIter != outerClusters.end();
            ++outerClusterIter )
      {
        /// Build a temporary Stub
        TTStub< T > tempTTStub( Id );
        tempTTStub.addClusterRef( edmNew::makeRefTo( TTClusterHandle, innerClusterIter ) );
        tempTTStub.addClusterRef( edmNew::makeRefTo( TTClusterHandle, outerClusterIter ) );

        /// Check for compatibility
        bool thisConfirmation = false;
        int thisDisplacement = 999999;
        int thisOffset = 0;

        theStubFindingAlgoHandle->PatternHitCorrelation( thisConfirmation, thisDisplacement, thisOffset, tempTTStub );

        /// If the Stub is above threshold
        if ( thisConfirmation )
        {
          tempTTStub.setTriggerDisplacement( thisDisplacement );
          tempTTStub.setTriggerOffset( thisOffset );

          /// Put in the output
          if ( maxStubs == 0 )
          {
            /// This means that ALL stubs go into the output
            tempInner->push_back( *innerClusterIter );
            tempOuter->push_back( *outerClusterIter );
            tempOutput->push_back( tempTTStub );
          }
          else
          {
            /// This means that only some of them do
            /// Put in the temporary output
            int chip = tempTTStub.getTriggerPosition() / chipSize; /// Find out which ASIC
            if ( moduleStubs.find( chip ) == moduleStubs.end() ) /// Already a stub for this ASIC?
            {
              /// No, so new entry
              std::vector< TTStub< T > > tempStubs;
              tempStubs.clear();
              tempStubs.push_back( tempTTStub );
              moduleStubs.insert( std::pair< int, std::vector< TTStub< T > > >( chip, tempStubs ) );
            }
            else
            {
              /// Already existing entry
              moduleStubs[chip].push_back( tempTTStub );
            }
          }
        } /// Stub accepted
/* NP 2014 02 25
* this is commented to avoid memory exhaustion in hi PU events
else
{
tempRejected->push_back( tempTTStub );
} /// Stub rejected
*/
      } /// End of nested loop
    } /// End of loop over pairs of Clusters

    /// If we are working with max no. stub/ROC, then clean the temporary output
    /// and store only the selected stubs
    if ( moduleStubs.empty() == false )
    {
      /// Loop over ROC's
      /// the ROC ID is not important
      for ( auto const & is : moduleStubs )
      {
        /// Put the stubs into the output
        if ( is.second.size() <= maxStubs )
        {
          for ( auto const & ts: is.second )
          {
            tempInner->push_back( *(ts.getClusterRef(0)) );
            tempOuter->push_back( *(ts.getClusterRef(1)) );
            tempOutput->push_back( ts );
          }
        }
        else
        {
          /// Sort them and pick up only the first N.
          std::vector< std::pair< unsigned int, double > > bendMap;
          for ( unsigned int i = 0; i < is.second.size(); ++i )
          {
            bendMap.push_back( std::pair< unsigned int, double >( i, is.second[i].getTriggerBend() ) );
          }
          std::sort( bendMap.begin(), bendMap.end(), TTStubBuilder< T >::SortStubBendPairs );

          for ( unsigned int i = 0; i < maxStubs; ++i )
          {
            /// Put the highest momenta (lowest bend) stubs into the event
            tempInner->push_back( *(is.second[bendMap[i].first].getClusterRef(0)) );
            tempOuter->push_back( *(is.second[bendMap[i].first].getClusterRef(1)) );
            tempOutput->push_back( is.second[bendMap[i].first] );
          }
/* NP 2014 02 25
* this is commented to avoid memory exhaustion in hi PU events
for ( unsigned int i = maxStubs; i < is.second.size(); ++i )
{
/// Reject the rest
tempRejected->push_back( is.second[bendMap[i].first] );
}
*/
        }
      } /// End of loop over temp output
    } /// End store only the selected stubs if max no. stub/ROC is set

    /// Create the FastFillers
    if ( tempInner->size() > 0 )
    {
      typename edmNew::DetSetVector< TTCluster< T > >::FastFiller innerOutputFiller( *TTClusterDSVForOutput, id0 );
      for ( unsigned int m = 0; m < tempInner->size(); m++ )
      {
        innerOutputFiller.push_back( tempInner->at(m) );
      }
      if ( innerOutputFiller.empty() )
        innerOutputFiller.abort();
    }

    if ( tempOuter->size() > 0 )
    {
      typename edmNew::DetSetVector< TTCluster< T > >::FastFiller outerOutputFiller( *TTClusterDSVForOutput, id1 );
      for ( unsigned int m = 0; m < tempOuter->size(); m++ )
      {
        outerOutputFiller.push_back( tempOuter->at(m) );
      }
      if ( outerOutputFiller.empty() )
        outerOutputFiller.abort();
    }

    if ( tempOutput->size() > 0 )
    {
      typename edmNew::DetSetVector< TTStub< T > >::FastFiller tempOutputFiller( *TTStubDSVForOutputTemp, DetId(Id.rawId()) );
      for ( unsigned int m = 0; m < tempOutput->size(); m++ )
      {
        tempOutputFiller.push_back( tempOutput->at(m) );
      }
      if ( tempOutputFiller.empty() )
        tempOutputFiller.abort();
    }

/* NP 2014 02 25
* this is commented to avoid memory exhaustion in hi PU events
if ( tempRejected->size() > 0 )
{
typename edmNew::DetSetVector< TTStub< T > >::FastFiller rejectedOutputFiller( *TTStubDSVForOutputRejected, DetId(Id.rawId()) );
for ( unsigned int m = 0; m < tempRejected->size(); m++ )
{
rejectedOutputFiller.push_back( tempRejected->at(m) );
}
if ( rejectedOutputFiller.empty() )
rejectedOutputFiller.abort();
}
*/

  } /// End of loop over detector elements

  /// Put output in the event (1)
  /// Get also the OrphanHandle of the accepted clusters
  edm::OrphanHandle< edmNew::DetSetVector< TTCluster< T > > > TTClusterAcceptedHandle = iEvent.put( TTClusterDSVForOutput, "ClusterAccepted" );

  /// Now, correctly reset the output
  typename edmNew::DetSetVector< TTStub< T > >::const_iterator stubDetIter;

  for ( stubDetIter = TTStubDSVForOutputTemp->begin();
        stubDetIter != TTStubDSVForOutputTemp->end();
        ++stubDetIter )
  {
    /// Get the DetId and prepare the FastFiller
    DetId thisStackedDetId = stubDetIter->id();
    typename edmNew::DetSetVector< TTStub< T > >::FastFiller acceptedOutputFiller( *TTStubDSVForOutputAccepted, thisStackedDetId );

    /// Get its DetUnit
    const StackedTrackerDetUnit* thisUnit = theStackedTracker->idToStack( thisStackedDetId );
    DetId id0 = thisUnit->stackMember(0);
    DetId id1 = thisUnit->stackMember(1);

    /// Check that everything is ok in the maps
    /// Redundant up to (*)
    if ( theStackedTracker->findPairedDetector( id0 ) != id1 ||
         theStackedTracker->findPairedDetector( id1 ) != id0 )
    {
      std::cerr << "A L E R T! error in detector association within Pt module (detector-to-detector)" << std::endl;
      continue;
    }

    if ( theStackedTracker->findStackFromDetector( id0 ) != thisStackedDetId ||
         theStackedTracker->findStackFromDetector( id1 ) != thisStackedDetId )
    {
      std::cerr << "A L E R T! error in detector association within Pt module (detector-to-module)" << std::endl;
      continue;
    }

    /// Go on only if both detectors have clusters
    if ( TTClusterAcceptedHandle->find( id0 ) == TTClusterAcceptedHandle->end() ||
         TTClusterAcceptedHandle->find( id1 ) == TTClusterAcceptedHandle->end() )
      continue;

    /// (*)

    /// Get the DetSets of the clusters
    edmNew::DetSet< TTCluster< T > > innerClusters = (*TTClusterAcceptedHandle)[ id0 ];
    edmNew::DetSet< TTCluster< T > > outerClusters = (*TTClusterAcceptedHandle)[ id1 ];

    /// Get the DetSet of the stubs
    edmNew::DetSet< TTStub< T > > theseStubs = (*TTStubDSVForOutputTemp)[ thisStackedDetId ];

    /// Prepare the new DetSet to replace the current one
    /// Loop over the stubs
    typename edmNew::DetSet< TTCluster< T > >::iterator clusterIter;
    typename edmNew::DetSet< TTStub< T > >::iterator stubIter;
    for ( stubIter = theseStubs.begin();
          stubIter != theseStubs.end();
          ++stubIter )
    {
      /// Create a temporary stub
      TTStub< T > tempTTStub( stubIter->getDetId() );

      /// Compare the clusters stored in the stub with the ones of this module
      edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > innerClusterToBeReplaced = stubIter->getClusterRef(0);
      edm::Ref< edmNew::DetSetVector< TTCluster< T > >, TTCluster< T > > outerClusterToBeReplaced = stubIter->getClusterRef(1);

      bool innerOK = false;
      bool outerOK = false;

      for ( clusterIter = innerClusters.begin();
            clusterIter != innerClusters.end() && !innerOK;
            ++clusterIter )
      {
        if ( clusterIter->getHits() == innerClusterToBeReplaced->getHits() )
        {
          tempTTStub.addClusterRef( edmNew::makeRefTo( TTClusterAcceptedHandle, clusterIter ) );
          innerOK = true;
        }
      }

      for ( clusterIter = outerClusters.begin();
            clusterIter != outerClusters.end() && !outerOK;
            ++clusterIter )
      {
        if ( clusterIter->getHits() == outerClusterToBeReplaced->getHits() )
        {
          tempTTStub.addClusterRef( edmNew::makeRefTo( TTClusterAcceptedHandle, clusterIter ) );
          outerOK = true;
        }
      }

      /// If no compatible clusters were found, skip to the next one
      if ( !innerOK || !outerOK )
        continue;

      tempTTStub.setTriggerDisplacement( stubIter->getTriggerDisplacement() );
      tempTTStub.setTriggerOffset( stubIter->getTriggerOffset() );

      acceptedOutputFiller.push_back( tempTTStub );

    } /// End of loop over stubs of this module

    if ( acceptedOutputFiller.empty() )
      acceptedOutputFiller.abort();

  } /// End of loop over stub DetSetVector

  /// Put output in the event (2)
  iEvent.put( TTStubDSVForOutputAccepted, "StubAccepted" );
  iEvent.put( TTStubDSVForOutputRejected, "StubRejected" );
}

/// Sort routine for stub ordering
template< typename T >
bool TTStubBuilder< T >::SortStubBendPairs( const std::pair< unsigned int, double >& left, const std::pair< unsigned int, double >& right )
{
  return left.second < right.second;
}

#endif
