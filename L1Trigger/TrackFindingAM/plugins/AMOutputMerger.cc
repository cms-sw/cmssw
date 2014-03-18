/*! \class   AMOutputMerger
 *
 *
 *  \update by S.Viret 
 *  \date   2014, Feb 17
 *
 */

#ifndef AM_MERGER_H
#define AM_MERGER_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

class AMOutputMerger : public edm::EDProducer
{
  public:
    /// Constructor
    explicit AMOutputMerger( const edm::ParameterSet& iConfig );

    /// Destructor;
    ~AMOutputMerger();

  private:

  /// Data members
  double                        mMagneticField;
  const StackedTrackerGeometry  *theStackedTracker;
  edm::InputTag                 TTClustersInputTag;
  edm::InputTag                 TTStubsInputTag;
  std::string                   TTStubOutputTag;
  std::vector< edm::InputTag >  TTPatternsInputTags;
  std::string                   TTPatternOutputTag;
  std::vector<int>              stored_IDs;

  /// Mandatory methods
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );

  bool inPattern(int j);
}; /// Close class

/*! \brief   Implementation of methods
 */

/// Constructors
AMOutputMerger::AMOutputMerger( const edm::ParameterSet& iConfig )
{
  TTClustersInputTag  = iConfig.getParameter< edm::InputTag >( "TTInputClusters" );
  TTStubsInputTag     = iConfig.getParameter< edm::InputTag >( "TTInputStubs" );
  TTPatternsInputTags = iConfig.getParameter< std::vector< edm::InputTag > >( "TTInputPatterns" );

  TTStubOutputTag     = iConfig.getParameter< std::string >( "TTFiltStubsName" );
  TTPatternOutputTag  = iConfig.getParameter< std::string >( "TTPatternsName" );

  produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( TTPatternOutputTag );
  produces<  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > >( TTStubOutputTag );
}

/// Destructor
AMOutputMerger::~AMOutputMerger() {}

/// Begin run
void AMOutputMerger::beginRun( const edm::Run& run, const edm::EventSetup& iSetup )
{
  /// Get the geometry references
  edm::ESHandle< StackedTrackerGeometry > StackedTrackerGeomHandle;
  iSetup.get< StackedTrackerGeometryRecord >().get( StackedTrackerGeomHandle );
  theStackedTracker = StackedTrackerGeomHandle.product();

  /// Get magnetic field
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  mMagneticField = (floor(mMagneticFieldStrength*10.0 + 0.5))/10.0;
}

/// End run
void AMOutputMerger::endRun( const edm::Run& run, const edm::EventSetup& iSetup ) {}

/// Implement the producer
void AMOutputMerger::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  /// Prepare output

  /// Get the Stubs/Cluster already stored
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  iEvent.getByLabel( TTStubsInputTag, TTStubHandle );

  edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > TTClusterHandle;
  iEvent.getByLabel( TTClustersInputTag, TTClusterHandle );

  // The container for filtered patterns / stubs 

  std::auto_ptr< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTracksForOutput( new std::vector< TTTrack< Ref_PixelDigi_ > > );
  std::auto_ptr< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubsForOutput( new edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > );

  std::vector< edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > > TTPatternHandle;

  TTPatternHandle.clear();
  TTPatternHandle.resize(static_cast<int>(TTPatternsInputTags.size()));

  for ( unsigned int m = 0; m < TTPatternsInputTags.size(); m++ )
    iEvent.getByLabel( TTPatternsInputTags.at(m), TTPatternHandle.at(m) );


  //
  // First step is pretty simple, we just create a map of all the clusters 
  // contained in the event
  //

  unsigned int stub_n = 0;
  std::map< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > , unsigned int > stubMap;

  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator inputIter;
  edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator stubIter;

  for ( inputIter = TTStubHandle->begin(); inputIter != TTStubHandle->end(); ++inputIter )
  {
    for ( stubIter = inputIter->begin(); stubIter != inputIter->end(); ++stubIter )
    {
      ++stub_n;

      /// Make the Ref to be put in the Track
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = makeRefTo( TTStubHandle, stubIter );

      stubMap.insert( std::make_pair( tempStubRef, stub_n ) );
    }
  }

  //
  // In the second step, we merge all the patterns into a single containe
  // because they are stored in vectors
  //
 
  std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterTTTrack;

  for ( unsigned j = 0; j < TTPatternsInputTags.size(); ++j )
  {
    edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTPatterns = TTPatternHandle.at(j);

    if ( TTPatterns->size() > 0 )
    {      
      for ( iterTTTrack = TTPatterns->begin();
	    iterTTTrack != TTPatterns->end();
	    ++iterTTTrack )
      {

	TTTrack< Ref_PixelDigi_ > tempTTPatt(iterTTTrack->getStubRefs());
	
	tempTTPatt.setSector(iterTTTrack->getSector());
	tempTTPatt.setWedge(iterTTTrack->getWedge());
	tempTTPatt.setMomentum(iterTTTrack->getMomentum());
	tempTTPatt.setPOCA(iterTTTrack->getPOCA());

	TTTracksForOutput->push_back(tempTTPatt);
      }
    }
  }

  // Get the OrphanHandle of the accepted patterns
  
  edm::OrphanHandle< std::vector< TTTrack< Ref_PixelDigi_ > > > TTPatternAcceptedHandle = iEvent.put( TTTracksForOutput, TTPatternOutputTag );
 

  //
  // Third step, we flag the stubs contained in the patterns stored
  // 
  //

  bool found;

 
  stored_IDs.clear();

  if ( TTPatternAcceptedHandle->size() > 0 )
  {
    /// Loop over Patterns
    unsigned int tkCnt = 0;

    for ( iterTTTrack = TTPatternAcceptedHandle->begin();
	  iterTTTrack != TTPatternAcceptedHandle->end();
	  ++iterTTTrack )
    {
      edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr( TTPatternAcceptedHandle, tkCnt++ );

      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_  > >, TTStub< Ref_PixelDigi_  > > > trackStubs = tempTrackPtr->getStubRefs();

      // Loop over stubs contained in the pattern to recover the info

      for(unsigned int i=0;i<trackStubs.size();i++)
      {
	found=false;

	for(unsigned int l = 0; l < stored_IDs.size(); ++l )
	{
	  if (found) continue;
	  if (stored_IDs.at(l)==int(stubMap[ trackStubs.at(i) ])) found=true;
	}

	if (!found) stored_IDs.push_back(stubMap[ trackStubs.at(i) ]);

      }
    }
  }

  //
  // Last step, we recreate the filtered stub container from there
  // 
  //

  unsigned int j2 = 0;

  for ( inputIter = TTStubHandle->begin(); inputIter != TTStubHandle->end(); ++inputIter )
  {
    DetId thisStackedDetId = inputIter->id();

    /// Get its DetUnit

    const StackedTrackerDetUnit* thisUnit = theStackedTracker->idToStack( thisStackedDetId );

    DetId id0 = thisUnit->stackMember(0);
    DetId id1 = thisUnit->stackMember(1);
 
    /// Check that everything is ok in the maps
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

    /// Go on only if both detectors have Clusters
    if ( TTClusterHandle->find( id0 ) == TTClusterHandle->end() ||
         TTClusterHandle->find( id1 ) == TTClusterHandle->end() )
      continue;

    /// Create the vector of stubs to be passed to the FastFiller
    std::vector< TTStub< Ref_PixelDigi_ > > *tempOutput = new std::vector< TTStub< Ref_PixelDigi_ > >();
    tempOutput->clear();

    for ( stubIter = inputIter->begin(); stubIter != inputIter->end(); ++stubIter )
    {
      ++j2;

      if (!AMOutputMerger::inPattern(j2)) continue;

      TTStub< Ref_PixelDigi_ > tempTTStub( stubIter->getDetId() );

      tempTTStub.addClusterRef(stubIter->getClusterRef(0));
      tempTTStub.addClusterRef(stubIter->getClusterRef(1));
      tempTTStub.setTriggerDisplacement( stubIter->getTriggerDisplacement() );
      tempTTStub.setTriggerOffset( stubIter->getTriggerOffset() );
      tempOutput->push_back( tempTTStub );
    }

    /// Create the FastFiller

    if ( tempOutput->size() > 0 )
    {
      typename edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::FastFiller tempOutputFiller( *TTStubsForOutput, thisStackedDetId );
      for ( unsigned int m = 0; m < tempOutput->size(); m++ )
      {
        tempOutputFiller.push_back( tempOutput->at(m) );
      }
      if ( tempOutputFiller.empty() )
        tempOutputFiller.abort();
    }
  }

  /// Put in the event content
  iEvent.put( TTStubsForOutput, TTStubOutputTag);  
}



bool AMOutputMerger::inPattern(int j)
{
  for(unsigned l = 0; l < stored_IDs.size(); ++l )
  {    
    if (stored_IDs.at(l)==j) return true;
  }

  return false;
}

// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(AMOutputMerger);

#endif

