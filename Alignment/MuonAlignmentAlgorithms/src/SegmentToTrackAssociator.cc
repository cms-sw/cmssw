/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/07/16 07:14:42 $
 *  $Revision: 1.3 $
 *  \author P. Martinez - IFCA
 */


#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"



#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Alignment/MuonAlignmentAlgorithms/interface/SegmentToTrackAssociator.h"


#include <vector>

SegmentToTrackAssociator::SegmentToTrackAssociator( const edm::ParameterSet& iConfig )
{

  theDTSegmentLabel = iConfig.getParameter<edm::InputTag>( "segmentsDT" );
  theCSCSegmentLabel = iConfig.getParameter<edm::InputTag>( "segmentsCSC" );
    
}


SegmentToTrackAssociator::~SegmentToTrackAssociator() {}


void SegmentToTrackAssociator::clear()
{
  indexCollectionDT.clear();
  indexCollectionCSC.clear();
}

MuonTransientTrackingRecHit::MuonRecHitContainer
  SegmentToTrackAssociator::associate( const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track, std::string TrackRefitterType)
{

  // The segment collections
  edm::Handle<DTRecSegment4DCollection> DTSegments;
  iEvent.getByLabel(theDTSegmentLabel, DTSegments); 

  edm::Handle<CSCSegmentCollection> CSCSegments;
  iEvent.getByLabel(theCSCSegmentLabel, CSCSegments);

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get( theTrackingGeometry );
	  
  MuonTransientTrackingRecHit::MuonRecHitContainer SelectedSegments;
	  
  DTRecSegment4DCollection::const_iterator segmentDT;
  CSCSegmentCollection::const_iterator segmentCSC;
	  
  std::vector<int> positionDT;
  std::vector<int> positionCSC;
  std::vector<TrackingRecHit *> my4DTrack;
	  
  //Loop over the hits of the track
  for( unsigned int counter = 0; counter != track.recHitsSize()-1; counter++ )
  {
	    
    TrackingRecHitRef myRef = track.recHit( counter );
    
    const TrackingRecHit *rechit = myRef.get();
    const GeomDet* geomDet = theTrackingGeometry->idToDet( rechit->geographicalId() );
	    
    //It's a DT Hit
    if( geomDet->subDetector() == GeomDetEnumerators::DT )
    {
	    
      //Take the layer associated to this hit
      DTLayerId myLayer( rechit->geographicalId().rawId() );
	      
      int NumberOfDTSegment = 0;
      //Loop over segments
      for( segmentDT = DTSegments->begin(); segmentDT != DTSegments->end(); ++segmentDT ) {
	
	//By default the chamber associated to this Segment is new
	bool isNewChamber = true;
		
	//Loop over segments already included in the vector of segments in the actual track
	for( std::vector<int>::iterator positionIt = positionDT.begin();
	    positionIt != positionDT.end(); positionIt++ )
	{
	  //If this segment has been used before isNewChamber = false
	  if(NumberOfDTSegment == *positionIt) isNewChamber = false;
	}
	
	//Loop over vectors of segments associated to previous tracks
	for( std::vector<std::vector<int> >::iterator collect = indexCollectionDT.begin();
	    collect != indexCollectionDT.end(); ++collect)
	{
	  //Loop over segments associated to a track
	  for( std::vector<int>::iterator positionIt = (*collect).begin();
	      positionIt != (*collect).end(); positionIt++ )
	  {
	    //If this segment was used in a previos track then isNewChamber = false
	    if( NumberOfDTSegment == *positionIt ) isNewChamber = false;
	  }
	}
	
	//If the chamber is new
	if( isNewChamber )
	{
	  DTChamberId myChamber( (*segmentDT).geographicalId().rawId() );
	  //If the layer of the hit belongs to the chamber of the 4D Segment
	  if( myLayer.wheel() == myChamber.wheel() &&
	     myLayer.station() == myChamber.station() &&
	     myLayer.sector() == myChamber.sector() )
	  {
	    //push position of the segment and tracking rechit
	    positionDT.push_back( NumberOfDTSegment );
	    const GeomDet* DTgeomDet = theTrackingGeometry->idToDet( myChamber );
	    SelectedSegments.push_back( MuonTransientTrackingRecHit::specificBuild( DTgeomDet, (TrackingRecHit *) &*segmentDT ) );

	    //edm::LogWarning("Alignment") << "TagSeg: " << "NumberOfDTSegment " << NumberOfDTSegment << " Wheel " << myChamber.wheel() << " Sector " <<  myChamber.sector() << " Chamber " << myChamber.station() << std::endl;

	  }
	}
	NumberOfDTSegment++;
      }
      //In case is a CSC
    }
    else if ( geomDet->subDetector() == GeomDetEnumerators::CSC )
    {
      
      //Take the layer associated to this hit
      CSCDetId myLayer( rechit->geographicalId().rawId() );
      
      int NumberOfCSCSegment = 0;
      //Loop over 4Dsegments
      for( segmentCSC = CSCSegments->begin(); segmentCSC != CSCSegments->end(); segmentCSC++ )
      {
	
	//By default the chamber associated to the segment is new
	bool isNewChamber = true;
	//Loop over segments in the current track
	for( std::vector<int>::iterator positionIt = positionCSC.begin();
	    positionIt != positionCSC.end(); positionIt++ )
	{
	  //If this segment has been used then newchamber = false
	  if( NumberOfCSCSegment == *positionIt ) isNewChamber = false;
	}
	//Loop over vectors of segments in previous tracks
	for( std::vector<std::vector<int> >::iterator collect = indexCollectionCSC.begin();
	    collect != indexCollectionCSC.end(); ++collect )
	{
	  //Loop over segments in a track
	  for( std::vector<int>::iterator positionIt = (*collect).begin();
	      positionIt != (*collect).end(); positionIt++ )
	  {
	    //If the segment was used in a previous track isNewChamber = false
	    if( NumberOfCSCSegment == *positionIt ) isNewChamber = false;
	  }
	}
	//If the chamber is new
	if( isNewChamber )
	{
	  CSCDetId myChamber( (*segmentCSC).geographicalId().rawId() );
	  //If the chambers are the same
	  if( myLayer.chamberId() == myChamber.chamberId() )
	  {
	    //push
	    positionCSC.push_back( NumberOfCSCSegment );
	    const GeomDet* CSCgeomDet = theTrackingGeometry->idToDet( myChamber );
	    SelectedSegments.push_back(MuonTransientTrackingRecHit::specificBuild( CSCgeomDet, (TrackingRecHit *) &*segmentCSC ) );
	  }
	}
	NumberOfCSCSegment++;
      }
    }
  }
  
  indexCollectionDT.push_back(positionDT);
  indexCollectionCSC.push_back(positionCSC);


  if ( TrackRefitterType == "CosmicLike" )
  {
    std::reverse(SelectedSegments.begin(),SelectedSegments.end());
  }
 
  return SelectedSegments;

}

