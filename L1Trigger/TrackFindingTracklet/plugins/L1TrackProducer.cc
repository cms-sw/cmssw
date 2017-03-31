//////////////////////////
//  Producer by Anders  //
//     and Emmanuele    //
//    july 2012 @ CU    //
//////////////////////////


#ifndef L1TTRACK_PRDC_H
#define L1TTRACK_PRDC_H

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
//
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 
#include "DataFormats/Common/interface/DetSetVector.h"

#include "L1Trigger/TrackFindingTracklet/interface/slhcevent.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TBarrel.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TDisk.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.hh"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
//
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
////////////////////////
// FAST SIMULATION STUFF
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPtConsistency.h"

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

//////////////
// NAMESPACES
using namespace edm;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

/////////////////////////////////////
// this class is needed to make a map
// between different types of stubs
struct L1TStubCompare 
{
public:
  bool operator()(const L1TStub& x, const L1TStub& y) const {
    if (x.layer() != y.layer()) return (y.layer()>x.layer());
    else {
      if (x.ladder() != y.ladder()) return (y.ladder()>x.ladder());
      else {
	if (x.module() != y.module()) return (y.module()>x.module());
	else {
	  if (x.iz() != y.iz()) return (y.iz()>x.iz());
	  else return (x.iphi()>y.iphi());
	}
      }
    }
  }
};


class L1TrackProducer : public edm::EDProducer
{
public:

  /// Constructor/destructor
  explicit L1TrackProducer(const edm::ParameterSet& iConfig);
  virtual ~L1TrackProducer();

protected:
                     
private:

  int eventnum;

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  double phiWindowSF_;

  string asciiEventOutName_;
  std::ofstream asciiEventOut_;


  edm::ESHandle<TrackerTopology> tTopoHandle;
  edm::ESHandle<TrackerGeometry> tGeomHandle;

  edm::InputTag simTrackSrc_;
  edm::InputTag simVertexSrc_;
  edm::InputTag ttStubSrc_;
  edm::InputTag ttStubMCTruthSrc_;

  edm::InputTag bsSrc_;
  
  const edm::EDGetTokenT< edm::SimTrackContainer > simTrackToken_;
  const edm::EDGetTokenT< edm::SimVertexContainer > simVertexToken_;
  
  const edm::EDGetTokenT< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > > ttStubToken_;
  const edm::EDGetTokenT< TTStubAssociationMap< Ref_Phase2TrackerDigi_ > > ttStubMCTruthToken_;

  const edm::EDGetTokenT< reco::BeamSpot > bsToken_;
  

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
};


//////////////
// CONSTRUCTOR
L1TrackProducer::L1TrackProducer(edm::ParameterSet const& iConfig) :
  config(iConfig),
  simTrackSrc_(config.getParameter<edm::InputTag>("SimTrackSource")),
  simVertexSrc_(config.getParameter<edm::InputTag>("SimVertexSource")),
  ttStubSrc_(config.getParameter<edm::InputTag>("TTStubSource")),
  ttStubMCTruthSrc_(config.getParameter<edm::InputTag>("TTStubMCTruthSource")),
  bsSrc_(config.getParameter<edm::InputTag>("BeamSpotSource")),

  simTrackToken_(consumes< edm::SimTrackContainer >(simTrackSrc_)),
  simVertexToken_(consumes< edm::SimVertexContainer >(simVertexSrc_)),
  ttStubToken_(consumes< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > >(ttStubSrc_)),
  ttStubMCTruthToken_(consumes< TTStubAssociationMap< Ref_Phase2TrackerDigi_ > >(ttStubMCTruthSrc_)),
  bsToken_(consumes< reco::BeamSpot >(bsSrc_))

{

  produces< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >( "Level1TTTracks" ).setBranchAlias("Level1TTTracks");

  phiWindowSF_ = iConfig.getUntrackedParameter<double>("phiWindowSF",1.0);

  asciiEventOutName_ = iConfig.getUntrackedParameter<string>("asciiFileName","");

  eventnum=0;
  if (asciiEventOutName_!="") {
    asciiEventOut_.open(asciiEventOutName_.c_str());
  }

}

/////////////
// DESTRUCTOR
L1TrackProducer::~L1TrackProducer()
{
  if (asciiEventOutName_!="") {
    asciiEventOut_.close();
  }

}  

//////////
// END JOB
void L1TrackProducer::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
}

////////////
// BEGIN JOB
void L1TrackProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup )
{
}

//////////
// PRODUCE
void L1TrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef std::map< L1TStub, edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ >  >, L1TStubCompare > stubMapType;


  /// Prepare output
  std::unique_ptr< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > L1TkTracksForOutput( new std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > );

  stubMapType stubMap;

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  const TrackerGeometry*                                       theGeometry;


  /// Set pointers to Stacked Modules
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);


  ////////////////////////
  // GET MAGNETIC FIELD //

  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();

  ////////////
  // GET BS //
  ////////////
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByToken( bsToken_, beamSpotHandle );
  math::XYZPoint bsPosition=beamSpotHandle->position();

  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);

  SLHCEvent ev;
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());
  eventnum++;


  ///////////////////
  // GET SIMTRACKS //
  edm::Handle<edm::SimTrackContainer>   simTrackHandle;
  edm::Handle<edm::SimVertexContainer>  simVtxHandle;
  iEvent.getByToken( simTrackToken_, simTrackHandle );
  iEvent.getByToken( simVertexToken_, simVtxHandle );


  const TrackerTopology* const tTopo = tTopoHandle.product();
  const TrackerGeometry* const theTrackerGeom = tGeomHandle.product();


  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > >    Phase2TrackerDigiTTStubHandle;
  iEvent.getByToken( ttStubToken_,        Phase2TrackerDigiTTStubHandle );

  edm::Handle< TTStubAssociationMap< Ref_Phase2TrackerDigi_ > >    MCTruthTTStubHandle;
  iEvent.getByToken( ttStubMCTruthToken_,        MCTruthTTStubHandle );


  ////////////////////////
  /// LOOP OVER SimTracks
  SimTrackContainer::const_iterator iterSimTracks;
  for ( iterSimTracks = simTrackHandle->begin();
	iterSimTracks != simTrackHandle->end();
	++iterSimTracks ) {

    /// Get the corresponding vertex
    int vertexIndex = iterSimTracks->vertIndex();
    const SimVertex& theSimVertex = (*simVtxHandle)[vertexIndex];
    math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();
    GlobalPoint trkVtxCorr = GlobalPoint( trkVtxPos.x() - bsPosition.x(), 
					  trkVtxPos.y() - bsPosition.y(), 
					  trkVtxPos.z() - bsPosition.z() );
    
    double pt=iterSimTracks->momentum().pt();
    if (pt!=pt) pt=9999.999;
    ev.addL1SimTrack(iterSimTracks->trackId(),iterSimTracks->type(),pt,
		     iterSimTracks->momentum().eta(), 
		     iterSimTracks->momentum().phi(), 
		     trkVtxCorr.x(),
		     trkVtxCorr.y(),
		     trkVtxCorr.z());
       
  } /// End of Loop over SimTrack



  ////////////////////////////////
  /// COLLECT STUB INFORMATION ///
  ////////////////////////////////

  /// Maps to store TrackingParticle information
  std::map< unsigned int, std::vector< edm::Ptr< TrackingParticle > > > tpPerStubLayer;
  std::map< unsigned int, std::vector< edm::Ptr< TrackingParticle > > > tpPerStubDisk;


  // loop over stubs
  for (auto gd=theTrackerGeom->dets().begin(); gd != theTrackerGeom->dets().end(); gd++) {
    
    DetId detid = (*gd)->geographicalId();
    if(detid.subdetId()!=StripSubdetector::TOB && detid.subdetId()!=StripSubdetector::TID ) continue; // only run on OT
    if(!tTopo->isLower(detid) ) continue; // loop on the stacks: choose the lower arbitrarily
    DetId stackDetid = tTopo->stack(detid); // Stub module detid

    if (Phase2TrackerDigiTTStubHandle->find( stackDetid ) == Phase2TrackerDigiTTStubHandle->end() ) continue;

    // Get the DetSets of the Clusters
    edmNew::DetSet< TTStub< Ref_Phase2TrackerDigi_ > > stubs = (*Phase2TrackerDigiTTStubHandle)[ stackDetid ];
    const GeomDetUnit* det0 = theTrackerGeom->idToDetUnit( detid );
    const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >( det0 );
    const PixelTopology* topol = dynamic_cast< const PixelTopology* >( &(theGeomDet->specificTopology()) );
    
    // loop over stubs
    for ( auto stubIter = stubs.begin();stubIter != stubs.end();++stubIter ) {
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_  > >, TTStub< Ref_Phase2TrackerDigi_  > >
	tempStubPtr = edmNew::makeRefTo( Phase2TrackerDigiTTStubHandle, stubIter );
      
      MeasurementPoint coords = tempStubPtr->getClusterRef(0)->findAverageLocalCoordinatesCentered();      
      LocalPoint clustlp = topol->localPosition(coords);

      GlobalPoint posStub  =  theGeomDet->surface().toGlobal(clustlp);
      //double displStub    = tempStubPtr->getTriggerDisplacement();

      int layer=-999999;
      int ladder=-999999;
      int module=-999999;

      int strip=460;
      int simtrack=-1;

      double z=posStub.z();

      if ( detid.subdetId()==StripSubdetector::TOB ) {
	layer  = static_cast<int>(tTopo->layer(detid));
        module = static_cast<int>(tTopo->module(detid));
	ladder = static_cast<int>(tTopo->tobRod(detid));
	if (layer==1)
	  {
	    if (z<-15.0) {
	      module = static_cast<int>(tTopo->tobRod(detid));
	      ladder = static_cast<int>(tTopo->module(detid));
	    }
	    if (z>15.0) {
	      module = 18+static_cast<int>(tTopo->tobRod(detid));
	      ladder = static_cast<int>(tTopo->module(detid));
	    }
	    if (fabs(z)<15.0)  module = 11+static_cast<int>(tTopo->module(detid));
	  }

	if (layer==2)
	  {
	    if (z<-25.0) {
	      module = static_cast<int>(tTopo->tobRod(detid));
	      ladder = static_cast<int>(tTopo->module(detid));
	    }
	    if (z>25.0)  {
	      module = 23+static_cast<int>(tTopo->tobRod(detid));
	      ladder = static_cast<int>(tTopo->module(detid));
	    }
	    if (fabs(z)<25.0)  module = 12+static_cast<int>(tTopo->module(detid));
	  }

	if (layer==3)
	  {
	    if (z<-34.0) {
	      module = static_cast<int>(tTopo->tobRod(detid));
	      ladder = static_cast<int>(tTopo->module(detid));
	    }
	    if (z>34.0)  {
	      module = 28+static_cast<int>(tTopo->tobRod(detid));
	      ladder = static_cast<int>(tTopo->module(detid));
	    }
	    if (fabs(z)<34.0)  module = 13+static_cast<int>(tTopo->module(detid));
	  } 
      }
      else if ( detid.subdetId()==StripSubdetector::TID ) {
	layer  = 1000+static_cast<int>(tTopo->tidRing(detid));
	ladder =  static_cast<int>(tTopo->module(detid)); 
	module = static_cast<int>(tTopo->tidWheel(detid));
      }

      // clusters
      std::vector<bool> innerStack;
      std::vector<int> irphi;
      std::vector<int> iz;
      std::vector<int> iladder;
      std::vector<int> imodule;

      /// Get the Inner and Outer TTCluster
      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_Phase2TrackerDigi_> >, TTCluster<Ref_Phase2TrackerDigi_> > innerCluster = tempStubPtr->getClusterRef(0);

      const DetId innerDetId = innerCluster->getDetId();

      std::vector< int > innerrows= innerCluster->getRows();
      std::vector< int > innercols= innerCluster->getCols();

      for (unsigned int ihit=0;ihit<innerrows.size();ihit++){

	if (layer<1000) {
	  innerStack.push_back(true);
	  irphi.push_back(innerrows[ihit]);
	  iz.push_back(innercols[ihit]);
	  iladder.push_back(static_cast<int>(tTopo->tobRod(innerDetId)));
	  imodule.push_back(static_cast<int>(tTopo->module(innerDetId)));
	}
	else {
	  innerStack.push_back(true);
	  irphi.push_back(innerrows[ihit]);
	  iz.push_back(innercols[ihit]);
	  iladder.push_back(static_cast<int>(tTopo->tobRod(innerDetId)));
	  imodule.push_back(static_cast<int>(tTopo->module(innerDetId)));
	}    
      }


      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_Phase2TrackerDigi_> >, TTCluster<Ref_Phase2TrackerDigi_> > outerCluster = tempStubPtr->getClusterRef(1);

      const DetId outerDetId = outerCluster->getDetId();

      std::vector< int > outerrows= outerCluster->getRows();
      std::vector< int > outercols= outerCluster->getCols();

      for (unsigned int ihit=0;ihit<outerrows.size();ihit++){

	if (layer<1000) {
	  innerStack.push_back(true);
	  irphi.push_back(outerrows[ihit]);
	  iz.push_back(outercols[ihit]);
	  iladder.push_back(static_cast<int>(tTopo->tobRod(outerDetId)));
	  imodule.push_back(static_cast<int>(tTopo->module(outerDetId)));
	}
	else {
	  innerStack.push_back(true);
	  irphi.push_back(outerrows[ihit]);
	  iz.push_back(outercols[ihit]);
	  iladder.push_back(static_cast<int>(tTopo->tobRod(outerDetId)));
	  imodule.push_back(static_cast<int>(tTopo->module(outerDetId)));
	}    
      }

      
      if (irphi.size()!=0) {
      	strip=irphi[0];
      }

      if (ev.addStub(layer,ladder,module,strip,-1,tempStubPtr->getTriggerBend(),
		     posStub.x(),posStub.y(),posStub.z(),
		     innerStack,irphi,iz,iladder,imodule)) {
		
	L1TStub lastStub=ev.lastStub();
	stubMap[lastStub]=tempStubPtr;
      }
    
    }
  }

  //std::cout << "Will actually do L1 tracking:"<<std::endl;

  //////////////////////////
  // NOW RUN THE L1 tracking

  if (asciiEventOutName_!="") {
    ev.write(asciiEventOut_);
  }

#include "L1Tracking.icc"  
  
  for (unsigned itrack=0; itrack<purgedTracks.size(); itrack++) {
    L1TTrack track=purgedTracks.get(itrack);

    TTTrack<Ref_Phase2TrackerDigi_> aTrack;

    aTrack.setSector(999); //this is currently not retrained by the algorithm
    aTrack.setWedge(999); //not used by the tracklet implementations

    //First do the 4 parameter fit
    GlobalPoint bsPosition4par(0.0,0.0,track.z04par());
    aTrack.setPOCA(bsPosition4par,4);

    double pt4par=fabs(track.pt4par(mMagneticFieldStrength));
    
    GlobalVector p34par(GlobalVector::Cylindrical(pt4par, 
						  track.phi04par(), 
						  pt4par*sinh(track.eta4par())));

    aTrack.setMomentum(p34par,4);
    aTrack.setRInv(track.rinv4par(),4);
    aTrack.setChi2(track.chisq4par(),4);


    //Now do the 5 parameter fit
    GlobalPoint bsPosition5par(-track.d0()*sin(track.phi0()),track.d0()*cos(track.phi0()),track.z0());
    aTrack.setPOCA(bsPosition5par,5);
 
    double pt5par=fabs(track.pt(mMagneticFieldStrength));
    
    GlobalVector p35par(GlobalVector::Cylindrical(pt5par, 
						  track.phi0(), 
						  pt5par*sinh(track.eta())));

    aTrack.setMomentum(p35par,5);
    aTrack.setRInv(track.rinv(),5);
    aTrack.setChi2(track.chisq(),5);
        
    vector<L1TStub> stubs = track.getStubs();

    stubMapType::const_iterator it;
    for (vector<L1TStub>::const_iterator itstubs = stubs.begin(); 
	 itstubs != stubs.end(); itstubs++) {
      it=stubMap.find(*itstubs);
      if (it!=stubMap.end()) {
	aTrack.addStubRef(it->second);
      }
      else{
	cout << "Could not find stub in stub map"<<endl;
	cout << "stub:"<<itstubs->layer()<<" "
	     <<itstubs->ladder()<<" "
	     <<itstubs->module()<<" "
	     <<itstubs->iz()<<" "
	     <<itstubs->iphi()<<endl;

      }
    }

    // pt consistency
    //float consistency4par = StubPtConsistency::getConsistency(aTrack, theStackedGeometry, mMagneticFieldStrength, 4); 
    //aTrack.setStubPtConsistency(consistency4par, 4);
    aTrack.setStubPtConsistency(-1, 4);

    //float consistency5par = StubPtConsistency::getConsistency(aTrack, theStackedGeometry, mMagneticFieldStrength, 5); 
    //aTrack.setStubPtConsistency(consistency5par,5);
    aTrack.setStubPtConsistency(-1,5);


    L1TkTracksForOutput->push_back(aTrack);

  }


  iEvent.put( std::move(L1TkTracksForOutput), "Level1TTTracks");


} /// End of produce()


// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackProducer);

#endif
