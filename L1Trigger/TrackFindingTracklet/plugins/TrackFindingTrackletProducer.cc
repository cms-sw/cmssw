//////////////////////////
//  Producer by Anders  //
//     and Emmanuele    //
//    july 2012 @ CU    //
//////////////////////////

#ifndef L1TTRACK_PRDC_H
#define L1TTRACK_PRDC_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
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

#include "L1Trigger/TrackFindingTracklet/interface/slhcevent.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TBarrel.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TConstants.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TDisk.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TTrack.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TTracklet.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TTracklets.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TTracks.hh"
#include "L1Trigger/TrackFindingTracklet/interface/L1TWord.hh"

#include <memory>
#include <string>
#include <iostream>
#include <fstream>

// this class is needed to make a map
// between different types of stubs
class L1TStubCompare
{
public:
  bool operator()(const L1TStub& x, const L1TStub& y) {
    if (x.layer() != y.layer()) return (y.layer()-x.layer())>0;
    else {
      if (x.ladder() != y.ladder()) return (y.ladder()-x.ladder())>0;
      else {
  if (x.module() != y.module()) return (y.module()-x.module())>0;
  else {
    if (x.iz() != y.iz()) return (y.iz()-x.iz())>0;
    else return (x.iphi()-y.iphi())>0;
  }
      }
    }
  }
};


class TrackFindingTrackletProducer : public edm::EDProducer
{
public:
  /// Constructor/destructor
  explicit TrackFindingTrackletProducer(const edm::ParameterSet& iConfig);
  virtual ~TrackFindingTrackletProducer();

protected:

private:
  GeometryMap geom;
  int eventnum;
  edm::ParameterSet config;
  string geometry_;

  virtual void beginRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
};


// CONSTRUCTOR
TrackFindingTrackletProducer::TrackFindingTrackletProducer(edm::ParameterSet const& iConfig) // :   config(iConfig)
{
  produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( "TrackletBasedL1Tracks" ).setBranchAlias("TrackletBasedL1Tracks");
  geometry_ = iConfig.getUntrackedParameter<string>("geometry","");
}

// DESTRUCTOR
TrackFindingTrackletProducer::~TrackFindingTrackletProducer()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}

// END JOB
void TrackFindingTrackletProducer::endRun(edm::Run& run, const edm::EventSetup& iSetup)
{
  /// Things to be done at the exit of the event Loop
}

// BEGIN JOB
void TrackFindingTrackletProducer::beginRun(edm::Run& run, const edm::EventSetup& iSetup )
{
  eventnum=0;
  //std::cout << "TrackFindingTrackletProducer" << std::endl;
}

//////////
// PRODUCE
void TrackFindingTrackletProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef std::map< L1TStub, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > >, L1TStubCompare > stubMapType;

  /// Prepare output
  std::auto_ptr< std::vector< TTTrack< Ref_PixelDigi_ > > > TTTracksForOutput( new std::vector< TTTrack< Ref_PixelDigi_ > > );

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                  geometryHandle;
  const TrackerGeometry*                          theGeometry;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  theGeometry = &(*geometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product(); /// Note this is different
                                                        /// from the "global" geometry

  // Magnetic field
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();

  /// Beamspot
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel("BeamSpotFromSim","BeamSpot",recoBeamSpotHandle);
  math::XYZPoint bsPosition=recoBeamSpotHandle->position();

  //cout << "TrackFindingTrackletProducer: B="<<mMagneticFieldStrength
  //     <<" vx reco="<<bsPosition.x()
  //     <<" vy reco="<<bsPosition.y()
  //     <<" vz reco="<<bsPosition.z()
  //     <<endl;

  SLHCEvent ev;
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());
  eventnum++;

/*
  //cout << "Get simtracks"<<endl;

  ///////////////////
  // GET SIMTRACKS //
  edm::Handle<edm::SimTrackContainer>   simTrackHandle;
  edm::Handle<edm::SimVertexContainer>  simVtxHandle;
  //iEvent.getByLabel( "famosSimHits", simTrackHandle );
  //iEvent.getByLabel( "famosSimHits", simVtxHandle );
  iEvent.getByLabel( "g4SimHits", simTrackHandle );
  iEvent.getByLabel( "g4SimHits", simVtxHandle );

  //////////////////////
  // GET MC PARTICLES //
  edm::Handle<reco::GenParticleCollection> genpHandle;
  iEvent.getByLabel( "genParticles", genpHandle );
*/

  //cout << "Get pixel digis"<<endl;

  // GET PIXEL DIGIS
  edm::Handle<edm::DetSetVector<PixelDigi> >         pixelDigiHandle;
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  pixelDigiSimLinkHandle;
  iEvent.getByLabel("simSiPixelDigis", pixelDigiHandle);
  iEvent.getByLabel("simSiPixelDigis", pixelDigiSimLinkHandle);

  //cout << "Get stubs and clusters"<<endl;

  // GET THE PRIMITIVES
//  edm::Handle<std::vector< TTCluster< Ref_PixelDigi_ > > >  pixelDigiTTClusterHandle;
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > >     pixelDigiTTStubHandle;
//  iEvent.getByLabel("TTClustersFromPixelDigis", pixelDigiTTClusterHandle);
  iEvent.getByLabel("TTStubsFromPixelDigis", "StubAccepted", pixelDigiTTStubHandle);

  //cout << "Will loop over simtracks" <<endl;

/*
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
   
    
  } /// End of Loop over SimTracks
*/

  /// TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingVertexHandle );

  /// Loop over TrackingParticles
  if ( TrackingParticleHandle->size() > 0 )
  {
    unsigned int tpCnt = 0;
    std::vector< TrackingParticle >::const_iterator iterTP;
    for ( iterTP = TrackingParticleHandle->begin();
          iterTP != TrackingParticleHandle->end();
          ++iterTP )
    {

      /// Make the pointer
      edm::Ptr< TrackingParticle > tempTPPtr( TrackingParticleHandle, tpCnt++ );

      std::vector<SimTrack>::const_iterator iterSimTracks;

      for ( iterSimTracks = iterTP->g4Tracks().begin();
            iterSimTracks != iterTP->g4Tracks().end();
            ++iterSimTracks )
      {

        /// Get the corresponding vertex
        GlobalPoint trkVtxCorr = GlobalPoint( tempTPPtr->vertex().x() - bsPosition.x(),
                                              tempTPPtr->vertex().y() - bsPosition.y(),
                                              tempTPPtr->vertex().z() - bsPosition.z() );

        double pt=iterSimTracks->momentum().pt();
        if (pt!=pt) pt=9999.999;

        ev.addL1SimTrack(iterSimTracks->trackId(),iterSimTracks->type(),pt,
                         iterSimTracks->momentum().eta(),
                         iterSimTracks->momentum().phi(),
                         trkVtxCorr.x(),
                         trkVtxCorr.y(),
                         trkVtxCorr.z());
      }
    } /// End of Loop over SimTracks
  }

  //std::cout << "Will loop over digis:"<<std::endl;

  edm::DetSetVector<PixelDigi>::const_iterator iterDet;
  for ( iterDet = pixelDigiHandle->begin();
        iterDet != pixelDigiHandle->end();
        iterDet++ ) {

    /// Build Detector Id
    DetId tkId( iterDet->id );
    StackedTrackerDetId stdetid(tkId);
    /// Check if it is Pixel
    if ( tkId.subdetId() == 2 ) {

      PXFDetId pxfId(tkId);
      edm::DetSetVector<PixelDigiSimLink>::const_iterator itDigiSimLink1=pixelDigiSimLinkHandle->find(pxfId.rawId());
      if (itDigiSimLink1!=pixelDigiSimLinkHandle->end()){
        edm::DetSet<PixelDigiSimLink> digiSimLink = *itDigiSimLink1;
        //DetSet<PixelDigiSimLink> digiSimLink = (*pixelDigiSimLinkHandle)[ pxfId.rawId() ];
        edm::DetSet<PixelDigiSimLink>::const_iterator iterSimLink;
        /// Renormalize layer number from 5-14 to 0-9 and skip if inner pixels

        int disk = pxfId.disk();

        if (disk<4) {
          continue;
        }

        disk-=3;

        // Layer 0-20
        //DetId digiDetId = iterDet->id;
        //int sensorLayer = 0.5*(2*PXFDetId(digiDetId).layer() + (PXFDetId(digiDetId).ladder() + 1)%2 - 8);

        /// Loop over PixelDigis within Module and select those above threshold
        edm::DetSet<PixelDigi>::const_iterator iterDigi;
        for ( iterDigi = iterDet->data.begin();
              iterDigi != iterDet->data.end();
              iterDigi++ ) {

          /// Threshold (here it is NOT redundant)
          if ( iterDigi->adc() <= 30 ) continue;

          /// Try to learn something from PixelDigi position
          const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( tkId );
          MeasurementPoint mp( iterDigi->row() + 0.5, iterDigi->column() + 0.5 );
          GlobalPoint pdPos = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;

          int offset=1000;

          if (pxfId.side()==1) {
            offset=2000;
          }

          assert(pxfId.panel()==1);

          vector<int> simtrackids;
          /// Loop over PixelDigiSimLink to find the
          /// correct link to the SimTrack collection
          for ( iterSimLink = digiSimLink.data.begin();
                iterSimLink != digiSimLink.data.end();
                iterSimLink++) {

          /// When the channel is the same, the link is found
          if ( (int)iterSimLink->channel() == iterDigi->channel() &&
               iterSimLink->eventId().event()==0 &&
               iterSimLink->eventId().bunchCrossing()==0
              ) {

            /// Map wrt SimTrack Id
            unsigned int simTrackId = iterSimLink->SimTrackId();
            simtrackids.push_back(simTrackId);
          }
        }
        ev.addDigi(offset+disk,iterDigi->row(),iterDigi->column(),
                   9999999,pxfId.panel(),pxfId.module(),
                   pdPos.x(),pdPos.y(),pdPos.z(),simtrackids);
        }
      }
    }

    if ( tkId.subdetId() == 1 ) {
      /// Get the PixelDigiSimLink corresponding to this one
      PXBDetId pxbId(tkId);
      edm::DetSetVector<PixelDigiSimLink>::const_iterator itDigiSimLink=pixelDigiSimLinkHandle->find(pxbId.rawId());
      if (itDigiSimLink==pixelDigiSimLinkHandle->end()){
        continue;
      }
      edm::DetSet<PixelDigiSimLink> digiSimLink = *itDigiSimLink;
      //DetSet<PixelDigiSimLink> digiSimLink = (*pixelDigiSimLinkHandle)[ pxbId.rawId() ];
      edm::DetSet<PixelDigiSimLink>::const_iterator iterSimLink;
      /// Renormalize layer number from 5-14 to 0-9 and skip if inner pixels
      if ( pxbId.layer() < 5 ) {
        continue;
      }

      // Layer 0-20
      DetId digiDetId = iterDet->id;
      int sensorLayer = 0.5*(2*PXBDetId(digiDetId).layer() + (PXBDetId(digiDetId).ladder() + 1)%2 - 8);

      /// Loop over PixelDigis within Module and select those above threshold
      edm::DetSet<PixelDigi>::const_iterator iterDigi;
      for ( iterDigi = iterDet->data.begin();
            iterDigi != iterDet->data.end();
            iterDigi++ ) {

        /// Threshold (here it is NOT redundant)
        if ( iterDigi->adc() <= 30 ) continue;

        /// Try to learn something from PixelDigi position
        const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( tkId );
        MeasurementPoint mp( iterDigi->row() + 0.5, iterDigi->column() + 0.5 );
        GlobalPoint pdPos = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;

        /// Loop over PixelDigiSimLink to find the
        /// correct link to the SimTrack collection
        vector<int > simtrackids;
        for ( iterSimLink = digiSimLink.data.begin();
              iterSimLink != digiSimLink.data.end();
              iterSimLink++) {

          /// When the channel is the same, the link is found
          if ( (int)iterSimLink->channel() == iterDigi->channel() &&
               iterSimLink->eventId().event()==0 &&
               iterSimLink->eventId().bunchCrossing()==0
             ) {

            /// Map wrt SimTrack Id
            unsigned int simTrackId = iterSimLink->SimTrackId();
            simtrackids.push_back(simTrackId);
          }
        }
        ev.addDigi(sensorLayer,iterDigi->row(),iterDigi->column(),
                   pxbId.layer(),pxbId.ladder(),pxbId.module(),
                   pdPos.x(),pdPos.y(),pdPos.z(),simtrackids);
      }
    }
  }

  //cout << "Will loop over stubs" << endl;

  stubMapType stubMap;
  int iter=0;

  int stubcounter=0;

  /// Loop over L1TkStubs
  edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator iterDSV;
  edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator iterTTStub;
  for ( iterDSV = pixelDigiTTStubHandle->begin();
        iterDSV != pixelDigiTTStubHandle->end();
        ++iterDSV )
  {
    DetId thisStackedDetId = iterDSV->id();

    edmNew::DetSet< TTStub< Ref_PixelDigi_ > > theStubs = (*pixelDigiTTStubHandle)[ thisStackedDetId ];

    for ( iterTTStub = theStubs.begin();
          iterTTStub != theStubs.end();
          ++iterTTStub )
    {
      stubcounter++;

      double stubPt = theStackedGeometry->findRoughPt(mMagneticFieldStrength,&(*iterTTStub));
 
      if (stubPt>10000.0) stubPt=9999.99;
      GlobalPoint stubPosition = theStackedGeometry->findGlobalPosition(&(*iterTTStub));

      StackedTrackerDetId stubDetId = iterTTStub->getDetId();
      unsigned int iStack = stubDetId.iLayer();
      unsigned int iRing = stubDetId.iRing();
      unsigned int iPhi = stubDetId.iPhi();
      unsigned int iZ = stubDetId.iZ();

      std::vector<bool> innerStack;
      std::vector<int> irphi;
      std::vector<int> iz;
      std::vector<int> iladder;
      std::vector<int> imodule;

      if (iStack==999999) {
        iStack=1000+iRing;
      }

      /// Get the Inner and Outer TTCluster
      edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > innerCluster = iterTTStub->getClusterRef(0);

      const DetId innerDetId = theStackedGeometry->idToDet( innerCluster->getDetId(), 0 )->geographicalId();

      for (unsigned int ihit=0;ihit<innerCluster->getHits().size();ihit++){

        std::pair<int,int> rowcol=
          std::make_pair( innerCluster->getRows().at( ihit), innerCluster->getCols().at(ihit) );

        if (iStack<1000) {
          innerStack.push_back(true);
          irphi.push_back(rowcol.first);
          iz.push_back(rowcol.second);
          iladder.push_back(PXBDetId(innerDetId).ladder());
          imodule.push_back(PXBDetId(innerDetId).module());
        }
        else {
          innerStack.push_back(true);
          irphi.push_back(rowcol.first);
          iz.push_back(rowcol.second);
          iladder.push_back(PXFDetId(innerDetId).disk());
          imodule.push_back(PXFDetId(innerDetId).module());
        }
      }

      edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > outerCluster = iterTTStub->getClusterRef(1);

      const DetId outerDetId = theStackedGeometry->idToDet( outerCluster->getDetId(), 1 )->geographicalId();

      for (unsigned int ihit=0;ihit<outerCluster->getHits().size();ihit++){

        std::pair<int,int> rowcol=
          std::make_pair( outerCluster->getRows().at( ihit), outerCluster->getCols().at(ihit) );

        if (iStack<1000) {
          innerStack.push_back(false);
          irphi.push_back(rowcol.first);
          iz.push_back(rowcol.second);
          iladder.push_back(PXBDetId(outerDetId).ladder());
          imodule.push_back(PXBDetId(outerDetId).module());
        }
        else {
          innerStack.push_back(false);
          irphi.push_back(rowcol.first);
          iz.push_back(rowcol.second);
          iladder.push_back(PXFDetId(outerDetId).disk());
          imodule.push_back(PXFDetId(outerDetId).module());
        }
      }

      if (ev.addStub(iStack-1,iPhi,iZ,stubPt,
          stubPosition.x(),stubPosition.y(),stubPosition.z(),
          innerStack,irphi,iz,iladder,imodule)) {
        Stub *aStub = new Stub;
        *aStub = ev.stub(iter);
        iter++;

        //int theSimtrackId=ev.simtrackid(*aStub);
        int theSimtrackId=-1;

        L1TStub L1Stub(theSimtrackId, aStub->iphi(), aStub->iz(),
                       aStub->layer()+1, aStub->ladder()+1, aStub->module(),
                       aStub->x(), aStub->y(), aStub->z(),0.0,0.0,aStub->pt());
        delete aStub;

        stubMap.insert( make_pair(L1Stub, edmNew::makeRefTo( pixelDigiTTStubHandle, iterTTStub ) ) );
      }
    }
  }

  cout << "TrackFindingTrackletProducer: "<<stubcounter<<endl;

  //std::cout << "Will actually do L1 tracking:"<<std::endl;


  //////////////////////////
  // NOW RUN THE L1 tracking


  int mode = 0;

  // mode means:
  // 1 LB_6PS
  // 2 LB_4PS_2SS
  // 3 BE
  // 4 BE5D

  //cout << "geometry:"<<geometry_<<endl;

  if (geometry_=="LB_6PS") mode=1;
  if (geometry_=="LB_4PS_2SS") mode=2;
  if (geometry_=="BE") mode=3;
  if (geometry_=="BE5D") mode=4;


  assert(mode==1||mode==2||mode==3||mode==4);

#include "L1Tracking.icc"

  for (unsigned itrack=0; itrack<purgedTracks.size(); itrack++) {
    L1TTrack track=purgedTracks.get(itrack);

    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > TkStubs;
    std::vector< L1TStub > stubs = track.getStubs();

    stubMapType::iterator it;
    //cout << "stubmap size="<<stubMap.size()<<" "<<stubs.size()<<endl;
    for (it = stubMap.begin(); it != stubMap.end(); it++) {
      for (int j=0; j<(int)stubs.size(); j++) {
        if (it->first == stubs[j]) {
          //cout << "Found stub match"<<endl;
          TkStubs.push_back(it->second);
        }
      }
    }

    TTTrack< Ref_PixelDigi_ > TkTrack(TkStubs);
    //double frac;
    //TkTrack.setSimTrackId(track.simtrackid(frac));
    GlobalPoint bsPosition(0.0,0.0,track.z0()); //store the L1 track vertex position 
    TkTrack.setPOCA(bsPosition);  
    TkTrack.setChi2(track.chisq());
    TkTrack.setFitParNo(4);
    //short int charge=1;
    //if (track.pt(mMagneticFieldStrength)<0.0) charge=-1;
    //TkTrack.setCharge(charge);
    TkTrack.setRInv(track.rinv());

    // set simtrack ID (??) **this doesn't work, re-introduced get/set simtrack ID methods**
    //if (iEvent.isRealData() == false) TkTrack.checkSimTrack();

    TkTrack.setMomentum( GlobalVector ( GlobalVector::Cylindrical( fabs(track.pt(mMagneticFieldStrength)), 
                                                                   track.phi0(), 
                                                                   fabs(track.pt(mMagneticFieldStrength))*sinh(track.eta())) ) );

    TTTracksForOutput->push_back(TkTrack);
  }

  iEvent.put( TTTracksForOutput, "TrackletBasedL1Tracks");

} /// End of produce()

// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(TrackFindingTrackletProducer);

#endif

