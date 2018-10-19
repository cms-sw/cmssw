//
// Original Author:  Yetkin Yilmaz, Young Soo Park
//         Created:  Wed Jun 11 15:31:41 CEST 2008
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

using namespace std;
using namespace edm;
using namespace reco;

//
// class declaration
//

namespace reco {
class CentralityProducer : public edm::EDProducer {
   public:
      explicit CentralityProducer(const edm::ParameterSet&);
      ~CentralityProducer() override;

   private:
      void beginJob() override ;
      void produce(edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      
      // ----------member data ---------------------------

   bool recoLevel_;

   bool produceHFhits_;
   bool produceHFtowers_;
   bool produceEcalhits_;
   bool produceZDChits_;
   bool produceETmidRap_;
   bool producePixelhits_;
   bool produceTracks_;
   bool reuseAny_;
   bool producePixelTracks_;

  bool doPixelCut_;

  double midRapidityRange_;
  double trackPtCut_;
  double trackEtaCut_;
  double hfEtaCut_;

  bool lowGainZDC_;

  edm::EDGetTokenT<HFRecHitCollection>  srcHFhits_;	
  edm::EDGetTokenT<CaloTowerCollection>  srcTowers_;
  edm::EDGetTokenT<EcalRecHitCollection> srcEEhits_;
  edm::EDGetTokenT<EcalRecHitCollection> srcEBhits_;
  edm::EDGetTokenT<ZDCRecHitCollection> srcZDChits_;
  edm::EDGetTokenT<SiPixelRecHitCollection> srcPixelhits_;
  edm::EDGetTokenT<TrackCollection> srcTracks_;
  edm::EDGetTokenT<TrackCollection> srcPixelTracks_;
  edm::EDGetTokenT<VertexCollection> srcVertex_;
  edm::EDGetTokenT<Centrality> reuseTag_;
  
  bool useQuality_;
  reco::TrackBase::TrackQuality trackQuality_;

  edm::ESHandle<TrackerGeometry> tGeo;
  edm::ESHandle<CaloGeometry> cGeo;
  edm::ESHandle<TrackerTopology> topo_;

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
  CentralityProducer::CentralityProducer(const edm::ParameterSet& iConfig) {
   //register your products
   produceHFhits_ = iConfig.getParameter<bool>("produceHFhits");
   produceHFtowers_ = iConfig.getParameter<bool>("produceHFtowers");
   produceEcalhits_ = iConfig.getParameter<bool>("produceEcalhits");
   produceZDChits_ = iConfig.getParameter<bool>("produceZDChits");
   produceETmidRap_ = iConfig.getParameter<bool>("produceETmidRapidity");
   producePixelhits_ = iConfig.getParameter<bool>("producePixelhits");
   produceTracks_ = iConfig.getParameter<bool>("produceTracks");
   producePixelTracks_ = iConfig.getParameter<bool>("producePixelTracks");

   midRapidityRange_ = iConfig.getParameter<double>("midRapidityRange");
   trackPtCut_ = iConfig.getParameter<double>("trackPtCut");
   trackEtaCut_ = iConfig.getParameter<double>("trackEtaCut");

   hfEtaCut_ = iConfig.getParameter<double>("hfEtaCut");

   if(produceHFhits_)  srcHFhits_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("srcHFhits"));
   if(produceHFtowers_ || produceETmidRap_) srcTowers_ = consumes<CaloTowerCollection>(iConfig.getParameter<edm::InputTag>("srcTowers"));

   if(produceEcalhits_){
     srcEBhits_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("srcEBhits"));
     srcEEhits_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("srcEEhits"));
   }
   if(produceZDChits_){
     srcZDChits_ = consumes<ZDCRecHitCollection>(iConfig.getParameter<edm::InputTag>("srcZDChits"));
     lowGainZDC_ = iConfig.getParameter<bool>("lowGainZDC");
   }
   if(producePixelhits_){
     srcPixelhits_ = consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("srcPixelhits"));
     doPixelCut_ = iConfig.getParameter<bool>("doPixelCut");
     srcVertex_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("srcVertex"));
   }
   if(produceTracks_) {
     srcTracks_ = consumes<TrackCollection>(iConfig.getParameter<edm::InputTag>("srcTracks"));
     srcVertex_ = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("srcVertex"));
   }
   if(producePixelTracks_) srcPixelTracks_ = consumes<TrackCollection>(iConfig.getParameter<edm::InputTag>("srcPixelTracks"));
   
   reuseAny_ = iConfig.getParameter<bool>("reUseCentrality");
   if(reuseAny_) reuseTag_ = consumes<Centrality>(iConfig.getParameter<edm::InputTag>("srcReUse"));

   useQuality_   = iConfig.getParameter<bool>("UseQuality");
   trackQuality_ = TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));

   produces<Centrality>();
   
}


CentralityProducer::~CentralityProducer()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CentralityProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if(producePixelhits_) iSetup.get<TrackerDigiGeometryRecord>().get(tGeo);
  if(produceEcalhits_) iSetup.get<CaloGeometryRecord>().get(cGeo);
  if(producePixelhits_) iSetup.get<TrackerTopologyRcd>().get(topo_);

  auto creco = std::make_unique<Centrality>();
  Handle<Centrality> inputCentrality;

  if(reuseAny_) iEvent.getByToken(reuseTag_,inputCentrality);
  
  if(produceHFhits_){
     creco->etHFhitSumPlus_ = 0;
     creco->etHFhitSumMinus_ = 0;

     Handle<HFRecHitCollection> hits;
     iEvent.getByToken(srcHFhits_,hits);
     for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
	const HFRecHit & rechit = (*hits)[ ihit ];
        if(rechit.id().ieta() > 0 )
	   creco->etHFhitSumPlus_ += rechit.energy();
        if(rechit.id().ieta() < 0)
	   creco->etHFhitSumMinus_ += rechit.energy();
     }       
  }else{
    if(reuseAny_){
     creco->etHFhitSumMinus_ = inputCentrality->EtHFhitSumMinus();
     creco->etHFhitSumPlus_ = inputCentrality->EtHFhitSumPlus();
    }
  }
  
  if(produceHFtowers_ || produceETmidRap_){
     creco->etHFtowerSumPlus_ = 0;
     creco->etHFtowerSumMinus_ = 0;
     creco->etHFtowerSumECutPlus_ = 0;
     creco->etHFtowerSumECutMinus_ = 0;
     creco->etMidRapiditySum_ = 0;
     
     Handle<CaloTowerCollection> towers;

     iEvent.getByToken(srcTowers_,towers);

	for( size_t i = 0; i<towers->size(); ++ i){
	   const CaloTower & tower = (*towers)[ i ];
	   double eta = tower.eta();
	   bool isHF = tower.ietaAbs() > 29;
	   if(produceHFtowers_){
	      if(isHF && eta > 0){
		 creco->etHFtowerSumPlus_ += tower.pt();
         if(tower.energy() > 1.5) creco->etHFtowerSumECutPlus_ += tower.pt();
		 if(eta > hfEtaCut_) creco->etHFtruncatedPlus_ += tower.pt();
	      }
	      if(isHF && eta < 0){
		 creco->etHFtowerSumMinus_ += tower.pt();
         if(tower.energy() > 1.5) creco->etHFtowerSumECutMinus_ += tower.pt();
		 if(eta < -hfEtaCut_) creco->etHFtruncatedMinus_ += tower.pt();
	      }
	   }else{
    	     if(reuseAny_){
	      creco->etHFtowerSumMinus_ = inputCentrality->EtHFtowerSumMinus();
	      creco->etHFtowerSumPlus_ = inputCentrality->EtHFtowerSumPlus();
             creco->etHFtowerSumECutMinus_ = inputCentrality->EtHFtowerSumECutMinus();
             creco->etHFtowerSumECutPlus_ = inputCentrality->EtHFtowerSumECutPlus();
	      creco->etHFtruncatedMinus_ = inputCentrality->EtHFtruncatedMinus();
	      creco->etHFtruncatedPlus_ = inputCentrality->EtHFtruncatedPlus();
    	     }
	   }
	   if(produceETmidRap_){
	      if(std::abs(eta) < midRapidityRange_) creco->etMidRapiditySum_ += tower.pt()/(midRapidityRange_*2.);
	   }else if(reuseAny_) creco->etMidRapiditySum_ = inputCentrality->EtMidRapiditySum();
	}
  }else{
    if(reuseAny_){
     creco->etHFtowerSumMinus_ = inputCentrality->EtHFtowerSumMinus();
     creco->etHFtowerSumPlus_ = inputCentrality->EtHFtowerSumPlus();
     creco->etHFtowerSumECutMinus_ = inputCentrality->EtHFtowerSumECutMinus();
     creco->etHFtowerSumECutPlus_ = inputCentrality->EtHFtowerSumECutPlus();
     creco->etMidRapiditySum_ = inputCentrality->EtMidRapiditySum();
    }
  }
  
  if(produceEcalhits_){
     creco->etEESumPlus_ = 0;
     creco->etEESumMinus_ = 0;
     creco->etEBSum_ = 0;

     Handle<EcalRecHitCollection> ebHits;
     Handle<EcalRecHitCollection> eeHits;

     iEvent.getByToken(srcEBhits_,ebHits);
     iEvent.getByToken(srcEEhits_,eeHits);

     for(unsigned int i = 0; i < ebHits->size(); ++i){
        const EcalRecHit & hit= (*ebHits)[i];
        const GlobalPoint& pos=cGeo->getPosition(hit.id());
        double et = hit.energy()*(pos.perp()/pos.mag());
        creco->etEBSum_ += et;
     }

     for(unsigned int i = 0; i < eeHits->size(); ++i){
        const EcalRecHit & hit= (*eeHits)[i];
        const GlobalPoint& pos=cGeo->getPosition(hit.id());
        double et = hit.energy()*(pos.perp()/pos.mag());
        if(pos.z() > 0){
           creco->etEESumPlus_ += et;
        }else{
           creco->etEESumMinus_ += et;
        }
     }
  }else{
    if(reuseAny_){
      creco->etEESumMinus_ = inputCentrality->EtEESumMinus();
      creco->etEESumPlus_ = inputCentrality->EtEESumPlus();
      creco->etEBSum_ = inputCentrality->EtEBSum();
    }
  }
  
  if(producePixelhits_){
     creco->pixelMultiplicity_ = 0;
     const SiPixelRecHitCollection* rechits;
     Handle<SiPixelRecHitCollection> rchts;
     iEvent.getByToken(srcPixelhits_,rchts);
     rechits = rchts.product();
     int nPixel =0 ;
     int nPixel_plus =0 ;
     int nPixel_minus =0 ;
     for (SiPixelRecHitCollection::const_iterator it = rechits->begin(); it!=rechits->end();it++)
     {
        SiPixelRecHitCollection::DetSet hits = *it;
        DetId detId = DetId(hits.detId());
        SiPixelRecHitCollection::const_iterator recHitMatch = rechits->find(detId);
        const SiPixelRecHitCollection::DetSet recHitRange = *recHitMatch;
        for ( SiPixelRecHitCollection::DetSet::const_iterator recHitIterator = recHitRange.begin(); 
	      recHitIterator != recHitRange.end(); ++recHitIterator) {

	  // add selection if needed, now all hits.
	    const SiPixelRecHit * recHit = &(*recHitIterator);
	    const PixelGeomDetUnit* pixelLayer = dynamic_cast<const PixelGeomDetUnit*> (tGeo->idToDet(recHit->geographicalId()));
	    GlobalPoint gpos = pixelLayer->toGlobal(recHit->localPosition());
	    math::XYZVector rechitPos(gpos.x(),gpos.y(),gpos.z());
	    double eta = rechitPos.eta();
	    double abeta = std::abs(rechitPos.eta());
	    int clusterSize = recHit->cluster()->size();
           unsigned layer = topo_->layer(detId);
	  if(doPixelCut_){
          if (detId.det() == DetId::Tracker && detId.subdetId() == PixelSubdetector::PixelBarrel){
            if (layer==1 && 18*abeta-40>clusterSize) continue;
            if (layer==2 && 6*abeta-7.2>clusterSize) continue;
            if ((layer==3 || layer==4) && 4*abeta-2.4>clusterSize) continue;
          }
	  }
	  nPixel++;
         if(eta>=0) nPixel_plus++;
         if(eta<0) nPixel_minus++;
      
        } 
     }
     creco->pixelMultiplicity_ = nPixel;
     creco->pixelMultiplicityPlus_ = nPixel_plus;
     creco->pixelMultiplicityMinus_ = nPixel_minus;
  }else{
    if(reuseAny_){
     creco->pixelMultiplicity_ = inputCentrality->multiplicityPixel();
     creco->pixelMultiplicityPlus_ = inputCentrality->multiplicityPixelPlus();
     creco->pixelMultiplicityMinus_ = inputCentrality->multiplicityPixelMinus();
    }
  }

  if(produceTracks_){

     double vx=-999.;
     double vy=-999.;
     double vz=-999.;
     double vxError=-999.;
     double vyError=-999.;
     double vzError=-999.;
     math::XYZVector vtxPos(0,0,0);

     Handle<VertexCollection> recoVertices;
     iEvent.getByToken(srcVertex_,recoVertices);
     unsigned int daughter = 0;
     int greatestvtx = 0;
    
     for (unsigned int i = 0 ; i< recoVertices->size(); ++i){
      daughter = (*recoVertices)[i].tracksSize();
      if( daughter > (*recoVertices)[greatestvtx].tracksSize()) greatestvtx = i;
     }

     if(!recoVertices->empty()){
      vx = (*recoVertices)[greatestvtx].position().x();
      vy = (*recoVertices)[greatestvtx].position().y();
      vz = (*recoVertices)[greatestvtx].position().z();
      vxError = (*recoVertices)[greatestvtx].xError();
      vyError = (*recoVertices)[greatestvtx].yError();
      vzError = (*recoVertices)[greatestvtx].zError();
     }
    
     vtxPos = math::XYZVector(vx,vy,vz);

     Handle<TrackCollection> tracks;
     iEvent.getByToken(srcTracks_,tracks);
     int nTracks = 0;

     double trackCounter = 0;
     double trackCounterEta = 0;
     double trackCounterEtaPt = 0;

     for(unsigned int i = 0 ; i < tracks->size(); ++i){
       const Track& track = (*tracks)[i];
       if(useQuality_ && !track.quality(trackQuality_)) continue;

       if( track.pt() > trackPtCut_)  trackCounter++;
       if(std::abs(track.eta())<trackEtaCut_) {
	 trackCounterEta++;
	 if (track.pt() > trackPtCut_) trackCounterEtaPt++;
       }

       math::XYZPoint v1(vx,vy, vz);    
       double dz= track.dz(v1);
       double dzsigma2 = track.dzError()*track.dzError()+vzError*vzError;    
       double dxy= track.dxy(v1);
       double dxysigma2 = track.dxyError()*track.dxyError()+vxError*vyError;

       const double pterrcut = 0.1;
       const double dzrelcut = 3.0;
       const double dxyrelcut = 3.0;

       if( track.quality(trackQuality_) && 
	   track.pt()>0.4 && std::abs(track.eta())<2.4 && 
	   track.ptError()/track.pt() < pterrcut &&
	   dz*dz < dzrelcut*dzrelcut * dzsigma2 && 
	   dxy*dxy < dxyrelcut*dxyrelcut * dxysigma2 ){
	 nTracks++;
       }
     }

     creco->trackMultiplicity_ = nTracks;
     creco->ntracksPtCut_ = trackCounter; 
     creco->ntracksEtaCut_ = trackCounterEta;
     creco->ntracksEtaPtCut_ = trackCounterEtaPt;

  }else{
    if(reuseAny_){
     creco->trackMultiplicity_ = inputCentrality->Ntracks();
     creco->ntracksPtCut_= inputCentrality->NtracksPtCut();
     creco->ntracksEtaCut_= inputCentrality->NtracksEtaCut();
     creco->ntracksEtaPtCut_= inputCentrality->NtracksEtaPtCut();
    }
  }

  if(producePixelTracks_){
    Handle<TrackCollection> pixeltracks;
    iEvent.getByToken(srcPixelTracks_,pixeltracks);
    int nPixelTracks = pixeltracks->size();
    int nPixelTracksPlus = 0;
    int nPixelTracksMinus = 0;

    for (auto const& track : *pixeltracks){
        if(track.eta()<0) nPixelTracksMinus++;
        else nPixelTracksPlus++;
    }
    creco->nPixelTracks_ = nPixelTracks;
    creco->nPixelTracksPlus_ = nPixelTracksPlus;
    creco->nPixelTracksMinus_ = nPixelTracksMinus;
  }
  else{
    if(reuseAny_){
     creco->nPixelTracks_ = inputCentrality->NpixelTracks();
     creco->nPixelTracksPlus_ = inputCentrality->NpixelTracksPlus();
     creco->nPixelTracksMinus_ = inputCentrality->NpixelTracksMinus();
    }
  }

  if(produceZDChits_){
     creco->zdcSumPlus_ = 0;
     creco->zdcSumMinus_ = 0;
     
     Handle<ZDCRecHitCollection> hits;
     bool zdcAvailable = iEvent.getByToken(srcZDChits_,hits);
     if(zdcAvailable){
	for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
	   const ZDCRecHit & rechit = (*hits)[ ihit ];
	   if(rechit.id().zside() > 0 ) {
	     if(lowGainZDC_){
	       creco->zdcSumPlus_ += rechit.lowGainEnergy();
	     }else{
	       creco->zdcSumPlus_ += rechit.energy();
	     }
	   }
	   if(rechit.id().zside() < 0) {
             if(lowGainZDC_){
	      creco->zdcSumMinus_ += rechit.lowGainEnergy();
	     }else{
	       creco->zdcSumMinus_ += rechit.energy();
	     }
           }
	}
     }else{
	creco->zdcSumPlus_ = -9;
	creco->zdcSumMinus_ = -9;
     }
  }else{
    if(reuseAny_){
     creco->zdcSumMinus_ = inputCentrality->zdcSumMinus();
     creco->zdcSumPlus_ = inputCentrality->zdcSumPlus();
    }
  }
  
  iEvent.put(std::move(creco));

}

// ------------ method called once each job just before starting event loop  ------------
void 
CentralityProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CentralityProducer::endJob()
{
}


//define this as a plug-in
DEFINE_FWK_MODULE(CentralityProducer);

}
