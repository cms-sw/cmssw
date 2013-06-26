// -*- C++ -*-
//
// Package:    CentralityProducer
// Class:      CentralityProducer
// 
/**\class CentralityProducer CentralityProducer.cc RecoHI/CentralityProducer/src/CentralityProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz, Young Soo Park
//         Created:  Wed Jun 11 15:31:41 CEST 2008
// $Id: CentralityProducer.cc,v 1.40 2012/12/26 23:38:37 wmtan Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
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

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

using namespace std;

//
// class declaration
//
namespace reco{

class CentralityProducer : public edm::EDFilter {
   public:
      explicit CentralityProducer(const edm::ParameterSet&);
      ~CentralityProducer();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

   bool recoLevel_;

   bool doFilter_;
   bool produceHFhits_;
   bool produceHFtowers_;
   bool produceEcalhits_;
   bool produceBasicClusters_;
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

   edm::InputTag  srcHFhits_;	
   edm::InputTag  srcTowers_;
   edm::InputTag srcEEhits_;
   edm::InputTag srcEBhits_;
   edm::InputTag srcBasicClustersEE_;
   edm::InputTag srcBasicClustersEB_;
   edm::InputTag srcZDChits_;
   edm::InputTag srcPixelhits_;
   edm::InputTag srcTracks_;
   edm::InputTag srcPixelTracks_;

   edm::InputTag reuseTag_;

  bool useQuality_;
  reco::TrackBase::TrackQuality trackQuality_;

  const TrackerGeometry* trackGeo_;
  const CaloGeometry* caloGeo_;

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
  CentralityProducer::CentralityProducer(const edm::ParameterSet& iConfig) :
    trackGeo_(0),
    caloGeo_(0)
{
   //register your products
   doFilter_ = iConfig.getParameter<bool>("doFilter");
   produceHFhits_ = iConfig.getParameter<bool>("produceHFhits");
   produceHFtowers_ = iConfig.getParameter<bool>("produceHFtowers");
   produceBasicClusters_ = iConfig.getParameter<bool>("produceBasicClusters");
   produceEcalhits_ = iConfig.getParameter<bool>("produceEcalhits");
   produceZDChits_ = iConfig.getParameter<bool>("produceZDChits");
   produceETmidRap_ = iConfig.getParameter<bool>("produceETmidRapidity");
   producePixelhits_ = iConfig.getParameter<bool>("producePixelhits");
   produceTracks_ = iConfig.getParameter<bool>("produceTracks");
   producePixelTracks_ = iConfig.getParameter<bool>("producePixelTracks");

   midRapidityRange_ = iConfig.getParameter<double>("midRapidityRange");
   trackPtCut_ = iConfig.getParameter<double>("trackPtCut");
   trackEtaCut_ = iConfig.getParameter<double>("trackEtaCut");

   if(produceHFhits_)  srcHFhits_ = iConfig.getParameter<edm::InputTag>("srcHFhits");
   if(produceHFtowers_ || produceETmidRap_) srcTowers_ = iConfig.getParameter<edm::InputTag>("srcTowers");

   if(produceEcalhits_){
      srcEBhits_ = iConfig.getParameter<edm::InputTag>("srcEBhits");
      srcEEhits_ = iConfig.getParameter<edm::InputTag>("srcEEhits");
   }
   if(produceBasicClusters_){
      srcBasicClustersEE_ = iConfig.getParameter<edm::InputTag>("srcBasicClustersEE");
      srcBasicClustersEB_ = iConfig.getParameter<edm::InputTag>("srcBasicClustersEB");
   }
   if(produceZDChits_) srcZDChits_ = iConfig.getParameter<edm::InputTag>("srcZDChits");
   if(producePixelhits_){
     srcPixelhits_ = iConfig.getParameter<edm::InputTag>("srcPixelhits");
     doPixelCut_ = iConfig.getParameter<bool>("doPixelCut");
   }
   if(produceTracks_) srcTracks_ = iConfig.getParameter<edm::InputTag>("srcTracks");
   if(producePixelTracks_) srcPixelTracks_ = iConfig.getParameter<edm::InputTag>("srcPixelTracks");
   
   reuseAny_ = !produceHFhits_ || !produceHFtowers_ || !produceBasicClusters_ || !produceEcalhits_ || !produceZDChits_;
   if(reuseAny_) reuseTag_ = iConfig.getParameter<edm::InputTag>("srcReUse");

   useQuality_   = iConfig.getParameter<bool>("UseQuality");
   trackQuality_ = TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));

   produces<reco::Centrality>();
   
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
bool
CentralityProducer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  if(!trackGeo_ && doPixelCut_){
    edm::ESHandle<TrackerGeometry> tGeo;
    iSetup.get<TrackerDigiGeometryRecord>().get(tGeo);
    trackGeo_ = tGeo.product();
  }
  
  if(!caloGeo_ && 0){
    edm::ESHandle<CaloGeometry> cGeo;
    iSetup.get<CaloGeometryRecord>().get(cGeo);
    caloGeo_ = cGeo.product();
  }

  std::auto_ptr<Centrality> creco(new Centrality());
  Handle<Centrality> inputCentrality;

  if(reuseAny_) iEvent.getByLabel(reuseTag_,inputCentrality);
  
  if(produceHFhits_){
     creco->etHFhitSumPlus_ = 0;
     creco->etHFhitSumMinus_ = 0;

     Handle<HFRecHitCollection> hits;
     iEvent.getByLabel(srcHFhits_,hits);
     for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
	const HFRecHit & rechit = (*hits)[ ihit ];
        if(rechit.id().ieta() > 0 )
	   creco->etHFhitSumPlus_ += rechit.energy();
        if(rechit.id().ieta() < 0)
	   creco->etHFhitSumMinus_ += rechit.energy();
     }       
  }else{
     creco->etHFhitSumMinus_ = inputCentrality->EtHFhitSumMinus();
     creco->etHFhitSumPlus_ = inputCentrality->EtHFhitSumPlus();
  }
  
  if(produceHFtowers_ || produceETmidRap_){
     creco->etHFtowerSumPlus_ = 0;
     creco->etHFtowerSumMinus_ = 0;
     creco->etMidRapiditySum_ = 0;
     
     Handle<CaloTowerCollection> towers;

     iEvent.getByLabel(srcTowers_,towers);

	for( size_t i = 0; i<towers->size(); ++ i){
	   const CaloTower & tower = (*towers)[ i ];
	   double eta = tower.eta();
	   bool isHF = tower.ietaAbs() > 29;
	   if(produceHFtowers_){
	      if(isHF && eta > 0){
		 creco->etHFtowerSumPlus_ += tower.pt();
	      }
	      if(isHF && eta < 0){
		 creco->etHFtowerSumMinus_ += tower.pt();
	      }
	   }else{
	      creco->etHFtowerSumMinus_ = inputCentrality->EtHFtowerSumMinus();
	      creco->etHFtowerSumPlus_ = inputCentrality->EtHFtowerSumPlus();
	   }
	   if(produceETmidRap_){
	      if(fabs(eta) < midRapidityRange_) creco->etMidRapiditySum_ += tower.pt()/(midRapidityRange_*2.);
	   }else creco->etMidRapiditySum_ = inputCentrality->EtMidRapiditySum();
	}
  }else{
     creco->etHFtowerSumMinus_ = inputCentrality->EtHFtowerSumMinus();
     creco->etHFtowerSumPlus_ = inputCentrality->EtHFtowerSumPlus();
     creco->etMidRapiditySum_ = inputCentrality->EtMidRapiditySum();
  }
  
  if(produceBasicClusters_){
     creco->etEESumPlus_ = 0;
     creco->etEESumMinus_ = 0;
     creco->etEBSum_ = 0;
     
     Handle<BasicClusterCollection> clusters;
     iEvent.getByLabel(srcBasicClustersEE_, clusters);
     for( size_t i = 0; i<clusters->size(); ++ i){
	const BasicCluster & cluster = (*clusters)[ i ];
	double eta = cluster.eta();
	double tg = cluster.position().rho()/cluster.position().r();
	double et = cluster.energy()*tg;
	if(eta > 0)
	   creco->etEESumPlus_ += et;
	if(eta < 0)
	   creco->etEESumMinus_ += et;
     }
     
     iEvent.getByLabel(srcBasicClustersEB_, clusters);
     for( size_t i = 0; i<clusters->size(); ++ i){
	const BasicCluster & cluster = (*clusters)[ i ];
	double tg = cluster.position().rho()/cluster.position().r();
        double et = cluster.energy()*tg;
	creco->etEBSum_ += et;
     }
  }else{
     creco->etEESumMinus_ = inputCentrality->EtEESumMinus();
     creco->etEESumPlus_ = inputCentrality->EtEESumPlus();
     creco->etEBSum_ = inputCentrality->EtEBSum();
  }
  
  if(producePixelhits_){
     creco->pixelMultiplicity_ = 0;
     const SiPixelRecHitCollection* rechits;
     Handle<SiPixelRecHitCollection> rchts;
     iEvent.getByLabel(srcPixelhits_,rchts);
     rechits = rchts.product();
     int nPixel =0 ;
     for (SiPixelRecHitCollection::const_iterator it = rechits->begin(); it!=rechits->end();it++)
     {
        SiPixelRecHitCollection::DetSet hits = *it;
        DetId detId = DetId(hits.detId());
        SiPixelRecHitCollection::const_iterator recHitMatch = rechits->find(detId);
        const SiPixelRecHitCollection::DetSet recHitRange = *recHitMatch;
        for ( SiPixelRecHitCollection::DetSet::const_iterator recHitIterator = recHitRange.begin(); 
	      recHitIterator != recHitRange.end(); ++recHitIterator) {
	  // add selection if needed, now all hits.
	  if(doPixelCut_){
	    const SiPixelRecHit * recHit = &(*recHitIterator);
	    const PixelGeomDetUnit* pixelLayer = dynamic_cast<const PixelGeomDetUnit*> (trackGeo_->idToDet(recHit->geographicalId()));
	    GlobalPoint gpos = pixelLayer->toGlobal(recHit->localPosition());
	    math::XYZVector rechitPos(gpos.x(),gpos.y(),gpos.z());
	    double abeta = fabs(rechitPos.eta());
	    int clusterSize = recHit->cluster()->size();
            if (                abeta < 0.5 && clusterSize < 1) continue;
	    if ( abeta > 0.5 && abeta < 1   && clusterSize < 2) continue;
            if ( abeta > 1.  && abeta < 1.5 && clusterSize < 3) continue;
            if ( abeta > 1.5 && abeta < 2.  && clusterSize < 4) continue;
            if ( abeta > 2.  && abeta < 2.5 && clusterSize < 6) continue;
            if ( abeta > 2.5 && abeta < 5   && clusterSize < 9) continue;
	  }
	  nPixel++;
        } 
     }
     creco->pixelMultiplicity_ = nPixel;
  }else{
     creco->pixelMultiplicity_ = inputCentrality->multiplicityPixel();
  }

  if(produceTracks_){
     edm::Handle<reco::TrackCollection> tracks;
     iEvent.getByLabel(srcTracks_,tracks);
     int nTracks = 0;

     double trackCounter = 0;
     double trackCounterEta = 0;
     double trackCounterEtaPt = 0;

     for(unsigned int i = 0 ; i < tracks->size(); ++i){
       const Track& track = (*tracks)[i];
       if(useQuality_ && !track.quality(trackQuality_)) continue;
       nTracks++;

       if( track.pt() > trackPtCut_)  trackCounter++;
       if(fabs(track.eta())<trackEtaCut_) {
	 trackCounterEta++;
	 if (track.pt() > trackPtCut_) trackCounterEtaPt++;
       }
     }

     creco->trackMultiplicity_ = nTracks;
     creco->ntracksPtCut_ = trackCounter; 
     creco->ntracksEtaCut_ = trackCounterEta;
     creco->ntracksEtaPtCut_ = trackCounterEtaPt;

  }else{
     creco->trackMultiplicity_ = inputCentrality->Ntracks();
     creco->ntracksPtCut_= inputCentrality->NtracksPtCut();
     creco->ntracksEtaCut_= inputCentrality->NtracksEtaCut();
     creco->ntracksEtaPtCut_= inputCentrality->NtracksEtaPtCut();
  }

  if(producePixelTracks_){
    edm::Handle<reco::TrackCollection> pixeltracks;
    iEvent.getByLabel(srcPixelTracks_,pixeltracks);
    int nPixelTracks = pixeltracks->size();
    creco->nPixelTracks_ = nPixelTracks;
  }
  else{
    creco->nPixelTracks_ = inputCentrality->NpixelTracks();
  }

  if(produceZDChits_){
     creco->zdcSumPlus_ = 0;
     creco->zdcSumMinus_ = 0;
     
     Handle<ZDCRecHitCollection> hits;
     bool zdcAvailable = iEvent.getByLabel(srcZDChits_,hits);
     if(zdcAvailable){
	for( size_t ihit = 0; ihit<hits->size(); ++ ihit){
	   const ZDCRecHit & rechit = (*hits)[ ihit ];
	   if(rechit.id().zside() > 0 )
	      creco->zdcSumPlus_ += rechit.energy();
	   if(rechit.id().zside() < 0)
	      creco->zdcSumMinus_ += rechit.energy();
	}
     }else{
	creco->zdcSumPlus_ = -9;
	creco->zdcSumMinus_ = -9;
     }
  }else{
     creco->zdcSumMinus_ = inputCentrality->zdcSumMinus();
     creco->zdcSumPlus_ = inputCentrality->zdcSumPlus();
  }
  
  iEvent.put(creco);
  return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
CentralityProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CentralityProducer::endJob() {
}
}

//define this as a plug-in
DEFINE_FWK_MODULE(reco::CentralityProducer);
