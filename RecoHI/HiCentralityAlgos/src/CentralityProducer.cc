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
// $Id: CentralityProducer.cc,v 1.18 2010/07/07 09:36:36 yilmaz Exp $
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
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
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
   bool reuseAny_;

   double midRapidityRange_;

   edm::InputTag  srcHFhits_;	
   edm::InputTag  srcTowers_;
   edm::InputTag srcEEhits_;
   edm::InputTag srcEBhits_;
   edm::InputTag srcBasicClustersEE_;
   edm::InputTag srcBasicClustersEB_;
   edm::InputTag srcZDChits_;
   edm::InputTag srcPixelhits_;

   edm::InputTag reuseTag_;

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

CentralityProducer::CentralityProducer(const edm::ParameterSet& iConfig)
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
   midRapidityRange_ = iConfig.getParameter<double>("midRapidityRange");

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
   if(producePixelhits_) srcPixelhits_ = iConfig.getParameter<edm::InputTag>("srcPixelhits");
   
   reuseAny_ = !produceHFhits_ || !produceHFtowers_ || !produceBasicClusters_ || !produceEcalhits_ || !produceZDChits_;
   if(reuseAny_) reuseTag_ = iConfig.getParameter<edm::InputTag>("srcReUse");

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
     
     Handle<CaloTowerCollection> towers;
     iEvent.getByLabel(srcTowers_,towers);
	for( size_t i = 0; i<towers->size(); ++ i){
	   const CaloTower & tower = (*towers)[ i ];
	   double eta = tower.eta();
	   if(produceHFtowers_){
	      if(eta > 3)
		 creco->etHFtowerSumPlus_ += tower.pt();
	      if(eta < -3)
		 creco->etHFtowerSumMinus_ += tower.pt();
	   }else{
	      creco->etHFtowerSumMinus_ = inputCentrality->EtHFtowerSumMinus();
	      creco->etHFtowerSumPlus_ = inputCentrality->EtHFtowerSumPlus();
	   }
	   if(produceETmidRap_){
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
        // add selection if needed, now all hits.
        nPixel++;
     }
     creco->pixelMultiplicity_ = nPixel;

  }else{
     creco->pixelMultiplicity_ = inputCentrality->multiplicityPixel();
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
