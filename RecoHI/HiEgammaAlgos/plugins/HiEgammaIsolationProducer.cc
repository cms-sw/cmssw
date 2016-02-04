// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      HiEgammaIsolationProducer
// 
/**\class HiEgammaIsolationProducer HiEgammaIsolationProducer.cc PhysicsTools/PatAlgos/test/HiEgammaIsolationProducer.cc

 Description: Produce HI Egamma isolationsfor PAT

 Implementation:

*/ 
//
// Original Author:  Yen-Jie Lee
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "RecoHI/HiEgammaAlgos/interface/CxCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/RxCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/TxyCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/dRxyCalculator.h"

#include <string>

namespace edm { using ::std::advance; }

//
// class decleration
//

class HiEgammaIsolationProducer : public edm::EDProducer {
   public:
      explicit HiEgammaIsolationProducer(const edm::ParameterSet&);
      ~HiEgammaIsolationProducer();


   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
      edm::InputTag photons_;
      edm::InputTag barrelBCLabel_;
      edm::InputTag endcapBCLabel_;
      edm::InputTag hfLabel_;
      edm::InputTag hoLabel_;
      edm::InputTag hbheLabel_;
      edm::InputTag trackLabel_;
      
      std::string   label_;
      enum IsoMode { calcCx, calcRx, calcTxy, calcDRxy, calcErr };
      double x_;
      double y_;
      IsoMode var_;
      int mode_;
};

//
// constructors and destructor
//
HiEgammaIsolationProducer::HiEgammaIsolationProducer(const edm::ParameterSet& iConfig):
  photons_(iConfig.getParameter<edm::InputTag>("photons")),
  barrelBCLabel_(iConfig.getParameter<edm::InputTag>("barrelBasicCluster")),
  endcapBCLabel_(iConfig.getParameter<edm::InputTag>("endcapBasicCluster")),
  hfLabel_(iConfig.getParameter<edm::InputTag>("hfreco")),
  hoLabel_(iConfig.getParameter<edm::InputTag>("horeco")),
  hbheLabel_(iConfig.getParameter<edm::InputTag>("hbhereco")),
  trackLabel_(iConfig.getParameter<edm::InputTag>("track")),
  label_(iConfig.existsAs<std::string>("label") ? iConfig.getParameter<std::string>("label") : ""),
  x_(iConfig.getParameter<double>("x")),
  y_(iConfig.getParameter<double>("y")),
  var_(iConfig.getParameter<std::string>("iso") == "Cx" ? calcCx :
       iConfig.getParameter<std::string>("iso") == "Rx" ? calcRx :
       iConfig.getParameter<std::string>("iso") == "Txy" ? calcTxy :
       iConfig.getParameter<std::string>("iso") == "dRxy" ? calcDRxy : calcErr ),
  mode_( iConfig.getParameter<std::string>("mode") == "BackgroundSubtracted" ? 1 : 0)
{
      produces<edm::ValueMap<float> >();
}


HiEgammaIsolationProducer::~HiEgammaIsolationProducer()
{
}

// ------------ method called to for each event  ------------
void
HiEgammaIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   Handle<View<reco::Photon> > recoPhotons;
   iEvent.getByLabel(photons_, recoPhotons);
   //std::cout << "Got " << recoPhotons->size() << " photons" << std::endl;
   //std::cout << "mode "<<mode_<<std::endl;
   vector<float> floats(recoPhotons->size(), -100);
   
   CxCalculator   CxC(iEvent,iSetup,barrelBCLabel_,endcapBCLabel_);
   RxCalculator   RxC(iEvent,iSetup,hbheLabel_,hfLabel_,hoLabel_);
   TxyCalculator  TxyC(iEvent,iSetup,trackLabel_);
   dRxyCalculator dRxyC(iEvent,iSetup,trackLabel_);

   for (size_t i = 0; i < recoPhotons->size(); ++i) {
      if (var_ == calcRx) {
         if (mode_ == 1) {
            floats[i] = RxC.getCRx((*recoPhotons)[i].superCluster(),x_,0);
	 } else {
	    floats[i] = RxC.getRx((*recoPhotons)[i].superCluster(),x_,0);
	 }
      } else if (var_ == calcCx) {
         if (mode_ == 1) {
            floats[i] = CxC.getCCx((*recoPhotons)[i].superCluster(),x_,0);
	 } else {
	    floats[i] = CxC.getCx((*recoPhotons)[i].superCluster(),x_,0);
	 }
      } else if (var_ == calcTxy) {
         if (mode_ == 1) {
            // No background subtraction for the moment...
	    floats[i] = TxyC.getTxy((*recoPhotons)[i].superCluster(),x_,y_);
	 } else {
	    floats[i] = TxyC.getTxy((*recoPhotons)[i].superCluster(),x_,y_);
	 }
      } else if (var_ == calcDRxy) {
         if (mode_ == 1) {
            // No background subtraction for the moment...
	    floats[i] = dRxyC.getDRxy((*recoPhotons)[i].superCluster(),x_,y_);
	 } else {
	    floats[i] = dRxyC.getDRxy((*recoPhotons)[i].superCluster(),x_,y_);
	 }
      }
   }

   auto_ptr<ValueMap<float> > pis(new ValueMap<float>());
   ValueMap<float>::Filler floatfiller(*pis);
   floatfiller.insert(recoPhotons, floats.begin(), floats.end());
   floatfiller.fill();
   iEvent.put(pis);
   

}

//define this as a plug-in
DEFINE_FWK_MODULE(HiEgammaIsolationProducer);
