// -*- C++ -*-
//
// Package:    HcalQLPlotAnal
// Class:      HcalQLPlotAnal
// 
/**\class HcalQLPlotAnal HcalQLPlotAnal.cc RecoTBCalo/HcalQLPlotAnal/src/HcalQLPlotAnal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Phillip R. Dudero
//         Created:  Tue Jan 16 21:11:37 CST 2007
// $Id: HcalQLPlotAnal.cc,v 1.5 2007/12/31 18:43:18 ratnik Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotAnalAlgos.h"
#include <string>
//
// class declaration
//

class HcalQLPlotAnal : public edm::EDAnalyzer {
   public:
      explicit HcalQLPlotAnal(const edm::ParameterSet&);
      ~HcalQLPlotAnal();


   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  edm::InputTag hbheRHLabel_,hoRHLabel_,hfRHLabel_;
  edm::InputTag hcalDigiLabel_, hcalTrigLabel_;
  bool doCalib_;
  double calibFC2GeV_;
  HcalQLPlotAnalAlgos * algo_;

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
HcalQLPlotAnal::HcalQLPlotAnal(const edm::ParameterSet& iConfig) :
  hbheRHLabel_(iConfig.getUntrackedParameter<edm::InputTag>("hbheRHtag")),
  hoRHLabel_(iConfig.getUntrackedParameter<edm::InputTag>("hoRHtag")),
  hfRHLabel_(iConfig.getUntrackedParameter<edm::InputTag>("hfRHtag")),
  hcalDigiLabel_(iConfig.getUntrackedParameter<edm::InputTag>("hcalDigiTag")),
  hcalTrigLabel_(iConfig.getUntrackedParameter<edm::InputTag>("hcalTrigTag")),
  doCalib_(iConfig.getUntrackedParameter<bool>("doCalib",false)),
  calibFC2GeV_(iConfig.getUntrackedParameter<double>("calibFC2GeV",0.2))
{
  algo_ = new
    HcalQLPlotAnalAlgos(iConfig.getUntrackedParameter<std::string>("outputFilename").c_str(),
			iConfig.getParameter<edm::ParameterSet>("HistoParameters"));
}


HcalQLPlotAnal::~HcalQLPlotAnal()
{
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HcalQLPlotAnal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Step A/C: Get Inputs and process (repeatedly)
  edm::Handle<HcalTBTriggerData> trig;
  iEvent.getByLabel(hcalTrigLabel_,trig);
  if (!trig.isValid()) {
    edm::LogError("HcalQLPlotAnal::analyze") << "No Trigger Data found, skip event";
    return;
  } else {
    algo_->SetEventType(*trig);
  }
  edm::Handle<HBHEDigiCollection> hbhedg;
  iEvent.getByLabel(hcalDigiLabel_,hbhedg);
  if (!hbhedg.isValid()) {
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HBHE Digis/RecHits not found";
  } else {
    algo_->processDigi(*hbhedg);
  }
  edm::Handle<HBHERecHitCollection> hbherh;  
  iEvent.getByLabel(hbheRHLabel_,hbherh);
  if (!hbherh.isValid()) {
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HBHE Digis/RecHits not found";
  } else {
    algo_->processRH(*hbherh,*hbhedg);
  }

  edm::Handle<HODigiCollection> hodg; 
  iEvent.getByLabel(hcalDigiLabel_,hodg);
  if (!hodg.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HO Digis/RecHits not found";
  } else {
    algo_->processDigi(*hodg);
  }
  edm::Handle<HORecHitCollection> horh;
  iEvent.getByLabel(hoRHLabel_,horh);
  if (!horh.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HO Digis/RecHits not found";
  } else {
    algo_->processRH(*horh,*hodg);
  }
  
  edm::Handle<HFDigiCollection> hfdg;
  iEvent.getByLabel(hcalDigiLabel_,hfdg);

  if (!hfdg.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HF Digis/RecHits not found";
  } else {
    algo_->processDigi(*hfdg);
  }

  edm::Handle<HFRecHitCollection> hfrh;
  iEvent.getByLabel(hfRHLabel_,hfrh);
  if (!hfrh.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HF Digis/RecHits not found";
  } else {
    algo_->processRH(*hfrh,*hfdg);
  }

  if (doCalib_) {
    // No rechits as of yet...
    edm::Handle<HcalCalibDigiCollection> calibdg;
    iEvent.getByLabel(hcalDigiLabel_,calibdg);
    if (!calibdg.isValid()) {
      edm::LogWarning("HcalQLPlotAnal::analyze") << "Hcal Calib Digis not found";
    } else {
      algo_->processDigi(*calibdg,calibFC2GeV_);
    }
  }

}


// ------------ method called once each job just after ending the event loop  ------------
void 
HcalQLPlotAnal::endJob()
{
  algo_->end();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalQLPlotAnal);
