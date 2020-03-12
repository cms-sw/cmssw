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
  ~HcalQLPlotAnal() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag hcalDigiLabel_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbherec_;
  edm::EDGetTokenT<HORecHitCollection> tok_horec_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hfrec_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;
  edm::EDGetTokenT<HcalTBTriggerData> tok_tb_;
  bool doCalib_;
  double calibFC2GeV_;
  HcalQLPlotAnalAlgos* algo_;
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
HcalQLPlotAnal::HcalQLPlotAnal(const edm::ParameterSet& iConfig)
    : hcalDigiLabel_(iConfig.getUntrackedParameter<edm::InputTag>("hcalDigiTag")),
      doCalib_(iConfig.getUntrackedParameter<bool>("doCalib", false)),
      calibFC2GeV_(iConfig.getUntrackedParameter<double>("calibFC2GeV", 0.2)) {
  algo_ = new HcalQLPlotAnalAlgos(iConfig.getUntrackedParameter<std::string>("outputFilename").c_str(),
                                  iConfig.getParameter<edm::ParameterSet>("HistoParameters"));

  tok_hbherec_ = consumes<HBHERecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("hbheRHtag"));
  tok_horec_ = consumes<HORecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("hoRHtag"));
  tok_hfrec_ = consumes<HFRecHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("hfRHtag"));
  tok_hbhe_ = consumes<HBHEDigiCollection>(hcalDigiLabel_);
  tok_ho_ = consumes<HODigiCollection>(hcalDigiLabel_);
  tok_hf_ = consumes<HFDigiCollection>(hcalDigiLabel_);
  tok_calib_ = consumes<HcalCalibDigiCollection>(hcalDigiLabel_);
  tok_tb_ = consumes<HcalTBTriggerData>(iConfig.getUntrackedParameter<edm::InputTag>("hcalTrigTag"));
}

HcalQLPlotAnal::~HcalQLPlotAnal() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void HcalQLPlotAnal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Step A/C: Get Inputs and process (repeatedly)
  edm::Handle<HcalTBTriggerData> trig;
  iEvent.getByToken(tok_tb_, trig);
  if (!trig.isValid()) {
    edm::LogError("HcalQLPlotAnal::analyze") << "No Trigger Data found, skip event";
    return;
  } else {
    algo_->SetEventType(*trig);
  }
  edm::Handle<HBHEDigiCollection> hbhedg;
  iEvent.getByToken(tok_hbhe_, hbhedg);
  if (!hbhedg.isValid()) {
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HBHE Digis/RecHits not found";
  } else {
    algo_->processDigi(*hbhedg);
  }
  edm::Handle<HBHERecHitCollection> hbherh;
  iEvent.getByToken(tok_hbherec_, hbherh);
  if (!hbherh.isValid()) {
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HBHE Digis/RecHits not found";
  } else {
    algo_->processRH(*hbherh, *hbhedg);
  }

  edm::Handle<HODigiCollection> hodg;
  iEvent.getByToken(tok_ho_, hodg);
  if (!hodg.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HO Digis/RecHits not found";
  } else {
    algo_->processDigi(*hodg);
  }
  edm::Handle<HORecHitCollection> horh;
  iEvent.getByToken(tok_horec_, horh);
  if (!horh.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HO Digis/RecHits not found";
  } else {
    algo_->processRH(*horh, *hodg);
  }

  edm::Handle<HFDigiCollection> hfdg;
  iEvent.getByToken(tok_hf_, hfdg);

  if (!hfdg.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HF Digis/RecHits not found";
  } else {
    algo_->processDigi(*hfdg);
  }

  edm::Handle<HFRecHitCollection> hfrh;
  iEvent.getByToken(tok_hfrec_, hfrh);
  if (!hfrh.isValid()) {
    // can't find it!
    edm::LogWarning("HcalQLPlotAnal::analyze") << "One of HF Digis/RecHits not found";
  } else {
    algo_->processRH(*hfrh, *hfdg);
  }

  if (doCalib_) {
    // No rechits as of yet...
    edm::Handle<HcalCalibDigiCollection> calibdg;
    iEvent.getByToken(tok_calib_, calibdg);
    if (!calibdg.isValid()) {
      edm::LogWarning("HcalQLPlotAnal::analyze") << "Hcal Calib Digis not found";
    } else {
      algo_->processDigi(*calibdg, calibFC2GeV_);
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void HcalQLPlotAnal::endJob() { algo_->end(); }

//define this as a plug-in
DEFINE_FWK_MODULE(HcalQLPlotAnal);
