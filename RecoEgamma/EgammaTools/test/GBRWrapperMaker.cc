// -*- C++ -*-
//
// Package:    GBRWrapperMaker
// Class:      GBRWrapperMaker
//
/**\class GBRWrapperMaker GBRWrapperMaker.cc GBRWrap/GBRWrapperMaker/src/GBRWrapperMaker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Josh Bendavid
//         Created:  Tue Nov  8 22:26:45 CET 2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFile.h"
#include "CondFormats/GBRForest/interface/GBRForestD.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

//
// class declaration
//

class GBRWrapperMaker : public edm::one::EDAnalyzer<> {
public:
  explicit GBRWrapperMaker(const edm::ParameterSet&);
  ~GBRWrapperMaker() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
};

GBRWrapperMaker::GBRWrapperMaker(const edm::ParameterSet& iConfig) {}

//
// member functions
//

// ------------ method called for each event  ------------
void GBRWrapperMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  TFile* infile = new TFile("../data/GBRLikelihood_Clustering_746_bx25_HLT.root", "READ");
  edm::LogPrint("GBRWrapperMaker") << "Load forest";
  GBRForestD* gbreb = (GBRForestD*)infile->Get("EBCorrection");
  GBRForestD* gbrebvar = (GBRForestD*)infile->Get("EBUncertainty");
  GBRForestD* gbree = (GBRForestD*)infile->Get("EECorrection");
  GBRForestD* gbreevar = (GBRForestD*)infile->Get("EEUncertainty");

  edm::LogPrint("GBRWrapperMaker") << "Make objects";
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(*gbreb, poolDbService->beginOfTime(), "mustacheSC_online_EBCorrection");
    poolDbService->writeOneIOV(*gbrebvar, poolDbService->beginOfTime(), "mustacheSC_online_EBUncertainty");
    poolDbService->writeOneIOV(*gbree, poolDbService->beginOfTime(), "mustacheSC_online_EECorrection");
    poolDbService->writeOneIOV(*gbreevar, poolDbService->beginOfTime(), "mustacheSC_online_EEUncertainty");
  }
}

// ------------ method called once each job just before starting event loop  ------------
void GBRWrapperMaker::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void GBRWrapperMaker::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GBRWrapperMaker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GBRWrapperMaker);
