/****************************************************************************
* Author:
*  Grzegorz Sroka
****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"
#include "CondFormats/DataRecord/interface/PPSAssociationCutsRcd.h"

#include <memory>

class RetrievePPSAssociationCuts : public edm::one::EDAnalyzer<> {
public:
  explicit RetrievePPSAssociationCuts(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAssociationCuts, PPSAssociationCutsRcd> esToken_;
};

RetrievePPSAssociationCuts::RetrievePPSAssociationCuts(const edm::ParameterSet &iConfig) : esToken_(esConsumes()) {}

void RetrievePPSAssociationCuts::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAssociationCuts = iSetup.getData(esToken_);

  edm::LogInfo("PPS") << ppsAssociationCuts;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RetrievePPSAssociationCuts);