/****************************************************************************
* Author: 
*  Mateusz Kocot (mateuszkocot99@gmail.com)
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

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfigRun3v1.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRun3v1Rcd.h"

#include <memory>

class RetrievePPSAlignmentConfigRun3v1 : public edm::one::EDAnalyzer<> {
public:
  explicit RetrievePPSAlignmentConfigRun3v1(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAlignmentConfigRun3v1, PPSAlignmentConfigRun3v1Rcd> esToken_;
};

RetrievePPSAlignmentConfigRun3v1::RetrievePPSAlignmentConfigRun3v1(const edm::ParameterSet &iConfig) : esToken_(esConsumes()) {}

void RetrievePPSAlignmentConfigRun3v1::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAlignmentConfigRun3v1 = iSetup.getData(esToken_);

  edm::LogInfo("PPS") << ppsAlignmentConfigRun3v1;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RetrievePPSAlignmentConfigRun3v1);
