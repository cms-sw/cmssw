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

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRcd.h"

#include <memory>

class RetrievePPSAlignmentConfig : public edm::one::EDAnalyzer<> {
public:
  explicit RetrievePPSAlignmentConfig(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  edm::ESGetToken<PPSAlignmentConfig, PPSAlignmentConfigRcd> esToken_;
};

RetrievePPSAlignmentConfig::RetrievePPSAlignmentConfig(const edm::ParameterSet &iConfig) : esToken_(esConsumes()) {}

void RetrievePPSAlignmentConfig::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get the data
  const auto &ppsAlignmentConfig = iSetup.getData(esToken_);

  edm::LogInfo("PPS") << ppsAlignmentConfig;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RetrievePPSAlignmentConfig);
