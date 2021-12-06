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

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigurationRcd.h"

#include <memory>

class RetrievePPSAlignmentConfiguration : public edm::one::EDAnalyzer<> {
public:
  explicit RetrievePPSAlignmentConfiguration(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESGetToken<PPSAlignmentConfiguration, PPSAlignmentConfigurationRcd> esToken_;
};

RetrievePPSAlignmentConfiguration::RetrievePPSAlignmentConfiguration(const edm::ParameterSet& iConfig)
    : esToken_(esConsumes()) {}

void RetrievePPSAlignmentConfiguration::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the data
  const auto& ppsAlignmentConfiguration = iSetup.getData(esToken_);

  edm::LogInfo("PPS") << ppsAlignmentConfiguration;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RetrievePPSAlignmentConfiguration);
