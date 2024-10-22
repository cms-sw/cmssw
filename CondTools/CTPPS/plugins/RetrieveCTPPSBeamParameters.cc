// -*- C++ -*-
//
// Class:      RetrieveCTPPSBeamParameters
//
// Description: Test analyzer for reading CTPPS beam parameters condition data
//
//              Simple analyzer that retrieves CTTPSBeamParameters record from a sql
//              database file, as a test of offline conditions implementation.
//
// Original Author:  Wagner De Paula Carvalho
//         Created:  Wed, 21 Nov 2018 17:35:07 GMT
//
//==================================================================================

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include <cstdint>

class RetrieveCTPPSBeamParameters : public edm::one::EDAnalyzer<> {
public:
  explicit RetrieveCTPPSBeamParameters(const edm::ParameterSet& ps)
      : label_(ps.getParameter<std::string>("label")),
        tokenBeamParameters_(esConsumes<CTPPSBeamParameters, CTPPSBeamParametersRcd>(edm::ESInputTag("", label_))) {}

  ~RetrieveCTPPSBeamParameters() override = default;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::string label_;

  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> tokenBeamParameters_;
};

//---------------------------------------------------------------------------------------

void RetrieveCTPPSBeamParameters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("CTPPSBeamParameters") << "\n" << iSetup.getData(tokenBeamParameters_) << "\n";
}

DEFINE_FWK_MODULE(RetrieveCTPPSBeamParameters);
