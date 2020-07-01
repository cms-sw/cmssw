// -*- C++ -*-
//
// Package:    ArbitraryLogError
// Class:      ArbitraryLogError
//
/**\class ArbitraryLogError ArbitraryLogError.cc HLTrigger/ArbitraryLogError/src/ArbitraryLogError.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Original Author:  Andrea Bocci,Bld. 40 Room 4-A01,+41227671545,
//         Created:  Tue Nov 10 12:00:46 CET 2009
//
//

// system include files
#include <cstdint>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class decleration
//

class ArbitraryLogError : public edm::EDAnalyzer {
public:
  explicit ArbitraryLogError(const edm::ParameterSet&);
  ~ArbitraryLogError() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const std::string m_category;
  const bool m_severity;
  const uint32_t m_rate;
  uint32_t m_counter;
};

// CTOR
ArbitraryLogError::ArbitraryLogError(const edm::ParameterSet& config)
    : m_category(config.getParameter<std::string>("category")),
      m_severity(config.getParameter<std::string>("severity") == "Error"),
      m_rate(config.getParameter<uint32_t>("rate")),
      m_counter(0) {}

// DTOR
ArbitraryLogError::~ArbitraryLogError() = default;

// ------------ method called to for each event  ------------
void ArbitraryLogError::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (not(++m_counter % m_rate)) {
    if (m_severity)
      (edm::LogError(m_category));
    else
      (edm::LogWarning(m_category));
  }
}

// ------------ method called once each job just before starting event loop  ------------
void ArbitraryLogError::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void ArbitraryLogError::endJob() {}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ArbitraryLogError);
