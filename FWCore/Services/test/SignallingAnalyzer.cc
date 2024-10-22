// -*- C++ -*-
//
// Package:     FWCore/Services/test
// Class  :     SignallingAnalyzer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sat, 22 Mar 2014 23:07:05 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include <csignal>

class SignallingAnalyzer : public edm::global::EDAnalyzer<> {
public:
  SignallingAnalyzer(edm::ParameterSet const& iPSet) : m_signal(iPSet.getUntrackedParameter<std::string>("signal")) {}

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final {
    if (m_signal == "INT") {
      raise(SIGINT);
    }
    if (m_signal == "ABRT") {
      raise(SIGABRT);
    }
    if (m_signal == "SEGV") {
      raise(SIGSEGV);
    }
    if (m_signal == "TERM") {
      raise(SIGTERM);
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iConf) {
    edm::ParameterSetDescription desc;
    desc.ifValue(edm::ParameterDescription<std::string>("signal", "INT", false),
                 edm::allowedValues<std::string>("INT", "ABRT", "SEGV", "TERM"))
        ->setComment("which signal to raise.");
    iConf.addDefault(desc);
  }

private:
  std::string const m_signal;
};

DEFINE_FWK_MODULE(SignallingAnalyzer);
