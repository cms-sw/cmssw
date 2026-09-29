// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     ConcurrentLumiAnalyzer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 25 Jun 2026 14:53:01 GMT
//

// system include files
#include <chrono>
#include <thread>

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edmtest {
  class ConcurrentLumiAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ConcurrentLumiAnalyzer(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
      auto lumi = iEvent.luminosityBlock();
      int count = 0;
      if (lumi == 1) {
        while (keepHolding_) {
          using namespace std::chrono_literals;
          std::this_thread::sleep_for(1000ms);
          ++count;
          if (count > 10) {
            throw cms::Exception("WaitedTooLong");
          }
        }
      }
      if (lumi == 3) {
        keepHolding_ = false;
      }
    }
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

  private:
    mutable std::atomic<bool> keepHolding_{true};
  };
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::ConcurrentLumiAnalyzer);
