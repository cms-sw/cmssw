
/*----------------------------------------------------------------------

Global EDAnalyzer for testing concurrent IOVs in the CondCore subsystem.

This module is written to run by the configuration
CondCore/ESSources/test/TestConcurrentIOVsCondCore_cfg.py only.
See that configuration for more comments and description.

----------------------------------------------------------------------*/

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <cmath>
#include <iostream>

namespace edmtest {

  constexpr unsigned nLumisToTest = 200;

  class TestConcurrentIOVsCondCore : public edm::global::EDAnalyzer<> {
  public:
    explicit TestConcurrentIOVsCondCore(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

    void endJob() override;

    void busyWait(char const* msg, unsigned int iterations) const;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    class BeamSpotTestObject {
    public:
      unsigned int lumi_;
      double x_ = 0;
      double y_ = 0;
      double z_ = 0;
      unsigned int start_ = 0;
      unsigned int end_ = 0;
    };

    const double pi_;

    edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> const esTokenBeamSpotObjects_;

    // This can be mutable because it is designed particularly to
    // be used in one unit test where there is one event per lumi
    // and the vector is accessed by lumi number and does not change
    // size.
    CMS_THREAD_SAFE mutable std::vector<BeamSpotTestObject> testObjects_;
  };

  TestConcurrentIOVsCondCore::TestConcurrentIOVsCondCore(edm::ParameterSet const&)
      : pi_(std::acos(-1)),
        esTokenBeamSpotObjects_{esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>(edm::ESInputTag("", ""))},
        testObjects_(nLumisToTest) {}

  void TestConcurrentIOVsCondCore::analyze(edm::StreamID,
                                           edm::Event const& event,
                                           edm::EventSetup const& eventSetup) const {
    unsigned int lumi = event.eventAuxiliary().luminosityBlock();

    // These hard coded lumi numbers come from prior knowledge of
    // the IOV ranges in the database. They are intentionally
    // selected to force multiple IOVs to run concurrently.
    // There is a wait each time the IOV changes so the first
    // event of each IOV will take a long time to process.
    // The number of iterations varies just to make things
    // interesting and make early IOVs finish faster than
    // earlier ones in many cases.
    if (lumi == 1) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 30 * 200 * 1000);
    } else if (lumi == 5) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 29 * 200 * 1000);
    } else if (lumi == 6) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 28 * 200 * 1000);
    } else if (lumi == 7) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 27 * 200 * 1000);
    } else if (lumi == 11) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 26 * 200 * 1000);
    } else if (lumi == 15) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 25 * 200 * 1000);
    } else if (lumi == 16) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 24 * 200 * 1000);
    } else if (lumi == 20) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 23 * 200 * 1000);
    } else if (lumi == 21) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 22 * 200 * 1000);
    } else if (lumi == 25) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 21 * 200 * 1000);
    } else if (lumi == 30) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 20 * 200 * 1000);
    } else if (lumi == 35) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 19 * 200 * 1000);
    } else if (lumi == 40) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 18 * 200 * 1000);
    } else if (lumi == 46) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 17 * 200 * 1000);
    } else if (lumi == 48) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 16 * 200 * 1000);
    } else if (lumi == 49) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 15 * 200 * 1000);
    } else if (lumi == 50) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 14 * 200 * 1000);
    } else if (lumi == 79) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 13 * 200 * 1000);
    } else if (lumi == 90) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 12 * 200 * 1000);
    } else if (lumi == 94) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 11 * 200 * 1000);
    } else if (lumi == 99) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 10 * 200 * 1000);
    } else if (lumi == 104) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 9 * 200 * 1000);
    } else if (lumi == 113) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 8 * 200 * 1000);
    } else if (lumi == 114) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 7 * 200 * 1000);
    } else if (lumi == 118) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 6 * 200 * 1000);
    } else if (lumi == 123) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 5 * 200 * 1000);
    } else if (lumi == 128) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 4 * 200 * 1000);
    } else if (lumi == 130) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 3 * 200 * 1000);
    } else if (lumi == 132) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 10 * 200 * 1000);
    } else if (lumi == 133) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 9 * 200 * 1000);
    } else if (lumi == 134) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 8 * 200 * 1000);
    } else if (lumi == 161) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 7 * 200 * 1000);
    } else if (lumi == 165) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 6 * 200 * 1000);
    } else if (lumi == 170) {
      busyWait("TestConcurrentIOVsCondCore::analyze", 5 * 200 * 1000);
    }

    edm::ESHandle<BeamSpotObjects> beamSpotObjects = eventSetup.getHandle(esTokenBeamSpotObjects_);
    BeamSpotObjectsRcd beamSpotObjectsRcd = eventSetup.get<BeamSpotObjectsRcd>();
    edm::ValidityInterval iov = beamSpotObjectsRcd.validityInterval();

    if (lumi < nLumisToTest) {
      BeamSpotTestObject& testObject = testObjects_[lumi];
      testObject.lumi_ = lumi;
      testObject.x_ = beamSpotObjects->GetX();
      testObject.y_ = beamSpotObjects->GetY();
      testObject.z_ = beamSpotObjects->GetZ();
      testObject.start_ = iov.first().luminosityBlockNumber();
      testObject.end_ = iov.last().luminosityBlockNumber();
    }
    edm::LogAbsolute("TestConcurrentIOVsCondCore::busyWait")
        << "TestConcurrentIOVsCondCore::analyze finishing lumi " << lumi << std::endl;
  }

  void TestConcurrentIOVsCondCore::endJob() {
    // This creates output that can be compared to a reference file.
    // The original reference file was created using a version
    // of the code before concurrent IOVs were implemented.
    for (auto const& object : testObjects_) {
      std::cout << "TestConcurrentIOVsCondCore: lumi = " << object.lumi_ << " position: (" << object.x_ << ", "
                << object.y_ << ", " << object.z_ << ")  iov = " << object.start_ << ":" << object.end_ << std::endl;
    }
  }

  void TestConcurrentIOVsCondCore::busyWait(char const* msg, unsigned int iterations) const {
    // This is just doing meaningless work trying to use up time.
    // Print it out and use cos so the compiler doesn't optimize it away.
    edm::LogAbsolute("TestConcurrentIOVsCondCore::busyWait") << "Start TestConcurrentIOVsCondCore::busyWait " << msg;
    double sum = 0.;
    const double stepSize = pi_ / iterations;
    for (unsigned int i = 0; i < iterations; ++i) {
      sum += stepSize * cos(i * stepSize);
    }
    edm::LogAbsolute("TestConcurrentIOVsCondCore::busyWait")
        << "Stop TestConcurrentIOVsCondCore::busyWait " << msg << " " << sum;
  }

  void TestConcurrentIOVsCondCore::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addDefault(desc);
  }

  DEFINE_FWK_MODULE(TestConcurrentIOVsCondCore);
}  // namespace edmtest
