#include <cmath>
#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edmtest {

  class UnitTestClient_C : public edm::global::EDAnalyzer<> {
  public:
    explicit UnitTestClient_C(edm::ParameterSet const&) {}

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
  };

  void UnitTestClient_C::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const {
    int i = 145;
    edm::LogWarning("cat_A") << "Test of std::hex: " << i << std::hex << " in hex is " << i;
    edm::LogWarning("cat_A") << "Test of std::setw(n) and std::setfill('c'): "
                             << "The following should read ++abcdefg $$$12: " << std::setfill('+') << std::setw(9)
                             << "abcdefg" << std::setw(5) << std::setfill('$') << 12;
    double d = M_PI;
    edm::LogWarning("cat_A") << "Test of std::setprecision(p): "
                             << "Pi with precision 12 is " << std::setprecision(12) << d;
    edm::LogWarning("cat_A") << "Test of spacing: "
                             << "The following should read a b c dd: "
                             << "a" << std::setfill('+') << "b" << std::hex << "c" << std::setw(2) << "dd";

    edm::LogWarning("cat_A").format("Test of format hex: {0} in hex is {0:x}", i);
    edm::LogWarning("cat_A")
        .format("Test of format fill and width: ")
        .format("The following should read ++abcdefg $$$12: {:+>9} {:$>5}", "abcdefg", 12);
    edm::LogWarning("cat_A").format("Test of format precision: Pi with precision 12 is {:.12g}", d);
    edm::LogWarning("cat_A").format(
        "Test of format spacing: The following should read a b cc: {} {:+>} {:>2}", "a", "b", "cc");

    edm::LogWarning("cat_A").printf("Test of printf hex: %d in hex is %x", i, i);
    edm::LogWarning("cat_A")
        .printf("Test of printf fill and width: ")
        .printf("The following should read   abcdefg 00012: %9s %05d", "abcdefg", 12);
    edm::LogWarning("cat_A").printf("Test of printf precision: Pi with precision 12 is %.12g", d);
    edm::LogWarning("cat_A").printf(
        "Test of printf spacing: The following should read a b cc: %-2s%s%3s", "a", "b", "cc");
  }

}  // namespace edmtest

using edmtest::UnitTestClient_C;
DEFINE_FWK_MODULE(UnitTestClient_C);
