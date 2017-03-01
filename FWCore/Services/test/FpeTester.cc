#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

void do_division(double num, double denom) {
  // We pass in arguments to avoid optimizations.
  // num must be nonzero, and denom must be zero.
  double ratio = num/denom;
  assert(ratio == std::numeric_limits<double>::infinity());
}

void do_invalid(double num, double denom) {
  // We pass in arguments to avoid optimizations.
  // The values passed in must both be zero.
  double ratio = num/denom;
  assert(ratio != ratio);
}

void do_underflow(long double base, long double expo) {
  // We pass in arguments to avoid optimizations.
  // num should be small, and denom should be large.
  long double result = std::pow(base, expo);
  assert(result == 0.0);
}

void do_overflow(long double base, long double expo) {
  // We pass in arguments to avoid optimizations.
  // both base and expo should be large.
  long double result = std::pow(base, expo);
  assert(result == std::numeric_limits<long double>::infinity());
}

namespace edmtest {
  class FpeTester : public edm::EDAnalyzer {
  public:
    explicit FpeTester(edm::ParameterSet const& params);
    virtual ~FpeTester();
    virtual void analyze(edm::Event const& event, edm::EventSetup const&);
  private:
    std::string testname_;
  };

  FpeTester::FpeTester(edm::ParameterSet const& params) :
    EDAnalyzer(),
    testname_(params.getParameter<std::string>("testname")) {
  }

  FpeTester::~FpeTester() {
  }

  [[clang::optnone]]
  void
  FpeTester::analyze(edm::Event const&, edm::EventSetup const&) {
    if (testname_ == "division") {
      do_division(1.0, 0.0);
    } else if (testname_ == "invalid") {
      do_invalid(0.0, 0.0);
    } else if (testname_ == "underflow") {
      do_underflow(std::numeric_limits<long double>::min(), std::numeric_limits<long double>::max());
    } else if (testname_ == "overflow") {
      do_overflow(std::numeric_limits<long double>::max(), std::numeric_limits<long double>::max());
    }
  }
}

using edmtest::FpeTester;
DEFINE_FWK_MODULE(FpeTester);
