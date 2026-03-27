#include <iostream>
#include <iomanip>
#include <cmath>
#include <format>

#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionSoA.h"
#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

class SchemaEvolutionAnalyzer : public edm::global::EDAnalyzer<> {
public:
  SchemaEvolutionAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, soaToken_{consumes(source_)} {}

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    std::cout << "Starting SchemaEvolutionAnalyzer.analyze" << std::endl;
    portabletest::SchemaEvolutionHostCollection const& product = event.get(soaToken_);
    auto const& view = product.const_view();
    std::cout << source_.encode() << ".size() = " << view.metadata().size() << '\n';

    constexpr float EPS_F = 1e-6f;
    constexpr double EPS_D = 1e-12;

    bool ok = true;

    auto report = [&](const std::string& columnName, int i, const auto expected, const auto actual) {
      std::cout << std::setprecision(17) << columnName << " mismatch at i=" << i << " expected=" << expected
                << " actual=" << actual << std::endl;
      ok = false;
    };

    portabletest::SEEigenVector::Scalar counter = 0;
    for (int i = 0; i < view.metadata().size(); i++) {
      auto element = view[i];

      float expectedFloat = static_cast<float>(i) + 0.1f;
      float actualFloat = element.cOneFloat();

      if (std::abs(actualFloat - expectedFloat) >= EPS_F) {
        report("cOneFloat", i, expectedFloat, actualFloat);
      }

      int expectedInt = static_cast<int>(i);
      int actualInt = element.cTwoInt();

      if (actualInt != expectedInt) {
        report("cTwoInt", i, expectedInt, actualInt);
      }

      double expectedDouble = std::sin(static_cast<double>(i)) * 1e3;
      double actualDouble = element.cThreeDouble();

      if (std::abs(actualDouble - expectedDouble) >= EPS_D) {
        report("cThreeDouble", i, expectedDouble, actualDouble);
      }

      portabletest::SEArray expectedArray{{i, i + 1, i + 2}};
      portabletest::SEArray actualArray = element.cFourArray();
      for (portabletest::SEArray::size_type j = 0; j < actualArray.size(); ++j) {
        if (actualArray[j] != expectedArray[j]) {
          report("cFourArray[" + std::to_string(j) + "]", i, expectedArray[j], actualArray[j]);
        }
      }

      portabletest::SEEigenVector expectedVec;

      for (int r = 0; r < portabletest::SEEigenVector::RowsAtCompileTime; ++r) {
        for (int c = 0; c < portabletest::SEEigenVector::ColsAtCompileTime; ++c) {
          expectedVec(r, c) = static_cast<portabletest::SEEigenVector::Scalar>(counter++);
        }
      }

      portabletest::SEEigenVector actualVec = element.eOneVector3d();

      for (int r = 0; r < portabletest::SEEigenVector::RowsAtCompileTime; ++r) {
        for (int c = 0; c < portabletest::SEEigenVector::ColsAtCompileTime; ++c) {
          std::cout << "Checking eOneVector3d[" << r << "][" << c << "]: expected=" << expectedVec(r, c)
                    << " actual=" << actualVec(r, c) << std::endl;
          if (std::abs(actualVec(r, c) - expectedVec(r, c)) >= EPS_D) {
            report("eOneVector3d[" + std::to_string(r) + "][" + std::to_string(c) + "]",
                   i,
                   expectedVec(r, c),
                   actualVec(r, c));
          }
        }
      }
    }

    // Scalars
    if (view.sOneInt() != 42) {
      report("sOneInt", 0, 42, view.sOneInt());
    }

    if (std::abs(view.sTwoFloat() - (1.0f / 3.0f)) >= EPS_F) {
      report("sTwoFloat", 0, 1.0f / 3.0f, view.sTwoFloat());
    }

    if (std::abs(view.sThreeDouble() - (1.0 / 10.0)) >= EPS_D) {
      report("sThreeDouble", 0, 1.0 / 10.0, view.sThreeDouble());
    }

    // Final result
    if (ok) {
      std::cout << "All checks passed." << std::endl;
    } else {
      std::cout << "Some checks FAILED." << std::endl;
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<portabletest::SchemaEvolutionHostCollection> soaToken_;
  ;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SchemaEvolutionAnalyzer);
