// -*- C++ -*-
//
// Package:    TestSpecParAnalyzer
// Class:      TestSpecParAnalyzer
//
/**\class TestSpecParAnalyzer TestSpecParAnalyzer.cc test/TestSpecParAnalyzer/src/TestSpecParAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//

#include <memory>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

class TestSpecParAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TestSpecParAnalyzer(const edm::ParameterSet&);
  ~TestSpecParAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  std::string specName_;
  std::string specStrValue_;
  double specDblValue_;
};

TestSpecParAnalyzer::TestSpecParAnalyzer(const edm::ParameterSet& iConfig)
    : specName_(iConfig.getParameter<std::string>("specName")),
      specStrValue_(iConfig.getUntrackedParameter<std::string>("specStrValue", "frederf")),
      specDblValue_(iConfig.getUntrackedParameter<double>("specDblValue", 0.0)) {}

TestSpecParAnalyzer::~TestSpecParAnalyzer() {}

void TestSpecParAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::cout << "Here I am " << std::endl;
  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get("", pDD);
  const DDCompactView& cpv(*pDD);
  if (specStrValue_ != "frederf") {
    std::cout << "specName = " << specName_ << " and specStrValue = " << specStrValue_ << std::endl;
    DDSpecificsMatchesValueFilter filter{DDValue(specName_, specStrValue_, 0.0)};
    DDFilteredView fv(cpv, filter);
    bool doit = fv.firstChild();
    std::vector<const DDsvalues_type*> spec = fv.specifics();
    std::vector<const DDsvalues_type*>::const_iterator spit = spec.begin();
    while (doit) {
      spec = fv.specifics();
      spit = spec.begin();
      std::cout << fv.geoHistory() << std::endl;
      for (; spit != spec.end(); ++spit) {
        DDsvalues_type::const_iterator it = (**spit).begin();
        for (; it != (**spit).end(); it++) {
          std::cout << "\t" << it->second.name() << std::endl;
          if (it->second.isEvaluated()) {
            for (double i : it->second.doubles()) {
              std::cout << "\t\t" << i << std::endl;
            }
          } else {
            for (const auto& i : it->second.strings()) {
              std::cout << "\t\t" << i << std::endl;
            }
          }
        }
      }
      doit = fv.next();
    }

  } else {
    std::cout << "double spec value not implemented" << std::endl;
  }

  std::cout << "finished" << std::endl;
}

DEFINE_FWK_MODULE(TestSpecParAnalyzer);
