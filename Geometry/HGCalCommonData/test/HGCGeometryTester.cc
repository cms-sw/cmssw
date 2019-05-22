// -*- C++ -*-
//
// Package:    HGCGeometryTester
// Class:      HGCGeometryTester
//
/**\class HGCGeometryTester HGCGeometryTester.cc test/HGCGeometryTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2014/02/07
// $Id: HGCGeometryTester.cc,v 1.0 2014/02/07 14:06:07 sunanda Exp $
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCGeometryTester : public edm::one::EDAnalyzer<> {
 public:
  explicit HGCGeometryTester(const edm::ParameterSet&);
  ~HGCGeometryTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

 private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  bool square;
};

HGCGeometryTester::HGCGeometryTester(const edm::ParameterSet& iC):
  ddToken_{esConsumes<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{})}
{
  square = iC.getUntrackedParameter<bool>("SquareType", true);
}

HGCGeometryTester::~HGCGeometryTester() {}

// ------------ method called to produce the data  ------------
void HGCGeometryTester::analyze(const edm::Event& iEvent,
                                const edm::EventSetup& iSetup) {
  const auto& pDD = iSetup.getData(ddToken_);

  // parse the DD for sensitive volumes
  DDExpandedView eview(pDD);
  std::map<std::string, std::pair<double, double> > svPars;
  do {
    const DDLogicalPart& logPart = eview.logicalPart();
    const std::string& name = logPart.name().fullname();

    // only EE sensitive volumes for the moment
    if ((name.find("HGCal") != std::string::npos) &&
        (name.find("Sensitive") != std::string::npos)) {
      size_t pos = name.find("Sensitive") + 9;
      int layer = atoi(name.substr(pos, name.size() - 1).c_str());
      if (svPars.find(name) == svPars.end()) {
        // print half height and widths for the trapezoid
        std::vector<double> solidPar = eview.logicalPart().solid().parameters();
        if (square) {
          svPars[name] = std::pair<double, double>(
              solidPar[3], 0.5 * (solidPar[4] + solidPar[5]));
          std::cout << name << " Layer " << layer << " " << solidPar[3] << " "
                    << solidPar[4] << " " << solidPar[5] << std::endl;
        } else {
          svPars[name] = std::pair<double, double>(
              solidPar[0], 0.5 * (solidPar[2] - solidPar[1]));
          std::cout << name << " Layer " << layer << " " << solidPar[0] << " "
                    << solidPar[1] << " " << solidPar[2] << std::endl;
        }
      }
    }
  } while (eview.next());
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCGeometryTester);
