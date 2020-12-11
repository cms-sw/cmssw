// -*- C++ -*-
//
// Package:    HcalSimNumberingTester
// Class:      HcalSimNumberingTester
//
/**\class HcalSimNumberingTester HcalSimNumberingTester.cc test/HcalSimNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2013/12/26
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

class HcalSimNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalSimNumberingTester(const edm::ParameterSet&);
  ~HcalSimNumberingTester() override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> token_;
};

HcalSimNumberingTester::HcalSimNumberingTester(const edm::ParameterSet&)
    : token_{esConsumes<HcalDDDSimConstants, HcalSimNumberingRecord>(edm::ESInputTag{})} {}

HcalSimNumberingTester::~HcalSimNumberingTester() {}

// ------------ method called to produce the data  ------------
void HcalSimNumberingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HcalDDDSimConstants hdc = iSetup.getData(token_);

  edm::LogVerbatim("HcalGeom") << "about to getConst for 0..1";
  for (int i = 0; i <= 1; ++i) {
    std::vector<std::pair<double, double> > gcons = hdc.getConstHBHE(i);
    edm::LogVerbatim("HcalGeom") << "Geometry Constants for [" << i << "] with " << gcons.size() << "  elements";
    for (unsigned int k = 0; k < gcons.size(); ++k)
      edm::LogVerbatim("HcalGeom") << "Element[" << k << "] = " << gcons[k].first << " : " << gcons[k].second;
  }
  for (int i = 0; i < 4; ++i)
    edm::LogVerbatim("HcalGeom") << "MaxDepth[" << i << "] = " << hdc.getMaxDepth(i);
  hdc.printTiles();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalSimNumberingTester);
