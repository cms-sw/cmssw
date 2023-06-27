// -*- C++ -*-
//
// Package:    HcalHitRelabellerTester
// Class:      HcalHitRelabellerTester
//
/**\class HcalHitRelabellerTester HcalHitRelabellerTester.cc test/HcalHitRelabellerTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2023/06/24
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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"

class HcalHitRelabellerTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalHitRelabellerTester(const edm::ParameterSet&);
  ~HcalHitRelabellerTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  bool nd_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> token_;
};

HcalHitRelabellerTester::HcalHitRelabellerTester(const edm::ParameterSet& ps)
    : nd_(ps.getParameter<bool>("neutralDensity")),
      token_{esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>(edm::ESInputTag{})} {
  edm::LogVerbatim("HCalGeom") << "Construct HcalHitRelabellerTester with Neutral Density: " << nd_;
}

void HcalHitRelabellerTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("neutralDensity", true);
  descriptions.add("hcalHitRelabellerTester", desc);
}

// ------------ method called to produce the data  ------------
void HcalHitRelabellerTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HcalDDDRecConstants* theRecNumber = &iSetup.getData(token_);
  std::unique_ptr<HcalHitRelabeller> relabel = std::make_unique<HcalHitRelabeller>(nd_);
  relabel->setGeometry(theRecNumber);
  std::vector<int> etas = {-29, -26, -22, -19, -16, -13, -10, -7, -4, -1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29};
  std::vector<int> layers = {1, 2};
  const int iphi = 63;
  edm::LogVerbatim("HCalGeom") << "HcalHitRelabellerTester:: Testing " << etas.size() << " eta, " << layers.size()
                               << " layer values and iphi = " << iphi;
  for (const auto& eta : etas) {
    int ieta = std::abs(eta);
    int det = (ieta <= 16) ? 1 : 2;
    int zside = (eta >= 0) ? 1 : -1;
    for (const auto& lay : layers) {
      if (ieta == 16)
        det = (lay <= 3) ? 1 : 2;
      int depth = theRecNumber->findDepth(det, ieta, iphi, zside, lay);
      if (depth > 0) {
        uint32_t id = HcalTestNumbering::packHcalIndex(det, zside, depth, ieta, iphi, lay);
        double wt = relabel->energyWt(id);
        edm::LogVerbatim("HCalGeom") << "Det " << det << " Z " << zside << " Eta" << ieta << "  Phi " << iphi << " Lay "
                                     << lay << " Depth " << depth << " Layer0Wt " << wt;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalHitRelabellerTester);
