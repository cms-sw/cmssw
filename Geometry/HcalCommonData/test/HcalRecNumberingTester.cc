// -*- C++ -*-
//
// Package:    HcalRecNumberingTester
// Class:      HcalRecNumberingTester
//
/**\class HcalRecNumberingTester HcalRecNumberingTester.cc test/HcalRecNumberingTester.cc

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
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#define EDM_ML_DEBUG

class HcalRecNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalRecNumberingTester(const edm::ParameterSet&);
  ~HcalRecNumberingTester() override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> token_;
};

HcalRecNumberingTester::HcalRecNumberingTester(const edm::ParameterSet&)
    : token_{esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>(edm::ESInputTag{})} {}

HcalRecNumberingTester::~HcalRecNumberingTester() {}

// ------------ method called to produce the data  ------------
void HcalRecNumberingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HcalDDDRecConstants& hdc = iSetup.getData(token_);
  for (int i = 0; i < 4; ++i)
    edm::LogVerbatim("HcalGeom") << "MaxDepth[" << i << "] = " << hdc.getMaxDepth(i);
  edm::LogVerbatim("HcalGeom") << "about to getPhiOff and getPhiBin for 0..2";
  int neta = hdc.getNEta();
  edm::LogVerbatim("HcalGeom") << neta << " eta bins with phi off set for "
                               << "barrel = " << hdc.getPhiOff(0) << ", endcap = " << hdc.getPhiOff(1);
  for (int i = 0; i < neta; ++i) {
    std::pair<double, double> etas = hdc.getEtaLimit(i);
    double fbin = hdc.getPhiBin(i);
    std::vector<int> depths = hdc.getDepth(i, false);
    edm::LogVerbatim("HcalGeom") << "EtaBin[" << i << "]: EtaLimit = (" << etas.first << ":" << etas.second
                                 << ")  phiBin = " << fbin << " and " << depths.size() << " depths";
    for (unsigned int k = 0; k < depths.size(); ++k) {
      edm::LogVerbatim("HcalGeom") << "[" << k << "] " << depths[k];
    }
  }
  for (int type = 0; type < 2; ++type) {
    std::pair<int, int> etar = hdc.getEtaRange(type);
    edm::LogVerbatim("HcalGeom") << "Detector type: " << type << " with eta ranges " << etar.first << ":"
                                 << etar.second;
    for (int eta = etar.first; eta <= etar.second; ++eta) {
      std::vector<std::pair<int, double>> phis = hdc.getPhis(type + 1, eta);
      for (auto& phi : phis) {
        edm::LogVerbatim("HcalGeom") << "Type:Eta:phi " << type << ":" << eta << ":" << phi.first
                                     << " Depth range (+z) " << hdc.getMinDepth(type, eta, phi.first, 1) << ":"
                                     << hdc.getMaxDepth(type, eta, phi.first, 1) << " (-z) "
                                     << hdc.getMinDepth(type, eta, phi.first, -1) << ":"
                                     << hdc.getMaxDepth(type, eta, phi.first, -1);
      }
    }
  }
  std::vector<HcalDDDRecConstants::HcalEtaBin> hbar = hdc.getEtaBins(0);
  std::vector<HcalDDDRecConstants::HcalEtaBin> hcap = hdc.getEtaBins(1);
  edm::LogVerbatim("HcalGeom") << "Topology Mode " << hdc.getTopoMode() << " HB with " << hbar.size()
                               << " eta sectors and HE with " << hcap.size() << " eta sectors";
  std::vector<HcalCellType> hbcell = hdc.HcalCellTypes(HcalBarrel);
  edm::LogVerbatim("HcalGeom") << "HB with " << hbcell.size() << " cells";
  unsigned int i1(0), i2(0), i3(0), i4(0);
  for (const auto& cell : hbcell) {
    edm::LogVerbatim("HcalGeom") << "HB[" << i1 << "] det " << cell.detType() << " zside " << cell.zside() << ":"
                                 << cell.halfSize() << " RO " << cell.actualReadoutDirection() << " eta "
                                 << cell.etaBin() << ":" << cell.etaMin() << ":" << cell.etaMax() << " phi "
                                 << cell.nPhiBins() << ":" << cell.nPhiModule() << ":" << cell.phiOffset() << ":"
                                 << cell.phiBinWidth() << ":" << cell.unitPhi() << " depth " << cell.depthSegment()
                                 << ":" << cell.depth() << ":" << cell.depthMin() << ":" << cell.depthMax() << ":"
                                 << cell.depthType();
    ++i1;
    std::vector<std::pair<int, double>> phis = cell.phis();
    edm::LogVerbatim("HcalGeom") << "Phis (" << phis.size() << ") :";
    for (const auto& phi : phis)
      edm::LogVerbatim("HcalGeom") << " [" << phi.first << ", " << phi.second << "]";
  }
  std::vector<HcalCellType> hecell = hdc.HcalCellTypes(HcalEndcap);
  edm::LogVerbatim("HcalGeom") << "HE with " << hecell.size() << " cells";
  for (const auto& cell : hecell) {
    edm::LogVerbatim("HcalGeom") << "HE[" << i2 << "] det " << cell.detType() << " zside " << cell.zside() << ":"
                                 << cell.halfSize() << " RO " << cell.actualReadoutDirection() << " eta "
                                 << cell.etaBin() << ":" << cell.etaMin() << ":" << cell.etaMax() << " phi "
                                 << cell.nPhiBins() << ":" << cell.nPhiModule() << ":" << cell.phiOffset() << ":"
                                 << cell.phiBinWidth() << ":" << cell.unitPhi() << " depth " << cell.depthSegment()
                                 << ":" << cell.depth() << ":" << cell.depthMin() << ":" << cell.depthMax() << ":"
                                 << cell.depthType();
    ++i2;
    std::vector<std::pair<int, double>> phis = cell.phis();
    edm::LogVerbatim("HcalGeom") << "Phis (" << phis.size() << ") :";
    for (const auto& phi : phis)
      edm::LogVerbatim("HcalGeom") << " [" << phi.first << ", " << phi.second << "]";
  }
  std::vector<HcalCellType> hfcell = hdc.HcalCellTypes(HcalForward);
  edm::LogVerbatim("HcalGeom") << "HF with " << hfcell.size() << " cells";
  for (const auto& cell : hfcell) {
    edm::LogVerbatim("HcalGeom") << "HF[" << i3 << "] det " << cell.detType() << " zside " << cell.zside() << ":"
                                 << cell.halfSize() << " RO " << cell.actualReadoutDirection() << " eta "
                                 << cell.etaBin() << ":" << cell.etaMin() << ":" << cell.etaMax() << " phi "
                                 << cell.nPhiBins() << ":" << cell.nPhiModule() << ":" << cell.phiOffset() << ":"
                                 << cell.phiBinWidth() << ":" << cell.unitPhi() << " depth " << cell.depthSegment()
                                 << ":" << cell.depth() << ":" << cell.depthMin() << ":" << cell.depthMax() << ":"
                                 << cell.depthType();
    ++i3;
  }
  std::vector<HcalCellType> hocell = hdc.HcalCellTypes(HcalOuter);
  edm::LogVerbatim("HcalGeom") << "HO with " << hocell.size() << " cells";
  for (const auto& cell : hocell) {
    edm::LogVerbatim("HcalGeom") << "HO[" << i4 << "] det " << cell.detType() << " zside " << cell.zside() << ":"
                                 << cell.halfSize() << " RO " << cell.actualReadoutDirection() << " eta "
                                 << cell.etaBin() << ":" << cell.etaMin() << ":" << cell.etaMax() << " phi "
                                 << cell.nPhiBins() << ":" << cell.nPhiModule() << ":" << cell.phiOffset() << ":"
                                 << cell.phiBinWidth() << ":" << cell.unitPhi() << " depth " << cell.depthSegment()
                                 << ":" << cell.depth() << ":" << cell.depthMin() << ":" << cell.depthMax() << ":"
                                 << cell.depthType();
    ++i4;
  }
  for (int type = 0; type <= 1; ++type) {
    std::vector<HcalDDDRecConstants::HcalActiveLength> act = hdc.getThickActive(type);
    edm::LogVerbatim("HcalGeom") << "Hcal type " << type << " has " << act.size() << " eta/depth segments";
    for (const auto& active : act) {
      edm::LogVerbatim("HcalGeom") << "zside " << active.zside << " ieta " << active.ieta << " depth " << active.depth
                                   << " type " << active.stype << " eta " << active.eta << " active thickness "
                                   << active.thick;
    }
  }

  // Test merging
  std::vector<int> phiSp;
  HcalSubdetector subdet = HcalSubdetector(hdc.dddConstants()->ldMap()->validDet(phiSp));
  if (subdet == HcalBarrel || subdet == HcalEndcap) {
    int type = (int)(subdet - 1);
    std::pair<int, int> etas = hdc.getEtaRange(type);
    for (int eta = etas.first; eta <= etas.second; ++eta) {
      for (int k : phiSp) {
        int zside = (k > 0) ? 1 : -1;
        int iphi = (k > 0) ? k : -k;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HcalGeom") << "Look for Subdet " << subdet << " Zside " << zside << " Eta " << eta << " Phi "
                                     << iphi << " depths " << hdc.getMinDepth(type, eta, iphi, zside) << ":"
                                     << hdc.getMaxDepth(type, eta, iphi, zside);
#endif
        std::vector<HcalDetId> ids;
        for (int depth = hdc.getMinDepth(type, eta, iphi, zside); depth <= hdc.getMaxDepth(type, eta, iphi, zside);
             ++depth) {
          HcalDetId id(subdet, zside * eta, iphi, depth);
          HcalDetId hid = hdc.mergedDepthDetId(id);
          hdc.unmergeDepthDetId(hid, ids);
          edm::LogVerbatim("HcalGeom") << "Input ID " << id << " Merged ID " << hid << " containing " << ids.size()
                                       << " IDS:";
          for (auto id : ids)
            edm::LogVerbatim("HcalGeom") << " " << id;
        }
      }
    }
  }
  // R,Z of cells
  for (const auto& cell : hbcell) {
    int ieta = cell.etaBin() * cell.zside();
    double rz = hdc.getRZ(HcalBarrel, ieta, cell.phis()[0].first, cell.depthSegment());
    edm::LogVerbatim("HcalGeom") << "HB (eta=" << ieta << ", phi=" << cell.phis()[0].first
                                 << ", depth=" << cell.depthSegment() << ") r/z = " << rz;
  }
  for (const auto& cell : hecell) {
    int ieta = cell.etaBin() * cell.zside();
    double rz = hdc.getRZ(HcalEndcap, ieta, cell.phis()[0].first, cell.depthSegment());
    edm::LogVerbatim("HcalGeom") << "HE (eta=" << ieta << ", phi=" << cell.phis()[0].first
                                 << ", depth=" << cell.depthSegment() << ") r/z = " << rz;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalRecNumberingTester);
