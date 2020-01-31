// -*- C++ -*-
//
// Package:    HcalScintillatorTester
// Class:      HcalScintillatorTester
//
/**\class HcalScintillatorTester HcalScintillatorTester.cc test/HcalScintillatorTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2020/01/24
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "DataFormats/Math/interface/GeantUnits.h"

using namespace geant_units::operators;

class HcalScintillatorTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalScintillatorTester(const edm::ParameterSet&);
  ~HcalScintillatorTester() override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> token_;
};

HcalScintillatorTester::HcalScintillatorTester(const edm::ParameterSet&)
    : token_{esConsumes<HcalDDDSimConstants, HcalSimNumberingRecord>(edm::ESInputTag{})} {}

HcalScintillatorTester::~HcalScintillatorTester() {}

// ------------ method called to produce the data  ------------
void HcalScintillatorTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (auto pHSNDC = iSetup.getHandle(token_)) {
    edm::LogVerbatim("HCalGeom") << "about to de-reference the edm::ESHandle<HcalDDDSimConstants> pHSNDC";
    const HcalDDDSimConstants hdc(*pHSNDC);

    // Layers for HB
    edm::LogVerbatim("HCalGeom") << "\n\nPrint out the parameters for HB"
                                 << "\n===============================\n";
    for (int zs = 0; zs <= 1; ++zs) {
      int zside = 2 * zs - 1;
      const auto etab = hdc.getiEtaRange(0);
      for (int eta = etab.first; eta <= etab.second; ++eta) {
        double etaL = hdc.parameter()->etaTable.at(eta - 1);
        double thetaL = 2. * atan(exp(-etaL));
        double etaH = hdc.parameter()->etaTable.at(eta);
        double thetaH = 2. * atan(exp(-etaH));
        for (int depth = hdc.getMinDepth(1, eta, 1, zside, false); depth <= hdc.getMaxDepth(1, eta, 1, zside, false);
             ++depth) {
          int layL = hdc.getLayerFront(1, eta, 1, zside, depth);
          int layH = hdc.getLayerBack(1, eta, 1, zside, depth);
          edm::LogVerbatim("HCalGeom") << "\nHB: zSide " << zside << " iEta " << eta << " Depth " << depth << " Layers "
                                       << layL << ":" << layH;
          for (int lay = (layL - 1); lay < layH; ++lay) {
            std::vector<double> area(2, 0), length(2, 0), width(2, 0);
            int kk(0);
            for (unsigned int k = 0; k < hdc.parameter()->layHB.size(); ++k) {
              if (lay == hdc.parameter()->layHB[k]) {
                double zmin = hdc.parameter()->rhoxHB[k] * std::cos(thetaL) / std::sin(thetaL);
                double zmax = hdc.parameter()->rhoxHB[k] * std::cos(thetaH) / std::sin(thetaH);
                double dz = (std::min(zmax, hdc.parameter()->dxHB[lay]) - zmin);
                if (dz > 0) {
                  width[kk] = hdc.parameter()->dyHB[k];
                  length[kk] = dz;
                  area[kk] = dz * width[kk];
                  ++kk;
                }
              }
            }
            if (kk > 1) {
              edm::LogVerbatim("HCalGeom")
                  << "Layer " << (lay + 1) << " Length " << length[0] << ":" << length[1] << " width " << width[0]
                  << ":" << width[1] << " Area " << area[0] << ":" << area[1];
            } else if (kk > 0) {
              edm::LogVerbatim("HCalGeom")
                  << "Layer " << (lay + 1) << " Length " << length[0] << " width " << width[0] << " Area " << area[0];
            }
          }
        }
      }
    }

    //Layers for HE
    edm::LogVerbatim("HCalGeom") << "\n\nPrint out the parameters for HE"
                                 << "\n===============================\n";
    for (int zs = 0; zs <= 1; ++zs) {
      int zside = 2 * zs - 1;
      const auto etae = hdc.getiEtaRange(1);
      for (int eta = etae.first; eta <= etae.second; ++eta) {
        double etaL = hdc.parameter()->etaTable.at(eta - 1);
        double thetaL = 2. * atan(exp(-etaL));
        double etaH = hdc.parameter()->etaTable.at(eta);
        double thetaH = 2. * atan(exp(-etaH));
        double phib = hdc.parameter()->phibin[eta - 1];
        int nphi = (phib > 6._deg) ? 1 : 2;
        for (int depth = hdc.getMinDepth(2, eta, 1, zside, false); depth <= hdc.getMaxDepth(2, eta, 1, zside, false);
             ++depth) {
          int layL = hdc.getLayerFront(2, eta, 1, zside, depth);
          int layH = hdc.getLayerBack(2, eta, 1, zside, depth);
          edm::LogVerbatim("HCalGeom") << "\nHE: zSide " << zside << " iEta " << eta << " Depth " << depth << " Layers "
                                       << layL << ":" << layH;
          for (int lay = (layL - 1); lay < layH; ++lay) {
            std::vector<double> area(4, 0), delr(4, 0), dely(4, 0);
            int kk(0);
            for (unsigned int k = 0; k < hdc.parameter()->layHE.size(); ++k) {
              if (lay == hdc.parameter()->layHE[k]) {
                double rmin = hdc.parameter()->zxHE[k] * std::tan(thetaH);
                double rmax = hdc.parameter()->zxHE[k] * std::tan(thetaL);
                if ((lay != 0 || eta == 18) &&
                    (lay != 1 || (eta == 18 && (hdc.parameter()->rhoxHE[k] - hdc.parameter()->dyHE[k] > 1000)) ||
                     (eta != 18 && (hdc.parameter()->rhoxHE[k] - hdc.parameter()->dyHE[k] < 1000))) &&
                    ((rmin + 30) < (hdc.parameter()->rhoxHE[k] + hdc.parameter()->dyHE[k])) &&
                    (rmax > (hdc.parameter()->rhoxHE[k] - hdc.parameter()->dyHE[k]))) {
                  rmin = std::max(rmin, (hdc.parameter()->rhoxHE[k] - hdc.parameter()->dyHE[k]));
                  rmax = std::min(rmax, (hdc.parameter()->rhoxHE[k] + hdc.parameter()->dyHE[k]));
                  double dr = rmax - rmin;
                  if (dr > 0) {
                    double dx1 = rmin * std::tan(phib);
                    double dx2 = rmax * std::tan(phib);
                    if (nphi == 1) {
                      double dy = 0.5 * (dx1 + dx2 - 4. * hdc.parameter()->dx1HE[k]);
                      delr[kk] = dr;
                      dely[kk] = dy;
                      area[kk] = dr * dy;
                      ++kk;
                    } else {
                      double dy1 = 0.5 * (dx1 + dx2 - 2. * hdc.parameter()->dx1HE[k]);
                      delr[kk] = dr;
                      dely[kk] = dy1;
                      area[kk] = dr * dy1;
                      double dy2 = 0.5 * ((rmax + rmin) * tan(10._deg) - 4 * hdc.parameter()->dx1HE[k]) - dy1;
                      delr[kk + 2] = dr;
                      dely[kk + 2] = dy2;
                      area[kk + 2] = dr * dy2;
                      ++kk;
                    }
                  }
                }
              }
            }
            if (area[0] > 0 && area[1] > 0) {
              if (nphi == 1) {
                edm::LogVerbatim("HCalGeom")
                    << "Layer " << (lay + 1) << " Length " << delr[0] << ":" << delr[1] << " width " << dely[0] << ":"
                    << dely[1] << " Area " << area[0] << ":" << area[1];
              } else {
                edm::LogVerbatim("HCalGeom")
                    << "Layer " << (lay + 1) << " Length " << delr[0] << ":" << delr[1] << ":" << delr[2] << ":"
                    << delr[3] << " width " << dely[0] << ":" << dely[1] << ":" << dely[2] << ":" << dely[3] << " Area "
                    << area[0] << ":" << area[1] << ":" << area[2] << ":" << area[3];
              }
            }
          }
        }
      }
    }
  } else {
    edm::LogVerbatim("HCalGeom") << "No record found with HcalDDDSimConstants";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalScintillatorTester);
