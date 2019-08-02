//  Original Authors:  S. Taroni, N. Marinelli
//  University of Notre Dame - US
//  Created:
//
//
//

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryBDTG.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"  // can I use a egammatools here?
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>

#include <iostream>
#include <memory>

#include <fstream>
#include <ostream>
#include <string>

namespace {

  struct XtalMatrix {
    float rEn[9];
    float sumE8;
    float ieta[9];
    float iphi[9];
  };

  std::vector<float> xtalMatrixToVector(XtalMatrix const &xtalMatrix) {
    std::vector<float> v(xtalMatrix.rEn, xtalMatrix.rEn + 4);
    v.insert(v.end(), xtalMatrix.rEn + 5, xtalMatrix.rEn + 9);
    v.push_back(xtalMatrix.sumE8);
    v.insert(v.end(), xtalMatrix.ieta, xtalMatrix.ieta + 9);
    v.insert(v.end(), xtalMatrix.iphi, xtalMatrix.iphi + 9);
    return v;
  }
}  // namespace

template <>
void EcalDeadChannelRecoveryBDTG<EBDetId>::setParameters(const edm::ParameterSet &ps) {
  gbrForestNoCrack_ = createGBRForest(ps.getParameter<edm::FileInPath>("bdtWeightFileNoCracks"));
  gbrForestCrack_ = createGBRForest(ps.getParameter<edm::FileInPath>("bdtWeightFileCracks"));
}

template <>
void EcalDeadChannelRecoveryBDTG<EEDetId>::setParameters(const edm::ParameterSet &ps) {}

template <>
double EcalDeadChannelRecoveryBDTG<EEDetId>::recover(
    const EEDetId id, const EcalRecHitCollection &hit_collection, double single8Cut, double sum8Cut, bool &acceptFlag) {
  return 0;
}

template <>
double EcalDeadChannelRecoveryBDTG<EBDetId>::recover(
    const EBDetId id, const EcalRecHitCollection &hit_collection, double single8Cut, double sum8Cut, bool &acceptFlag) {
  XtalMatrix mx;

  bool isCrack = false;
  int cellIndex = 0.;
  double neighTotEn = 0.;
  float val = 0.;

  //find the matrix around id
  std::vector<DetId> m3x3aroundDC = EcalClusterTools::matrixDetId(topology_, id, 1);
  if (m3x3aroundDC.size() < 9) {
    acceptFlag = false;
    return 0;
  }

  //  Loop over all cells in the vector "NxNaroundDC", and for each cell find it's energy
  //  (from the EcalRecHits collection).
  for (auto const &theCells : m3x3aroundDC) {
    EBDetId cell = EBDetId(theCells);
    if (cell == id) {
      int iEtaCentral = std::abs(cell.ieta());
      int iPhiCentral = cell.iphi();

      if (iEtaCentral < 2 || std::abs(iEtaCentral - 25) < 2 || std::abs(iEtaCentral - 45) < 2 ||
          std::abs(iEtaCentral - 65) < 2 || iEtaCentral > 83 || (int(iPhiCentral + 0.5) % 20 == 0))
        isCrack = true;
    }
    if (!cell.null()) {
      EcalRecHitCollection::const_iterator goS_it = hit_collection.find(cell);
      if (goS_it != hit_collection.end() && cell != id) {
        if (goS_it->energy() < single8Cut) {
          acceptFlag = false;
          return 0.;
        } else {
          neighTotEn += goS_it->energy();
          mx.rEn[cellIndex] = goS_it->energy();
          mx.iphi[cellIndex] = cell.iphi();
          mx.ieta[cellIndex] = cell.ieta();
          cellIndex++;
        }
      } else if (cell == id) {  // the cell is the central one
        mx.rEn[cellIndex] = 0;
        cellIndex++;
      } else {  //goS_it is not in the rechitcollection
        acceptFlag = false;
        return 0.;
      }
    } else {  //cell is null
      acceptFlag = false;
      return 0.;
    }
  }
  if (cellIndex > 0 && neighTotEn >= single8Cut * 8. && neighTotEn >= sum8Cut) {
    bool allneighs = true;
    mx.sumE8 = neighTotEn;
    for (unsigned int icell = 0; icell < 9; icell++) {
      if (mx.rEn[icell] < single8Cut && icell != 4) {
        allneighs = false;
      }
      mx.rEn[icell] = mx.rEn[icell] / neighTotEn;
    }
    if (allneighs == true) {
      // evaluate the regression
      if (isCrack) {
        val = exp(gbrForestCrack_->GetResponse(xtalMatrixToVector(mx).data()));
      } else {
        val = exp(gbrForestNoCrack_->GetResponse(xtalMatrixToVector(mx).data()));
      }

      acceptFlag = true;
      //return the estimated energy
      return val;
    } else {
      acceptFlag = false;
      return 0;
    }
  } else {
    acceptFlag = false;
    return 0;
  }
}

template class EcalDeadChannelRecoveryBDTG<EBDetId>;
template class EcalDeadChannelRecoveryBDTG<EEDetId>;  //not used.
