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

#include <ostream>
#include <string>
#include <fstream>

template <>
void EcalDeadChannelRecoveryBDTG<EBDetId>::addVariables(TMVA::Reader *reader) {
  for (int i = 0; i < 9; ++i) {
    if (i == 4)
      continue;
    reader->AddVariable("E" + std::to_string(i + 1) + "/(E1+E2+E3+E4+E6+E7+E8+E9)", &(mx_.rEn[i]));
  }
  reader->AddVariable("E1+E2+E3+E4+E6+E7+E8+E9", &(mx_.sumE8));
  for (int i = 0; i < 9; ++i)
    reader->AddVariable("iEta" + std::to_string(i + 1), &(mx_.ieta[i]));
  for (int i = 0; i < 9; ++i)
    reader->AddVariable("iPhi" + std::to_string(i + 1), &(mx_.iphi[i]));
}
template <>
void EcalDeadChannelRecoveryBDTG<EBDetId>::loadFile() {
  readerNoCrack = std::make_unique<TMVA::Reader>("!Color:!Silent");
  readerCrack = std::make_unique<TMVA::Reader>("!Color:!Silent");

  addVariables(readerNoCrack.get());
  addVariables(readerCrack.get());

  reco::details::loadTMVAWeights(readerNoCrack.get(), "BDTG", bdtWeightFileNoCracks_.fullPath());
  reco::details::loadTMVAWeights(readerCrack.get(), "BDTG", bdtWeightFileCracks_.fullPath());
}

template <typename T>
EcalDeadChannelRecoveryBDTG<T>::EcalDeadChannelRecoveryBDTG() {}

template <typename T>
EcalDeadChannelRecoveryBDTG<T>::~EcalDeadChannelRecoveryBDTG() {}

template <>
void EcalDeadChannelRecoveryBDTG<EBDetId>::setParameters(const edm::ParameterSet &ps) {
  bdtWeightFileNoCracks_ = ps.getParameter<edm::FileInPath>("bdtWeightFileNoCracks");
  bdtWeightFileCracks_ = ps.getParameter<edm::FileInPath>("bdtWeightFileCracks");

  loadFile();
}

template <>
void EcalDeadChannelRecoveryBDTG<EEDetId>::setParameters(const edm::ParameterSet &ps) {}

template <>
double EcalDeadChannelRecoveryBDTG<EEDetId>::recover(
    const EEDetId id, const EcalRecHitCollection &hit_collection, double single8Cut, double sum8Cut, bool *acceptFlag) {
  return 0;
}

template <>
double EcalDeadChannelRecoveryBDTG<EBDetId>::recover(
    const EBDetId id, const EcalRecHitCollection &hit_collection, double single8Cut, double sum8Cut, bool *acceptFlag) {
  bool isCrack = false;
  int cellIndex = 0.;
  double neighTotEn = 0.;
  float val = 0.;

  //find the matrix around id
  std::vector<DetId> m3x3aroundDC = EcalClusterTools::matrixDetId(topology_, id, 1);
  if (m3x3aroundDC.size() < 9) {
    *acceptFlag = false;
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
          *acceptFlag = false;
          return 0.;
        } else {
          neighTotEn += goS_it->energy();
          mx_.rEn[cellIndex] = goS_it->energy();
          mx_.iphi[cellIndex] = cell.iphi();
          mx_.ieta[cellIndex] = cell.ieta();
          cellIndex++;
        }
      } else if (cell == id) {  // the cell is the central one
        mx_.rEn[cellIndex] = 0;
        cellIndex++;
      } else {  //goS_it is not in the rechitcollection
        *acceptFlag = false;
        return 0.;
      }
    } else {  //cell is null
      *acceptFlag = false;
      return 0.;
    }
  }
  if (cellIndex > 0 && neighTotEn >= single8Cut * 8. && neighTotEn >= sum8Cut) {
    bool allneighs = true;
    mx_.sumE8 = neighTotEn;
    for (unsigned int icell = 0; icell < 9; icell++) {
      if (mx_.rEn[icell] < single8Cut && icell != 4) {
        allneighs = false;
      }
      mx_.rEn[icell] = mx_.rEn[icell] / neighTotEn;
    }
    if (allneighs == true) {
      // evaluate the regression
      if (isCrack) {
        val = exp((readerCrack->EvaluateRegression("BDTG"))[0]);
      } else {
        val = exp((readerNoCrack->EvaluateRegression("BDTG"))[0]);
      }

      *acceptFlag = true;
      //return the estimated energy
      return val;
    } else {
      *acceptFlag = false;
      return 0;
    }
  } else {
    *acceptFlag = false;
    return 0;
  }
}

template class EcalDeadChannelRecoveryBDTG<EBDetId>;
template class EcalDeadChannelRecoveryBDTG<EEDetId>;  //not used.
