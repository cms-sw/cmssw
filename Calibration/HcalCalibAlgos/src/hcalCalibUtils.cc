#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "TString.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Calibration/HcalCalibAlgos/interface/hcalCalibUtils.h"

//#include "Calibration/HcalCalibAlgos/plugins/CommonUsefulStuff.h"
#include "Calibration/HcalCalibAlgos/interface/CommonUsefulStuff.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

void sumDepths(std::vector<TCell>& selectCells) {
  // Assignes teh sum of the energy in cells with the same iEta, iPhi to the cell with depth=1.
  // All cells with depth>1 are removed form the container. If
  // the cell at depth=1 is not present: create it and follow the procedure.

  if (selectCells.empty())
    return;

  std::vector<TCell> selectCellsDepth1;
  std::vector<TCell> selectCellsHighDepth;

  //
  // NB: Here we add depth 3 for iEta==16 in *HE* to the value in the barrel
  // this approach is reflected in several of the following loops: make sure
  // to check when making changes.
  //
  // In some documents it is described as having depth 1, the mapping in CMSSW uses depth 3.

  for (std::vector<TCell>::iterator i_it = selectCells.begin(); i_it != selectCells.end(); ++i_it) {
    if (HcalDetId(i_it->id()).depth() == 1) {
      selectCellsDepth1.push_back(*i_it);
    } else {
      selectCellsHighDepth.push_back(*i_it);
    }
  }

  // case where depth 1 has zero energy, but higher depths with same (iEta, iPhi) have energy.
  // For iEta<15 there is one depth -> selectCellsHighDepth is empty and we do not get in the loop.
  for (std::vector<TCell>::iterator i_it2 = selectCellsHighDepth.begin(); i_it2 != selectCellsHighDepth.end();
       ++i_it2) {
    // protect against corrupt data
    if (HcalDetId(i_it2->id()).ietaAbs() < 15 && HcalDetId(i_it2->id()).depth() > 1) {
      edm::LogWarning("HcalCalib") << "ERROR!!! there are no HB cells with depth>1 for iEta<15!\n"
                                   << "Check the input data...\nHCalDetId: " << HcalDetId(i_it2->id());
      return;
    }

    bool foundDepthOne = false;
    for (std::vector<TCell>::iterator i_it = selectCellsDepth1.begin(); i_it != selectCellsDepth1.end(); ++i_it) {
      if (HcalDetId(i_it->id()).ieta() == HcalDetId(i_it2->id()).ieta() &&
          HcalDetId(i_it->id()).iphi() == HcalDetId(i_it2->id()).iphi())
        foundDepthOne = true;
      continue;
    }
    if (!foundDepthOne) {  // create entry for depth 1 with 0 energy

      UInt_t newId;
      if (abs(HcalDetId(i_it2->id()).ieta()) == 16)
        newId = HcalDetId(HcalBarrel, HcalDetId(i_it2->id()).ieta(), HcalDetId(i_it2->id()).iphi(), 1);
      else
        newId =
            HcalDetId(HcalDetId(i_it2->id()).subdet(), HcalDetId(i_it2->id()).ieta(), HcalDetId(i_it2->id()).iphi(), 1);

      selectCellsDepth1.push_back(TCell(newId, 0.0));
    }
  }

  for (std::vector<TCell>::iterator i_it = selectCellsDepth1.begin(); i_it != selectCellsDepth1.end(); ++i_it) {
    for (std::vector<TCell>::iterator i_it2 = selectCellsHighDepth.begin(); i_it2 != selectCellsHighDepth.end();
         ++i_it2) {
      if (HcalDetId(i_it->id()).ieta() == HcalDetId(i_it2->id()).ieta() &&
          HcalDetId(i_it->id()).iphi() == HcalDetId(i_it2->id()).iphi()) {
        i_it->SetE(i_it->e() + i_it2->e());
        i_it2->SetE(0.0);  // paranoid, aren't we...
      }
    }
  }

  // replace the original vectors with the new ones
  selectCells = selectCellsDepth1;

  return;
}

void combinePhi(std::vector<TCell>& selectCells) {
  // Map: NxN -> N cluster
  // Comine the targetE of cells with the same iEta

  if (selectCells.empty())
    return;

  // new container for the TCells
  // dummy cell id created with iEta; iPhi=1; depth
  // if combinePhi() is run after combining depths, depth=1
  std::vector<TCell> combinedCells;

  std::map<UInt_t, std::vector<Float_t> > etaSliceE;  // keyed by id of cell with iEta and **iPhi=1**

  // map the cells to the eta ring
  std::vector<TCell>::iterator i_it = selectCells.begin();
  for (; i_it != selectCells.end(); ++i_it) {
    DetId id = HcalDetId(i_it->id());
    UInt_t thisKey = HcalDetId(HcalDetId(id).subdet(), HcalDetId(id).ieta(), 1, HcalDetId(id).depth());
    etaSliceE[thisKey].push_back(i_it->e());
  }

  std::map<UInt_t, std::vector<Float_t> >::iterator m_it = etaSliceE.begin();
  for (; m_it != etaSliceE.end(); ++m_it) {
    combinedCells.push_back(TCell(m_it->first, accumulate(m_it->second.begin(), m_it->second.end(), 0.0)));
  }

  // replace the original TCell vector with the new one
  selectCells = combinedCells;
}

void combinePhi(std::vector<TCell>& selectCells, std::vector<TCell>& combinedCells) {
  // Map: NxN -> N cluster
  // Comine the targetE of cells with the same iEta

  if (selectCells.empty())
    return;

  std::map<UInt_t, std::vector<Float_t> > etaSliceE;  // keyed by id of cell with iEta and **iPhi=1**

  // map the cells to the eta ring
  std::vector<TCell>::iterator i_it = selectCells.begin();
  for (; i_it != selectCells.end(); ++i_it) {
    DetId id = HcalDetId(i_it->id());
    UInt_t thisKey = HcalDetId(HcalDetId(id).subdet(), HcalDetId(id).ieta(), 1, HcalDetId(id).depth());
    etaSliceE[thisKey].push_back(i_it->e());
  }

  std::map<UInt_t, std::vector<Float_t> >::iterator m_it = etaSliceE.begin();
  for (; m_it != etaSliceE.end(); ++m_it) {
    combinedCells.push_back(TCell(m_it->first, accumulate(m_it->second.begin(), m_it->second.end(), 0.0)));
  }
}

void getIEtaIPhiForHighestE(std::vector<TCell>& selectCells, Int_t& iEtaMostE, UInt_t& iPhiMostE) {
  std::vector<TCell> summedDepthsCells = selectCells;

  sumDepths(summedDepthsCells);
  std::vector<TCell>::iterator highCell = summedDepthsCells.begin();

  // sum depths locally to get highest energy tower

  Float_t highE = -999;

  for (std::vector<TCell>::iterator it = summedDepthsCells.begin(); it != summedDepthsCells.end(); ++it) {
    if (highE < it->e()) {
      highCell = it;
      highE = it->e();
    }
  }

  iEtaMostE = HcalDetId(highCell->id()).ieta();
  iPhiMostE = HcalDetId(highCell->id()).iphi();

  return;
}

//
// Remove RecHits outside the 3x3 cluster and replace the  vector that will
// be used in the minimization. Acts on "event" level.
// This can not be done for iEta>20 due to segmentation => in principle the result should be restricted
// to iEta<20. Attempted to minimize affect at the boundary without a sharp jump.

void filterCells3x3(std::vector<TCell>& selectCells, Int_t iEtaMaxE, UInt_t iPhiMaxE) {
  std::vector<TCell> filteredCells;

  Int_t dEta, dPhi;

  for (std::vector<TCell>::iterator it = selectCells.begin(); it != selectCells.end(); ++it) {
    Bool_t passDEta = false;
    Bool_t passDPhi = false;

    dEta = HcalDetId(it->id()).ieta() - iEtaMaxE;
    dPhi = HcalDetId(it->id()).iphi() - iPhiMaxE;

    if (dPhi > 36)
      dPhi -= 72;
    if (dPhi < -36)
      dPhi += 72;

    if (abs(dEta) <= 1 || (iEtaMaxE * HcalDetId(it->id()).ieta() == -1))
      passDEta = true;

    if (abs(iEtaMaxE) <= 20) {
      if (abs(HcalDetId(it->id()).ieta()) <= 20) {
        if (abs(dPhi) <= 1)
          passDPhi = true;
      } else {
        // iPhi is labelled by odd numbers
        if (iPhiMaxE % 2 == 0) {
          if (abs(dPhi) <= 1)
            passDPhi = true;
        } else {
          if (dPhi == -2 || dPhi == 0)
            passDPhi = true;
        }
      }

    }  // if hottest cell with iEta<=20

    else {
      if (abs(HcalDetId(it->id()).ieta()) <= 20) {
        if (abs(dPhi) <= 1 || dPhi == 2)
          passDPhi = true;
      } else {
        if (abs(dPhi) <= 2)
          passDPhi = true;
      }
    }  // if hottest cell with iEta>20

    if (passDEta && passDPhi)
      filteredCells.push_back(*it);
  }

  selectCells = filteredCells;

  return;
}

//
// Remove RecHits outside the 5x5 cluster and replace the  vector that will
// be used in the minimization. Acts on "event" level
// In principle the ntuple should be produced with 5x5 already precelected
//
// Size for iEta>20 is 3x3, but the segmentation changes by x2 in phi.
// There is some bias in the selection of towers near the boundary

void filterCells5x5(std::vector<TCell>& selectCells, Int_t iEtaMaxE, UInt_t iPhiMaxE) {
  std::vector<TCell> filteredCells;

  Int_t dEta, dPhi;

  for (std::vector<TCell>::iterator it = selectCells.begin(); it != selectCells.end(); ++it) {
    dEta = HcalDetId(it->id()).ieta() - iEtaMaxE;
    dPhi = HcalDetId(it->id()).iphi() - iPhiMaxE;

    if (dPhi > 36)
      dPhi -= 72;
    if (dPhi < -36)
      dPhi += 72;

    bool passDPhi = (abs(dPhi) < 3);

    bool passDEta = (abs(dEta) < 3 || (iEtaMaxE * HcalDetId(it->id()).ieta() == -2));
    // includes  +/- eta boundary

    if (passDPhi && passDEta)
      filteredCells.push_back(*it);
  }

  selectCells = filteredCells;

  return;
}

// this is for the problematic layer near the HB/HE boundary
// sum depths 1,2 in towers 15,16

void sumSmallDepths(std::vector<TCell>& selectCells) {
  if (selectCells.empty())
    return;

  std::vector<TCell> newCells;          // holds unaffected cells to which the modified ones are added
  std::vector<TCell> manipulatedCells;  // the ones that are combined

  for (std::vector<TCell>::iterator i_it = selectCells.begin(); i_it != selectCells.end(); ++i_it) {
    if ((HcalDetId(i_it->id()).ietaAbs() == 15 && HcalDetId(i_it->id()).depth() <= 2) ||
        (HcalDetId(i_it->id()).ietaAbs() == 16 && HcalDetId(i_it->id()).depth() <= 2)) {
      manipulatedCells.push_back(*i_it);
    } else {
      newCells.push_back(*i_it);
    }
  }

  // if the list is empty there is nothing to manipulate
  // leave the original vector unchanged

  if (manipulatedCells.empty()) {
    newCells.clear();
    return;
  }

  // See what cells are needed to hold the combined information:
  // Make holders for depth=1 for each (iEta,iPhi)
  // if a cell with these values is present in "manupulatedCells"
  std::vector<UInt_t> dummyIds;     // to keep track of kreated cells
  std::vector<TCell> createdCells;  // cells that need to be added or they exists;

  for (std::vector<TCell>::iterator i_it = manipulatedCells.begin(); i_it != manipulatedCells.end(); ++i_it) {
    UInt_t dummyId =
        HcalDetId(HcalDetId(i_it->id()).subdet(), HcalDetId(i_it->id()).ieta(), HcalDetId(i_it->id()).iphi(), 1);
    if (find(dummyIds.begin(), dummyIds.end(), dummyId) == dummyIds.end()) {
      dummyIds.push_back(dummyId);
      createdCells.push_back(TCell(dummyId, 0.0));
    }
  }

  for (std::vector<TCell>::iterator i_it = createdCells.begin(); i_it != createdCells.end(); ++i_it) {
    for (std::vector<TCell>::iterator i_it2 = manipulatedCells.begin(); i_it2 != manipulatedCells.end(); ++i_it2) {
      if (HcalDetId(i_it->id()).ieta() == HcalDetId(i_it2->id()).ieta() &&
          HcalDetId(i_it->id()).iphi() == HcalDetId(i_it2->id()).iphi() && HcalDetId(i_it2->id()).depth() <= 2) {
        i_it->SetE(i_it->e() + i_it2->e());
      }
    }
  }

  for (std::vector<TCell>::iterator i_it = createdCells.begin(); i_it != createdCells.end(); ++i_it) {
    newCells.push_back(*i_it);
  }

  // replace the original vectors with the new ones
  selectCells = newCells;

  return;
}

void filterCellsInCone(std::vector<TCell>& selectCells,
                       const GlobalPoint hitPositionHcal,
                       Float_t maxConeDist,
                       const CaloGeometry* theCaloGeometry) {
  std::vector<TCell> filteredCells;

  for (std::vector<TCell>::iterator it = selectCells.begin(); it != selectCells.end(); ++it) {
    GlobalPoint recHitPoint;
    DetId id = it->id();
    if (id.det() == DetId::Hcal) {
      recHitPoint = (static_cast<const HcalGeometry*>(theCaloGeometry->getSubdetectorGeometry(id)))->getPosition(id);
    } else {
      recHitPoint = GlobalPoint(theCaloGeometry->getPosition(id));
    }

    if (getDistInPlaneSimple(hitPositionHcal, recHitPoint) <= maxConeDist)
      filteredCells.push_back(*it);
  }

  selectCells = filteredCells;

  return;
}

// From Jim H. => keep till the code is included centrally
/*
double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint) {
  
  // Simplified version of getDistInPlane
  // Assume track direction is origin -> point of hcal intersection
  
  const GlobalVector caloIntersectVector(caloPoint.x(), 
					 caloPoint.y(), 
					 caloPoint.z());

  const GlobalVector caloIntersectUnitVector = caloIntersectVector.unit();
  
  const GlobalVector rechitVector(rechitPoint.x(),
				  rechitPoint.y(),
				  rechitPoint.z());

  const GlobalVector rechitUnitVector = rechitVector.unit();

  double dotprod = caloIntersectUnitVector.dot(rechitUnitVector);
  double rechitdist = caloIntersectVector.mag()/dotprod;
  
  
  const GlobalVector effectiveRechitVector = rechitdist*rechitUnitVector;
  const GlobalPoint effectiveRechitPoint(effectiveRechitVector.x(),
					 effectiveRechitVector.y(),
					 effectiveRechitVector.z());
  
  
  GlobalVector distance_vector = effectiveRechitPoint-caloPoint;
  
  if (dotprod > 0.)
    {
      return distance_vector.mag();
    }
  else
    {
      return 999999.;
    
    }

    
}
*/
