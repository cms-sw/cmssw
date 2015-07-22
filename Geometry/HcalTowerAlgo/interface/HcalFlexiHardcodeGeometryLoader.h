#ifndef GEOMETRY_HCALTOWERALGO_HCALFLEXIHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_HCALFLEXIHARDCODEGEOMETRYLOADER_H 1

/** \class HcalFlexiHardcodeGeometryLoader
 *
 * \author F.Ratnikov, UMd
*/

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>

class HcalTopology;
class HcalDDDRecConstants;

class HcalFlexiHardcodeGeometryLoader {

public:
  HcalFlexiHardcodeGeometryLoader(const edm::ParameterSet&);
  
  CaloSubdetectorGeometry* load(const HcalTopology& fTopology, const HcalDDDRecConstants& hcons);

private:

  struct HBHOCellParameters {
    HBHOCellParameters (int f_eta, int f_depth, int f_phiFirst, int f_nphi, int f_phiStep, double f_phiStart, double f_dPhi, double f_rMin, double f_rMax, double f_etaMin, double f_etaMax)
    : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), nPhi(f_nphi), phiStep(f_phiStep), phiStart(f_phiStart), dphi(f_dPhi), rMin(f_rMin), rMax(f_rMax), etaMin(f_etaMin), etaMax(f_etaMax)
    {}
 
    int eta;
    int depth;
    int phiFirst;
    int nPhi;
    int phiStep;
    double phiStart;
    double dphi;
    double rMin;
    double rMax;
    double etaMin;
    float etaMax;
  };

  struct HECellParameters {
    HECellParameters (int f_eta, int f_depth, int f_phiFirst, int f_nphi, int f_phiStep, double f_phiStart, double f_dPhi, double f_zMin, double f_zMax, double f_etaMin, double f_etaMax)
    : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), nPhi(f_nphi), phiStep(f_phiStep), phiStart(f_phiStart), dphi(f_dPhi), zMin(f_zMin), zMax(f_zMax), etaMin(f_etaMin), etaMax(f_etaMax)
    {}
 
    int eta;
    int depth;
    int phiFirst;
    int nPhi;
    int phiStep;
    double phiStart;
    double dphi;
    double zMin;
    double zMax;
    double etaMin;
    double etaMax;
  };

  struct HFCellParameters {
    HFCellParameters (int f_eta, int f_depth, int f_phiFirst, int f_phiStep, int f_dPhi, float f_zMin, float f_zMax, float f_rMin, float f_rMax)
      : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), phiStep(f_phiStep), dphi(f_dPhi), zMin(f_zMin), zMax(f_zMax), rMin(f_rMin), rMax(f_rMax)
    {}
 
    int eta;
    int depth;
    int phiFirst;
    int phiStep;
    int dphi;
    float zMin;
    float zMax;
    float rMin;
    float rMax;
  };

  std::vector <HBHOCellParameters> makeHBCells (const HcalDDDRecConstants& hcons);
  std::vector <HBHOCellParameters> makeHOCells ();
  std::vector <HECellParameters> makeHECells (const HcalDDDRecConstants& hcons);
  std::vector <HECellParameters> makeHECells_H2 ();
  std::vector <HFCellParameters> makeHFCells ();

  void fillHBHO (CaloSubdetectorGeometry* fGeometry, const std::vector <HBHOCellParameters>& fCells, bool fHB);
  void fillHE (CaloSubdetectorGeometry* fGeometry, const std::vector <HECellParameters>& fCells);
  void fillHF (CaloSubdetectorGeometry* fGeometry, const std::vector <HFCellParameters>& fCells);

  int    MAX_HCAL_PHI;
  double DEGREE2RAD;
};

#endif
