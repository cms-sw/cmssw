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
class HcalGeometry;

class HcalFlexiHardcodeGeometryLoader {

public:
  HcalFlexiHardcodeGeometryLoader();

  CaloSubdetectorGeometry* load(const HcalTopology& fTopology, const HcalDDDRecConstants& hcons);

private:

  struct HBHOCellParameters {
    HBHOCellParameters (int f_eta, int f_depth, int f_phi, double f_phi0, double f_dPhi, double f_rMin, double f_rMax, double f_etaMin, double f_etaMax)
    : ieta(f_eta), depth(f_depth), iphi(f_phi), phi(f_phi0), dphi(f_dPhi), rMin(f_rMin), rMax(f_rMax), etaMin(f_etaMin), etaMax(f_etaMax)
    {}

    int ieta;
    int depth;
    int iphi;
    double phi;
    double dphi;
    double rMin;
    double rMax;
    double etaMin;
    float etaMax;
  };

  struct HECellParameters {
    HECellParameters (int f_eta, int f_depth, int f_phi, double f_phi0, double f_dPhi, double f_zMin, double f_zMax, double f_etaMin, double f_etaMax)
    : ieta(f_eta), depth(f_depth), iphi(f_phi), phi(f_phi0), dphi(f_dPhi), zMin(f_zMin), zMax(f_zMax), etaMin(f_etaMin), etaMax(f_etaMax)
    {}

    int ieta;
    int depth;
    int iphi;
    double phi;
    double dphi;
    double zMin;
    double zMax;
    double etaMin;
    double etaMax;
  };

  struct HFCellParameters {
    HFCellParameters (int f_eta, int f_depth, int f_phiFirst, int f_phiStep, int f_nPhi, int f_dPhi, float f_zMin, float f_zMax, float f_rMin, float f_rMax)
    : eta(f_eta), depth(f_depth), phiFirst(f_phiFirst), phiStep(f_phiStep), nPhi(f_nPhi), dphi(f_dPhi), zMin(f_zMin), zMax(f_zMax), rMin(f_rMin), rMax(f_rMax)
    {}

    int eta;
    int depth;
    int phiFirst;
    int phiStep;
    int nPhi;
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
  std::vector <HFCellParameters> makeHFCells (const HcalDDDRecConstants& hcons);

  void fillHBHO (HcalGeometry* fGeometry, const std::vector <HBHOCellParameters>& fCells, bool fHB);
  void fillHE (HcalGeometry* fGeometry, const std::vector <HECellParameters>& fCells);
  void fillHF (HcalGeometry* fGeometry, const std::vector <HFCellParameters>& fCells);

  int    MAX_HCAL_PHI;
  double DEGREE2RAD;
  bool   isBH_;
};

#endif
