#ifndef hcalCalibUtils_h
#define hcalCalibUtils_h

#include <vector>
#include <map>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Calibration/HcalCalibAlgos/src/TCell.h"


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"


void sumDepths(std::vector<TCell> &selectCells);   // replaces original collection

// sum "small" depths in towers 16,17: depths 1,2 in HB
void sumSmallDepths(std::vector<TCell> &selectCells);   // replaces original collection

void combinePhi(std::vector<TCell> &selectCells);  // replaces original collection

void combinePhi(std::vector<TCell> &selectCells, std::vector<TCell> &combinedCells);



void getIEtaIPhiForHighestE(std::vector<TCell>& selectCells, Int_t& iEta, UInt_t& iPhi);
void filterCells3x3        (std::vector<TCell>& selectCells, Int_t iEta, UInt_t iPhi);
void filterCells5x5        (std::vector<TCell>& selectCells, Int_t iEta, UInt_t iPhi);


void filterCellsInCone     (std::vector<TCell>& selectCells, const GlobalPoint hitPositionHcal, 
			    Float_t maxConeDist, const CaloGeometry* theCaloGeometry);

// Jim's implementation fo the cone

double getDistInPlaneSimple(const GlobalPoint caloPoint, const GlobalPoint rechitPoint);



//void makeTextFile(std::map<Int_t, Float_t> &coef);

#endif
