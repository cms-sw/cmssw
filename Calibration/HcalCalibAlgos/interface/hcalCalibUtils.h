#ifndef hcalCalibUtils_h
#define hcalCalibUtils_h

#include <vector>
#include <map>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "TCell.h"

void sumDepths(std::vector<TCell> &selectCells);   // replaces original collection
void combinePhi(std::vector<TCell> &selectCells);  // replaces original collection

void combinePhi(std::vector<TCell> &selectCells, std::vector<TCell> &combinedCells);

void getIEtaIPhiForHighestE(std::vector<TCell>& selectCells, Int_t& iEta, UInt_t& iPhi);
void filterCells3x3        (std::vector<TCell>& selectCells, Int_t iEta, UInt_t iPhi);
void filterCells5x5        (std::vector<TCell>& selectCells, Int_t iEta, UInt_t iPhi);

//void makeTextFile(std::map<Int_t, Float_t> &coef);


// This function is provided in the new version of HcalDetId (not included yet as of 2_1_10)
// remove when the it is included in the releases - also the implementation in the .cc file
bool validDetId( HcalSubdetector sd, int ies, int ip, int dp); 

#endif
