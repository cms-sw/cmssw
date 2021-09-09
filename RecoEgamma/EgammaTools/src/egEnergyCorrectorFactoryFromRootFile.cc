// -*- C++ -*-
//
// Package:     RecoEgamma/EgammaTools
// Class  :     egEnergyCorrectorFactoryFromRootFile
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 03 Sep 2021 18:57:07 GMT
//

// system include files
#include <TFile.h>

// user include files
#include "RecoEgamma/EgammaTools/interface/egEnergyCorrectorFactoryFromRootFile.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"

//
// constants, enums and typedefs
//

EGEnergyCorrector::Initializer egEnergyCorrectorFactoryFromRootFile(const char *fileName) {
  EGEnergyCorrector::Initializer ret;
  std::unique_ptr<TFile> fgbr(TFile::Open(fileName, "READ"));
  ret.readereb_.reset((GBRForest *)fgbr->Get("EBCorrection"));
  ret.readerebvariance_.reset((GBRForest *)fgbr->Get("EBUncertainty"));
  ret.readeree_.reset((GBRForest *)fgbr->Get("EECorrection"));
  ret.readereevariance_.reset((GBRForest *)fgbr->Get("EEUncertainty"));
  fgbr->Close();
  return ret;
}
