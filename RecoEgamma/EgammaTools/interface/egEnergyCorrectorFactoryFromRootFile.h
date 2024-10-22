#ifndef RecoEgamma_EgammaTools_egEnergyCorrectorFactoryFromRootFile_h
#define RecoEgamma_EgammaTools_egEnergyCorrectorFactoryFromRootFile_h
// -*- C++ -*-
//
// Package:     RecoEgamma/EgammaTools
// Class  :     egEnergyCorrectorFactoryFromRootFile
//
/**\class egEnergyCorrectorFactoryFromRootFile egEnergyCorrectorFactoryFromRootFile.h "RecoEgamma/EgammaTools/interface/egEnergyCorrectorFactoryFromRootFile.h"

 Description: function to setup initialization ofr EGEnergyCorrector based on data stored in a file

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 03 Sep 2021 18:54:50 GMT
//

// system include files
#include <string>

// user include files
#include "EGEnergyCorrector.h"

// forward declarations

EGEnergyCorrector::Initializer egEnergyCorrectorFactoryFromRootFile(const char* fileName);

#endif
