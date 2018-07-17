//--------------------------------------------------------------------------------------------------
//
// GRBForestTools
//
// Utility to read a TMVA weights file with a BDT into a GRBForest.
//
// Author: Jonas Rembser
//--------------------------------------------------------------------------------------------------


#ifndef RecoEgamma_EgammaTools_GBRForestTools_h
#define RecoEgamma_EgammaTools_GBRForestTools_h

#include <vector>
#include <string>

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TMVA/MethodBDT.h"
#include "TMVA/Reader.h"

#include "CommonTools/Utils/interface/TMVAZipReader.h"

class GBRForestTools
{
  public:
    GBRForestTools() {}

    static std::unique_ptr<const GBRForest> createGBRForest(const std::string &weightFile);
    static std::unique_ptr<const GBRForest> createGBRForest(const edm::FileInPath &weightFile);

    // Overloaded versions which are taking string vectors by reference to strore the variable names in
    static std::unique_ptr<const GBRForest> createGBRForest(const std::string &weightFile, std::vector<std::string> &varNames);
    static std::unique_ptr<const GBRForest> createGBRForest(const edm::FileInPath &weightFile, std::vector<std::string> &varNames);

};

#endif
