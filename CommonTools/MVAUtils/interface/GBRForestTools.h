#ifndef CommonTools_MVAUtils_GBRForestTools_h
#define CommonTools_MVAUtils_GBRForestTools_h

//--------------------------------------------------------------------------------------------------
//
// GRBForestTools
//
// Utility to parse an XML weights files specifying an ensemble of decision trees into a GRBForest.
//
// Author: Jonas Rembser
//--------------------------------------------------------------------------------------------------


#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>

// Create a GBRForest from an XML weight file
std::unique_ptr<const GBRForest> createGBRForest(const std::string     &weightsFile);
std::unique_ptr<const GBRForest> createGBRForest(const edm::FileInPath &weightsFile);

// Overloaded versions which are taking string vectors by reference to strore the variable names in
std::unique_ptr<const GBRForest> createGBRForest(const std::string     &weightsFile, std::vector<std::string> &varNames);
std::unique_ptr<const GBRForest> createGBRForest(const edm::FileInPath &weightsFile, std::vector<std::string> &varNames);

#endif
