#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>

//--------------------------------------------------------------------------------------------------
//
// GRBForestTools
//
// Utility to parse an XML weights files specifying an ensemble of decision trees into a GRBForest.
//
// Author: Jonas Rembser
//--------------------------------------------------------------------------------------------------


// Create a GBRForest from an XML weight file
std::unique_ptr<const GBRForest> createGBRForest(const std::string     &weightFile);
std::unique_ptr<const GBRForest> createGBRForest(const edm::FileInPath &weightFile);

// Overloaded versions which are taking string vectors by reference to strore the variable names in
std::unique_ptr<const GBRForest> createGBRForest(const std::string     &weightFile, std::vector<std::string> &varNames);
std::unique_ptr<const GBRForest> createGBRForest(const edm::FileInPath &weightFile, std::vector<std::string> &varNames);
