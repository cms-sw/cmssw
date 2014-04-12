//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/MuonCommonData/plugins/DDMuonAngular.h"
#include "Geometry/MuonCommonData/plugins/DDGEMAngular.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDMuonAngular,      "muon:DDMuonAngular");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDGEMAngular,       "muon:DDGEMAngular");
