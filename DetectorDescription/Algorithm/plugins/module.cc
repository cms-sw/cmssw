#include <string>

#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Algorithm/interface/DDAngular.h"
#include "DetectorDescription/Algorithm/interface/DDLinear.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

DEFINE_EDM_PLUGIN( DDAlgorithmFactory, DDAngular, "global:DDAngular");
DEFINE_EDM_PLUGIN( DDAlgorithmFactory, DDLinear, "global:DDLinear");

