#include <string>

#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DDAngular.h"
#include "DDLinear.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

DEFINE_EDM_PLUGIN( DDAlgorithmFactory, DDAngular, "global:DDAngular");
DEFINE_EDM_PLUGIN( DDAlgorithmFactory, DDLinear, "global:DDLinear");

