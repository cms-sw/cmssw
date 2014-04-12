#ifndef CalibMuon_DTCalibration_DTVDriftPluginFactory_h
#define CalibMuon_DTCalibration_DTVDriftPluginFactory_h

/** \class DTVDriftPluginFactory
 *  Factory of seal plugins for vDrfit computation.
 *  The plugins are concrete implementations of DTVDriftBaseAlgo.
 *
 *  \author A. Vilela Pereira
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
}
namespace dtCalibration {
  class DTVDriftBaseAlgo;
}

typedef edmplugin::PluginFactory<dtCalibration::DTVDriftBaseAlgo *(const edm::ParameterSet &)> DTVDriftPluginFactory;
#endif
