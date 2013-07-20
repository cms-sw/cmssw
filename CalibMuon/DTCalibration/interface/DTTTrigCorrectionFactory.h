#ifndef CalibMuon_DTTTrigCorrectionFactory_H
#define CalibMuon_DTTTrigCorrectionFactory_H

/** \class DTTTrigCorrectionFactory
 *  Factory of seal plugins for TTrig DB corrections.
 *  The plugins are concrete implementations of DTTTrigBaseCorrection case class.
 *
 *  $Date: 2012/03/02 19:47:32 $
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
}
namespace dtCalibration {
  class DTTTrigBaseCorrection;
}

typedef edmplugin::PluginFactory<dtCalibration::DTTTrigBaseCorrection *(const edm::ParameterSet &)> DTTTrigCorrectionFactory;
#endif
