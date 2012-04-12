#ifndef CalibMuon_DTTTrigCorrectionFactory_H
#define CalibMuon_DTTTrigCorrectionFactory_H

/** \class DTTTrigCorrectionFactory
 *  Factory of seal plugins for TTrig DB corrections.
 *  The plugins are concrete implementations of DTTTrigBaseCorrection case class.
 *
 *  $Date: 2007/04/17 22:46:21 $
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
}
class DTTTrigBaseCorrection;

typedef edmplugin::PluginFactory<DTTTrigBaseCorrection *(const edm::ParameterSet &)> DTTTrigCorrectionFactory;
#endif

