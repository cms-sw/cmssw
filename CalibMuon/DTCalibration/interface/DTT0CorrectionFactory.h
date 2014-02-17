#ifndef CalibMuon_DTT0CorrectionFactory_H
#define CalibMuon_DTT0CorrectionFactory_H

/** \class DTT0CorrectionFactory
 *  Factory of seal plugins for TTrig DB corrections.
 *  The plugins are concrete implementations of DTT0BaseCorrection case class.
 *
 *  $Date: 2012/03/02 19:47:31 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
  class ParameterSet;
}
namespace dtCalibration {
  class DTT0BaseCorrection;
}

typedef edmplugin::PluginFactory<dtCalibration::DTT0BaseCorrection *(const edm::ParameterSet &)> DTT0CorrectionFactory;
#endif
