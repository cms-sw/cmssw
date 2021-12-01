#ifndef CalibMuon_DTT0CorrectionFactory_H
#define CalibMuon_DTT0CorrectionFactory_H

/** \class DTT0CorrectionFactory
 *  Factory of seal plugins for TTrig DB corrections.
 *  The plugins are concrete implementations of DTT0BaseCorrection case class.
 *
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm
namespace dtCalibration {
  class DTT0BaseCorrection;
}

typedef edmplugin::PluginFactory<dtCalibration::DTT0BaseCorrection *(const edm::ParameterSet &, edm::ConsumesCollector)>
    DTT0CorrectionFactory;
#endif
