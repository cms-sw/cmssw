#ifndef Alignment_CommonAlignmentAlgorithm_IntegratedCalibrationPluginFactory_h
#define Alignment_CommonAlignmentAlgorithm_IntegratedCalibrationPluginFactory_h

/// \class IntegratedCalibrationPluginfactory
///  Plugin factory for calibrations integrated into the alignment framework
///
///  \author G. Flucke - DESY
///  date: July 2012
///  $Revision: 1.1 $
///  $Date: 2012/08/10 09:07:21 $
///  (last update by $Author: flucke $)

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

typedef edmplugin::PluginFactory<IntegratedCalibrationBase* (const edm::ParameterSet&)>
		  IntegratedCalibrationPluginFactory;

#endif
