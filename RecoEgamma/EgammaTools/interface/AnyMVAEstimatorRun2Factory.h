#ifndef RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Factory_H
#define RecoEgamma_EgammaTools_AnyMVAEstimatorRun2Factory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

// This plugin factory typedef is not defined in the main header file
// "AnyMVAEstimatorRun2Base.h", because there are usecases of generating the
// dictionaries for AnyMVAEstimatorRun2Base on the fly (see notes in
// ElectronMVAEstimatorRun2.h for more details). This doesn't work if
// PluginFactory.h is included in the header file because of conflicting C++
// modules.

typedef edmplugin::PluginFactory<AnyMVAEstimatorRun2Base*(const edm::ParameterSet&)> AnyMVAEstimatorRun2Factory;

#endif
