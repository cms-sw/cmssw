#ifndef FWCore_Framework_ModuleAbilityEnums_h
#define FWCore_Framework_ModuleAbilityEnums_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleAbilityEnums
//
/**\class ModuleAbilityEnums ModuleAbilityEnums.h "FWCore/Framework/interface/ModuleAbilityEnums.h"

 Description: Enums used internally by framework to determine abilities of a module

 Usage:
    These are used internally by the framework

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 07 May 2013 18:11:24 GMT
//

// system include files

// user include files

// forward declarations
namespace edm {
  namespace module {
    typedef unsigned char AbilitiesType;

    enum class Abilities {
      kGlobalCache,
      kStreamCache,
      kInputProcessBlockCache,
      kRunCache,
      kLuminosityBlockCache,
      kRunSummaryCache,
      kLuminosityBlockSummaryCache,
      kWatchProcessBlock,
      kBeginProcessBlockProducer,
      kEndProcessBlockProducer,
      kBeginRunProducer,
      kEndRunProducer,
      kBeginLuminosityBlockProducer,
      kEndLuminosityBlockProducer,
      kOneSharedResources,
      kOneWatchRuns,
      kOneWatchLuminosityBlocks,
      kStreamWatchRuns,
      kStreamWatchLuminosityBlocks,
      kWatchInputFiles,
      kExternalWork,
      kAccumulator,
      kTransformer
    };
  }  // namespace module
}  // namespace edm

#endif
