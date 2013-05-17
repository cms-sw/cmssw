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
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  namespace module {
    typedef  unsigned char AbilitiesType;
    
    enum class Abilities {
      kGlobalCache,
      kStreamCache,
      kRunCache,
      kLuminosityBlockCache,
      kRunSummaryCache,
      kLuminosityBlockSummaryCache,
      kBeginRunProducer,
      kEndRunProducer,
      kBeginLuminosityBlockProducer,
      kEndLuminosityBlockProducer,
      kOneSharedResources,
      kOneWatchRuns,
      kOneWatchLuminosityBlocks
    };
    
    namespace AbilityBits {
      enum Bits {
        kGlobalCache=1,
        kStreamCache=2,
        kRunCache=4,
        kLuminosityBlockCache=8,
        kRunSummaryCache=16,
        kLuminosityBlockSummaryCache=32,
        kBeginRunProducer=64,
        kEndRunProducer=128,
        kOneSharedResources=256,
        kOneWatchRuns=512,
        kOneWatchLuminosityBlocks=1024
      };
    }
    
    namespace AbilityToTransitions {
      enum Bits {
        kBeginStream=AbilityBits::kStreamCache,
        kEndStream=AbilityBits::kStreamCache,
        
        kGlobalBeginRun=AbilityBits::kRunCache|AbilityBits::kRunSummaryCache|AbilityBits::kOneWatchRuns,
        kGlobalEndRun=AbilityBits::kRunCache|AbilityBits::kRunSummaryCache|AbilityBits::kEndRunProducer|AbilityBits::kOneWatchRuns,
        kStreamBeginRun=AbilityBits::kStreamCache,
        kStreamEndRun=AbilityBits::kStreamCache|AbilityBits::kRunSummaryCache,
        
        kGlobalBeginLuminosityBlock=AbilityBits::kLuminosityBlockCache|AbilityBits::kLuminosityBlockSummaryCache|AbilityBits::kOneWatchLuminosityBlocks,
        kGlobalEndLuminosityBlock=AbilityBits::kLuminosityBlockCache|AbilityBits::kLuminosityBlockSummaryCache|AbilityBits::kOneWatchLuminosityBlocks,
        kStreamBeginLuminosityBlock=AbilityBits::kStreamCache|AbilityBits::kLuminosityBlockSummaryCache,
        kStreamEndLuminosityBlock=AbilityBits::kStreamCache|AbilityBits::kLuminosityBlockSummaryCache
        
      };
    }
  }
}


#endif
