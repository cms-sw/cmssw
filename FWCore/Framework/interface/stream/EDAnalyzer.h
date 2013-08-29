#ifndef FWCore_Framework_stream_EDAnalyzer_h
#define FWCore_Framework_stream_EDAnalyzer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDAnalyzer
// 
/**\class edm::stream::EDAnalyzer EDAnalyzer.h "FWCore/Framework/interface/stream/EDAnalyzer.h"

 Description: Base class for stream based EDAnalyzers

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 21:41:42 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/AbilityToImplementor.h"
#include "FWCore/Framework/interface/stream/CacheContexts.h"
#include "FWCore/Framework/interface/stream/Contexts.h"
#include "FWCore/Framework/interface/stream/AbilityChecker.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerBase.h"
// forward declarations
namespace edm {
  namespace stream {
    template< typename... T>
    class EDAnalyzer : public AbilityToImplementor<T>::Type...,
                       public EDAnalyzerBase
    {
      
    public:
      typedef CacheContexts<T...> CacheTypes;
      
      typedef typename CacheTypes::GlobalCache GlobalCache;
      typedef typename CacheTypes::RunCache RunCache;
      typedef typename CacheTypes::LuminosityBlockCache LuminosityBlockCache;
      typedef RunContextT<RunCache,GlobalCache> RunContext;
      typedef LuminosityBlockContextT<LuminosityBlockCache,
                                     RunCache,
                                     GlobalCache> LuminosityBlockContext;
      typedef typename CacheTypes::RunSummaryCache RunSummaryCache;
      typedef typename CacheTypes::LuminosityBlockSummaryCache LuminosityBlockSummaryCache;

      typedef AbilityChecker<T...> HasAbility;
      
      
      EDAnalyzer() = default;
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
    protected:
      using EDAnalyzerBase::currentContext;
      using EDAnalyzerBase::callWhenNewProductsRegistered;
      
    private:
      EDAnalyzer(const EDAnalyzer&) = delete; // stop default
      
      const EDAnalyzer& operator=(const EDAnalyzer&) = delete; // stop default
      
      // ---------- member data --------------------------------
      
    };
    
  }
}


#endif
