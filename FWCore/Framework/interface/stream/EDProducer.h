#ifndef FWCore_Framework_stream_EDProducer_h
#define FWCore_Framework_stream_EDProducer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDProducer
// 
/**\class edm::stream::EDProducer EDProducer.h "FWCore/Framework/interface/stream/EDProducer.h"

 Description: Base class for stream based EDProducers

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
#include "FWCore/Framework/interface/stream/EDProducerBase.h"
// forward declarations
namespace edm {
  namespace stream {
    template< typename... T>
    class EDProducer : public AbilityToImplementor<T>::Type...,
                       public EDProducerBase
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
      
      
      EDProducer() = default;
      //virtual ~EDProducer();
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      
    private:
      EDProducer(const EDProducer&) = delete; // stop default
      
      const EDProducer& operator=(const EDProducer&) = delete; // stop default
      
      // ---------- member data --------------------------------
      
    };
    
  }
}


#endif
