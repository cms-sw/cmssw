#ifndef FWCore_Framework_stream_StreamWorker_h
#define FWCore_Framework_stream_StreamWorker_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     StreamWorker
// 
/**\class StreamWorker StreamWorker.h "StreamWorker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 05 Aug 2013 14:04:38 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/stream/EDProducerAdaptor.h"

// forward declarations
namespace edm {
  namespace stream {
    template<typename T>
    class StreamWorker;
    
    template<>
    class StreamWorker<EDProducerAdaptorBase> : public WorkerT<EDProducerAdaptorBase>
    {
      
    public:
      //Doesn't work in gcc4.7 using WorkerT<EDProducerAdaptorBase>::WorkerT;
      StreamWorker(std::unique_ptr<EDProducerAdaptorBase>&& iMod,
                   ModuleDescription const& iDesc,
                   WorkerParams const& iParams):
      WorkerT(std::move(iMod), iDesc,iParams) {}


      template<typename ModType>
      static std::unique_ptr<EDProducerAdaptorBase> makeModule(ModuleDescription const&,
                                                               ParameterSet const& pset) {
        std::unique_ptr<EDProducerAdaptor<ModType>> module = std::unique_ptr<EDProducerAdaptor<ModType>>(new EDProducerAdaptor<ModType>(pset));
        return std::unique_ptr<EDProducerAdaptorBase>(module.release());
      }
    };
    
  }
}


#endif
