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

// forward declarations
namespace edm {
  namespace stream {
    template<typename ABase, typename ModType> struct BaseToAdaptor;
    
    template<typename T>
    class StreamWorker : public WorkerT<T>
    {
      
    public:
      //Doesn't work in gcc4.7 using WorkerT<EDProducerAdaptorBase>::WorkerT;
      StreamWorker(std::unique_ptr<T>&& iMod,
                   ModuleDescription const& iDesc,
                   WorkerParams const& iParams):
      WorkerT<T>(std::move(iMod), iDesc,iParams) {}


      template<typename ModType>
      static std::unique_ptr<T> makeModule(ModuleDescription const&,
                                                               ParameterSet const& pset) {
        typedef typename BaseToAdaptor<T,ModType>::Type Adaptor;
        std::unique_ptr<Adaptor> module = std::unique_ptr<Adaptor>(new Adaptor(pset));
        return std::unique_ptr<T>(module.release());
      }
    };
    
  }
}


#endif
