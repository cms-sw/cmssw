#ifndef FWCore_Framework_stream_EDProducerAdaptor_h
#define FWCore_Framework_stream_EDProducerAdaptor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDProducerAdaptor
// 
/**\class edm::stream::EDProducerAdaptor EDProducerAdaptor.h "EDProducerAdaptor.h"

 Description: Adapts an edm::stream::EDProducer<> to work with an edm::Worker

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:18 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"
#include "FWCore/Framework/interface/stream/ProducingModuleAdaptor.h"
#include "FWCore/Framework/src/MakeModuleHelper.h"
// forward declarations

namespace edm {
  namespace stream {
    template<typename ABase, typename ModType> struct BaseToAdaptor;

    template<typename T> using EDProducerAdaptor = ProducingModuleAdaptor<T,EDProducerBase, EDProducerAdaptorBase>;

    template<typename ModType> struct BaseToAdaptor<EDProducerAdaptorBase,ModType> {
      typedef EDProducerAdaptor<ModType> Type;
    };
  }
  
  template<>
  class MakeModuleHelper<edm::stream::EDProducerAdaptorBase>
  {
    typedef edm::stream::EDProducerAdaptorBase Base;
  public:
    template<typename ModType>
    static std::unique_ptr<Base> makeModule(ParameterSet const& pset) {
      typedef typename stream::BaseToAdaptor<Base,ModType>::Type Adaptor;
      std::unique_ptr<Adaptor> module = std::unique_ptr<Adaptor>(new Adaptor(pset));
      return std::unique_ptr<Base>(module.release());
    }
  };
}

#endif
