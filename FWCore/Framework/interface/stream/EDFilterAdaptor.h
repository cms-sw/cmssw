#ifndef FWCore_Framework_stream_EDFilterAdaptor_h
#define FWCore_Framework_stream_EDFilterAdaptor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDFilterAdaptor
// 
/**\class edm::stream::EDFilterAdaptor EDFilterAdaptor.h "EDFilterAdaptor.h"

 Description: Adapts an edm::stream::EDFilter<> to work with an edm::Worker

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:18 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDFilterAdaptorBase.h"
#include "FWCore/Framework/interface/stream/ProducingModuleAdaptor.h"
#include "FWCore/Framework/src/MakeModuleHelper.h"

// forward declarations

namespace edm {
  namespace stream {

    template<typename T> using EDFilterAdaptor = ProducingModuleAdaptor<T,EDFilterBase, EDFilterAdaptorBase>;

    template<typename ABase, typename ModType> struct BaseToAdaptor;

    template<typename ModType> struct BaseToAdaptor<EDFilterAdaptorBase,ModType> {
      typedef EDFilterAdaptor<ModType> Type;
    };
  }
  
  template<>
  class MakeModuleHelper<edm::stream::EDFilterAdaptorBase>
  {
    typedef edm::stream::EDFilterAdaptorBase Base;
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
