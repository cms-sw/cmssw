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
#include "FWCore/Framework/interface/stream/StreamWorker.h"
// forward declarations

namespace edm {
  namespace stream {
    template<typename T> using EDProducerAdaptor = ProducingModuleAdaptor<T,EDProducerBase, EDProducerAdaptorBase>;

    template<typename ModType> struct BaseToAdaptor<EDProducerAdaptorBase,ModType> {
      typedef EDProducerAdaptor<ModType> Type;
    };
  }
}

#endif
