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
#include "FWCore/Framework/interface/stream/StreamWorker.h"
// forward declarations

namespace edm {
  namespace stream {

    template<typename T> using EDFilterAdaptor = ProducingModuleAdaptor<T,EDFilterBase, EDFilterAdaptorBase>;

    template<typename ModType> struct BaseToAdaptor<EDFilterAdaptorBase,ModType> {
      typedef EDFilterAdaptor<ModType> Type;
    };
  }
}

#endif
