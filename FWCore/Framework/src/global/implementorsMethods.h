#ifndef FWCore_Framework_global_implementorsMethods_h
#define FWCore_Framework_global_implementorsMethods_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     implementorsMethods
//
/**\file implementorsMethods.h "FWCore/Framework/src/global/implementorsMethods.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 20:13:53 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/implementors.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"

// forward declarations

namespace edm {
  namespace global {
    namespace impl {
      template <typename T>
      void BeginRunProducer<T>::doBeginRunProduce_(Run& rp, EventSetup const& c) {
        this->globalBeginRunProduce(rp, c);
      }

      template <typename T>
      void EndRunProducer<T>::doEndRunProduce_(Run& rp, EventSetup const& c) {
        this->globalEndRunProduce(rp, c);
      }

      template <typename T>
      void BeginLuminosityBlockProducer<T>::doBeginLuminosityBlockProduce_(LuminosityBlock& rp, EventSetup const& c) {
        this->globalBeginLuminosityBlockProduce(rp, c);
      }

      template <typename T>
      void EndLuminosityBlockProducer<T>::doEndLuminosityBlockProduce_(LuminosityBlock& rp, EventSetup const& c) {
        this->globalEndLuminosityBlockProduce(rp, c);
      }

      template <typename T>
      void ExternalWork<T>::doAcquire_(StreamID s,
                                       Event const& ev,
                                       edm::EventSetup const& es,
                                       WaitingTaskWithArenaHolder& holder) {
        this->acquire(s, ev, es, holder);
      }
    }  // namespace impl
  }    // namespace global
}  // namespace edm

#endif
