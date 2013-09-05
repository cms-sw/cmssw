#ifndef FWCore_Framework_one_implementorsMethods_h
#define FWCore_Framework_one_implementorsMethods_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     implementorsMethods
// 
/**\file implementorsMethods.h "FWCore/Framework/src/one/implementorsMethods.h"

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
#include "FWCore/Framework/interface/one/implementors.h"

// forward declarations

namespace edm {
  namespace one {
    namespace impl {
      template< typename T>
      void RunWatcher<T>::doBeginRun_(Run const& rp, EventSetup const& c) {
        this->beginRun(rp,c);
      }
      template< typename T>
      void RunWatcher<T>::doEndRun_(Run const& rp, EventSetup const& c) {
        this->endRun(rp,c);
      }

      
      template< typename T>
      void LuminosityBlockWatcher<T>::doBeginLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) {
        this->beginLuminosityBlock(rp,c);
      }
      template< typename T>
      void LuminosityBlockWatcher<T>::doEndLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) {
        this->endLuminosityBlock(rp,c);
      }

      template< typename T>
      void BeginRunProducer<T>::doBeginRunProduce_(Run& rp, EventSetup const& c) {
        this->beginRunProduce(rp,c);
      }

      template< typename T>
      void EndRunProducer<T>::doEndRunProduce_(Run& rp, EventSetup const& c) {
        this->endRunProduce(rp,c);
      }

      template< typename T>
      void BeginLuminosityBlockProducer<T>::doBeginLuminosityBlockProduce_(LuminosityBlock& rp, EventSetup const& c) {
        this->beginLuminosityBlockProduce(rp,c);
      }
      
      template< typename T>
      void EndLuminosityBlockProducer<T>::doEndLuminosityBlockProduce_(LuminosityBlock& rp, EventSetup const& c) {
        this->endLuminosityBlockProduce(rp,c);
      }
    }
  }
}


#endif
