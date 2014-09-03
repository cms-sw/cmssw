#ifndef FWCore_Framework_stream_makeGlobal_h
#define FWCore_Framework_stream_makeGlobal_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     makeGlobal
// 
/** Helper functions for making stream modules

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 22 May 2014 13:55:01 GMT
//

// system include files

// user include files

// forward declarations

namespace edm {
  namespace stream {
    namespace impl {
      template<typename T, typename G>
      std::unique_ptr<G> makeGlobal(edm::ParameterSet const& iPSet, G const*) {
        return T::initializeGlobalCache(iPSet);
      }
      template<typename T>
      dummy_ptr makeGlobal(edm::ParameterSet const& iPSet, void const*) {
        return dummy_ptr();
      }
      
      template< typename T, typename G>
      T* makeStreamModule(edm::ParameterSet const& iPSet,
                          G const* iGlobal) {
        return new T(iPSet,iGlobal);
      }
      
      template< typename T>
      T* makeStreamModule(edm::ParameterSet const& iPSet,
                          void const* ) {
        return new T(iPSet);
      }
    }
  }
}
#endif
