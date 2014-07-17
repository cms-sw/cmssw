#ifndef Framework_ESWatcher_h
#define Framework_ESWatcher_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESWatcher
// 
/**\class ESWatcher ESWatcher.h FWCore/Framework/interface/ESWatcher.h

 Description: Watches an EventSetup Record and reports when it changes

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 22 18:19:24 EDT 2006
//

// system include files
#include <functional>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations

namespace edm {
  template <class T>
  class ESWatcher
  {

   public:
    
    struct NullFunction {
      void operator()(const T& ) {}
    };
    
    ESWatcher() :callback_(NullFunction()), cacheId_(0) {}
    
    template <class TFunc>
    ESWatcher(TFunc iFunctor):callback_(iFunctor),cacheId_(0) {}
    
    template <class TObj, class TMemFunc>
    ESWatcher(TObj const& iObj, TMemFunc iFunc):
    callback_(std::bind(iFunc,iObj,std::placeholders::_1)),
    cacheId_(0)
     {}
      //virtual ~ESWatcher();

      // ---------- const member functions ---------------------
    
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      bool check(const edm::EventSetup& iSetup) {
        const T& record = iSetup.template get<T>();
        bool result = cacheId_ != record.cacheIdentifier();
        if(result) {
          callback_(record);
        }
        cacheId_ = record.cacheIdentifier();
        return result;
      }
   private:
      ESWatcher(const ESWatcher&); // stop default

      const ESWatcher& operator=(const ESWatcher&); // stop default

      // ---------- member data --------------------------------
      std::function<void (const T&)> callback_;
      unsigned long long cacheId_;
};
}

#endif
