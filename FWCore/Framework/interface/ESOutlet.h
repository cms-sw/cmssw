#ifndef Framework_ESOutlet_h
#define Framework_ESOutlet_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESOutlet
// 
/**\class ESOutlet ESOutlet.h FWCore/Framework/interface/ESOutlet.h

 Description: An outlet which gets its data from the EventSetup and passes it to an edm::ExtensionCord

 Usage:
    If you have a framework module (e.g. an EDProducer) which internally holds objects and these objects need access
to data in the EventSetup then you can use the edm::ESOutlet and an edm::ExtensionCord to pass the EventSetup data
from the EDProducer to the object which needs the data.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 22 15:07:28 EDT 2006
// $Id: ESOutlet.h,v 1.1 2006/09/22 19:54:49 chrjones Exp $
//

// system include files
#include <string>

// user include files
#include "FWCore/Utilities/interface/OutletBase.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations

namespace edm {
  template <class T, class TRec>
  class ESOutlet :private OutletBase<T>
  {
    class Getter : public extensioncord::ECGetterBase<T> {
public:
      Getter(const edm::EventSetup& iES,
             const std::string& iLabel = std::string()) :
      es_(&iES),
      label_(iLabel) {}
private:
      virtual const T* getImpl() const {
        ESHandle<T> data;
        es_->template get<TRec>().get(label_,data);
        return &(*data);
      }
      const edm::EventSetup* es_;
      const std::string label_;
    };
    
    
   public:
      ESOutlet(const edm::EventSetup& iES,
               ExtensionCord<T>& iCord):
       OutletBase<T>(iCord),
       getter_(iES)  {
         this->setGetter(&getter_);
      }

      ESOutlet( const edm::EventSetup& iES,
              const std::string& iLabel,
              ExtensionCord<T>& iCord):
       getter_(iES,iLabel) {
         this->setGetter( &getter_);
      }
    
    //virtual ~ESOutlet();

   private:
      ESOutlet(const ESOutlet&); // stop default

      const ESOutlet& operator=(const ESOutlet&); // stop default

      // ---------- member data --------------------------------
      Getter getter_;

  };
}

#endif
