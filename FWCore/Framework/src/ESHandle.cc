#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
      eventsetup::ComponentDescription const* 
      ESHandleBase::description() const { 
         if(!description_) {
            throw edm::Exception(edm::errors::InvalidReference,"NullPointer");
         }
         return description_; 
      }
      
}
