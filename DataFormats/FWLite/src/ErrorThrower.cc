// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ErrorThrower
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 23 10:06:39 EDT 2008
//

// system include files

// user include files
#include "DataFormats/FWLite/interface/ErrorThrower.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include <ostream>

using namespace fwlite;
//
// constants, enums and typedefs
//
namespace {
   class NoProductErrorThrower : public ErrorThrower {
   public:
      NoProductErrorThrower(const std::type_info& iType, const char*iModule, const char*iInstance, const char*iProcess):
      type_(&iType), module_(iModule), instance_(iInstance), process_(iProcess) {}
      
      void throwIt() const override {

         edm::TypeID type(*type_);
         throw edm::Exception(edm::errors::ProductNotFound)<<"A branch was found for \n  type ='"<<type.className()<<"'\n  module='"<<module_
         <<"'\n  productInstance='"<<((0!=instance_)?instance_:"")<<"'\n  process='"<<((0!=process_)?process_:"")<<"'\n"
         "but no data is available for this Event";
      }
      virtual ErrorThrower* clone() const override {
         return new NoProductErrorThrower(*this);
      }

   private:
      const std::type_info* type_;
      const char* module_;
      const char* instance_;
      const char* process_;
   };
   
   class NoBranchErrorThrower : public ErrorThrower {
   public:
      NoBranchErrorThrower(const std::type_info& iType, const char*iModule, const char*iInstance, const char*iProcess):
      type_(&iType), module_(iModule), instance_(iInstance), process_(iProcess) {}
      
      void throwIt() const override {
         
         edm::TypeID type(*type_);
         throw edm::Exception(edm::errors::ProductNotFound)<<"No branch was found for \n  type ='"<<type.className()<<"'\n  module='"<<module_
         <<"'\n  productInstance='"<<((0!=instance_)?instance_:"")<<"'\n  process='"<<((0!=process_)?process_:"")<<"'";
      }
      
      virtual ErrorThrower* clone() const override {
         return new NoBranchErrorThrower(*this);
      }
      
   private:
      const std::type_info* type_;
      const char* module_;
      const char* instance_;
      const char* process_;
   };
   
   class UnsetErrorThrower : public ErrorThrower {
      void throwIt() const override {
         throw cms::Exception("UnsetHandle")<<"The fwlite::Handle was never set";
      }
      
      virtual ErrorThrower* clone() const override {
         return new UnsetErrorThrower(*this);
      }
      
   };
}
//
// static data member definitions
//

//
// constructors and destructor
//
ErrorThrower::ErrorThrower()
{
}

// ErrorThrower::ErrorThrower(const ErrorThrower& rhs)
// {
//    // do actual copying here;
// }

ErrorThrower::~ErrorThrower()
{
}

//
// assignment operators
//
// const ErrorThrower& ErrorThrower::operator=(const ErrorThrower& rhs)
// {
//   //An exception safe implementation is
//   ErrorThrower temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
ErrorThrower* 
ErrorThrower::unsetErrorThrower() {
   return new UnsetErrorThrower();
}

ErrorThrower* 
ErrorThrower::errorThrowerBranchNotFoundException(const std::type_info& iType, const char* iModule, const char* iInstance, const char* iProcess){
   return new NoBranchErrorThrower(iType,iModule,iInstance,iProcess);
}

ErrorThrower* 
ErrorThrower::errorThrowerProductNotFoundException(const std::type_info& iType, const char* iModule, const char* iInstance, const char* iProcess){
   return new NoProductErrorThrower(iType,iModule,iInstance,iProcess);
}
