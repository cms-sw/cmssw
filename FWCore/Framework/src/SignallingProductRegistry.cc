// -*- C++ -*-
//
// Package:     Framework
// Class  :     SignallingProductRegistry
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 23 16:52:50 CEST 2005
// $Id: SignallingProductRegistry.cc,v 1.1 2005/10/07 19:16:19 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;
//
// member functions
//
namespace {
   struct StackGuard {
      StackGuard(const std::string& iTypeName, std::map<std::string,unsigned int>& ioStack, bool iFromListener):
      numType_(++ioStack[iTypeName]), itr_(ioStack.find(iTypeName)),fromListener_(iFromListener)
      { if(iFromListener) {++(itr_->second);} }

      ~StackGuard() {
         --(itr_->second);
         if(fromListener_){
            --(itr_->second);
         }
      }
      
      unsigned int numType_;
      std::map<std::string,unsigned int>::iterator itr_;
      bool fromListener_;
   };
}

void SignallingProductRegistry::addCalled(BranchDescription const& iProd, bool iFromListener)
{
   StackGuard guard(iProd.fullClassName_, typeAddedStack_, iFromListener);
   if(guard.numType_ > 2) {
      throw cms::Exception("CircularReference")
      <<"Attempted to register the production of "<<iProd.fullClassName_
      <<" from module "
      <<iProd.module.moduleLabel_
      <<" with product instance \""
      <<iProd.productInstanceName_
      <<"\"\n"
      <<"However, this was in reaction to a registration of a production for the same type \n"
      <<"from another module who was also listening to product registrations.\n"
      <<"This can lead to circular Event::get* calls.\n"
      <<"Please reconfigure job so it does not contain both of the modules.";
   }
   productAddedSignal_(iProd);
}
