// -*- C++ -*-
//
// Package:     FWLite
// Class  :     EventBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Charles Plager
//         Created:  
// $Id: 
//

// system include files
#include <iostream>
#include "Reflex/Type.h"

// user include files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

static const edm::ProductID s_id;
static const edm::BranchDescription s_branch;
static const edm::Provenance s_prov(s_branch,s_id);

namespace {
   //This is used by the shared_ptr required to be passed to BasicHandle to keep the shared_ptr from doing the delete
   struct null_deleter
   {
      void operator()(void const *) const
      {
      }
   };
}
   
namespace fwlite
{
   EventBase::EventBase()
   {
   }

   EventBase::~EventBase()
   {
   }

   edm::BasicHandle 
   EventBase::getByLabelImpl(const std::type_info& iWrapperInfo, const std::type_info& /*iProductInfo*/, const edm::InputTag& iTag) const 
   {
      edm::EDProduct* prod=0;
      void* prodPtr = &prod;
      getByLabel(iWrapperInfo, 
                 iTag.label().c_str(), 
                 iTag.instance().empty()?static_cast<const char*>(0):iTag.instance().c_str(),
                 iTag.process().empty()?static_cast<const char*> (0):iTag.process().c_str(),
                 prodPtr);
      if(0==prod) {
         edm::TypeID productType(iWrapperInfo);
         boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
         *whyFailed
         << "getByLabel: Found zero products matching all criteria\n"
         << "Looking for type: " << productType << "\n"
         << "Looking for module label: " << iTag.label() << "\n"
         << "Looking for productInstanceName: " << iTag.instance() << "\n"
         << (iTag.process().empty() ? "" : "Looking for process: ") << iTag.process() << "\n";
         
         edm::BasicHandle failed(whyFailed);
         return failed;
      }
      edm::BasicHandle value(boost::shared_ptr<edm::EDProduct>(prod,null_deleter()),&s_prov);
      return value;
   }
}
