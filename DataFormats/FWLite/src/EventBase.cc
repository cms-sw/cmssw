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
//

// system include files
#include <iostream>

// user include files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace {
   //This is used by the shared_ptr required to be passed to BasicHandle to keep the shared_ptr from doing the delete
   struct null_deleter
   {
      void operator()(void *) const
      {
      }
   };
}

static const edm::ProductID s_id;
static edm::ConstBranchDescription s_branch = edm::ConstBranchDescription( edm::BranchDescription() );
static const edm::Provenance s_prov(boost::shared_ptr<edm::ConstBranchDescription>(&s_branch, null_deleter()) ,s_id);



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
      if(!prod->isPresent()) {
         edm::TypeID productType(iWrapperInfo);
         boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound));
         *whyFailed
         << "getByLabel: Found zero products matching all criteria\n"
         << "Looking for type: " << productType << "\n"
         << "Looking for module label: " << iTag.label() << "\n"
         << "Looking for productInstanceName: " << iTag.instance() << "\n"
         << (iTag.process().empty() ? "" : "Looking for process: ") << iTag.process() << "\n"
         << "The data is registered in the file but is not available for this event\n";
         edm::BasicHandle failed(whyFailed);
         return failed;
      }
      edm::BasicHandle value(boost::shared_ptr<edm::EDProduct>(prod,null_deleter()),&s_prov);
      return value;
   }
}
