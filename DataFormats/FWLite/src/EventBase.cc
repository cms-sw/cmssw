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
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

static const edm::ProductID s_id;
static edm::BranchDescription const s_branch = edm::BranchDescription(edm::BranchDescription());
static const edm::Provenance s_prov(std::shared_ptr<edm::BranchDescription const>(&s_branch, edm::do_nothing_deleter()), s_id);

namespace fwlite
{
   EventBase::EventBase()
   {
   }

   EventBase::~EventBase()
   {
   }

   edm::BasicHandle
   EventBase::getByLabelImpl(std::type_info const& iWrapperInfo, std::type_info const& /*iProductInfo*/, const edm::InputTag& iTag) const {
      edm::WrapperBase const* prod = nullptr;
      void* prodPtr = &prod;
      getByLabel(iWrapperInfo,
                 iTag.label().c_str(),
                 iTag.instance().empty()?static_cast<char const*>(0):iTag.instance().c_str(),
                 iTag.process().empty()?static_cast<char const*> (0):iTag.process().c_str(),
                 prodPtr);
      if(prod == nullptr || !prod->isPresent()) {
        edm::TypeID productType(iWrapperInfo);

        edm::BasicHandle failed(edm::makeHandleExceptionFactory([=]()->std::shared_ptr<cms::Exception>{
          std::shared_ptr<cms::Exception> whyFailed(std::make_shared<edm::Exception>(edm::errors::ProductNotFound));
          *whyFailed
          << "getByLabel: Found zero products matching all criteria\n"
          << "Looking for type: " << productType << "\n"
          << "Looking for module label: " << iTag.label() << "\n"
          << "Looking for productInstanceName: " << iTag.instance() << "\n"
          << (iTag.process().empty() ? "" : "Looking for process: ") << iTag.process() << "\n"
          << "The data is registered in the file but is not available for this event\n";
          return whyFailed;
        }));
         return failed;
      }

      edm::BasicHandle value(prod, &s_prov);
      return value;
   }
}
