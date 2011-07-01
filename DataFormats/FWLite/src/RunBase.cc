// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     RunBase
//
/**\class RunBase RunBase.h DataFormats/FWLite/interface/RunBase.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Eric Vaandering
//         Created:  Wed Jan  13 15:01:20 EDT 2007
//

// system include files
#include <iostream>

// user include files
#include "DataFormats/FWLite/interface/RunBase.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

static const edm::ProductID s_id;
static edm::ConstBranchDescription s_branch = edm::ConstBranchDescription(edm::BranchDescription());
static const edm::Provenance s_prov(boost::shared_ptr<edm::ConstBranchDescription>(&s_branch, edm::do_nothing_deleter()), s_id);

namespace fwlite
{
   RunBase::RunBase()
   {
   }

   RunBase::~RunBase()
   {
   }

   edm::BasicHandle
   RunBase::getByLabelImpl(std::type_info const& iWrapperInfo, std::type_info const& /*iProductInfo*/, const edm::InputTag& iTag) const {
      edm::WrapperHolder edp;
      getByLabel(iWrapperInfo,
                 iTag.label().c_str(),
                 iTag.instance().empty()?static_cast<char const*>(0):iTag.instance().c_str(),
                 iTag.process().empty()?static_cast<char const*> (0):iTag.process().c_str(),
                 edp);
      if(!edp.isValid() || !edp.isPresent()) {
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
      edm::BasicHandle value(edp, &s_prov);
      return value;
   }
}
