// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     LuminosityBlockBase
//
/**\class LuminosityBlockBase LuminosityBlockBase.h DataFormats/FWLite/interface/LuminosityBlockBase.h

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
#include "DataFormats/FWLite/interface/LuminosityBlockBase.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace {
   //This is used by the shared_ptr required to be passed to BasicHandle to keep the shared_ptr from doing the delete
   struct null_deleter
   {
      void operator()(void const *) const
      {
      }
   };
}
static const edm::ProductID s_id;
static edm::ConstBranchDescription s_branch = edm::ConstBranchDescription( edm::BranchDescription() );
static const edm::Provenance s_prov(boost::shared_ptr<edm::ConstBranchDescription>(&s_branch, null_deleter()) ,s_id);


namespace fwlite
{
   LuminosityBlockBase::LuminosityBlockBase()
   {
   }

   LuminosityBlockBase::~LuminosityBlockBase()
   {
   }

   edm::BasicHandle
   LuminosityBlockBase::getByLabelImpl(const std::type_info& iWrapperInfo, const std::type_info& /*iProductInfo*/, const edm::InputTag& iTag) const
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
