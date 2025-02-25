#include "DataProductsRNTuple.h"
#include "ROOT/RNTuple.hxx"
#include "IOPool/Common/interface/getWrapperBasePtr.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace edm::input;
using namespace ROOT::Experimental;

namespace {
  std::string fixName(std::string_view iName) {
    if (not iName.empty() and iName.back() == '.') {
      iName.remove_suffix(1);
    }
    return std::string(iName);
  }

  std::unique_ptr<ROOT::RNTuple> get_and_check_RNTuple(TFile* iFile, std::string const& iName) {
    auto ret = std::unique_ptr<ROOT::RNTuple>(iFile->Get<ROOT::RNTuple>(iName.c_str()));
    if (not ret) {
      throw edm::Exception(edm::errors::FileReadError)
          << "The entry for '" << iName << "' does not exist or is not an RNTuple";
    }
    return ret;
  }

}  // namespace

DataProductsRNTuple::DataProductsRNTuple(TFile* iFile,
                                         std::string const& iName,
                                         std::string const& iAux,
                                         ROOT::Experimental::RNTupleReadOptions const& iOps)
    : reader_(RNTupleReader::Open(*get_and_check_RNTuple(iFile, iName), iOps)) {
  auxDesc_ = reader_->GetDescriptor().FindFieldId(iAux);
}

bool DataProductsRNTuple::setupToReadProductIfAvailable(ProductDescription& iProduct) {
  auto fixedName = fixName(iProduct.branchName());
  auto desc = reader_->GetDescriptor().FindFieldId(fixedName);
  if (desc == ROOT::Experimental::kInvalidDescriptorId) {
    return false;
  }
  iProduct.initFromDictionary();
  iProduct.setOnDemand(true);
  infos_.emplace(iProduct.branchID().id(), ProductInfo(iProduct.wrappedName(), desc, std::move(fixedName)));
  return true;
}

TClass const* DataProductsRNTuple::WrapperFactory::wrapperBase() {
  static TClass const* const s_base = TClass::GetClass("edm::WrapperBase");
  return s_base;
}

DataProductsRNTuple::WrapperFactory::WrapperFactory(std::string const& iTypeName)
    : wrapperClass_(TClass::GetClass(iTypeName.c_str())) {
  offsetToWrapperBase_ = wrapperClass_->GetBaseClassOffset(wrapperBase());
}

void DataProductsRNTuple::WrapperFactory::Deleter::operator()(void* iPtr) const { class_->Destructor(iPtr); }

std::unique_ptr<void, DataProductsRNTuple::WrapperFactory::Deleter> DataProductsRNTuple::WrapperFactory::newWrapper()
    const {
  return std::unique_ptr<void, Deleter>(wrapperClass_->New(), Deleter(wrapperClass_));
}

std::shared_ptr<edm::WrapperBase> DataProductsRNTuple::WrapperFactory::toWrapperBase(
    std::unique_ptr<void, Deleter> iProduct) const {
  return getWrapperBasePtr(iProduct.release(), offsetToWrapperBase_);
}

std::shared_ptr<edm::WrapperBase> DataProductsRNTuple::dataProduct(edm::BranchID const& iProduct, int iEntry) {
  auto const& info = infos_.find(iProduct.id());
  if (info == infos_.end()) {
    throw cms::Exception("RNTupleError") << " unable to find branch id " << iProduct.id() << " for entry " << iEntry;
  }

  //std::cout <<"dataProduct "<<info->second.name_<<std::endl;
  auto product = info->second.factory_.newWrapper();
  if (not info->second.view_) {
    info->second.view_ = reader_->GetView<void>(info->second.descriptor_, product.get());
  } else {
    info->second.view_->BindRawPtr(product.get());
  }
  (*info->second.view_)(iEntry);

  return info->second.factory_.toWrapperBase(std::move(product));
}
