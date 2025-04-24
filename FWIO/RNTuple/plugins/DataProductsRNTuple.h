#if !defined(DataProductsRNTuple_h)
#define DataProductsRNTuple_h

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "ROOT/RNTupleReader.hxx"
#include "ROOT/RNTupleReadOptions.hxx"
#include "TFile.h"
#include "TClass.h"
#include <string>
#include <memory>
#include <optional>

namespace edm::input {
  class DataProductsRNTuple {
  public:
    DataProductsRNTuple(TFile*,
                        std::string const& iName,
                        std::string const& iAux,
                        ROOT::RNTupleReadOptions const& iOpts);

    bool setupToReadProductIfAvailable(ProductDescription&);

    std::shared_ptr<edm::WrapperBase> dataProduct(edm::BranchID const&, int iEntry);

    template <typename T>
    ROOT::RNTupleView<T> auxView(std::shared_ptr<T> oStorage) {
      return reader_->GetView(auxDesc_, std::move(oStorage));
    }

    ROOT::NTupleSize_t numberOfEntries() const { return reader_->GetNEntries(); }

    ROOT::DescriptorId_t findDescriptorID(std::string const& iFieldName) {
      return reader_->GetDescriptor().FindFieldId(iFieldName);
    }

    template <typename T>
    ROOT::RNTupleView<T> viewFor(ROOT::DescriptorId_t iID, std::shared_ptr<T> oStorage) {
      return reader_->GetView(iID, std::move(oStorage));
    }

    void printInfo(std::ostream& iStream) { reader_->PrintInfo(ROOT::ENTupleInfo::kMetrics, iStream); }

  private:
    struct WrapperFactory {
      struct Deleter {
        Deleter(TClass* iClass) : class_(iClass) {}
        void operator()(void*) const;
        TClass* class_;
      };

      WrapperFactory(std::string const& iTypeName);

      std::unique_ptr<void, Deleter> newWrapper() const;
      std::shared_ptr<edm::WrapperBase> toWrapperBase(std::unique_ptr<void, Deleter>) const;
      TClass* wrapperClass_;
      Int_t offsetToWrapperBase_;

      static TClass const* wrapperBase();
    };

    struct ProductInfo {
      ProductInfo(std::string const& iTypeName, ROOT::DescriptorId_t iDesc, std::string iName)
          : factory_(iTypeName), descriptor_(iDesc), name_(std::move(iName)) {}
      WrapperFactory factory_;
      ROOT::DescriptorId_t descriptor_;
      std::string name_;
      std::optional<ROOT::RNTupleView<void>> view_;
    };

    std::unique_ptr<ROOT::RNTupleReader> reader_;
    std::unordered_map<edm::BranchID::value_type, ProductInfo> infos_;
    ROOT::DescriptorId_t auxDesc_;
  };
}  // namespace edm::input

#endif
