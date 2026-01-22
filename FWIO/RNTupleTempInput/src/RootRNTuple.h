#ifndef FWIO_RNTupleTempInput_RootRNTuple_h
#define FWIO_RNTupleTempInput_RootRNTuple_h

/*----------------------------------------------------------------------

RootRNTuple.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "Rtypes.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <variant>

#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleReader.hxx"
#include "ROOT/RNTupleView.hxx"
class TClass;

namespace edm::rntuple_temp {
  class InputFile;
  class RootDelayedReaderBase;

  namespace rootrntuple {
    using EntryNumber = IndexIntoFile::EntryNumber_t;
    struct ProductInfo {
      ProductInfo(ProductDescription const& prod) : productDescription_(prod) {}
      ProductInfo(ProductInfo const&) = delete;
      ProductInfo& operator=(ProductInfo const&) = delete;
      ProductInfo(ProductInfo&&) = default;
      ProductInfo& operator=(ProductInfo&&) = delete;
      void setField(ROOT::RFieldToken token, ROOT::RNTupleView<void> view, TClass const* wrapperBaseTClass);
      std::unique_ptr<WrapperBase> newWrapper() const;
      bool valid() const { return view_.has_value(); }
      ROOT::RNTupleView<void>& view() const { return view_.value(); }
      ProductDescription const& productDescription() const { return productDescription_; }

      ROOT::RFieldToken token() const { return token_; }

    private:
      ProductDescription const productDescription_;
      ROOT::RFieldToken token_;
      mutable std::optional<ROOT::RNTupleView<void>> view_;
      //All access to a ROOT file is serialized
      TClass* classCache_ = nullptr;
      Int_t offsetToWrapperBase_ = 0;
    };

    class ProductMap {
    public:
      using Map = std::unordered_map<unsigned int, ProductInfo>;

      void reserve(Map::size_type iSize) { map_.reserve(iSize); }
      void insert(edm::BranchID const& iKey, ProductInfo iInfo) { map_.emplace(iKey.id(), std::move(iInfo)); }
      ProductInfo const* find(BranchID const& iKey) const { return find(iKey.id()); }
      ProductInfo const* find(unsigned int iKey) const {
        auto itFound = map_.find(iKey);
        if (itFound == map_.end()) {
          return nullptr;
        }
        return &itFound->second;
      }

      using const_iterator = Map::const_iterator;
      const_iterator begin() const { return map_.cbegin(); }
      const_iterator end() const { return map_.cend(); }
      Map::size_type size() const { return map_.size(); }

    private:
      Map map_;
    };
  }  // namespace rootrntuple

  class RootRNTuple {
  public:
    using ProductMap = rootrntuple::ProductMap;
    using EntryNumber = rootrntuple::EntryNumber;
    struct Options {
      bool useClusterCache = true;
      bool promptReading = false;
      bool enableIMT = true;

      Options usingDefaultNonEventOptions() const { return {}; }
    };

    RootRNTuple(std::shared_ptr<InputFile> filePtr,
                BranchType const& branchType,
                unsigned int nIndexes,
                Options const& options,
                InputType inputType);

    RootRNTuple(std::shared_ptr<InputFile> filePtr,
                BranchType const& branchType,
                std::string const& processName,
                unsigned int nIndexes,
                Options const& options,
                InputType inputType);

    ~RootRNTuple();

    RootRNTuple(RootRNTuple const&) = delete;             // Disallow copying and moving
    RootRNTuple& operator=(RootRNTuple const&) = delete;  // Disallow copying and moving

    bool isValid() const;
    void numberOfBranchesToAdd(ProductMap::Map::size_type iSize) { branches_.reserve(iSize); }
    void addBranch(ProductDescription const& prod, std::string const& oldBranchName);
    void dropBranch(std::string const& oldBranchName);
    void getEntry(ROOT::RNTupleView<void>& view, EntryNumber entry) const;
    void getEntryForAllBranches(std::unordered_map<unsigned int, std::unique_ptr<edm::WrapperBase>>&) const;
    void setPresence(ProductDescription& prod, std::string const& oldBranchName);

    bool next() { return ++entryNumber_ < entries_; }
    bool nextWithCache();
    bool current() const { return entryNumber_ < entries_ && entryNumber_ >= 0; }
    bool current(EntryNumber entry) const { return entry < entries_ && entry >= 0; }
    void rewind() { entryNumber_ = 0; }
    void rewindToInvalid() { entryNumber_ = IndexIntoFile::invalidEntry; }
    void close();
    bool skipEntries(unsigned int& offset);
    EntryNumber const& entryNumber() const { return entryNumber_; }
    EntryNumber const& entryNumberForIndex(unsigned int index) const;
    EntryNumber const& entries() const { return entries_; }
    void setEntryNumber(EntryNumber theEntryNumber);
    void insertEntryForIndex(unsigned int index);
    std::vector<std::string> const& branchNames() const { return branchNames_; }
    RootDelayedReaderBase* rootDelayedReader() const;
    DelayedReader* resetAndGetRootDelayedReader() const;
    template <typename T>
    void fillAux(T*& pAux) {
      try {
        if (std::holds_alternative<std::monostate>(auxView_)) {
          auxView_ = reader_->GetView(auxDesc_, pAux);
        }
        auto& view = std::get<ROOT::RNTupleView<T>>(auxView_);
        view.BindRawPtr(pAux);
        view(entryNumber_);
      } catch (cms::Exception const& e) {
        throw Exception(errors::FileReadError, "", e);
      } catch (std::exception const& e) {
        Exception t(errors::FileReadError);
        t << e.what();
        throw t;
      }
    }

    std::optional<ROOT::RNTupleView<void>> view(std::string_view iName);
    ROOT::DescriptorId_t descriptorFor(std::string_view iName) { return reader_->GetDescriptor().FindFieldId(iName); }

    void fillEntry(ROOT::RNTupleView<void>& view) { getEntry(view, entryNumber_); }
    void fillEntry(ROOT::RNTupleView<void>& view, EntryNumber entryNumber) { getEntry(view, entryNumber); }

    void fillEntry(ROOT::DescriptorId_t id, EntryNumber entryNumber, void* iData) {
      auto view = reader_->GetView(id, iData);
      getEntry(view, entryNumber);
    }

    ProductMap const& branches() const;

    BranchType branchType() const { return branchType_; }
    std::string const& processName() const { return processName_; }

    void setSignals(
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource);

  private:
    void init(std::string const& productTreeName, Options const& options);

    RootRNTuple(std::shared_ptr<InputFile> filePtr,
                BranchType const& branchType,
                unsigned int nIndexes,
                bool promptRead,
                InputType inputType);

    void setCacheSize(unsigned int cacheSize);
    void setTreeMaxVirtualSize(int treeMaxVirtualSize);

    std::shared_ptr<InputFile> filePtr_;
    // We use bare pointers for pointers to some ROOT entities.
    // Root owns them and uses bare pointers internally.
    // Therefore,using smart pointers here will do no good.
    std::unique_ptr<ROOT::RNTupleReader> reader_;
    BranchType branchType_;
    std::string processName_;
    ROOT::DescriptorId_t auxDesc_ = ROOT::kInvalidDescriptorId;
    EntryNumber entries_ = 0;
    EntryNumber entryNumber_ = IndexIntoFile::invalidEntry;
    std::unique_ptr<std::vector<EntryNumber>> entryNumberForIndex_;
    std::vector<std::string> branchNames_;
    ProductMap branches_;
    unsigned int cacheSize_ = 0;
    unsigned long treeAutoFlush_ = 0;
    bool promptRead_;
    std::unique_ptr<RootDelayedReaderBase> rootDelayedReader_;
    std::variant<std::monostate,
                 ROOT::RNTupleView<edm::EventAuxiliary>,
                 ROOT::RNTupleView<edm::LuminosityBlockAuxiliary>,
                 ROOT::RNTupleView<edm::RunAuxiliary>>
        auxView_;
  };
}  // namespace edm::rntuple_temp
#endif
