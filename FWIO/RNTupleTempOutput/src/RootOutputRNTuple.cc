
#include "RootOutputRNTuple.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/AbstractServices/interface/RootHandlers.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TBranch.h"
#include "TBranchElement.h"
#include "TCollection.h"
#include "TFile.h"
#include "Rtypes.h"
#include "RVersion.h"

#include <limits>

#include "oneapi/tbb/task_arena.h"

namespace edm {

  RootOutputRNTuple::RootOutputRNTuple(std::shared_ptr<TFile> filePtr,
                                       BranchType const& branchType,
                                       std::string const& processName)
      : filePtr_(filePtr),
        name_(processName.empty() ? BranchTypeToProductTreeName(branchType)
                                  : BranchTypeToProductTreeName(branchType, processName)),
        model_(ROOT::RNTupleModel::Create()),
        producedBranches_(),
        auxBranches_() {}

  namespace {
    template <typename T, typename U>
    struct Zip {
      T const& first;
      U const& second;

      Zip(T const& t, U const& u) : first(t), second(u) { assert(t.size() == u.size()); }

      struct iterator {
        using value_type = std::pair<typename T::value_type, typename U::value_type>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::input_iterator_tag;

        typename T::const_iterator t_iter;
        typename U::const_iterator u_iter;

        iterator(typename T::const_iterator t, typename U::const_iterator u) : t_iter(t), u_iter(u) {}

        value_type operator*() const { return std::make_pair(*t_iter, *u_iter); }
        iterator& operator++() {
          ++t_iter;
          ++u_iter;
          return *this;
        }
        iterator operator++(int) {
          iterator tmp = *this;
          ++(*this);
          return tmp;
        }
        bool operator==(iterator const& other) const { return t_iter == other.t_iter && u_iter == other.u_iter; }
        bool operator!=(iterator const& other) const { return !(*this == other); }
      };

      iterator begin() const { return iterator(first.begin(), second.begin()); }
      iterator end() const { return iterator(first.end(), second.end()); }
    };

    template <typename T, typename U>
    Zip<T, U> zip(T const& t, U const& u) {
      return Zip<T, U>(t, u);
    }
  }  // namespace

  void RootOutputRNTuple::fill() {
    // Isolate the fill operation so that IMT doesn't grab other large tasks
    // that could lead to RNTupleTempOutputModule stalling
    std::exception_ptr e;
    oneapi::tbb::this_task_arena::isolate([&] {
      try {
        auto entry = writer_->CreateRawPtrWriteEntry();
        for (auto z = zip(producedBranchPointers_, producedBranches_); auto prod : z) {
          entry->BindRawPtr(prod.second, *prod.first);
        }
        for (auto z = zip(auxBranchPointers_, auxBranches_); auto aux : z) {
          entry->BindRawPtr(aux.second, *aux.first);
        }
        writer_->Fill(*entry);
      } catch (...) {
        e = std::current_exception();
      }
    });
    if (e) {
      std::rethrow_exception(e);
    }
  }

  void RootOutputRNTuple::finishInitialization(Config const& config) {
    ROOT::RNTupleWriteOptions options;
    switch (config.compressionAlgo) {
      case Config::CompressionAlgos::kLZMA:
        options.SetCompression(ROOT::RCompressionSetting::EAlgorithm::kLZMA, config.compressionLevel);
        break;
      case Config::CompressionAlgos::kZSTD:
        options.SetCompression(ROOT::RCompressionSetting::EAlgorithm::kZSTD, config.compressionLevel);
        break;
      case Config::CompressionAlgos::kZLIB:
        options.SetCompression(ROOT::RCompressionSetting::EAlgorithm::kZLIB, config.compressionLevel);
        break;
      case Config::CompressionAlgos::kLZ4:
        options.SetCompression(ROOT::RCompressionSetting::EAlgorithm::kLZ4, config.compressionLevel);
        break;
      default:
        throw edm::Exception(edm::errors::Configuration)
            << "Unknown compression algorithm enum value: " << static_cast<int>(config.compressionAlgo) << "\n";
    }
    options.SetApproxZippedClusterSize(config.approxZippedClusterSize);
    options.SetMaxUnzippedClusterSize(config.maxUnzippedClusterSize);
    options.SetInitialUnzippedPageSize(config.initialUnzippedPageSize);
    options.SetMaxUnzippedPageSize(config.maxUnzippedPageSize);
    options.SetPageBufferBudget(config.pageBufferBudget);
    options.SetUseBufferedWrite(config.useBufferedWrite);
    options.SetUseDirectIO(config.useDirectIO);
    writer_ = ROOT::RNTupleWriter::Append(std::move(model_), name_, *filePtr_, options);
  }

  namespace {
    /* By default RNTuple will take a multi-byte intrinsic data type and break
it into multiple output fields to separate the high-bytes from the low-bytes (or mantessa from exponent).
This typically allows for better compression. Empirically we have found that some important
member data of some classes actually take more space on disk when this is done.
This function allows one to override the default RNTuple behavior and instead store
all bytes of a data type in one field. To do that one must find the storage type (typeName) and
explicitly pass the correct variable to `SetColumnRepresentatives`).
     */
    void noSplitField(ROOT::RFieldBase& iField) {
      auto const& typeName = iField.GetTypeName();
      if (typeName == "std::uint16_t") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt16}});
      } else if (typeName == "std::uint32_t") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt32}});
      } else if (typeName == "std::uint64_t") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt64}});
      } else if (typeName == "std::int16_t") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kInt16}});
      } else if (typeName == "std::int32_t") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kInt32}});
      } else if (typeName == "std::int64_t") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kInt64}});
      } else if (typeName == "float") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}});
      } else if (typeName == "double") {
        iField.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal64}});
      }
    }

    void findSubFieldsForNoSplitThenApply(ROOT::RFieldBase& iField, std::vector<std::string> const& iNoSplitFields) {
      for (auto const& name : iNoSplitFields) {
        if (name.starts_with(iField.GetFieldName())) {
          bool found = false;
          for (auto& subfield : iField) {
            if (subfield.GetQualifiedFieldName() == name) {
              found = true;
              noSplitField(subfield);
              break;
            }
          }
          if (not found) {
            throw edm::Exception(edm::errors::Configuration)
                << "The data product was found but the requested subfield '" << name << "' is not part of the class";
          }
        }
      }
    }
  }  // namespace

  void RootOutputRNTuple::addField(std::string const& branchName,
                                   std::string const& className,
                                   void const** pProd,
                                   bool useStreamer,
                                   std::vector<std::string> const& iNoSplitFields) {
    const bool noSplitSubFields = (iNoSplitFields.size() == 1 and iNoSplitFields[0] == "all") ? true : false;
    if (useStreamer) {
      auto field = std::make_unique<ROOT::RStreamerField>(branchName, className);
      model_->AddField(std::move(field));
    } else {
      auto field = ROOT::RFieldBase::Create(branchName, className).Unwrap();
      if (noSplitSubFields) {
        //use the 'conventional' way to store fields
        for (auto& subfield : *field) {
          noSplitField(subfield);
        }
      } else if (not iNoSplitFields.empty()) {
        findSubFieldsForNoSplitThenApply(*field, iNoSplitFields);
      }
      model_->AddField(std::move(field));
    }
    producedBranches_.push_back(model_->GetToken(branchName));
    producedBranchPointers_.push_back(pProd);
  }

  void RootOutputRNTuple::close() {
    // The TFile was just closed.
    // Just to play it safe, zero all pointers to quantities in the file.
    auxBranches_.clear();
    producedBranches_.clear();
    writer_ = nullptr;   // propagate_const<T> has no reset() function
    filePtr_ = nullptr;  // propagate_const<T> has no reset() function
  }
}  // namespace edm
