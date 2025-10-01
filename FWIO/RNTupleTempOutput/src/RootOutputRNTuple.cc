
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
                                       int splitLevel,
                                       int treeMaxVirtualSize,
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
        auto entry = writer_->CreateEntry();
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

  void RootOutputRNTuple::finishInitialization() {
    writer_ = ROOT::RNTupleWriter::Append(std::move(model_), name_, *filePtr_, ROOT::RNTupleWriteOptions());
  }

  void RootOutputRNTuple::addField(std::string const& branchName,
                                   std::string const& className,
                                   void const** pProd,
                                   int splitLevel,
                                   int basketSize,
                                   bool produced) {
    auto field = ROOT::RFieldBase::Create(branchName, className).Unwrap();
    model_->AddField(std::move(field));
    producedBranches_.push_back(model_->GetToken(branchName));
    producedBranchPointers_.push_back(const_cast<void**>(pProd));
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
