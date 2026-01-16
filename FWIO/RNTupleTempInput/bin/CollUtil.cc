#include "CollUtil.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"

#include "TBasket.h"
#include "TBranch.h"
#include "TFile.h"
#include "TIterator.h"
#include "TKey.h"
#include "TList.h"
#include "TObject.h"
#include "TTree.h"
#include "ROOT/RNTupleReader.hxx"
#include "ROOT/RNTupleDescriptor.hxx"
#include "ROOT/RNTuple.hxx"

#include <iomanip>
#include <iostream>
#include <ranges>

namespace edm::rntuple_temp {

  // Get a file handler
  TFile *openFileHdl(std::string const &fname) {
    TFile *hdl = TFile::Open(fname.c_str(), "read");

    if (nullptr == hdl) {
      std::cout << "ERR Could not open file " << fname.c_str() << std::endl;
      exit(1);
    }
    return hdl;
  }

  // Print every tree in a file
  void printTrees(TFile *hdl) {
    hdl->ls();
    TList *keylist = hdl->GetListOfKeys();
    TIterator *iter = keylist->MakeIterator();
    TKey *key = nullptr;
    while ((key = (TKey *)iter->Next())) {
      TObject *obj = hdl->Get(key->GetName());
      if (obj->IsA() == TTree::Class()) {
        obj->Print();
      }
    }
    return;
  }

  // number of entries in a tree
  Long64_t numEntries(TFile *hdl, std::string const &trname) {
    auto tuple = hdl->Get<ROOT::RNTuple>(trname.c_str());
    if (tuple) {
      auto reader = ROOT::RNTupleReader::Open(*tuple);
      return reader->GetNEntries();
    } else {
      // Try as a TTree
      TTree *tree = (TTree *)hdl->Get(trname.c_str());
      if (tree) {
        return tree->GetEntries();
      }
    }
    std::cout << "ERR cannot find a RNTuple named \"" << trname << "\"" << std::endl;
    return -1;
  }

  namespace {
    void addBranchSizes(TBranch *branch, Long64_t &size) {
      size += branch->GetTotalSize();  // Includes size of branch metadata
      // Now recurse through any subbranches.
      Long64_t nB = branch->GetListOfBranches()->GetEntries();
      for (Long64_t i = 0; i < nB; ++i) {
        TBranch *btemp = (TBranch *)branch->GetListOfBranches()->At(i);
        addBranchSizes(btemp, size);
      }
    }
    Long64_t branchCompressedSizes(TBranch *branch) { return branch->GetZipBytes("*"); }
  }  // namespace

  void printBranchNames(TTree *tree) {
    if (tree != nullptr) {
      Long64_t nB = tree->GetListOfBranches()->GetEntries();
      for (Long64_t i = 0; i < nB; ++i) {
        Long64_t size = 0LL;
        TBranch *btemp = (TBranch *)tree->GetListOfBranches()->At(i);
        addBranchSizes(btemp, size);
        std::cout << "Branch " << i << " of " << tree->GetName() << " tree: " << btemp->GetName()
                  << " Total size = " << size << " Compressed size = " << branchCompressedSizes(btemp) << std::endl;
      }
    } else {
      std::cout << "Missing Events tree?\n";
    }
  }

  void longBranchPrint(TTree *tr) {
    if (tr != nullptr) {
      Long64_t nB = tr->GetListOfBranches()->GetEntries();
      for (Long64_t i = 0; i < nB; ++i) {
        tr->GetListOfBranches()->At(i)->Print();
      }
    } else {
      std::cout << "Missing Events tree?\n";
    }
  }

  namespace {
    Long64_t storageForField(ROOT::RFieldDescriptor const &iField, ROOT::RNTupleDescriptor const &iDesc) {
      Long64_t storage = 0;
      for (auto &col : iDesc.GetColumnIterable(iField)) {
        if (col.IsAliasColumn()) {
          continue;
        }
        auto id = col.GetPhysicalId();

        for (auto &cluster : iDesc.GetClusterIterable()) {
          auto columnRange = cluster.GetColumnRange(id);
          if (columnRange.IsSuppressed()) {
            continue;
          }
          const auto &pageRange = cluster.GetPageRange(id);
          for (const auto &page : pageRange.GetPageInfos()) {
            storage += page.GetLocator().GetNBytesOnStorage();
          }
        }
      }
      return storage;
    }

    Long64_t storageForFieldAndSubFields(ROOT::RFieldDescriptor const &iField, ROOT::RNTupleDescriptor const &iDesc) {
      Long64_t storage = storageForField(iField, iDesc);
      for (auto const &sub : iDesc.GetFieldIterable(iField)) {
        storage += storageForFieldAndSubFields(sub, iDesc);
      }
      return storage;
    }
  }  // namespace

  void printFieldNames(ROOT::RNTupleReader &reader) {
    auto &desc = reader.GetDescriptor();
    auto topLevel = desc.GetTopLevelFields();
    unsigned int index = 0;
    for (auto const &topField : topLevel) {
      Long64_t storage = storageForFieldAndSubFields(topField, desc);
      std::cout << "Field " << index << " of " << desc.GetName() << " RNTuple: " << topField.GetFieldName()
                << " Bytes On Storage = " << storage << std::endl;

      ++index;
    }
  }

  void longFieldPrint(ROOT::RNTupleReader &reader) { reader.PrintInfo(ROOT::ENTupleInfo::kStorageDetails); }

  namespace {
    class BranchBasketBytes {
    public:
      BranchBasketBytes(TBranch const *branch)
          : basketFirstEntry_(branch->GetBasketEntry()),
            basketBytes_(branch->GetBasketBytes()),
            branchName_(branch->GetName()),
            maxBaskets_(branch->GetMaxBaskets()) {}

      bool isAlignedWithClusterBoundaries() const { return isAligned_; }

      std::string_view name() const { return branchName_; }

      // Processes "next cluster" for the branch, calculating the
      // number of bytes and baskets in the cluster
      //
      // @param[in] clusterBegin        Begin entry number for the cluster
      // @param[in] clusterEnd          End entry number (exclusive) for the cluster
      // @param[out] nonAlignedBranches Branch name is added to the set if the basket boundary
      //                                does not align with cluster boundary
      //
      // @return Tuple of the number of bytes and baskets in the cluster
      std::tuple<Long64_t, unsigned> bytesInNextCluster(Long64_t clusterBegin,
                                                        Long64_t clusterEnd,
                                                        std::set<std::string_view> &nonAlignedBranches) {
        if (basketFirstEntry_[iBasket_] != clusterBegin) {
          std::cout << "Branch " << branchName_ << " iBasket " << iBasket_ << " begin entry "
                    << basketFirstEntry_[iBasket_] << " does not align with cluster boundary, expected " << clusterBegin
                    << std::endl;
          exit(1);
        }

        Long64_t bytes = 0;
        unsigned nbaskets = 0;
        for (; iBasket_ < maxBaskets_ and basketFirstEntry_[iBasket_] < clusterEnd; ++iBasket_) {
          bytes += basketBytes_[iBasket_];
          ++nbaskets;
        }
        if (basketFirstEntry_[iBasket_] != clusterEnd) {
          nonAlignedBranches.insert(branchName_);
          isAligned_ = false;
          return std::tuple(0, 0);
        }
        return std::tuple(bytes, nbaskets);
      }

    private:
      Long64_t const *basketFirstEntry_;
      Int_t const *basketBytes_;
      std::string_view branchName_;
      Int_t maxBaskets_;
      Long64_t iBasket_ = 0;
      bool isAligned_ = true;
    };

    std::vector<BranchBasketBytes> makeBranchBasketBytes(TBranch *branch) {
      std::vector<BranchBasketBytes> ret;

      TObjArray *subBranches = branch->GetListOfBranches();
      if (subBranches and subBranches->GetEntries() > 0) {
        // process sub-branches if there are any
        auto const nbranches = subBranches->GetEntries();
        for (Long64_t iBranch = 0; iBranch < nbranches; ++iBranch) {
          auto vec = makeBranchBasketBytes(dynamic_cast<TBranch *>(subBranches->At(iBranch)));
          ret.insert(ret.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
        }
      } else {
        ret.emplace_back(branch);
      }
      return ret;
    }

    template <typename T>
    void processClusters(TTree *tr, T printer, const std::string &limitToBranch = "") {
      TTree::TClusterIterator clusterIter = tr->GetClusterIterator(0);
      Long64_t const nentries = tr->GetEntries();

      // Keep the state of each branch basket index so that we don't
      // have to iterate through everything on every cluster
      std::vector<BranchBasketBytes> processors;
      {
        TObjArray *branches = tr->GetListOfBranches();
        Long64_t const nbranches = branches->GetEntries();
        for (Long64_t iBranch = 0; iBranch < nbranches; ++iBranch) {
          auto branch = dynamic_cast<TBranch *>(branches->At(iBranch));
          if (limitToBranch.empty() or
              std::string_view(branch->GetName()).find(limitToBranch) != std::string_view::npos) {
            auto vec = makeBranchBasketBytes(branch);
            processors.insert(
                processors.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
          }
        }
      }

      printer.header(tr, processors);
      // Record branches whose baskets do not align with cluster boundaries
      std::set<std::string_view> nonAlignedBranches;
      {
        Long64_t clusterBegin;
        while ((clusterBegin = clusterIter()) < nentries) {
          Long64_t clusterEnd = clusterIter.GetNextEntry();
          printer.beginCluster(clusterBegin, clusterEnd);
          for (auto &p : processors) {
            if (p.isAlignedWithClusterBoundaries()) {
              auto const [bytes, baskets] = p.bytesInNextCluster(clusterBegin, clusterEnd, nonAlignedBranches);
              printer.processBranch(bytes, baskets);
            }
          }
          printer.endCluster();
        }
      }

      if (not nonAlignedBranches.empty()) {
        std::cout << "\nThe following branches had baskets whose entry boundaries did not align with the cluster "
                     "boundaries. Their baskets are excluded from the cluster size calculation above starting from the "
                     "first basket that did not align with a cluster boundary."
                  << std::endl;
        for (auto &name : nonAlignedBranches) {
          std::cout << "  " << name << std::endl;
        }
      }
    }
  }  // namespace

  void clusterPrint(TTree *tr) {
    struct ClusterPrinter {
      void header(TTree const *tr, std::vector<BranchBasketBytes> const &branchProcessors) const {
        std::cout << "Printing cluster boundaries in terms of tree entries of the tree " << tr->GetName()
                  << ". Note that the end boundary is exclusive." << std::endl;
        std::cout << std::setw(15) << "Begin" << std::setw(15) << "End" << std::setw(15) << "Entries" << std::setw(15)
                  << "Max baskets" << std::setw(15) << "Bytes" << std::endl;
      }

      void beginCluster(Long64_t clusterBegin, Long64_t clusterEnd) {
        bytes_ = 0;
        maxbaskets_ = 0;
        std::cout << std::setw(15) << clusterBegin << std::setw(15) << clusterEnd << std::setw(15)
                  << (clusterEnd - clusterBegin);
      }

      void processBranch(Long64_t bytes, unsigned int baskets) {
        bytes_ += bytes;
        maxbaskets_ = std::max(baskets, maxbaskets_);
      }

      void endCluster() const { std::cout << std::setw(15) << maxbaskets_ << std::setw(15) << bytes_ << std::endl; }

      Long64_t bytes_ = 0;
      unsigned int maxbaskets_ = 0;
    };
    processClusters(tr, ClusterPrinter{});
  }

  void basketPrint(TTree *tr, const std::string &branchName) {
    struct BasketPrinter {
      void header(TTree const *tr, std::vector<BranchBasketBytes> const &branchProcessors) const {
        std::cout << "Printing cluster boundaries in terms of tree entries of the tree " << tr->GetName()
                  << ". Note that the end boundary is exclusive." << std::endl;
        std::cout << "\nBranches for which number of baskets in each cluster are printed\n";
        for (int i = 0; auto const &p : branchProcessors) {
          std::cout << "[" << i << "] " << p.name() << std::endl;
          ++i;
        }
        std::cout << "\n"
                  << std::setw(15) << "Begin" << std::setw(15) << "End" << std::setw(15) << "Entries" << std::setw(15);
        for (auto i : std::views::iota(0U, branchProcessors.size())) {
          std::cout << std::setw(5) << (std::string("[") + std::to_string(i) + "]");
        }
        std::cout << std::endl;
      }

      void beginCluster(Long64_t clusterBegin, Long64_t clusterEnd) const {
        std::cout << std::setw(15) << clusterBegin << std::setw(15) << clusterEnd << std::setw(15)
                  << (clusterEnd - clusterBegin);
      }

      void processBranch(Long64_t bytes, unsigned int baskets) const { std::cout << std::setw(5) << baskets; }

      void endCluster() const { std::cout << std::endl; }
    };
    processClusters(tr, BasketPrinter{}, branchName);
  }

  void clusterPrint(ROOT::RNTupleReader &) {}
  void pagePrint(ROOT::RNTupleReader &, std::string const &) {}

  std::string getUuid(ROOT::RNTupleReader *uuidTree) {
    FileID fid;
    auto entry = uuidTree->GetModel().CreateEntry();
    assert(entry);
    entry->BindRawPtr(poolNames::fileIdentifierBranchName(), &fid);
    uuidTree->LoadEntry(0, *entry);
    return fid.fid();
  }

  void printUuids(ROOT::RNTupleReader *uuidTree) { std::cout << "UUID: " << getUuid(uuidTree) << std::endl; }

  static void postIndexIntoFilePrintEventLists(TFile *tfl, ROOT::RNTupleReader &metaDataReader) {
    if (metaDataReader.GetModel().GetFieldNames().find(poolNames::indexIntoFileBranchName()) ==
        metaDataReader.GetModel().GetFieldNames().end()) {
      std::cout << "IndexIntoFile not found this indicates a problem with the file.";
      return;
    }
    IndexIntoFile indexIntoFile;
    auto metaDataEntry = metaDataReader.GetModel().CreateEntry();
    metaDataEntry->BindRawPtr(poolNames::indexIntoFileBranchName(), &indexIntoFile);
    metaDataReader.LoadEntry(0, *metaDataEntry);

    //need to read event # from the EventAuxiliary branch
    auto *eventsTuple = tfl->Get<ROOT::RNTuple>(poolNames::eventTreeName().c_str());
    assert(nullptr != eventsTuple);
    auto eventsReader = ROOT::RNTupleReader::Open(*eventsTuple);
    if (eventsReader->GetModel().GetFieldNames().find("EventAuxiliary") ==
        eventsReader->GetModel().GetFieldNames().end()) {
      std::cout << "Failed to find EventAuxiliary Field in Events RNTuple.  Something is wrong with this file."
                << std::endl;
      return;
    }

    EventAuxiliary eventAuxiliary;
    auto eventAuxEntry = eventsReader->GetModel().CreateEntry();
    eventAuxEntry->BindRawPtr("EventAuxiliary", &eventAuxiliary);
    std::cout << "\nPrinting IndexIntoFile contents.  This includes a list of all Runs, LuminosityBlocks\n"
              << "and Events stored in the root file.\n\n";
    std::cout << std::setw(15) << "Run" << std::setw(15) << "Lumi" << std::setw(15) << "Event" << std::setw(15)
              << "TTree Entry"
              << "\n";

    for (IndexIntoFile::IndexIntoFileItr it = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder),
                                         itEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
         it != itEnd;
         ++it) {
      IndexIntoFile::EntryType t = it.getEntryType();
      std::cout << std::setw(15) << it.run() << std::setw(15) << it.lumi();
      EventNumber_t eventNum = 0;
      std::string type;
      switch (t) {
        case IndexIntoFile::kRun:
          type = "(Run)";
          break;
        case IndexIntoFile::kLumi:
          type = "(Lumi)";
          break;
        case IndexIntoFile::kEvent:
          eventsReader->LoadEntry(it.entry(), *eventAuxEntry);
          eventNum = eventAuxiliary.id().event();
          break;
        default:
          break;
      }
      std::cout << std::setw(15) << eventNum << std::setw(15) << it.entry() << " " << type << std::endl;
    }

    if (indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::firstAppearanceOrder)) {
      std::cout << "Events are sorted such that fast copy is possible in the \"noEventSort = false\" mode\n";
    } else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the \"noEventSort = false\" mode\n";
    }

    // This will not work unless the other nonpersistent parts of the Index are filled first
    // I did not have time to implement this yet.
    // if (indexIntoFile.iterationWillBeInEntryOrder(IndexIntoFile::numericalOrder)) {
    //   std::cout << "Events are sorted such that fast copy is possible in the \"noEventSort\" mode\n";
    // } else {
    //   std::cout << "Events are sorted such that fast copy is NOT possible in the \"noEventSort\" mode\n";
    // }
    std::cout << "(Note that other factors can prevent fast copy from occurring)\n\n";
  }

  void printEventLists(TFile *tfl) {
    auto metaDataTuple = tfl->Get<ROOT::RNTuple>(edm::poolNames::metaDataTreeName().c_str());
    assert(nullptr != metaDataTuple);
    auto reader = ROOT::RNTupleReader::Open(*metaDataTuple);

    postIndexIntoFilePrintEventLists(tfl, *reader);
  }

  static void postIndexIntoFilePrintEventsInLumis(ROOT::RNTupleReader &metaDataReader) {
    IndexIntoFile indexIntoFile;
    auto metaDataEntry = metaDataReader.GetModel().CreateEntry();
    metaDataEntry->BindRawPtr(poolNames::indexIntoFileBranchName(), &indexIntoFile);
    metaDataReader.LoadEntry(0, *metaDataEntry);

    if (metaDataReader.GetModel().GetFieldNames().find(poolNames::indexIntoFileBranchName()) ==
        metaDataReader.GetModel().GetFieldNames().end()) {
      std::cout << "IndexIntoFile not found.  If this input file was created with release 1_8_0 or later\n"
                   "this indicates a problem with the file.  This condition should be expected with\n"
                   "files created with earlier releases and printout of the event list will fail.\n";
      return;
    }
    std::cout << "\n"
              << std::setw(15) << "Run" << std::setw(15) << "Lumi" << std::setw(15) << "# Events"
              << "\n";

    unsigned long nEvents = 0;
    unsigned long runID = 0;
    unsigned long lumiID = 0;

    for (IndexIntoFile::IndexIntoFileItr it = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder),
                                         itEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
         it != itEnd;
         ++it) {
      IndexIntoFile::EntryType t = it.getEntryType();
      switch (t) {
        case IndexIntoFile::kRun:
          break;
        case IndexIntoFile::kLumi:
          if (runID != it.run() || lumiID != it.lumi()) {
            //print the previous one
            if (lumiID != 0) {
              std::cout << std::setw(15) << runID << std::setw(15) << lumiID << std::setw(15) << nEvents << "\n";
            }
            nEvents = 0;
            runID = it.run();
            lumiID = it.lumi();
          }
          break;
        case IndexIntoFile::kEvent:
          ++nEvents;
          break;
        default:
          break;
      }
    }
    //print the last one
    if (lumiID != 0) {
      std::cout << std::setw(15) << runID << std::setw(15) << lumiID << std::setw(15) << nEvents << "\n";
    }
    std::cout << "\n";
  }

  void printEventsInLumis(TFile *tfl) {
    auto metaDataTuple = tfl->Get<ROOT::RNTuple>(edm::poolNames::metaDataTreeName().c_str());
    assert(nullptr != metaDataTuple);
    auto reader = ROOT::RNTupleReader::Open(*metaDataTuple);

    postIndexIntoFilePrintEventsInLumis(*reader);
  }
}  // namespace edm::rntuple_temp
