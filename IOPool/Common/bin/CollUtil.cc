#include "IOPool/Common/bin/CollUtil.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"

#include "TBranch.h"
#include "TFile.h"
#include "TIterator.h"
#include "TKey.h"
#include "TList.h"
#include "TObject.h"
#include "TTree.h"

#include <iomanip>
#include <iostream>

namespace edm {

  // Get a file handler
  TFile* openFileHdl(std::string const& fname) {
    TFile *hdl = TFile::Open(fname.c_str(), "read");

    if (0 == hdl) {
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
    TKey *key = 0;
    while ((key = (TKey*)iter->Next())) {
      TObject *obj = hdl->Get(key->GetName());
      if (obj->IsA() == TTree::Class()) {
        obj->Print();
      }
    }
    return;
  }

  // number of entries in a tree
  Long64_t numEntries(TFile *hdl, std::string const& trname) {
    TTree *tree = (TTree*)hdl->Get(trname.c_str());
    if (tree) {
      return tree->GetEntries();
    } else {
      std::cout << "ERR cannot find a TTree named \"" << trname << "\""
                << std::endl;
      return -1;
    }
  }

  namespace {
    void addBranchSizes(TBranch *branch, Long64_t& size) {
      size += branch->GetTotalSize(); // Includes size of branch metadata
      // Now recurse through any subbranches.
      Long64_t nB = branch->GetListOfBranches()->GetEntries();
      for (Long64_t i = 0; i < nB; ++i) {
        TBranch *btemp = (TBranch *)branch->GetListOfBranches()->At(i);
        addBranchSizes(btemp, size);
      }
    }
  }

  void printBranchNames(TTree *tree) {
    if (tree != 0) {
      Long64_t nB = tree->GetListOfBranches()->GetEntries();
      for (Long64_t i = 0; i < nB; ++i) {
        Long64_t size = 0LL;
        TBranch *btemp = (TBranch *)tree->GetListOfBranches()->At(i);
        addBranchSizes(btemp, size);
        std::cout << "Branch " << i << " of " << tree->GetName() << " tree: " << btemp->GetName() << " Total size = " << size << std::endl;
      }
    } else {
      std::cout << "Missing Events tree?\n";
    }
  }

  void longBranchPrint(TTree *tr) {
    if (tr != 0) {
      Long64_t nB = tr->GetListOfBranches()->GetEntries();
      for (Long64_t i = 0; i < nB; ++i) {
        tr->GetListOfBranches()->At(i)->Print();
      }
    } else {
      std::cout << "Missing Events tree?\n";
    }
  }

  std::string getUuid(TTree *uuidTree) {
    FileID fid;
    FileID *fidPtr = &fid;
    uuidTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    uuidTree->GetEntry(0);
    return fid.fid();
  }

  void printUuids(TTree *uuidTree) {
    FileID fid;
    FileID *fidPtr = &fid;
    uuidTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    uuidTree->GetEntry(0);

    std::cout << "UUID: " << fid.fid() << std::endl;
  }

  static void preIndexIntoFilePrintEventLists(TFile*, FileFormatVersion const& fileFormatVersion, TTree *metaDataTree) {
    FileIndex fileIndex;
    FileIndex *findexPtr = &fileIndex;
    if (metaDataTree->FindBranch(poolNames::fileIndexBranchName().c_str()) != 0) {
      TBranch *fndx = metaDataTree->GetBranch(poolNames::fileIndexBranchName().c_str());
      fndx->SetAddress(&findexPtr);
      fndx->GetEntry(0);
    } else {
      std::cout << "FileIndex not found.  If this input file was created with release 1_8_0 or later\n"
                   "this indicates a problem with the file.  This condition should be expected with\n"
        "files created with earlier releases and printout of the event list will fail.\n";
      return;
    }

    std::cout << "\n" << fileIndex;

    std::cout << "\nFileFormatVersion = " << fileFormatVersion << ".  ";
    if (fileFormatVersion.fastCopyPossible()) std::cout << "This version supports fast copy\n";
    else std::cout << "This version does not support fast copy\n";

    if (fileIndex.allEventsInEntryOrder()) {
      std::cout << "Events are sorted such that fast copy is possible in the \"noEventSort = False\" mode\n";
    } else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the \"noEventSort = False\" mode\n";
    }

    fileIndex.sortBy_Run_Lumi_EventEntry();
    if (fileIndex.allEventsInEntryOrder()) {
      std::cout << "Events are sorted such that fast copy is possible in the \"noEventSort\" mode\n";
    } else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the \"noEventSort\" mode\n";
    }
    std::cout << "(Note that other factors can prevent fast copy from occurring)\n\n";
  }

  static void postIndexIntoFilePrintEventLists(TFile* tfl, FileFormatVersion const& fileFormatVersion, TTree *metaDataTree) {
    IndexIntoFile indexIntoFile;
    IndexIntoFile *findexPtr = &indexIntoFile;
    if (metaDataTree->FindBranch(poolNames::indexIntoFileBranchName().c_str()) != 0) {
      TBranch *fndx = metaDataTree->GetBranch(poolNames::indexIntoFileBranchName().c_str());
      fndx->SetAddress(&findexPtr);
      fndx->GetEntry(0);
    } else {
      std::cout << "IndexIntoFile not found.  If this input file was created with release 1_8_0 or later\n"
                   "this indicates a problem with the file.  This condition should be expected with\n"
                   "files created with earlier releases and printout of the event list will fail.\n";
      return;
    }
    //need to read event # from the EventAuxiliary branch
    TTree* eventsTree = dynamic_cast<TTree*>(tfl->Get(poolNames::eventTreeName().c_str()));
    TBranch* eventAuxBranch = 0;
    assert(0 != eventsTree);
    char const* const kEventAuxiliaryBranchName = "EventAuxiliary";
    if(eventsTree->FindBranch(kEventAuxiliaryBranchName) != 0){
      eventAuxBranch = eventsTree->GetBranch(kEventAuxiliaryBranchName);
    } else {
      std::cout << "Failed to find " << kEventAuxiliaryBranchName << " branch in Events TTree.  Something is wrong with this file." << std::endl;
      return;
    }
    EventAuxiliary eventAuxiliary;
    EventAuxiliary* eAPtr = &eventAuxiliary;
    eventAuxBranch->SetAddress(&eAPtr);
    std::cout << "\nPrinting IndexIntoFile contents.  This includes a list of all Runs, LuminosityBlocks\n"
       << "and Events stored in the root file.\n\n";
    std::cout << std::setw(15) << "Run"
       << std::setw(15) << "Lumi"
       << std::setw(15) << "Event"
       << std::setw(15) << "TTree Entry"
       << "\n";

    for(IndexIntoFile::IndexIntoFileItr it = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder),
                                        itEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
                                        it != itEnd; ++it) {
      IndexIntoFile::EntryType t = it.getEntryType();
      std::cout << std::setw(15) << it.run() << std::setw(15) << it.lumi();
      EventNumber_t eventNum = 0;
      std::string type;
      switch(t) {
        case IndexIntoFile::kRun:
          type = "(Run)";
        break;
        case IndexIntoFile::kLumi:
          type = "(Lumi)";
        break;
        case IndexIntoFile::kEvent:
          eventAuxBranch->GetEntry(it.entry());
          eventNum = eventAuxiliary.id().event();
        break;
        default:
        break;
      }
      std::cout << std::setw(15) << eventNum << std::setw(15) << it.entry() << " " << type << std::endl;
    }

    std::cout << "\nFileFormatVersion = " << fileFormatVersion << ".  ";
    if (fileFormatVersion.fastCopyPossible()) std::cout << "This version supports fast copy\n";
    else std::cout << "This version does not support fast copy\n";

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
    TTree *metaDataTree = dynamic_cast<TTree *>(tfl->Get(poolNames::metaDataTreeName().c_str()));
    assert(0 != metaDataTree);

    FileFormatVersion fileFormatVersion;
    FileFormatVersion *fftPtr = &fileFormatVersion;
    if (metaDataTree->FindBranch(poolNames::fileFormatVersionBranchName().c_str()) != 0) {
      TBranch *fft = metaDataTree->GetBranch(poolNames::fileFormatVersionBranchName().c_str());
      fft->SetAddress(&fftPtr);
      fft->GetEntry(0);
    }
    if(fileFormatVersion.hasIndexIntoFile()) {
      postIndexIntoFilePrintEventLists(tfl, fileFormatVersion, metaDataTree);
    } else {
      preIndexIntoFilePrintEventLists(tfl, fileFormatVersion, metaDataTree);
    }
  }

  static void preIndexIntoFilePrintEventsInLumis(TFile*, FileFormatVersion const& fileFormatVersion, TTree *metaDataTree) {
    FileIndex fileIndex;
    FileIndex *findexPtr = &fileIndex;
    if (metaDataTree->FindBranch(poolNames::fileIndexBranchName().c_str()) != 0) {
      TBranch *fndx = metaDataTree->GetBranch(poolNames::fileIndexBranchName().c_str());
      fndx->SetAddress(&findexPtr);
      fndx->GetEntry(0);
    } else {
      std::cout << "FileIndex not found.  If this input file was created with release 1_8_0 or later\n"
      "this indicates a problem with the file.  This condition should be expected with\n"
      "files created with earlier releases and printout of the event list will fail.\n";
      return;
    }

    std::cout <<"\n"<< std::setw(15) << "Run"
    << std::setw(15) << "Lumi"
    << std::setw(15) << "# Events"
    << "\n";
    unsigned long nEvents = 0;
    unsigned long runID = 0;
    unsigned long lumiID = 0;
    for(std::vector<FileIndex::Element>::const_iterator it = fileIndex.begin(), itEnd = fileIndex.end(); it != itEnd; ++it) {
      if(it->getEntryType() == FileIndex::kEvent) {
        ++nEvents;
      }
      else if(it->getEntryType() == FileIndex::kLumi) {
        if(runID !=it->run_ || lumiID !=it->lumi_) {
          //print the previous one
          if(lumiID !=0) {
            std::cout << std::setw(15) << runID
            << std::setw(15) << lumiID
            << std::setw(15) << nEvents<<"\n";
          }
          nEvents=0;
          runID = it->run_;
          lumiID = it->lumi_;
        }
      }
    }
    //print the last one
    if(lumiID !=0) {
      std::cout << std::setw(15) << runID
      << std::setw(15) << lumiID
      << std::setw(15) << nEvents<<"\n";
    }

    std::cout << "\n";
  }

  static void postIndexIntoFilePrintEventsInLumis(TFile* tfl, FileFormatVersion const& fileFormatVersion, TTree *metaDataTree) {
    IndexIntoFile indexIntoFile;
    IndexIntoFile *findexPtr = &indexIntoFile;
    if (metaDataTree->FindBranch(poolNames::indexIntoFileBranchName().c_str()) != 0) {
      TBranch *fndx = metaDataTree->GetBranch(poolNames::indexIntoFileBranchName().c_str());
      fndx->SetAddress(&findexPtr);
      fndx->GetEntry(0);
    } else {
      std::cout << "IndexIntoFile not found.  If this input file was created with release 1_8_0 or later\n"
      "this indicates a problem with the file.  This condition should be expected with\n"
      "files created with earlier releases and printout of the event list will fail.\n";
      return;
    }
    std::cout <<"\n"<< std::setw(15) << "Run"
    << std::setw(15) << "Lumi"
    << std::setw(15) << "# Events"
    << "\n";

    unsigned long nEvents = 0;
    unsigned long runID = 0;
    unsigned long lumiID = 0;

    for(IndexIntoFile::IndexIntoFileItr it = indexIntoFile.begin(IndexIntoFile::firstAppearanceOrder),
        itEnd = indexIntoFile.end(IndexIntoFile::firstAppearanceOrder);
        it != itEnd; ++it) {
      IndexIntoFile::EntryType t = it.getEntryType();
      switch(t) {
        case IndexIntoFile::kRun:
          break;
        case IndexIntoFile::kLumi:
          if(runID != it.run() || lumiID != it.lumi()) {
            //print the previous one
            if(lumiID !=0) {
              std::cout << std::setw(15) << runID
              << std::setw(15) << lumiID
              << std::setw(15) << nEvents<<"\n";
            }
            nEvents=0;
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
    if(lumiID !=0) {
      std::cout << std::setw(15) << runID
      << std::setw(15) << lumiID
      << std::setw(15) << nEvents<<"\n";
    }
    std::cout << "\n";
  }

  void printEventsInLumis(TFile *tfl) {
    TTree *metaDataTree = dynamic_cast<TTree *>(tfl->Get(poolNames::metaDataTreeName().c_str()));
    assert(0 != metaDataTree);

    FileFormatVersion fileFormatVersion;
    FileFormatVersion *fftPtr = &fileFormatVersion;
    if (metaDataTree->FindBranch(poolNames::fileFormatVersionBranchName().c_str()) != 0) {
      TBranch *fft = metaDataTree->GetBranch(poolNames::fileFormatVersionBranchName().c_str());
      fft->SetAddress(&fftPtr);
      fft->GetEntry(0);
    }
    if(fileFormatVersion.hasIndexIntoFile()) {
      postIndexIntoFilePrintEventsInLumis(tfl, fileFormatVersion, metaDataTree);
    } else {
      preIndexIntoFilePrintEventsInLumis(tfl, fileFormatVersion, metaDataTree);
    }
  }
}
