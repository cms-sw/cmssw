#include "IOPool/Common/bin/CollUtil.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include <iostream>
#include <iomanip>

#include "TFile.h" 
#include "TList.h"
#include "TIterator.h"
#include "TKey.h"
#include "TTree.h"
#include "TObject.h"
#include "TBranch.h"

namespace edm {

  // Get a file handler
  TFile* openFileHdl(const std::string& fname) {
    
    TFile *hdl= TFile::Open(fname.c_str(),"read");

    if ( 0== hdl ) {
      std::cout << "ERR Could not open file " << fname.c_str() << std::endl;
      exit(1);
    }
    return hdl;
  }

  // Print every tree in a file
  void printTrees(TFile *hdl) {

    hdl->ls();
    TList *keylist= hdl->GetListOfKeys();
    TIterator *iter= keylist->MakeIterator();
    TKey *key;
    while ( (key= (TKey*)iter->Next()) ) {
      TObject *obj= hdl->Get(key->GetName());
      if ( obj->IsA() == TTree::Class() ) {
	obj->Print();
      }
    }
    return;
  }

  // number of entries in a tree
  Long64_t numEntries(TFile *hdl, const std::string& trname) {

    TTree *tree= (TTree*)hdl->Get(trname.c_str());
    if ( tree ) {
      return tree->GetEntries();
    } else {
      std::cout << "ERR cannot find a TTree named \"" << trname << "\"" 
		<< std::endl;
      return -1;
    }
  }


  void printBranchNames( TTree *tree) {

    if ( tree != 0 ) {
      Long64_t nB=tree->GetListOfBranches()->GetEntries();
      for ( Long64_t i=0; i<nB; ++i) {
	TBranch *btemp = (TBranch *)tree->GetListOfBranches()->At(i);
    	std::cout << "Branch " << i <<" of " << tree->GetName() <<" tree: " << btemp->GetName() << " Total size = " << btemp->GetTotalSize() << std::endl;
      }
    }
    else{
      std::cout << "Missing Events tree?\n";
    }

  }

  void longBranchPrint( TTree *tr) {

    if ( tr != 0 ) {
      Long64_t nB=tr->GetListOfBranches()->GetEntries();
      for ( Long64_t i=0; i<nB; ++i) {    
	tr->GetListOfBranches()->At(i)->Print();
      }
    }
    else{
      std::cout << "Missing Events tree?\n";
    }
    
  }

  void printUuids(TTree *uuidTree) {
    FileID fid;
    FileID *fidPtr = &fid;
    uuidTree->SetBranchAddress(poolNames::fileIdentifierBranchName().c_str(), &fidPtr);
    uuidTree->GetEntry(0);

    std::cout << "UUID: " << fid.fid() << std::endl;
  }

  static void preIndexIntoFilePrintEventLists(TFile* tfl, const FileFormatVersion& fileFormatVersion, TTree *metaDataTree) {

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
      std::cout << "Events are sorted such that fast copy is possible in the default mode\n";
    }
    else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the default mode\n";
    }

    fileIndex.sortBy_Run_Lumi_EventEntry();
    if (fileIndex.allEventsInEntryOrder()) {
      std::cout << "Events are sorted such that fast copy is possible in the \"noEventSort\" mode\n";
    }
    else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the \"noEventSort\" mode\n";
    }
    std::cout << "(Note that other factors can prevent fast copy from occurring)\n\n";
  }

  static void postIndexIntoFilePrintEventLists(TFile* tfl, const FileFormatVersion& fileFormatVersion, TTree *metaDataTree) {

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
    TTree* eventsTree = dynamic_cast<TTree*>(tfl->Get(edm::poolNames::eventTreeName().c_str()));
    TBranch* eventAuxBranch=0;
    assert(0!=eventsTree);
    const char* const kEventAuxiliaryBranchName = "EventAuxiliary";
    if(eventsTree->FindBranch(kEventAuxiliaryBranchName)!=0){
      eventAuxBranch = eventsTree->GetBranch(kEventAuxiliaryBranchName);
    } else {
      std::cout <<"Failed to find "<<kEventAuxiliaryBranchName<<" branch in Events TTree.  Something is wrong with this file."<<std::endl;
      return;
    }

    EventAuxiliary eventAuxiliary;
    EventAuxiliary* eAPtr=&eventAuxiliary;
    eventAuxBranch->SetAddress(&eAPtr);
    std::cout << "\nPrinting IndexIntoFile contents.  This includes a list of all Runs, LuminosityBlocks\n"
       << "and Events stored in the root file.\n\n";
    std::cout << std::setw(15) << "Run"
       << std::setw(15) << "Lumi"
       << std::setw(15) << "Event"
       << std::setw(15) << "TTree Entry"
       << "\n";
    
    for(IndexIntoFile::IndexIntoFileItr it = indexIntoFile.begin(false), itEnd = indexIntoFile.end(false);
    it != itEnd;
    ++it) {
      IndexIntoFile::EntryType t = it.getEntryType();
      std::cout << std::setw(15)<<it.run()<< std::setw(15)<<it.lumi();
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
      std::cout << std::setw(15)<<eventNum << std::setw(15) << it.entry()<<" "<<type<<std::endl;
    }

    std::cout << "\nFileFormatVersion = " << fileFormatVersion << ".  ";
    if (fileFormatVersion.fastCopyPossible()) std::cout << "This version supports fast copy\n";
    else std::cout << "This version does not support fast copy\n";

    if (indexIntoFile.allEventsInEntryOrder(false)) {
      std::cout << "Events are sorted such that fast copy is possible in the default mode\n";
    }
    else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the default mode\n";
    }

    if (indexIntoFile.allEventsInEntryOrder(false)) {
      std::cout << "Events are sorted such that fast copy is possible in the \"noEventSort\" mode\n";
    }
    else {
      std::cout << "Events are sorted such that fast copy is NOT possible in the \"noEventSort\" mode\n";
    }
    std::cout << "(Note that other factors can prevent fast copy from occurring)\n\n";
  }
  
  void printEventLists(TFile *tfl) {
    TTree *metaDataTree = dynamic_cast<TTree *>(tfl->Get(poolNames::metaDataTreeName().c_str()));

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
  
}
