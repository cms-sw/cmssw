#include "IOPool/Common/bin/CollUtil.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileIndex.h"

#include <iostream>

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

  void printEventLists(TFile *tfl) {

    FileFormatVersion fileFormatVersion;
    FileFormatVersion *fftPtr = &fileFormatVersion;

    FileIndex fileIndex;
    FileIndex *findexPtr = &fileIndex;

    TTree *metaDataTree = dynamic_cast<TTree *>(tfl->Get(poolNames::metaDataTreeName().c_str()));
    metaDataTree->SetBranchAddress(poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);
    if (metaDataTree->FindBranch(poolNames::fileIndexBranchName().c_str()) != 0) {
      metaDataTree->SetBranchAddress(poolNames::fileIndexBranchName().c_str(), &findexPtr);
    }
    else {
      std::cout << "FileIndex not found.  If this input file was created with release 1_8_0 or later\n"
                   "this indicates a problem with the file.  This condition should be expected with\n"
	"files created with earlier releases and printout of the event list will fail.\n";
      return;
    }
    metaDataTree->GetEntry(0);

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
}
