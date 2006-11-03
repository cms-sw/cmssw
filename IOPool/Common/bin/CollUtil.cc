#include "IOPool/Common/bin/CollUtil.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"

#include <iostream>
#include <string>

#include "TObject.h"
#include "TKey.h"
#include "TList.h"
#include "TIterator.h"
#include "TTree.h"
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
  long int numEntries(TFile *hdl, const std::string& trname) {

    TTree *tree= (TTree*)hdl->Get(trname.c_str());
    if ( tree ) {
      return tree->GetEntries();
    } else {
      std::cout << "ERR cannot find a TTree named \"" << trname << "\"" 
		<< std::endl;
      return -1;
    }
  }

  // show event if for the specified events
  //  void showEvents(TFile *hdl, const std::string& trname, const std::string& evtstr) {
  void showEvents(TFile *hdl, const std::string& trname, const long iLow, const long iHigh) {

    TTree *tree= (TTree*)hdl->Get(trname.c_str());
    
    if ( tree != 0 ) {

      EventAux* evtAux_=0;
      TBranch *evtAuxBr = tree->GetBranch("EventAux");

      tree->SetBranchAddress("EventAux",&evtAux_);
      long int max= tree->GetEntries();
      for (long int i=iLow; i<= iHigh && i< max; i++) {
	evtAuxBr->GetEntry(i);
	if ( evtAux_ != 0 ) {
	  Timestamp time_=evtAux_->time();
	  EventID id_=evtAux_->id();
	  std::cout << id_ << "  time: " << time_.value() << std::endl;
	}
	else{
	  std::cout << "Event: " << i << " Nonsense EventAux object? " << std::endl;
	}
      }
      
    } else {
      std::cout << "ERR cannot find a TTree named \"" << trname << "\""
                << std::endl;
      return;
    }

    return;
  }

  void printBranchNames( TTree *tree) {

    if ( tree != 0 ) {
      int nB=tree->GetListOfBranches()->GetEntries();
      for ( int i=0; i<nB; i++) {
    	std::cout << "Branch " << i <<" of " << tree->GetName() <<" tree: " << tree->GetListOfBranches()->At(i)->GetName() << std::endl;
      }
    }
    else{
      std::cout << "Missing Events tree?\n";
    }

  }

}
