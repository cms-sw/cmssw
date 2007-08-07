#include "IOPool/Common/bin/CollUtil.h"
#include "TFile.h" 
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
//#include "DataFormats/Common/interface/EDProduct.h"

#include <iostream>

#include "TObject.h"
#include "TKey.h"
#include "TList.h"
#include "TIterator.h"
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

  // show event if for the specified events
  //  void showEvents(TFile *hdl, const std::string& trname, const std::string& evtstr) {
  void showEvents(TFile *hdl, const std::string& trname, const Long64_t& iLow, const Long64_t& iHigh) {

    TTree *tree= (TTree*)hdl->Get(trname.c_str());
    
    if ( tree != 0 ) {

      EventAuxiliary* evtAux_=0;
      TBranch *evtAuxBr = tree->GetBranch("EventAuxiliary");

      tree->SetBranchAddress("EventAuxiliary",&evtAux_);
      Long64_t max= tree->GetEntries();
      for (Long64_t i = iLow; i <= iHigh && i < max; ++i) {
	evtAuxBr->GetEntry(i);
	if ( evtAux_ != 0 ) {
	  Timestamp time_=evtAux_->time();
	  EventID id_=evtAux_->id();
	  std::cout << id_ << "  time: " << time_.value() << std::endl;
	}
	else{
	  std::cout << "Event: " << i << " Nonsense EventAuxiliary object? " << std::endl;
	}
      }
      
    } else {
      std::cout << "ERR cannot find a TTree named \"" << trname << "\""
                << std::endl;
      return;
    }

    return;
  }

//   void showEventsAndEntries(TFile *hdl, const std::string& trname, const Long64_t& iLow, const Long64_t& iHigh) {

//     TTree *tree= (TTree*)hdl->Get(trname.c_str());
    
//     if ( tree != 0 ) {

//       EventAuxiliary* evtAux_=0;
//       TBranch *evtAuxBr = tree->GetBranch("EventAuxiliary");
//       tree->SetBranchAddress("EventAuxiliary",&evtAux_);
//       Long64_t max= tree->GetEntries();
//       int entrycounter = 0;
//       for (Long64_t i=iLow; i <= iHigh && i < max; ++i) {
// 	evtAuxBr->GetEntry(i);
// 	if ( evtAux_ != 0 ) {
// 	  Timestamp time_=evtAux_->time();
// 	  EventID id_=evtAux_->id();
// 	  std::cout << id_ << "  time: " << time_.value() << std::endl;
	  
// 	  // Now count # of entries in each branch for this event
// 	  Long64_t nB=tree->GetListOfBranches()->GetEntries();
// 	  std::cout << "No. of branches = " << nB << std::endl;
	  
// 	  for ( Long64_t j = 0; j < nB; ++j) {
// 	    TBranch *br = (TBranch *)tree->GetListOfBranches()->At(j);
//  	    TString branchName = br->GetName();
// 	    branchName+="obj.";
//  	    std::cout << "The branch name is " << branchName << std::endl;
// 	    // 	    TClass *cp = gROOT->GetClass(br->GetClassName());
// 	    TClass *cp = gROOT->GetClass(branchName);
//  	    std::cout << "GotClass " << cp->GetName() << std::endl;	  
// 	    EDProduct *p = static_cast<EDProduct *>(cp->New());
// 	    std::cout << "Got EDProduct" << std::endl;
// 	    br->SetAddress(&p);
// 	    br->GetEntry(i);
// 	  }
// 	}
// 	else{
// 	  std::cout << "Event: " << i << " Nonsense EventAuxiliary object? " << std::endl;
// 	}
//       }
      
//     } else {
//       std::cout << "ERR cannot find a TTree named \"" << trname << "\""
//                 << std::endl;
//       return;
//     }
    
//     return;
//   }


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

  void printUuids( TTree *uuidTree) {
    char uuidCA[1024];
      uuidTree->SetBranchAddress("db_string",uuidCA);
      uuidTree->GetEntry(0);

      // Then pick out relevent piece of this string
      // 9A440868-8058-DB11-85E3-00304885AB94 from
      // [NAME=FID][VALUE=9A440868-8058-DB11-85E3-00304885AB94]

      std::string uuidStr(uuidCA);
      std::string::size_type start=uuidStr.find("VALUE=");
      if ( start == std::string::npos ) {
	std::cout << "Seemingly invalid db_string entry in ##Params tree?\n";
	std::cout << uuidStr << std::endl;
      }
      else{
	std::string::size_type stop=uuidStr.find("]",start);
	if ( stop == std::string::npos ) {
	  std::cout << "Seemingly invalid db_string entry in ##Params tree?\n";
	  std::cout << uuidStr << std::endl;
	}
	else{
	  //Everything is Ok - just proceed...
	  std::string result=uuidStr.substr(start+6,stop-start-6);
	  std::cout << "UUID: " << result << std::endl;
	}
      }    
  }

  void printEventLists( std::string remainingEvents, int numevents, TFile *tfl, bool entryoption) {
    bool keepgoing=true;
    while ( keepgoing ) {
      int iLow(-1),iHigh(-2);
      // split by commas
      std::string::size_type pos= remainingEvents.find_first_of(",");
      std::string evtstr=remainingEvents;
	
      if ( pos == std::string::npos ) {
	keepgoing=false;
      }
      else{
	evtstr=remainingEvents.substr(0,pos);
	remainingEvents=remainingEvents.substr(pos+1);
      }
      
      pos= evtstr.find_first_of("-");
      if ( pos == std::string::npos ) {
	iLow= (int)atof(evtstr.c_str());
	iHigh= iLow;
      } else {
	iLow= (int)atof(evtstr.substr(0,pos).c_str());
	iHigh= (int)atof(evtstr.substr(pos+1).c_str());
      }
      
      //    edm::showEvents(tfile,"Events",vm["events"].as<std::string>());
      if ( iLow < 1 ) iLow=1;
      if ( iHigh > numevents ) iHigh=numevents;
      
      // shift by one.. C++ starts at 0
      iLow--;
      iHigh--;
      if(entryoption==false)
	showEvents(tfl,"Events",iLow,iHigh);
//       else if(entryoption==true)
// 	showEventsAndEntries(tfl,"Events",iLow,iHigh);
    }
    
  }
  
}
