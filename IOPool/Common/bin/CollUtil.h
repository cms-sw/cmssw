#ifndef Modules_CollUtil_h
#define Modules_CollUtil_h

#include <vector>
#include <string>

#include "TFile.h"
#include "TTree.h"

namespace edm {
  
  TFile* openFileHdl(const std::string& fname) ;
  void printTrees(TFile *hdl);
  void printBranchNames(TTree *tree);
  Long64_t numEntries(TFile *hdl, const std::string& trname);
  void showEvents(TFile *hdl, const std::string& trname, const Long64_t& firstEv, const Long64_t& lastEv);
/*   void showEventsAndEntries(TFile *hdl, const std::string& trname, const int firstEv, const int lastEv); */
  void longBranchPrint(TTree *tr);
  void printUuids(TTree *uuidTree);
  void printEventLists(std::string remainingEvents, int numevents, TFile *tfl, bool displayEntries);
  //  void showEvents(TFile *hdl, const std::string& trname, const std::string& evtstr);

}

#endif

    
