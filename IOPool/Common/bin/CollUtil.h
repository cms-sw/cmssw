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
  long int numEntries(TFile *hdl, const std::string& trname);
  void showEvents(TFile *hdl, const std::string& trname, const long firstEv, const long lastEv);
  //  void showEvents(TFile *hdl, const std::string& trname, const std::string& evtstr);

}

#endif

    
