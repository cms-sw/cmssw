#ifndef Modules_CollUtil_h
#define Modules_CollUtil_h

#include "Rtypes.h"

#include <string>

class TFile;
class TTree;

namespace edm {
  
  TFile* openFileHdl(const std::string& fname) ;
  void printTrees(TFile *hdl);
  Long64_t numEntries(TFile *hdl, const std::string& trname);
  void printBranchNames(TTree *tree);
  void longBranchPrint(TTree *tr);
  std::string getUuid(TTree *uuidTree);
  void printUuids(TTree *uuidTree);
  void printEventLists(TFile *tfl);
}

#endif
