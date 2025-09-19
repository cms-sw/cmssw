#ifndef Modules_CollUtil_h
#define Modules_CollUtil_h

#include "Rtypes.h"

#include <string>

class TFile;
class TTree;

namespace ROOT {
  class RNTupleReader;
}

namespace edm::rntuple_temp {

  TFile *openFileHdl(const std::string &fname);
  void printTrees(TFile *hdl);
  Long64_t numEntries(TFile *hdl, const std::string &trname);
  void printBranchNames(TTree *tree);
  void longBranchPrint(TTree *tr);
  void clusterPrint(TTree *tr);
  void basketPrint(TTree *tr, const std::string &branchName);
  std::string getUuid(ROOT::RNTupleReader *uuidTree);
  void printUuids(ROOT::RNTupleReader *uuidTree);
  void printEventLists(TFile *tfl);
  void printEventsInLumis(TFile *tfl);
}  // namespace edm::rntuple_temp

#endif
