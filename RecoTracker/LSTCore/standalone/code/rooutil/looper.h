//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef looper_cc
#define looper_cc

// C/C++
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdarg.h>
#include <functional>
#include <cmath>

// ROOT
#include "TBenchmark.h"
#include "TBits.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TChainElement.h"
#include "TTreeCache.h"
#include "TTreePerfStats.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TString.h"
#include "TLorentzVector.h"
#include "Math/LorentzVector.h"

#include "printutil.h"
#include "eventindexmap.h"
#include "treeutil.h"

//#include "cpptqdm/tqdm.h"

namespace RooUtil {

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Looper class
  ///////////////////////////////////////////////////////////////////////////////////////////////
  class Looper {
    // Members
    TChain* tchain;
    TBenchmark* bmark;
    TObjArray* listOfFiles;
    TObjArrayIter* fileIter;
    TFile* tfile;
    TTree* ttree;
    TTreePerfStats* ps;
    unsigned int nEventsTotalInChain;
    unsigned int nEventsTotalInTree;
    int nEventsToProcess;
    unsigned int nEventsProcessed;
    unsigned int indexOfEventInTTree;
    bool fastmode;
    TreeUtil* treeclass;
    TStopwatch my_timer;
    int bar_id;
    int print_rate;
    bool doskim;
    TString skimfilename;
    TFile* skimfile;
    TTree* skimtree;
    unsigned int nEventsSkimmed;
    std::vector<TString> skimbrfiltpttn;
    bool silent;
    bool isinit;
    bool use_treeclass_progress;
    bool isnewfileopened;
    //        bool use_tqdm_progress_bar;
    unsigned int nskipped_batch;
    unsigned int nskipped;
    unsigned int nbatch_skip_threshold;
    unsigned int nbatch_to_skip;
    unsigned int nskipped_threshold;
    unsigned int ncounter;
    //        tqdm bar;
    EventIndexMap eventindexmap;
    TEventList* teventlist;

  public:
    // Functions
    Looper();
    Looper(TChain* chain, TreeUtil* treeclass, int nEventsToProcess = -1);
    ~Looper();
    void init(TChain* chain, TreeUtil* treeclass, int nEventsToProcess);
    void setTChain(TChain* c);
    void setTreeClass(TreeUtil* t);
    void printCurrentEventIndex();
    void setSilent(bool s = true) { silent = s; }
    void setEventIndexMap(TString file) { eventindexmap.load(file); }
    bool allEventsInTreeProcessed();
    bool allEventsInChainProcessed();
    bool nextEvent();
    bool isNewFileInChain();
    TTree* getTree() { return ttree; }
    TChain* getTChain() { return tchain; }
    unsigned int getNEventsProcessed() { return nEventsProcessed; }
    void setSkim(TString ofilename);
    void setSkimBranchFilterPattern(std::vector<TString> x) { skimbrfiltpttn = x; }
    void fillSkim();
    void saveSkim();
    TTree* getSkimTree() { return skimtree; }
    void setSkimMaxSize(Long64_t maxsize) { skimtree->SetMaxTreeSize(maxsize); }
    TTreePerfStats* getTTreePerfStats() { return ps; }
    unsigned int getCurrentEventIndex() { return indexOfEventInTTree - 1; }
    TFile* getCurrentFile() { return tfile; }
    TString getCurrentFileBaseName() { return gSystem->BaseName(tfile->GetName()); }
    TString getCurrentFileName() { return TString(tfile->GetName()); }
    TString getListOfFileNames();
    TString getCurrentFileTitle() { return TString(tfile->GetTitle()); }
    unsigned int getNEventsTotalInChain() { return nEventsTotalInChain; }
    void setNbatchToSkip(unsigned int n) { nbatch_to_skip = n; }
    void setNbadEventThreshold(unsigned int n) { nskipped_threshold = n; }
    void setNbadEventThresholdToTriggerBatchSkip(unsigned int n) { nbatch_skip_threshold = n; }
    bool handleBadEvent();
    void printStatus();
    void printSkippedBadEventStatus();
    void setFastMode(bool f = true) { fastmode = f; }
    void addCount() { ncounter++; }
    void resetCounter() { ncounter = 0; }
    bool doesBranchExist(TString bname);
    TString getSkimFileName() { return skimfilename; }
    TFile* getSkimFile() { return skimfile; }

  private:
    void setFileList();
    void setNEventsToProcess();
    bool nextTree();
    bool nextEventInTree();
    void initProgressBar();
    void printProgressBar(bool force = false);
    void createSkimTree();
    void copyAddressesToSkimTree();
  };

}  // namespace RooUtil

#endif
