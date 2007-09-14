#include "TSelectorEntries.h"
#include "TTree.h"

Long64_t GetEntriesMulti(const char* selection, TTree *tree, Int_t candscut, Long64_t nentries = 1000000, Long64_t firstentry = 0) {
  TSelectorEntries *selector;
  Long64_t rows = 0;
  Long_t entry, entryNumber, localEntry;
  Int_t cands = 0;
  Int_t ndata = 0;
  Int_t j = 0;

  selector = new TSelectorEntries(selection);
  nentries = tree->GetEntries();
  tree->SetNotify(selector);
  selector->SetOption("");
  selector->Begin(tree);       //<===call user initialisation function
  selector->SlaveBegin(tree);  //<===call user initialisation function
  selector->Notify();
  rows = 0;
  nentries = 1000000;  
  if (selector->GetAbort() != TSelector::kAbortProcess
      && (selector->Version() != 0 || selector->GetStatus() != -1)) {
    //loop on entries (elist or all entries)
    //    formula = new TTreeFormula("Selection", isoCut.c_str(), tree);
    //    formula->SetQuickLoad(kTRUE);

    for (entry=firstentry;entry<firstentry+nentries;entry++) {
      //    entryNumber = tree->GetEntryNumber(entry);
      //  if (entryNumber < 0) break;
      localEntry = tree->LoadTree(entry);
      if (localEntry < 0) break;
      ndata = selector->fSelect->GetNdata();
      cands = 0;
      if (selector->fSelect) {
        // Always call EvalInstance(0) to insure the loading
	// of the branches.
	if (ndata == 0) selector->fSelect->EvalInstance(0);
	else {
	  j = 0;
	  for (j = 0; j < ndata; j++) {
	    if (selector->fSelect->EvalInstance(j)) {
	      ++cands;
	    }
	  }
	}
      }
      if (cands >= candscut) {
        ++rows;
      }
    }
    //    delete formula; formula = 0;
  }
  if (selector->Version() != 0 || selector->GetStatus() != -1) {
    selector->SlaveTerminate();   //<==call user termination function
    selector->Terminate();        //<==call user termination function
  }
  delete selector;
  return rows;
}

void ApplyCut(const char* selection, const char* formulastring, TTree *tree, Int_t candscut, Long64_t nentries = 1000000, Long64_t firstentry = 0) {
  TSelectorEntries *selector;
  Long64_t rows = 0;
  Long_t entry, entryNumber, localEntry;
  Int_t cands = 0;
  Int_t ndata = 0;
  Int_t j = 0;

  selector = new TSelectorEntries(selection);
  nentries = tree->GetEntries();
  tree->SetNotify(selector);
  selector->SetOption("");
  selector->Begin(tree);       //<===call user initialisation function
  selector->SlaveBegin(tree);  //<===call user initialisation function
  selector->Notify();
  rows = 0;
  //  nentries = 1000000;  
  if (selector->GetAbort() != TSelector::kAbortProcess
      && (selector->Version() != 0 || selector->GetStatus() != -1)) {
    //loop on entries (elist or all entries)
    //    formula = new TTreeFormula("Selection", isoCut.c_str(), tree);
    //    formula->SetQuickLoad(kTRUE);
    
    for (entry=firstentry;entry<firstentry+nentries;entry++) {
      //    entryNumber = tree->GetEntryNumber(entry);
      //  if (entryNumber < 0) break;
      localEntry = tree->LoadTree(entry);
      if (localEntry < 0) break;
      tree->GetEntry(entry);
      ndata = selector->fSelect->GetNdata();
      cands = 0;
      Double_t *values = (Double_t *)(tree->GetLeaf(formulastring)->GetValuePointer());
      TTreeFormula *formula = new TTreeFormula("test", formulastring, tree);
      if (selector->fSelect) {
        // Always call EvalInstance(0) to insure the loading
	// of the branches.
	if (ndata == 0) selector->fSelect->EvalInstance(0);
	else {
	  j = 0;
	  for (j = 0; j < ndata; j++) {
	    if (selector->fSelect->EvalInstance(j)) {
	      ++cands;
	    }
	  }
	}
      }
      if (cands >= candscut) {
        ++rows;
      }
    }
    //    delete formula; formula = 0;
  }
  if (selector->Version() != 0 || selector->GetStatus() != -1) {
    selector->SlaveTerminate();   //<==call user termination function
    selector->Terminate();        //<==call user termination function
  }
  delete selector;
}
