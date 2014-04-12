#include "TChain.h"
#include "TTreeFormula.h"
#include "TTreeCache.h"
#include "vector"
#include "iostream"

void GetEntriesMulti(TChain *events, const char* selection, Int_t ncandscut, std::vector<Long64_t>& pass) {
  Long64_t rows = 0;
  Long64_t entry = 0;
  Int_t cands = 0;
  Int_t ndata = 0;
  Int_t j = 0;
  Int_t currentTree = 0;
  vector<Long64_t> entries;
  Long64_t nentries = events->GetEntries();
  TTreeFormula *selFormula= new TTreeFormula("Selection", selection, events);
  for (entry = 0; entry < nentries; entry++) {
    events->LoadTree(entry);
    selFormula->UpdateFormulaLeaves();
    selFormula->ResetDimensions();
    if (currentTree < events->GetTreeNumber() && entry != 0) {
      cout<<rows<<endl;
      pass.push_back(rows);
      rows = 0;
      currentTree = events->GetTreeNumber();
    }
    ndata = selFormula->GetNdata();
    cands = 0;
    if (selFormula) {
      // Always call EvalInstance(0) to insure the loading
      // of the branches.
      if (ndata == 0) selFormula->EvalInstance(0);
      else {
	j = 0;
	for (j = 0; j < ndata; j++) {
	  if (selFormula->EvalInstance(j)) {
	    ++cands;
	  }
	}
      }
    }
    if (cands >= ncandscut) {
      ++rows;
    }
  }
  cout<<rows<<endl<<endl;
  pass.push_back(rows);
  delete selFormula;
}
