TTree *CopyTreeMulti(const char *selection, TTree *input, Long64_t ncandscut, Long64_t nentries = 1000000, Long64_t firstentry = 0)
{
  // based on TTree::CopyTree
  //
  // copy a Tree with selection
  // make a clone of this Tree header.
  // then copy the selected entries
  // multiple candidate cuts possible
  //
  // selection is a standard selection expression (see TTreePlayer::Draw)
  // option is reserved for possible future use
  // nentries is the number of entries to process (default is all)
  // first is the first entry to process (default is 0)
  //
  // IMPORTANT: The copied tree stays connected with this tree until this tree
  //            is deleted.  In particular, any changes in branch addresses
  //            in this tree are forwarded to the clone trees.  Any changes
  //            made to the branch addresses of the copied trees are over-ridden
  //            anytime this tree changes its branch addresses.
  //            Once this tree is deleted, all the addresses of the copied tree
  //            are reset to their default values.
  //
  // The following example illustrates how to copy some events from the Tree
  // generated in $ROOTSYS/test/Event
  //
  //   gSystem->Load("libEvent");
  //   TFile f("Event.root");
  //   TTree *T = (TTree*)f.Get("T");
  //   Event *event = new Event();
  //   T->SetBranchAddress("event",&event);
  //   TFile f2("Event2.root","recreate");
  //   TTree *T2 = T->CopyTree("fNtrack<595");
  //   T2->Write();


  // we make a copy of the tree header
  TTree *tree = input->CloneTree(0);
  if (tree == 0) return 0;

  // The clone should not delete any shared i/o buffers.
  TObjArray* branches = tree->GetListOfBranches();
  Int_t nb = branches->GetEntriesFast();
  for (Int_t i = 0; i < nb; ++i) {
    TBranch* br = (TBranch*) branches->UncheckedAt(i);
    if (br->InheritsFrom("TBranchElement")) {
      ((TBranchElement*) br)->ResetDeleteObject();
    }
  }

  Long64_t entry,entryNumber;
  nentries = input->GetEntries();

  // Compile selection expression if there is one
  TTreeFormula *select = 0; // no need to interfer with fSelect since we
  // handle the loop explicitly below and can call
  // UpdateFormulaLeaves ourselves.
  if (strlen(selection)) {
    select = new TTreeFormula("Selection",selection,input);
    if (!select || !select->GetNdim()) { delete select; }
  }

  //loop on the specified entries
  Int_t tnumber = -1;
  for (entry=firstentry;entry<firstentry+nentries;entry++) {
    Long64_t ncand = 0;
    entryNumber = input->GetEntryNumber(entry);
    if (entryNumber < 0) break;
    Long64_t localEntry = input->LoadTree(entryNumber);
    if (localEntry < 0) break;
    if (tnumber != input->GetTreeNumber()) {
      tnumber = input->GetTreeNumber();
      if (select) select->UpdateFormulaLeaves();
    }
    if (select) {
      Int_t ndata = select->GetNdata();
      for(Int_t current = 0; current<ndata; current++) {
	if (select->EvalInstance(current) != 0) ncand++;
      }
      if (ncand < ncandscut) continue;
    }
    input->GetEntry(entryNumber);
    tree->Fill();
  }
  return tree;
}
