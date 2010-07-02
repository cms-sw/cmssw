// $Id: Stone_SKEL.cxx 2089 2008-11-23 20:31:03Z matevz $

#include "Fireworks/Core/interface/FWTSelectorToEventList.h"

#include "TEventList.h"
#include "TTreePlayer.h"
#include "TTreeFormula.h"


// FWTSelectorToEventList

//______________________________________________________________________________
//
// TTree selector for direct extraction into an TEventList -- no need
// to create it in TDirectory and name it.
//
// Own TTreePlayer is created and used directly in ProcessTree().
// This avoids usage of various global variables / state that would be
// used in the process of calling TTree::Draw(">>event_list").
//
// This can be called in a dedicated thread, but also do TFile::Open()
// there and get the tree out so that the TTree object is unique.

//______________________________________________________________________________
FWTSelectorToEventList::FWTSelectorToEventList(TTree*      tree,
                                               TEventList* evl,
                                               const char* sel) :
   TSelectorDraw(),
   fEvList(evl),
   fPlayer(new TTreePlayer)
{
   fInput.Add(new TNamed("varexp", ""));
   fInput.Add(new TNamed("selection", sel));
   SetInputList(&fInput);

   fPlayer->SetTree(tree);
}

//______________________________________________________________________________
FWTSelectorToEventList::~FWTSelectorToEventList()
{
   delete fPlayer;
}

//==============================================================================

//______________________________________________________________________________
Bool_t
FWTSelectorToEventList::Process(Long64_t entry)
{
   // Process entry.

   if (GetSelect()->EvalInstance(0) != 0)
      fEvList->Enter(entry);
   return kTRUE;
}

//______________________________________________________________________________
Long64_t
FWTSelectorToEventList::ProcessTree(TTree* t,
                                    Long64_t nentries,
                                    Long64_t firstentry)
{
   return fPlayer->Process(this, "", nentries, firstentry);
}
