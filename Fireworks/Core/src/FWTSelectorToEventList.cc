// $Id: FWTSelectorToEventList.cc,v 1.2 2010/07/02 13:06:00 matevz Exp $

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
// The event-list passed in constructor is not owned by the selector
// unless SetOwnEventList(kTRUE) is called -- then it is destroyed in
// the destructor.
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
   fPlayer(new TTreePlayer),
   fOwnEvList(kFALSE)
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
   if (fOwnEvList)
      delete fEvList;
}

//______________________________________________________________________________
void FWTSelectorToEventList::ClearEventList()
{
   fEvList->Clear();
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
FWTSelectorToEventList::ProcessTree(Long64_t nentries,
                                    Long64_t firstentry)
{
   return fPlayer->Process(this, "", nentries, firstentry);
}
