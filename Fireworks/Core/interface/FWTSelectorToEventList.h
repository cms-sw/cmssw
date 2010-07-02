// $Id: Stone_SKEL.h 2089 2008-11-23 20:31:03Z matevz $

#ifndef Fireworks_Core_FWTSelectorToEventList_h
#define Fireworks_Core_FWTSelectorToEventList_h

#include "TSelectorDraw.h"

class TTree;
class TEventList;
class TTreePlayer;

class FWTSelectorToEventList : public TSelectorDraw
{
private:
   TEventList  *fEvList;
   TTreePlayer *fPlayer;
   TList        fInput;

public:
   FWTSelectorToEventList(TTree* tree, TEventList* evl, const char* sel);
   virtual ~FWTSelectorToEventList();

   virtual Int_t    Version() const { return 1; }
   virtual Bool_t   Process(Long64_t entry);

   virtual Long64_t ProcessTree(TTree* t,
                                Long64_t nentries   = 1000000000,
                                Long64_t firstentry = 0);

   TEventList* GetEventList() const { return fEvList; }

   ClassDef(FWTSelectorToEventList, 0);
};

#endif
