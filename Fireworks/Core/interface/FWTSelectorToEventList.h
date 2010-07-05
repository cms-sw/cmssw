// $Id: FWTSelectorToEventList.h,v 1.2 2010/07/02 13:05:59 matevz Exp $

#ifndef Fireworks_Core_FWTSelectorToEventList_h
#define Fireworks_Core_FWTSelectorToEventList_h

#include "TSelectorEntries.h"

class TTree;
class TEventList;
class TTreePlayer;

class FWTSelectorToEventList : public TSelectorEntries
{
private:
   TEventList  *fEvList;
   TTreePlayer *fPlayer;
   TList        fInput;

public:
   FWTSelectorToEventList(TTree* tree, TEventList* evl, const char* sel);
   virtual ~FWTSelectorToEventList();

   virtual Bool_t   Process(Long64_t entry);

   virtual Long64_t ProcessTree(Long64_t nentries   = 1000000000,
                                Long64_t firstentry = 0);

   TEventList* GetEventList()  const { return fEvList; }
   void        ClearEventList();

   Bool_t GetOwnEventList() const   { return fOwnEvList; }
   void   SetOwnEventList(Bool_t o) { fOwnEvList = o; }

   ClassDef(FWTSelectorToEventList, 0);
};

#endif
