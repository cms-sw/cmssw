
#ifndef Fireworks_Core_FWTSelectorToEventList_h
#define Fireworks_Core_FWTSelectorToEventList_h

#include "TSelectorEntries.h"

class TTree;
class TEventList;
class TTreePlayer;

class FWTSelectorToEventList : public TSelectorEntries {
private:
  TEventList* fEvList;
  TTreePlayer* fPlayer;
  Bool_t fOwnEvList;

public:
  FWTSelectorToEventList(TTree* tree, TEventList* evl, const char* sel);
  ~FWTSelectorToEventList() override;

  Bool_t Process(Long64_t entry) override;

  virtual Long64_t ProcessTree(Long64_t nentries = 1000000000, Long64_t firstentry = 0);

  TEventList* GetEventList() const { return fEvList; }
  void ClearEventList();

  Bool_t GetOwnEventList() const { return fOwnEvList; }
  void SetOwnEventList(Bool_t o) { fOwnEvList = o; }

  ClassDefOverride(FWTSelectorToEventList, 0);
};

#endif
