#ifndef Fireworks_Core_FWTEventList_h
#define Fireworks_Core_FWTEventList_h

// There was a bug in ROOT ... fixed on Dec 9 2009:
//   http://root.cern.ch/viewcvs/trunk/tree/tree/interface/TEventList.cxx?view=log
//
// We need to keep this intermediate class until we switch to
// root-5.26 or later.

#include "TEventList.h"

class FWTEventList : public TEventList {
public:
  FWTEventList() : TEventList() {}
  FWTEventList(const char* name, const char* title = "", Int_t initsize = 0, Int_t delta = 0)
      : TEventList(name, title, initsize, delta) {}

  ~FWTEventList() override {}

  void Enter(Long64_t entry) override;
  void Add(const TEventList* list) override;

private:
  FWTEventList(const FWTEventList&);                   // stop default
  const FWTEventList& operator=(const FWTEventList&);  // stop default

  ClassDefOverride(FWTEventList, 0);
};

#endif
