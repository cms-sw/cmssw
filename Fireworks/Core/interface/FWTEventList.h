#ifndef Fireworks_Core_FWTEventList_h
#define Fireworks_Core_FWTEventList_h

#include "TEventList.h"

class FWTEventList : public TEventList
{
public:
   FWTEventList() : TEventList() {}
   FWTEventList(const char* name, const char* title = "", Int_t initsize = 0, Int_t delta = 0) : 
      TEventList(name, title, initsize, delta) {}

   virtual ~FWTEventList() {}

   virtual void	Enter(Long64_t entry);
   virtual void	Add(const TEventList* list);

private:
   FWTEventList(const FWTEventList&); // stop default
   const FWTEventList& operator=(const FWTEventList&); // stop default

   ClassDef(FWTEventList, 0);
};

#endif
