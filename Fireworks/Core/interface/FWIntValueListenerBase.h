#ifndef Fireworks_Core_FWIntValueListenerBase_h
#define Fireworks_Core_FWIntValueListenerBase_h

#include "Rtypes.h"

class FWIntValueListenerBase {
public:
   FWIntValueListenerBase() {
   }
   virtual ~FWIntValueListenerBase() {
   }

   // ---------- member functions ---------------------------
   void setValue(Int_t entry);
   virtual void setValueImp(Int_t entry) = 0;

   ClassDef(FWIntValueListenerBase, 0);

private:
   FWIntValueListenerBase(const FWIntValueListenerBase&); // stop default
   const FWIntValueListenerBase& operator=(const FWIntValueListenerBase&); // stop default
};

#endif
