#ifndef Fireworks_Core_FWIntValueListener_h
#define Fireworks_Core_FWIntValueListener_h

#include "Fireworks/Core/interface/FWIntValueListenerBase.h"
#include <sigc++/sigc++.h>

class FWIntValueListener : public FWIntValueListenerBase {
public:
   FWIntValueListener() : FWIntValueListenerBase() {
   }
   virtual ~FWIntValueListener() {
   }

   // ---------- member, functions -------------------------
   virtual void setValueImp(Int_t entry);
   sigc::signal<void,Int_t> valueChanged_;

private:
   FWIntValueListener(const FWIntValueListener&); // stop default
   const FWIntValueListener& operator=(const FWIntValueListener&); // stop default
};

#endif
