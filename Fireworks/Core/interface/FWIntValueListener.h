#ifndef Fireworks_Core_FWIntValueListener_h
#define Fireworks_Core_FWIntValueListener_h

#include "Fireworks/Core/interface/FWIntValueListenerBase.h"
#include <sigc++/sigc++.h>

class FWIntValueListener : public FWIntValueListenerBase {
public:
   FWIntValueListener() : FWIntValueListenerBase() {
   }
   ~FWIntValueListener() override {
   }

   // ---------- member, functions -------------------------
   void setValueImp(Int_t entry) override;
   sigc::signal<void,Int_t> valueChanged_;

private:
   FWIntValueListener(const FWIntValueListener&) = delete; // stop default
   const FWIntValueListener& operator=(const FWIntValueListener&) = delete; // stop default
};

#endif
