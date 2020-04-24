#ifndef Fireworks_Core_FWHLTValidator_h
#define Fireworks_Core_FWHLTValidator_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTValidator
//

#include "Fireworks/Core/src/FWValidatorBase.h"

namespace edm {
  class TriggerNames;
}

class FWHLTValidator: public FWValidatorBase {

public:
   FWHLTValidator(std::string& x):m_process(x){}
   ~FWHLTValidator() override {}

   void setProcess(const char* x) { m_process = x; m_triggerNames.clear(); }
   void fillOptions(const char* iBegin, const char* iEnd,
                            std::vector<std::pair<std::shared_ptr<std::string>, std::string> >& oOptions) const override;
private:
   FWHLTValidator(const FWHLTValidator&) = delete; // stop default
   const FWHLTValidator& operator=(const FWHLTValidator&) = delete; // stop default
  
   // ---------- member data --------------------------------
   std::string m_process;
   mutable std::vector<std::string> m_triggerNames;
};


#endif
