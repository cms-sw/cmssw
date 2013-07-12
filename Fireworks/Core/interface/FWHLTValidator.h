#ifndef Fireworks_Core_FWHLTValidator_h
#define Fireworks_Core_FWHLTValidator_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTValidator
// $Id: FWHLTValidator.h,v 1.3 2009/12/13 12:28:05 amraktad Exp $
//

#include "Fireworks/Core/src/FWValidatorBase.h"

namespace edm {
  class TriggerNames;
}

class FWHLTValidator: public FWValidatorBase {

public:
   FWHLTValidator(std::string& x):m_process(x){}
   virtual ~FWHLTValidator() {}

   void setProcess(const char* x) { m_process = x; m_triggerNames.clear(); }
   virtual void fillOptions(const char* iBegin, const char* iEnd,
                            std::vector<std::pair<boost::shared_ptr<std::string>, std::string> >& oOptions) const;
private:
   FWHLTValidator(const FWHLTValidator&); // stop default
   const FWHLTValidator& operator=(const FWHLTValidator&); // stop default
  
   // ---------- member data --------------------------------
   std::string m_process;
   mutable std::vector<std::string> m_triggerNames;
};


#endif
