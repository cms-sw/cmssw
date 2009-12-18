#ifndef Fireworks_Core_FWHLTValidator_h
#define Fireworks_Core_FWHLTValidator_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTValidator
// $Id: FWHLTValidator.h,v 1.2 2009/11/14 17:45:32 chrjones Exp $
//

#include "Fireworks/Core/src/FWValidatorBase.h"

namespace edm {
  class TriggerNames;
}

class FWHLTValidator: public FWValidatorBase {

public:
  FWHLTValidator():m_triggerNames(0){}
  virtual ~FWHLTValidator() {}

  virtual void fillOptions(const char* iBegin, const char* iEnd,
			   std::vector<std::pair<boost::shared_ptr<std::string>, std::string> >& oOptions) const;
private:
  FWHLTValidator(const FWHLTValidator&); // stop default
  const FWHLTValidator& operator=(const FWHLTValidator&); // stop default
  
  // ---------- member data --------------------------------
  mutable std::vector<std::string> m_triggerNames;
};


#endif
