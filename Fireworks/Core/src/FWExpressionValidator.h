#ifndef Fireworks_Core_FWExpressionValidator_h
#define Fireworks_Core_FWExpressionValidator_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWExpressionValidator
//
/**\class FWExpressionValidator FWExpressionValidator.h Fireworks/Core/interface/FWExpressionValidator.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 22 20:42:49 EDT 2008
//

// system include files
#include <vector>
#include <memory>
#include "FWCore/Reflection/interface/TypeWithDict.h"

// user include files
#include "Fireworks/Core/interface/FWValidatorBase.h"

// forward declarations
namespace fireworks {
  class OptionNode;
}

class FWExpressionValidator : public FWValidatorBase {
public:
  FWExpressionValidator();
  ~FWExpressionValidator() override;

  // ---------- const member functions ---------------------
  void fillOptions(const char* iBegin,
                   const char* iEnd,
                   std::vector<std::pair<std::shared_ptr<std::string>, std::string> >& oOptions) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void setType(const edm::TypeWithDict&);

  FWExpressionValidator(const FWExpressionValidator&) = delete;  // stop default

  const FWExpressionValidator& operator=(const FWExpressionValidator&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  edm::TypeWithDict m_type;
  std::vector<std::shared_ptr<fireworks::OptionNode> > m_options;
  std::vector<std::shared_ptr<fireworks::OptionNode> > m_builtins;
};

#endif
