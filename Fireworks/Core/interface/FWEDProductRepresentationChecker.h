#ifndef Fireworks_Core_FWEDProductRepresentationChecker_h
#define Fireworks_Core_FWEDProductRepresentationChecker_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEDProductRepresentationChecker
//
/**\class FWEDProductRepresentationChecker FWEDProductRepresentationChecker.h Fireworks/Core/interface/FWEDProductRepresentationChecker.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 15:20:05 EST 2008
//

// system include files
#include <string>

// user include files
#include "Fireworks/Core/interface/FWRepresentationCheckerBase.h"

// forward declarations

class FWEDProductRepresentationChecker : public FWRepresentationCheckerBase {
public:
  FWEDProductRepresentationChecker(const std::string& iTypeidName,
                                   const std::string& iPurpose,
                                   unsigned int iBitPackedViews,
                                   bool iRepresentsSubPart,
                                   bool iRequiresFF = false);

  // ---------- const member functions ---------------------
  FWRepresentationInfo infoFor(const std::string& iTypeName) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  FWEDProductRepresentationChecker(const FWEDProductRepresentationChecker&) = delete;  // stop default

  const FWEDProductRepresentationChecker& operator=(const FWEDProductRepresentationChecker&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const std::string m_typeidName;
};

#endif
