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
// $Id: FWExpressionValidator.h,v 1.5 2012/08/03 18:20:28 wmtan Exp $
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files
#include "Fireworks/Core/src/FWValidatorBase.h"

// forward declarations
namespace fireworks {
   class OptionNode;
}

class FWExpressionValidator : public FWValidatorBase {

public:
   FWExpressionValidator();
   virtual ~FWExpressionValidator();

   // ---------- const member functions ---------------------
   virtual void fillOptions(const char* iBegin, const char* iEnd,
                            std::vector<std::pair<boost::shared_ptr<std::string>, std::string> >& oOptions) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setType(const edm::TypeWithDict&);

private:
   FWExpressionValidator(const FWExpressionValidator&); // stop default

   const FWExpressionValidator& operator=(const FWExpressionValidator&); // stop default

   // ---------- member data --------------------------------
   edm::TypeWithDict m_type;
   std::vector<boost::shared_ptr<fireworks::OptionNode> > m_options;
   std::vector<boost::shared_ptr<fireworks::OptionNode> > m_builtins;

};


#endif
