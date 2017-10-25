#ifndef Fireworks_Core_FWModelExpressionSelector_h
#define Fireworks_Core_FWModelExpressionSelector_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelExpressionSelector
//
/**\class FWModelExpressionSelector FWModelExpressionSelector.h Fireworks/Core/interface/FWModelExpressionSelector.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Jan 23 10:37:13 EST 2008
//

// system include files
#include <string>

// user include files
#include <Rtypes.h>

// forward declarations
class FWEventItem;

class FWModelExpressionSelector
{

public:
   FWModelExpressionSelector() {
   }
   //virtual ~FWModelExpressionSelector();

   // ---------- const member functions ---------------------
   /** Throws an FWExpressionException if there is a problem */
   void select(FWEventItem* iItem, const std::string& iExpression, Color_t  iColor= -1) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWModelExpressionSelector(const FWModelExpressionSelector&) = delete;    // stop default

   const FWModelExpressionSelector& operator=(const FWModelExpressionSelector&) = delete;    // stop default

   // ---------- member data --------------------------------

};


#endif
