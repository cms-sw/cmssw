#ifndef Fireworks_Core_FWDigitSetProxyBuilder_h
#define Fireworks_Core_FWDigitSetProxyBuilder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDigitSetProxyBuilder
// 
/**\class FWDigitSetProxyBuilder FWDigitSetProxyBuilder.h Fireworks/Core/interface/FWDigitSetProxyBuilder.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Tue Oct 19 12:00:57 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

// forward declarations
class TEveDigitSet;
class TEveBoxSet;

class FWDigitSetProxyBuilder : public FWProxyBuilderBase
{

public:
   FWDigitSetProxyBuilder();
   virtual ~FWDigitSetProxyBuilder();

   // ---------- const member functions ---------------------

   virtual bool willHandleInteraction() const { return true; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   TEveBoxSet* addBoxSetToProduct(TEveElementList* product);
   void addBox(TEveBoxSet* set, const float* pnts);

private:
   FWDigitSetProxyBuilder(const FWDigitSetProxyBuilder&); // stop default

   const FWDigitSetProxyBuilder& operator=(const FWDigitSetProxyBuilder&); // stop default

   // ---------- member data --------------------------------

   virtual void modelChanges(const FWModelIds&, Product*);
 
   static TString getTooltip(TEveDigitSet* set, int idx);

   TEveDigitSet* digitSet(TEveElement* product);
};


#endif
