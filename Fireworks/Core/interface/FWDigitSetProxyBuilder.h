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
// $Id: FWDigitSetProxyBuilder.h,v 1.2 2010/11/09 18:37:30 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

// forward declarations
class TEveDigitSet;
class TEveBoxSet;
class FWDisplayProperties;

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

   // AMT: temproary structure since TEveBoxSet::BFreeBox_t is protected
   // this workaround should be  removed in next root patch
   struct BFreeBox_t
   {
      Int_t  fValue;  
      void  *fUserData;
      Float_t fVertices[8][3]; 
      BFreeBox_t(Int_t v=0) : fValue(v), fUserData(0) {}
   };


   TEveBoxSet* addBoxSetToProduct(TEveElementList* product);
   void addBox(TEveBoxSet* set, const float* pnts, const FWDisplayProperties& dp);
   TEveBoxSet* getBoxSet() const { return m_boxSet; } 

private:
   FWDigitSetProxyBuilder(const FWDigitSetProxyBuilder&); // stop default

   const FWDigitSetProxyBuilder& operator=(const FWDigitSetProxyBuilder&); // stop default

   // ---------- member data --------------------------------

   virtual void modelChanges(const FWModelIds&, Product*);
 
   static TString getTooltip(TEveDigitSet* set, int idx);

   TEveDigitSet* digitSet(TEveElement* product);

   TEveBoxSet* m_boxSet;	
};


#endif
