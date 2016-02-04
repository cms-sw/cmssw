#ifndef Fireworks_Core_FWStringParameterSetter_h
#define Fireworks_Core_FWStringParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWStringParameterSetter
// $Id: FWStringParameterSetter.h,v 1.2 2010/02/12 15:36:46 eulisse Exp $
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWStringParameter.h"

// forward declarations
class TGTextEntry;

class FWStringParameterSetter : public FWParameterSetterBase
{

public:
   FWStringParameterSetter();
   virtual ~FWStringParameterSetter();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void attach(FWParameterBase*) ;
   virtual TGFrame* build(TGFrame* iParent) ;
   void doUpdate();

private:
   FWStringParameterSetter(const FWStringParameterSetter&);    // stop default

   const FWStringParameterSetter& operator=(const FWStringParameterSetter&);    // stop default

   // ---------- member data --------------------------------
   FWStringParameter* m_param;
   TGTextEntry* m_widget;
};


#endif
