#ifndef Fireworks_Core_FWEnumParameterSetter_h
#define Fireworks_Core_FWEnumParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEnumParameterSetter
// 
/**\class FWEnumParameterSetter FWEnumParameterSetter.h Fireworks/Core/interface/FWEnumParameterSetter.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  matevz
//         Created:  Fri Apr 30 15:17:29 CEST 2010
// $Id: FWEnumParameterSetter.h,v 1.4 2012/05/09 04:51:04 amraktad Exp $
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"

// forward declarations
class TGComboBox;

class FWEnumParameterSetter : public FWParameterSetterBase
{

public:
   FWEnumParameterSetter();
   virtual ~FWEnumParameterSetter();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual TGFrame* build(TGFrame* iParent, bool labelBack = true);

   virtual void setEnabled(bool);

   void doUpdate(Int_t id);

  TGComboBox* getWidget() { return m_widget; }

private:
   FWEnumParameterSetter(const FWEnumParameterSetter&);                  // stop default
   const FWEnumParameterSetter& operator=(const FWEnumParameterSetter&); // stop default

   virtual void attach(FWParameterBase*);

   // ---------- member data --------------------------------

   FWEnumParameter *m_param;
   TGComboBox      *m_widget;
};

#endif
