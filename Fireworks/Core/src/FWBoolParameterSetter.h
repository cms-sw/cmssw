#ifndef Fireworks_Core_FWBoolParameterSetter_h
#define Fireworks_Core_FWBoolParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoolParameterSetter
//
/**\class FWBoolParameterSetter FWBoolParameterSetter.h Fireworks/Core/interface/FWBoolParameterSetter.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 11:22:26 CDT 2008
// $Id: FWBoolParameterSetter.h,v 1.6 2011/02/11 19:56:36 amraktad Exp $
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"

// forward declarations
class TGCheckButton;

class FWBoolParameterSetter : public FWParameterSetterBase
{

public:
   FWBoolParameterSetter();
   virtual ~FWBoolParameterSetter();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void attach(FWParameterBase*) ;
   virtual TGFrame* build(TGFrame* iParent, bool labelBack = true) ;
   virtual void setEnabled(bool);
   void doUpdate();

private:
   FWBoolParameterSetter(const FWBoolParameterSetter&);    // stop default

   const FWBoolParameterSetter& operator=(const FWBoolParameterSetter&);    // stop default

   // ---------- member data --------------------------------
   FWBoolParameter* m_param;
   TGCheckButton* m_widget;
};


#endif
