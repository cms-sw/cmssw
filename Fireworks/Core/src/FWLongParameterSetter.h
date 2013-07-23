#ifndef Fireworks_Core_FWLongParameterSetter_h
#define Fireworks_Core_FWLongParameterSetter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWLongParameterSetter
//
/**\class FWLongParameterSetter FWLongParameterSetter.h Fireworks/Core/interface/FWLongParameterSetter.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 11:22:26 CDT 2008
// $Id: FWLongParameterSetter.h,v 1.6 2012/02/22 03:45:59 amraktad Exp $
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWLongParameter.h"

// forward declarations
class TGNumberEntry;

class FWLongParameterSetter : public FWParameterSetterBase
{
public:
   FWLongParameterSetter();
   virtual ~FWLongParameterSetter();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual void     attach(FWParameterBase*);
   virtual TGFrame* build(TGFrame* iParent, bool labelBack = true);

   void doUpdate(Long_t);

private:
   FWLongParameterSetter(const FWLongParameterSetter&);                  // stop default
   const FWLongParameterSetter& operator=(const FWLongParameterSetter&); // stop default

   // ---------- member data --------------------------------

   FWLongParameter* m_param;
   TGNumberEntry*   m_widget;
};

#endif
