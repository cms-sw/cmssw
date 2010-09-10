#ifndef Fireworks_Core_CmsShowCommon_h
#define Fireworks_Core_CmsShowCommon_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowCommon
// 
/**\class CmsShowCommon CmsShowCommon.h Fireworks/Core/interface/CmsShowCommon.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Sep 10 14:51:07 CEST 2010
// $Id$
//

#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"

class CmsShowCommonPopup;
class FWColorManager;

class CmsShowCommon : public FWConfigurableParameterizable
{
   friend class CmsShowCommonPopup;

public:
   CmsShowCommon(FWColorManager*);
   virtual ~CmsShowCommon();

   // ---------- const member functions ---------------------
   virtual void addTo(FWConfiguration&) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);

   int gamma() { return m_gamma.value(); }
   void setGamma(int);
   void switchBackground();

protected:

private:
   CmsShowCommon(const CmsShowCommon&); // stop default

   const CmsShowCommon& operator=(const CmsShowCommon&); // stop default

   // ---------- member data --------------------------------
   FWColorManager*     m_colorManager;

   // colors
   FWBoolParameter     m_blackBackground;
   FWLongParameter     m_gamma;
   
   // scales
};


#endif
