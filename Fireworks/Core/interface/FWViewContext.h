#ifndef Fireworks_Core_FWViewContext_h
#define Fireworks_Core_FWViewContext_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContext
// 
/**\class FWViewContext FWViewContext.h Fireworks/Core/interface/FWViewContext.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Wed Apr 14 18:31:27 CEST 2010
// $Id: FWViewContext.h,v 1.2 2010/05/03 15:47:37 amraktad Exp $
//

// system include files
#include <sigc++/sigc++.h>
#include <map>
#include <string>

// user include files

// forward declarations
class FWViewEnergyScale;

class FWViewContext
{
public:
   FWViewContext();
   virtual ~FWViewContext();

   FWViewEnergyScale* getEnergyScale(const std::string&) const;
   void scaleChanged();
   void resetScale();
   void addScale( const std::string& name, FWViewEnergyScale* s) const;

   mutable sigc::signal<void, const FWViewContext*> scaleChanged_;
   
private:
   FWViewContext(const FWViewContext&); // stop default

   const FWViewContext& operator=(const FWViewContext&); // stop default

   // ---------- member data --------------------------------
   typedef std::map<std::string, FWViewEnergyScale*> Scales_t;
   typedef Scales_t::iterator Scales_i;

   mutable Scales_t m_scales;
};


#endif
