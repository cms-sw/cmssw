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
// $Id: FWViewContext.h,v 1.3 2010/06/18 19:51:24 amraktad Exp $
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


   bool getPlotEt() const { return m_plotEt; }
   bool getAutoScale() const { return m_autoScale; }

   void setPlotEt(bool x) { m_plotEt = x; }
   void setAutoScale(bool x) { m_autoScale = x; }

   mutable sigc::signal<void, const FWViewContext*> scaleChanged_;
   
private:
   FWViewContext(const FWViewContext&); // stop default

   const FWViewContext& operator=(const FWViewContext&); // stop default

   // ---------- member data --------------------------------
   typedef std::map<std::string, FWViewEnergyScale*> Scales_t;
   typedef Scales_t::iterator Scales_i;

   mutable Scales_t m_scales;

   // AT! tmp solution for PF scaling
   bool m_plotEt;
   bool m_autoScale;
};


#endif
