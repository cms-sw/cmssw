#ifndef Fireworks_Core_FWStringParameter_h
#define Fireworks_Core_FWStringParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWStringParameter
// $Id: FWStringParameter.h,v 1.3 2009/01/23 21:35:40 amraktad Exp $
//

// system include files
#include <sigc++/signal.h>

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"

// forward declarations

class FWStringParameter : public FWParameterBase
{

public:
   FWStringParameter(FWParameterizable* iParent,
		     const std::string& iName,
		     const std::string& iDefault = std::string() );
   template <class T>
   FWStringParameter(FWParameterizable* iParent,
		     const std::string& iName,
		     T iCallback,
		     const std::string& iDefault = std::string() ) :
      FWParameterBase(iParent,iName),
      m_value(iDefault)
   {
      changed_.connect(iCallback);
   }
   // ---------- const member functions ---------------------
   std::string value() const {
      return m_value;
   }
   virtual void addTo(FWConfiguration& ) const ;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&);
   void set(const std::string&);

   sigc::signal<void,std::string> changed_;

private:
   FWStringParameter(const FWStringParameter&);    // stop default

   const FWStringParameter& operator=(const FWStringParameter&);    // stop default

   // ---------- member data --------------------------------
   std::string m_value;
};


#endif
