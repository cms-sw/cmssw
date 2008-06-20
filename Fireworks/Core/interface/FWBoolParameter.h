#ifndef Fireworks_Core_FWBoolParameter_h
#define Fireworks_Core_FWBoolParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoolParameter
// 
/**\class FWBoolParameter FWBoolParameter.h Fireworks/Core/interface/FWBoolParameter.h

 Description: Provides access to a simple bool parameter

 Usage:
    If min and max values are both identical than no restriction is placed on the allowed value

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:34 EST 2008
// $Id: FWBoolParameter.h,v 1.1 2008/03/11 02:43:57 chrjones Exp $
//

// system include files
#include <sigc++/signal.h>

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"

// forward declarations

class FWBoolParameter : public FWParameterBase
{

   public:
      FWBoolParameter(FWParameterizable* iParent,
		      const std::string& iName,
		      bool iDefault=false );
      //virtual ~FWBoolParameter();
      template <class T>
      FWBoolParameter(FWParameterizable* iParent,
		      const std::string& iName,
		      T iCallback,
		      bool iDefault=false ):
      FWParameterBase(iParent,iName),
      m_value(iDefault)
      {
         changed_.connect(iCallback);
      }
      // ---------- const member functions ---------------------
      bool value() const {
         return m_value;
      }
      virtual void addTo(FWConfiguration& ) const ;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void setFrom(const FWConfiguration&);
      void set(bool);
   
      sigc::signal<void,bool> changed_;
   
   private:
      FWBoolParameter(const FWBoolParameter&); // stop default

      const FWBoolParameter& operator=(const FWBoolParameter&); // stop default

      // ---------- member data --------------------------------
      bool m_value;
};


#endif
