#ifndef Fireworks_Core_FWDoubleParameter_h
#define Fireworks_Core_FWDoubleParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDoubleParameter
// 
/**\class FWDoubleParameter FWDoubleParameter.h Fireworks/Core/interface/FWDoubleParameter.h

 Description: Provides access to a simple double parameter

 Usage:
    If min and max values are both identical than no restriction is placed on the allowed value

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:34 EST 2008
// $Id$
//

// system include files
#include <sigc++/signal.h>

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"

// forward declarations

class FWDoubleParameter : public FWParameterBase
{

   public:
      FWDoubleParameter(FWParameterizable* iParent,
                        const std::string& iName,
                        double iDefault=0,
                        double iMin=-1.,
                        double iMax=-1.);
      //virtual ~FWDoubleParameter();
      template <class T>
      FWDoubleParameter(FWParameterizable* iParent,
                        const std::string& iName,
                        T iCallback,
                        double iDefault=0.,
                        double iMin=-1.,
                        double iMax=-1.):
      FWParameterBase(iParent,iName),
      m_value(iDefault),
      m_min(iMin),
      m_max(iMax)
      {
         changed_.connect(iCallback);
      }
      // ---------- const member functions ---------------------
      double value() const {
         return m_value;
      }
      double min() const {
         return m_min;
      }
      double max() const {
         return m_max;
      }
   
      virtual void addTo(FWConfiguration& ) const ;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void setFrom(const FWConfiguration&);
      void set(double);
   
      sigc::signal<void,double> changed_;
   
   private:
      FWDoubleParameter(const FWDoubleParameter&); // stop default

      const FWDoubleParameter& operator=(const FWDoubleParameter&); // stop default

      // ---------- member data --------------------------------
      double m_value;
      double m_min;
      double m_max;
};


#endif
