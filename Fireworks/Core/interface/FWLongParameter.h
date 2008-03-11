#ifndef Fireworks_Core_FWLongParameter_h
#define Fireworks_Core_FWLongParameter_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWLongParameter
// 
/**\class FWLongParameter FWLongParameter.h Fireworks/Core/interface/FWLongParameter.h

 Description: Provides access to a simple double parameter

 Usage:
    If min and max values are both identical than no restriction is placed on the allowed value

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:34 EST 2008
// $Id: FWLongParameter.h,v 1.1 2008/03/11 02:43:57 chrjones Exp $
//

// system include files
#include <sigc++/signal.h>

// user include files
#include "Fireworks/Core/interface/FWParameterBase.h"

// forward declarations

class FWLongParameter : public FWParameterBase
{

   public:
      FWLongParameter(FWParameterizable* iParent,
                        const std::string& iName,
                        long iDefault=0,
                        long iMin=-1,
                        long iMax=-1);
      //virtual ~FWLongParameter();
      template <class T>
      FWLongParameter(FWParameterizable* iParent,
                        const std::string& iName,
                        T iCallback,
                        long iDefault=0,
                        long iMin=-1,
                        long iMax=-1):
      FWParameterBase(iParent,iName),
      m_value(iDefault),
      m_min(iMin),
      m_max(iMax)
      {
         changed_.connect(iCallback);
      }
      // ---------- const member functions ---------------------
      long value() const {
         return m_value;
      }
      long min() const {
         return m_min;
      }
      long max() const {
         return m_max;
      }
   
      virtual void addTo(FWConfiguration& ) const ;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void setFrom(const FWConfiguration&);
      void set(long);
   
      sigc::signal<void,long> changed_;
   
   private:
      FWLongParameter(const FWLongParameter&); // stop default

      const FWLongParameter& operator=(const FWLongParameter&); // stop default

      // ---------- member data --------------------------------
      long m_value;
      long m_min;
      long m_max;
};


#endif
