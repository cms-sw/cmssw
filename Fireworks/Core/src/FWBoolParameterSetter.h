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
// $Id: FWBoolParameterSetter.h,v 1.1 2008/03/11 02:43:55 chrjones Exp $
//

// system include files
#include <Rtypes.h>

// user include files
#include "Fireworks/Core/interface/FWParameterSetterBase.h"

// forward declarations
class FWBoolParameter;
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
      virtual TGFrame* build(TGFrame* iParent) ;
      void doUpdate();
   
   private:
      FWBoolParameterSetter(const FWBoolParameterSetter&); // stop default

      const FWBoolParameterSetter& operator=(const FWBoolParameterSetter&); // stop default

      // ---------- member data --------------------------------
      FWBoolParameter* m_param;
      TGCheckButton* m_widget;
};


#endif
