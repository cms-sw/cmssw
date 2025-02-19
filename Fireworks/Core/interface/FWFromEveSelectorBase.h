#ifndef Fireworks_Core_FWFromEveSelectorBase_h
#define Fireworks_Core_FWFromEveSelectorBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWFromEveSelectorBase
// 
/**\class FWFromEveSelectorBase FWFromEveSelectorBase.h Fireworks/Core/interface/FWFromEveSelectorBase.h

 Description: Abstract interface for objects carried as 'UserData' in a TEveElement and then used to select the appropriate Fireworks model(s)

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Oct 23 10:50:21 CDT 2009
// $Id: FWFromEveSelectorBase.h,v 1.2 2009/10/28 14:41:11 chrjones Exp $
//

// system include files

// user include files

// forward declarations

class FWFromEveSelectorBase
{

   public:
      FWFromEveSelectorBase();
      virtual ~FWFromEveSelectorBase();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void doSelect() = 0;
      virtual void doUnselect() = 0;

   private:
      //FWFromEveSelectorBase(const FWFromEveSelectorBase&); // stop default

      //const FWFromEveSelectorBase& operator=(const FWFromEveSelectorBase&); // stop default

      // ---------- member data --------------------------------

};


#endif
