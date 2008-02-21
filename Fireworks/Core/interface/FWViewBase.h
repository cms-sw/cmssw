#ifndef Fireworks_Core_FWViewBase_h
#define Fireworks_Core_FWViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewBase
// 
/**\class FWViewBase FWViewBase.h Fireworks/Core/interface/FWViewBase.h

 Description: Base class for all View instances

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 14:43:25 EST 2008
// $Id$
//

// system include files
#include <string>

// user include files

// forward declarations
class TGFrame;

class FWViewBase
{

   public:
      FWViewBase();
      virtual ~FWViewBase();

      // ---------- const member functions ---------------------
      virtual const std::string& typeName() const = 0;
   
      virtual TGFrame* frame() const = 0;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      FWViewBase(const FWViewBase&); // stop default

      const FWViewBase& operator=(const FWViewBase&); // stop default

      // ---------- member data --------------------------------

};


#endif
