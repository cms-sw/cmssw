#ifndef Fireworks_Core_FWEveValueScaled_h
#define Fireworks_Core_FWEveValueScaled_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveValueScaled
//
/**\class FWEveValueScaled FWEveValueScaled.h Fireworks/Core/interface/FWEveValueScaled.h

   Description: A 'mix-in' to be used with objects inheriting from TEveElement who need to
   be dynamically scaled

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Jul  2 15:48:10 EDT 2008
// $Id: FWEveValueScaled.h,v 1.2 2008/11/06 22:05:22 amraktad Exp $
//

// system include files

// user include files

// forward declarations

class FWEveValueScaled
{

public:
   FWEveValueScaled();
   virtual ~FWEveValueScaled();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setScale(float) = 0;


private:
   //FWEveValueScaled(const FWEveValueScaled&); // stop default

   //const FWEveValueScaled& operator=(const FWEveValueScaled&); // stop default

   // ---------- member data --------------------------------

};


#endif
