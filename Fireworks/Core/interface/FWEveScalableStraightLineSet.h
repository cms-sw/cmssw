#ifndef Fireworks_Core_FWEveScalableStraightLineSet_h
#define Fireworks_Core_FWEveScalableStraightLineSet_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveScalableStraightLineSet
//
/**\class FWEveScalableStraightLineSet FWEveScalableStraightLineSet.h Fireworks/Core/interface/FWEveScalableStraightLineSet.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jul  3 16:25:28 EDT 2008
// $Id: FWEveScalableStraightLineSet.h,v 1.2 2008/11/06 22:05:22 amraktad Exp $
//

// system include files

// user include files
#include "TEveScalableStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveValueScaled.h"

// forward declarations

class FWEveScalableStraightLineSet : public TEveScalableStraightLineSet, public FWEveValueScaled
{

public:
   FWEveScalableStraightLineSet(const Text_t* iName, const Text_t* iTitle="");
   //virtual ~FWEveScalableStraightLineSet();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setScale(float);

private:
   FWEveScalableStraightLineSet(const FWEveScalableStraightLineSet&);    // stop default

   const FWEveScalableStraightLineSet& operator=(const FWEveScalableStraightLineSet&);    // stop default

   // ---------- member data --------------------------------

};


#endif
