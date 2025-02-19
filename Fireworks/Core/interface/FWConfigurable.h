#ifndef Fireworks_Core_FWConfigurable_h
#define Fireworks_Core_FWConfigurable_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWConfigurable
//
/**\class FWConfigurable FWConfigurable.h Fireworks/Core/interface/FWConfigurable.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sun Feb 24 14:35:47 EST 2008
// $Id: FWConfigurable.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files

// user include files

// forward declarations
class FWConfiguration;

class FWConfigurable
{

public:
   FWConfigurable();
   virtual ~FWConfigurable();

   // ---------- const member functions ---------------------
   virtual void addTo(FWConfiguration&) const = 0;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void setFrom(const FWConfiguration&) = 0;

private:
   FWConfigurable(const FWConfigurable&);    // stop default

   const FWConfigurable& operator=(const FWConfigurable&);    // stop default

   // ---------- member data --------------------------------

};


#endif
