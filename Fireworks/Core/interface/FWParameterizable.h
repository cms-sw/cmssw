#ifndef Fireworks_Core_FWParameterizable_h
#define Fireworks_Core_FWParameterizable_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWParameterizable
//
/**\class FWParameterizable FWParameterizable.h Fireworks/Core/interface/FWParameterizable.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sat Feb 23 13:35:23 EST 2008
// $Id: FWParameterizable.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <vector>

// user include files

// forward declarations
class FWParameterBase;

class FWParameterizable
{

public:
   FWParameterizable();
   virtual ~FWParameterizable();

   typedef std::vector<FWParameterBase* >::const_iterator const_iterator;
   // ---------- const member functions ---------------------
   const_iterator begin() const {
      return m_parameters.begin();
   }

   const_iterator end() const {
      return m_parameters.end();
   }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   //base class implementation does not take ownership of added parameters
   void add(FWParameterBase*);

private:
   FWParameterizable(const FWParameterizable&);    // stop default

   const FWParameterizable& operator=(const FWParameterizable&);    // stop default

   // ---------- member data --------------------------------
   std::vector<FWParameterBase* > m_parameters;
};


#endif
