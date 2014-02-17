#ifndef Fireworks_Core_FWTypeToRepresentations_h
#define Fireworks_Core_FWTypeToRepresentations_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTypeToRepresentations
//
/**\class FWTypeToRepresentations FWTypeToRepresentations.h Fireworks/Core/interface/FWTypeToRepresentations.h

   Description: For a given C++ type, gives back a list of what 'Representations' are available

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 11:25:04 EST 2008
// $Id: FWTypeToRepresentations.h,v 1.2 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWRepresentationInfo.h"

// forward declarations
class FWRepresentationCheckerBase;

class FWTypeToRepresentations {

public:
   FWTypeToRepresentations();
   virtual ~FWTypeToRepresentations();

   // ---------- const member functions ---------------------
   const std::vector<FWRepresentationInfo>& representationsForType(const std::string& iTypeName) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void add( boost::shared_ptr<FWRepresentationCheckerBase> iChecker);
   void insert( const FWTypeToRepresentations& );

private:
   //FWTypeToRepresentations(const FWTypeToRepresentations&); // stop default

   //const FWTypeToRepresentations& operator=(const FWTypeToRepresentations&); // stop default

   // ---------- member data --------------------------------
   mutable std::map<std::string, std::vector<FWRepresentationInfo> > m_typeToReps;
   std::vector<boost::shared_ptr<FWRepresentationCheckerBase> > m_checkers;
};


#endif
