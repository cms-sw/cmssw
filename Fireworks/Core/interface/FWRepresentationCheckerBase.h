#ifndef Fireworks_Core_FWRepresentationCheckerBase_h
#define Fireworks_Core_FWRepresentationCheckerBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRepresentationCheckerBase
//
/**\class FWRepresentationCheckerBase FWRepresentationCheckerBase.h Fireworks/Core/interface/FWRepresentationCheckerBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 13:12:35 EST 2008
//

// system include files
#include <string>
// user include files

// forward declarations
class FWRepresentationInfo;

class FWRepresentationCheckerBase {

public:
   FWRepresentationCheckerBase(const std::string& iPurpose, unsigned int iBitPackedViews, bool iRepresentsSubPart, bool iRequiresFF = false);
   virtual ~FWRepresentationCheckerBase();

   // ---------- const member functions ---------------------
   const std::string& purpose() const;
   //virtual bool canWorkWith(const std::string& iTypeName) const = 0;
   virtual FWRepresentationInfo infoFor(const std::string& iTypeName) const = 0;

   unsigned int bitPackedViews() const;
   bool representsSubPart() const;

   bool requiresFF() const { return m_requiresFF; }
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWRepresentationCheckerBase(const FWRepresentationCheckerBase&) = delete; // stop default

   const FWRepresentationCheckerBase& operator=(const FWRepresentationCheckerBase&) = delete; // stop default

   // ---------- member data --------------------------------
   const std::string m_purpose;
   const unsigned int m_bitPackedViews;
   const bool  m_representsSubPart;
   const bool  m_requiresFF;

};


#endif
