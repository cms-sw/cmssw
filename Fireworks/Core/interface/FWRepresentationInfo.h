#ifndef Fireworks_Core_FWRepresentationInfo_h
#define Fireworks_Core_FWRepresentationInfo_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRepresentationInfo
//
/**\class FWRepresentationInfo FWRepresentationInfo.h Fireworks/Core/interface/FWRepresentationInfo.h

   Description: Collection of information about how a particular representation matches a data type

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 11 13:12:28 EST 2008
//

// system include files
#include <string>

// user include files

// forward declarations

class FWRepresentationInfo
{
public:
   FWRepresentationInfo(const std::string& iPurpose, unsigned int iProximity, unsigned int iBitPackedViews, bool iRepresentsSubPart, bool iRequiresFF) :
      m_purpose(iPurpose),
      m_proximity(iProximity),
      m_bitPackedViews(iBitPackedViews),
      m_representsSubPart(iRepresentsSubPart),
      m_requiresFF(iRequiresFF)
   {}

   FWRepresentationInfo() :
      m_purpose(),
      m_proximity(0xFFFFFFFF),
      m_bitPackedViews(0),
      m_representsSubPart(false),
      m_requiresFF(false)
   {}

   // ---------- const member functions ---------------------

   const std::string& purpose() const { return m_purpose; }

   ///measures how 'close' this representation is to the type in question, the large the number the farther away
   unsigned int proximity() const { return m_proximity; }

   bool isValid() const { return !m_purpose.empty(); }
   
   bool representsSubPart() const { return m_representsSubPart; }

   bool requiresFF() const { return m_requiresFF; }
   
   unsigned int bitPackedViews() const { return m_bitPackedViews; }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   //FWRepresentationInfo(const FWRepresentationInfo&); // stop default

   //const FWRepresentationInfo& operator=(const FWRepresentationInfo&); // stop default

   // ---------- member data --------------------------------
   std::string m_purpose;
   unsigned int m_proximity;
   unsigned int m_bitPackedViews;

   bool m_representsSubPart;
   bool m_requiresFF;
};


#endif
