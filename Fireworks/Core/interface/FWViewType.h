#ifndef Fireworks_Core_FWViewType_h
#define Fireworks_Core_FWViewType_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewType
// 
/**\class FWViewType FWViewType.h Fireworks/Core/interface/FWViewType.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Mon Mar 22 12:08:40 CET 2010
// $Id: FWViewType.h,v 1.22 2012/02/22 03:45:57 amraktad Exp $
//
#include <string>

class  FWViewType
{
public:
   class static_initializer
   {
   public:
      static_initializer();
   };

   static static_initializer init_statics;
   
   enum EType { kRhoPhi, kRhoZ, k3D, kISpy, kLego, kLegoHF, kGlimpse,
                kTable, kTableL1, kTableHLT,
                kRhoPhiPF, kLegoPFECAL,
                kGeometryTable,kOverlapTable,
                kTypeSize };
   
   enum EBit
   {
      k3DBit         = 1 << k3D,
      kRhoPhiBit     = 1 << kRhoPhi,
      kRhoZBit       = 1 << kRhoZ,
      kRhoPhiPFBit   = 1 << kRhoPhiPF,
      kISpyBit       = 1 << kISpy,
      kLegoBit       = 1 << kLego,
      kLegoHFBit     = 1 << kLegoHF,
      kLegoPFECALBit = 1 << kLegoPFECAL,
      kGlimpseBit    = 1 << kGlimpse,
      kTableBit      = 1 << kTable,
      kTableHLTBit   = 1 << kTableHLT,
      kTableL1Bit    = 1 << kTableL1,
      kGeometryBit   = 1 << kGeometryTable,
      kOverlapBit   = 1 << kOverlapTable
   };

   // shortcuts
   static const int kAllRPZBits;
   static const int kAll3DBits;
   static const int kAllLegoBits;

   static std::string sName[kTypeSize];

   static const std::string& idToName(int);
   static bool isProjected(int);
   static bool isLego(int);

   static const std::string&  checkNameWithViewVersion(const std::string& name, unsigned int viewVersion);
   
   FWViewType(EType);
   virtual ~FWViewType();

   const std::string& name() const;
   EType id() const;


private: 
   const EType m_id;
};

#endif
