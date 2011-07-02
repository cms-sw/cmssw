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
<<<<<<< FWViewType.h
// $Id: FWViewType.h,v 1.18 2011/07/01 00:03:58 amraktad Exp $
=======
// $Id: FWViewType.h,v 1.16 2011/01/26 11:47:06 amraktad Exp $
>>>>>>> 1.16
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
   
<<<<<<< FWViewType.h
   //   enum EType { kRhoPhi, kRhoZ, k3D, kISpy, kLego, kLegoHF, kGlimpse,	    
enum EType { k3D, kRhoPhi, kRhoZ, kISpy, kLego, kLegoHF, kGlimpse,
=======
   enum EType { k3D, kRhoPhi, kRhoZ, kISpy, kLego, kLegoHF, kGlimpse,
>>>>>>> 1.16
                kTable, kTableL1, kTableHLT,
                kRhoPhiPF, kLegoPFECAL,
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
      kTableL1Bit    = 1 << kTableL1
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
