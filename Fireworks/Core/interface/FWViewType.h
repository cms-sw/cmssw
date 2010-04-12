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
// $Id: FWViewType.h,v 1.3 2010/04/12 10:09:15 amraktad Exp $
//
#include <string>

class  FWViewType
{
public:
   enum EType { k3DE, kRhoPhi, kRhoZ, kISpy, kLego, kGlimpse, kSize };

   enum EBit
   {
      k3DEBit      = 1 << k3DE,
      kRhoPhiBit   = 1 << kRhoPhi,
      kRhoZBit     = 1 << kRhoZ,
      kISpyBit     = 1 << kISpy,
      kLegoBit     = 1 << kLego,
      kGlimpseBit  = 1 << kGlimpse
   };

   // shorcuts
   const static int kRPZBit = kRhoPhiBit | kRhoZBit;
   const static int k3DBit  = kISpyBit | k3DEBit;

   static  std::string  k3DEName;
   static  std::string  kRhoPhiName;
   static  std::string  kRhoZName;
   static  std::string  kISpyName;
   static  std::string  kLegoName;
   static  std::string  kGlimpseName;
   
   const static std::string& idToName(int);

   FWViewType(EType);
   virtual ~FWViewType();

   const  std::string& name() const;
   EType id() const;

private: 
   EType m_id;
};

#endif
