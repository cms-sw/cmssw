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
// $Id$
//
#include <string>

class  FWViewType
{
public:
   enum EType { k3D, kRhoPhi, kRhoZ, kLego, kGlimpse, kSize };

   enum EBit
   {
      k3DBit      = 1 << k3D,
      kRhoPhiBit  = 1 << kRhoPhi,
      kRhoZBit    = 1 << kRhoZ,
      kLegoBit    = 1 << kLego,
      kGlimpseBit = 1 << kGlimpse
   };
   // const static int kRPZBit = kRhoPhiBit |  kRhoZBit;

   static  std::string  k3DName;
   static  std::string  kRhoPhiName;
   static  std::string  kRhoZName;
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
