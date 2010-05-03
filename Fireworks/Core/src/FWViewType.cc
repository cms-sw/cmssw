// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewType
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Mar 26 12:25:02 CET 2010
// $Id: FWViewType.cc,v 1.8 2010/04/20 19:35:19 chrjones Exp $
//

// system include files

// user include files
#include "TMath.h"
#include "Fireworks/Core/interface/FWViewType.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

std::string   FWViewType::k3DName       = "3D";
std::string   FWViewType::kRhoPhiName   = "Rho Phi";
std::string   FWViewType::kRhoZName     = "Rho Z";
std::string   FWViewType::kISpyName     = "iSpy";
std::string   FWViewType::kLegoName     = "Lego";
std::string   FWViewType::kGlimpseName  = "Glimpse";

const int FWViewType::kAllRPZBits = kRhoPhiBit | kRhoZBit;
const int FWViewType::kAll3DBits  = kISpyBit | k3DBit;

//
// constructors and destructor
//
FWViewType::FWViewType(EType t):
   m_id(t)
{
}

FWViewType::~FWViewType()
{
}

//
// member functions
//

//
// const member functions
//
const std::string&
FWViewType::name() const
{
   return idToName(m_id);
}


FWViewType::EType 
FWViewType::id() const
{
   return m_id;
}
/*
const int 
FWViewType::value()const
{
   // needed for bitwise oprations
   return 1 << m_id;
}
*/
//
// static member functions
//
const std::string& 
FWViewType::idToName(int id)
{
   const static std::string errName = "Invalid ID";

   switch(id)
   {
      case k3D:
         return  k3DName;
      case kRhoPhi:
         return  kRhoPhiName;
      case kRhoZ:
         return  kRhoZName;
      case kISpy:
         return  kISpyName;
      case kGlimpse:
         return  kGlimpseName;
      case kLego:
         return  kLegoName;
      default:
         return errName;
   }
}

bool
FWViewType::isProjected(int id)
{
   return (id == kRhoPhi || id == kRhoZ);
}