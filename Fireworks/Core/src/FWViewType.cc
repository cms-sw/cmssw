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
// $Id: FWViewType.cc,v 1.17 2010/12/06 15:28:15 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewType.h"

 
//
// constants, enums and typedefs
//

//
// static data member definitions
//

const int FWViewType::kAllRPZBits  = kRhoPhiBit | kRhoZBit | kRhoPhiPFBit;
const int FWViewType::kAll3DBits   = kISpyBit | k3DBit;
const int FWViewType::kAllLegoBits = kLegoBit | kLegoHFBit | kLegoPFECALBit;

std::string FWViewType::sName[FWViewType::kTypeSize];

FWViewType::static_initializer::static_initializer()
{
  sName[k3D           ] = "3D";
  sName[kRhoPhi       ] = "Rho Phi";
  sName[kRhoZ         ] = "Rho Z";
  sName[kISpy         ] = "iSpy";
  sName[kGlimpse      ] = "Glimpse";
  sName[kLego         ] = "Lego";
  sName[kLegoHF       ] = "HF";
  sName[kTable        ] = "Table";
  sName[kTableHLT ] = "HLT Table";
  sName[kTableL1      ] = "L1 Table";
  sName[kLegoPFECAL   ] = "PF ECAL Lego";
  sName[kRhoPhiPF     ] = "PF Rho Phi";
}

FWViewType::static_initializer init_statics;


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
  return sName[m_id];
}


FWViewType::EType 
FWViewType::id() const
{
   return m_id;
}
//
// static member functions
//
const std::string& 
FWViewType::idToName(int id)
{
 
  return sName[id];
}

const std::string& 
FWViewType::checkNameWithViewVersion(const std::string& refName, unsigned int version)
{
   //  printf("version %d %s \n", version, refName.c_str());
   if (version < 2)
   {
      if (refName == "TriggerTable") 
         return idToName(FWViewType::kTableHLT);
      else if (refName == "L1TriggerTable")
         return idToName(FWViewType::kTableL1);
   }
   else if (version < 3)
   {
      if (refName == "3D Lego") 
         return idToName(FWViewType::kLego);
   }
   return refName;
}

bool
FWViewType::isProjected(int id)
{
   return (id == kRhoPhi || id == kRhoZ || id == kRhoPhiPF);
}


bool
FWViewType::isLego(int id)
{
   return (id == kLego || id == kLegoHF || id == kLegoPFECAL);
}
