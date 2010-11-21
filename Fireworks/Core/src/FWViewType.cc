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
// $Id: FWViewType.cc,v 1.15 2010/11/21 11:18:14 amraktad Exp $
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
  sName[kTableTrigger ] = "TriggerTable";
  sName[kTableL1      ] = "L1TriggerTable";
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
