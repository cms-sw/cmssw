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
// $Id: FWViewType.cc,v 1.23 2012/02/22 03:46:00 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/fwLog.h"
 
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
  sName[kRhoPhi       ] = "Rho Phi";
  sName[kRhoZ         ] = "Rho Z";
  sName[k3D           ] = "3D Tower";
  sName[kISpy         ] = "3D RecHit";
  sName[kGlimpse      ] = "Glimpse";
  sName[kLego         ] = "Lego";
  sName[kLegoHF       ] = "HF";
  sName[kTable        ] = "Table";
  sName[kTableHLT     ] = "HLT Table";
  sName[kTableL1      ] = "L1 Table";
  sName[kLegoPFECAL   ] = "PF ECAL Lego";
  sName[kRhoPhiPF     ] = "PF Rho Phi";
  sName[kGeometryTable] = "Geometry Table";
  sName[kOverlapTable ] = "Overlap Table";
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

const std::string& switchName (const std::string& old, FWViewType::EType id)
{
   fwLog(fwlog::kDebug) << old << " view has been renamed to "<< FWViewType::idToName(id) << ".\n";
   return  (const std::string&)FWViewType::idToName(id);
}

const std::string& 
FWViewType::checkNameWithViewVersion(const std::string& refName, unsigned int version)
{
   // printf("version %d %s \n", version, refName.c_str());fflush(stdout);
   if (version < 2)
   {
      if (refName == "TriggerTable") 
         return switchName(refName, FWViewType::kTableHLT);
      else if (refName == "L1TriggerTable")
         return switchName(refName, FWViewType::kTableL1);
   }
   if (version < 3)
   {
      if (refName == "3D Lego") 
         return switchName(refName, FWViewType::kLego);
   }
   if (version < 7)
   {
      if (refName == "iSpy") 
         return switchName(refName, FWViewType::kISpy);
      if (refName == "3D") 
         return switchName(refName, FWViewType::k3D);
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
