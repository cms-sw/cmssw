// -*- C++ -*-
//
// Package:     Core
// Class  :     Context
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 30 14:57:12 EDT 2008
// $Id: Context.cc,v 1.29 2010/09/15 11:48:42 amraktad Exp $
//

// system include files

// user include files
#include "TH2.h"
#include "TMath.h"
#include "TEveTrackPropagator.h"
#include "TEveCaloData.h"
#include "Fireworks/Core/interface/fw3dlego_xbins.h"

#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"

using namespace fireworks;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

const float Context::s_caloTransEta = 1.479; 
const float Context::s_caloTransAngle = 2*atan(exp(-s_caloTransEta));

// simplified
const float Context::s_caloZ  = 290; 
const float Context::s_caloR  = s_caloZ*tan(s_caloTransAngle);
// barrel
const float Context::s_caloR1 = 129;
const float Context::s_caloZ1 = s_caloR1/tan(s_caloTransAngle);
// endcap
const float Context::s_caloZ2 = 315.4;
const float Context::s_caloR2 = s_caloZ2*tan(s_caloTransAngle);
// calorimeter offset between TEveCalo and outlines (used by proxy builders)
const float Context::s_caloOffR = 10;
const float Context::s_caloOffZ = s_caloOffR/tan(s_caloTransAngle);


//
// constructors and destructor
//
Context::Context(FWModelChangeManager* iCM,
                 FWSelectionManager* iSM,
                 FWEventItemsManager* iEM,
                 FWColorManager* iColorM,
                 FWJobMetadataManager* iJMDM
                 ) :
  m_changeManager(iCM),
  m_selectionManager(iSM),
  m_eventItemsManager(iEM),
  m_colorManager(iColorM),
  m_metadataManager(iJMDM),
  m_geom(0),
  m_propagator(0),
  m_trackerPropagator(0),
  m_muonPropagator(0),
  m_magField(0),
  m_commonPrefs(0),
  m_caloData(0),
  m_caloDataHF(0),
  m_caloSplit(false)
{
  if (iColorM)
    m_commonPrefs = new CmsShowCommon(iColorM);
}

Context::~Context()
{
   delete m_magField;
   delete m_commonPrefs;
}

void
Context::initEveElements()
{
   m_magField = new FWMagField();

   float propagatorOffR = 5;
   float propagatorOffZ = propagatorOffR*caloZ1(false)/caloR1(false);

   // common propagator, helix stepper
   m_propagator = new TEveTrackPropagator();
   m_propagator->SetMagFieldObj(m_magField, false);
   m_propagator->SetMaxR(caloR2()-propagatorOffR);
   m_propagator->SetMaxZ(caloZ2()-propagatorOffZ);
   m_propagator->SetDelta(0.01);
   m_propagator->SetProjTrackBreaking(TEveTrackPropagator::kPTB_UseLastPointPos);
   m_propagator->SetRnrPTBMarkers(kTRUE);
   m_propagator->IncDenyDestroy();
   // tracker propagator
   m_trackerPropagator = new TEveTrackPropagator();
   m_trackerPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_trackerPropagator->SetMagFieldObj(m_magField, false);
   m_trackerPropagator->SetDelta(0.01);
   m_trackerPropagator->SetMaxR(caloR1()-propagatorOffR);
   m_trackerPropagator->SetMaxZ(caloZ2()-propagatorOffZ);
   m_trackerPropagator->SetProjTrackBreaking(TEveTrackPropagator::kPTB_UseLastPointPos);
   m_trackerPropagator->SetRnrPTBMarkers(kTRUE);
   m_trackerPropagator->IncDenyDestroy();
   // muon propagator
   m_muonPropagator = new TEveTrackPropagator();
   m_muonPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_muonPropagator->SetMagFieldObj(m_magField, false);
   m_muonPropagator->SetDelta(0.05);
   m_muonPropagator->SetMaxR(850.f);
   m_muonPropagator->SetMaxZ(1100.f);
   m_muonPropagator->SetProjTrackBreaking(TEveTrackPropagator::kPTB_UseLastPointPos);
   m_muonPropagator->SetRnrPTBMarkers(kTRUE);
   m_muonPropagator->IncDenyDestroy();

   // general calo data
   {
      m_caloData = new TEveCaloDataHist();
      m_caloData->IncDenyDestroy();

      // Phi range is always in the (-Pi, Pi) without a shift.
      // Set wrap to false for the optimisation on TEveCaloData::GetCellList().
      m_caloData->SetWrapTwoPi(false);

      Bool_t status = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE); //Keeps histogram from going into memory
      TH2F* dummy = new TH2F("background",
                             "background",
                             fw3dlego::xbins_n - 1, fw3dlego::xbins,
                             72, -1*TMath::Pi(), TMath::Pi());
      
      TH1::AddDirectory(status);
      Int_t sliceIndex = m_caloData->AddHistogram(dummy);
      (m_caloData)->RefSliceInfo(sliceIndex).Setup("background", 0., 0);
   }
   // HF calo data
   {
      m_caloDataHF = new TEveCaloDataVec(1);
      m_caloDataHF->IncDenyDestroy();
      m_caloDataHF->SetWrapTwoPi(false);
      m_caloDataHF->RefSliceInfo(0).Setup("bg", 0.3, kRed);
      m_caloDataHF->SetEtaBins(new TAxis(fw3dlego::xbins_hf_n - 1, fw3dlego::xbins_hf));
      Double_t off = 10 * TMath::DegToRad();
      m_caloDataHF->SetPhiBins(new TAxis(36, -TMath::Pi() -off, TMath::Pi() -off));
   }
}

void
Context::deleteEveElements()
{
   m_propagator->DecDenyDestroy();
   m_trackerPropagator->DecDenyDestroy();
   m_muonPropagator->DecDenyDestroy();
   m_caloData->DecDenyDestroy();
   m_caloDataHF->DecDenyDestroy();
}


CmsShowCommon* 
Context::commonPrefs() const
{
   return m_commonPrefs;
}

float Context::caloR1(bool offset)  const
{
   float v = m_caloSplit ? s_caloR1 : s_caloR;
   if (offset) v -= s_caloOffR;
   return v;
}

float Context::caloR2(bool offset) const
{
   float v = m_caloSplit ? s_caloR2 : s_caloR;
   if (offset) v -= s_caloOffR;
   return v;
}
float Context::caloZ1(bool offset) const
{ 
   float v = m_caloSplit ? s_caloZ1 : s_caloZ;
   if (offset) v -= s_caloOffZ;
   return v; 
}

float Context::caloZ2(bool offset) const
{ 
   float v = m_caloSplit ? s_caloZ2 : s_caloZ;
   if (offset) v -= s_caloOffZ; 
   return v;
}

bool Context::caloSplit() const
{
   return m_caloSplit;
}

//
// static member functions
//


float Context::caloTransEta()
{
   return s_caloTransEta;
}

float Context::caloTransAngle()
{
   return s_caloTransAngle;
}

double Context::caloMaxEta()
{
   using namespace  fw3dlego;
   return xbins_hf[xbins_hf_n -1];
}
