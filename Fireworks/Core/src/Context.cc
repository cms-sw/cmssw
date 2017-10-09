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
#include "Fireworks/Core/interface/FWBeamSpot.h"
#include "Fireworks/Core/interface/CmsShowCommon.h"

#include <boost/bind.hpp>

using namespace fireworks;

Context* Context::s_fwContext = NULL;

const float Context::s_caloTransEta = 1.479; 
const float Context::s_caloTransAngle = 2*atan(exp(-s_caloTransEta));

// simplified
const float Context::s_caloZ  = 290; 
const float Context::s_caloR  = s_caloZ*tan(s_caloTransAngle);

/*
// barrel
const float Context::s_caloR1 = 129;
const float Context::s_caloZ1 = s_caloR1/tan(s_caloTransAngle);
// endcap
const float Context::s_caloZ2 = 315.4;
const float Context::s_caloR2 = s_caloZ2*tan(s_caloTransAngle);
*/

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
  m_beamSpot(0),
  m_commonPrefs(0),
  m_maxEt(1.f),
  m_maxEnergy(1.f),
  m_hidePFBuilders(false),
  m_caloData(0),
  m_caloDataHF(0)
{
   if (iColorM) // unit test
     m_commonPrefs = new CmsShowCommon(this);

   s_fwContext = this;
}


Context::~Context()
{
   delete m_commonPrefs;
}

void
Context::initEveElements()
{
   m_magField = new FWMagField();
   m_beamSpot = new FWBeamSpot();
   

   float propagatorOffR = 5;
   float propagatorOffZ = propagatorOffR*caloZ1(false)/caloR1(false);

   // common propagator, helix stepper
   m_propagator = new TEveTrackPropagator();
   m_propagator->SetMagFieldObj(m_magField, false);
   m_propagator->SetMaxR(caloR2()-propagatorOffR);
   m_propagator->SetMaxZ(caloZ2()-propagatorOffZ);
   m_propagator->SetDelta(0.01);
   m_propagator->SetProjTrackBreaking(m_commonPrefs->getProjTrackBreaking());
   m_propagator->SetRnrPTBMarkers(m_commonPrefs->getRnrPTBMarkers());
   m_propagator->IncDenyDestroy();
   // tracker propagator
   m_trackerPropagator = new TEveTrackPropagator();
   m_trackerPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_trackerPropagator->SetMagFieldObj(m_magField, false);
   m_trackerPropagator->SetDelta(0.01);
   m_trackerPropagator->SetMaxR(caloR1()-propagatorOffR);
   m_trackerPropagator->SetMaxZ(caloZ2()-propagatorOffZ);
   m_trackerPropagator->SetProjTrackBreaking(m_commonPrefs->getProjTrackBreaking());
   m_trackerPropagator->SetRnrPTBMarkers(m_commonPrefs->getRnrPTBMarkers());
   m_trackerPropagator->IncDenyDestroy();
   // muon propagator
   m_muonPropagator = new TEveTrackPropagator();
   m_muonPropagator->SetStepper( TEveTrackPropagator::kRungeKutta );
   m_muonPropagator->SetMagFieldObj(m_magField, false);
   m_muonPropagator->SetDelta(0.05);
   m_muonPropagator->SetMaxR(850.f);
   m_muonPropagator->SetMaxZ(1100.f);
   m_muonPropagator->SetProjTrackBreaking(m_commonPrefs->getProjTrackBreaking());
   m_muonPropagator->SetRnrPTBMarkers(m_commonPrefs->getRnrPTBMarkers());
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
   // AMT: delete of eve-elements disabled to prevent crash on exit.
   // A lot of eve objects use this elements (e.g. TEveCalo, TEveTrack ...)
   // If want to have explicit delete make sure order of destruction
   // is correct: this should be called after all scenes are destroyed.
}



CmsShowCommon* 
Context::commonPrefs() const
{
   return m_commonPrefs;
}

void
Context::voteMaxEtAndEnergy(float et, float energy) const
{
   m_maxEt     = TMath::Max(et    , m_maxEt    );
   m_maxEnergy = TMath::Max(energy, m_maxEnergy);
}

void
Context::resetMaxEtAndEnergy() const
{
   // should not be zero, problems with infinte bbox 

   m_maxEnergy = 1.f;
   m_maxEt     = 1.f;
}

float
Context::getMaxEnergyInEvent(bool isEt) const
{
   return isEt ?  m_maxEt : m_maxEnergy;
}

//
// static member functions
//

float Context::caloR1(bool offset)
{
   return offset ? (s_caloR -offset) :s_caloR;
}

float Context::caloR2(bool offset)
{
  
   return offset ? (s_caloR -offset) :s_caloR;
}
float Context::caloZ1(bool offset)
{
   return offset ? (s_caloZ -offset) :s_caloZ;
}

float Context::caloZ2(bool offset)
{ 
   return offset ? (s_caloZ -offset) :s_caloZ;
}

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
   return fw3dlego::xbins_hf[fw3dlego::xbins_hf_n -1];
}

Context* Context::getInstance()
{
   return s_fwContext;
}
