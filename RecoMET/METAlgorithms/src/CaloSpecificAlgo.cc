// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      CaloSpecificAlgo
// 
// Original Author:  R. Cavanaugh (taken from F.Ratnikov, UMd)
//         Created:  June 6, 2006
// $Id: CaloSpecificAlgo.cc,v 1.32 2012/06/09 21:19:30 sakuma Exp $
//
//
#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

#include <iostream>

using namespace reco;

//____________________________________________________________________________||
// This algorithm adds calorimeter specific global event information to 
// the MET object which may be useful/needed for MET Data Quality Monitoring
// and MET cleaning. 
//____________________________________________________________________________||

//____________________________________________________________________________||
reco::CaloMET CaloSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > towers, CommonMETData met, bool noHF, double globalThreshold)
{ 
  SpecificCaloMETData specific;
  initializeSpecificCaloMETData(specific);

  double totalEt = 0.0; 
  double totalEm = 0.0;

  double sumEtInpHF = 0.0;
  double sumEtInmHF = 0.0;
  double MExInpHF = 0.0;
  double MEyInpHF = 0.0;
  double MExInmHF = 0.0;
  double MEyInmHF = 0.0;

  for(edm::View<Candidate>::const_iterator towerCand = towers->begin(); towerCand != towers->end(); ++towerCand ) 
    {
      const CaloTower* calotower = dynamic_cast<const CaloTower*> (&(*towerCand));
      if (!calotower) continue;
      if(calotower->et() < globalThreshold) continue;
      update_totalEt_totalEm(totalEt, totalEm, calotower, noHF);
      update_MaxTowerEm_MaxTowerHad(specific.MaxEtInEmTowers, specific.MaxEtInHadTowers, calotower, noHF);
      update_EmEtInEB_EmEtInEE(specific.EmEtInEB, specific.EmEtInEE, calotower);
      update_HadEtInHB_HadEtInHE_HadEtInHO_HadEtInHF_EmEtInHF(specific.HadEtInHB, specific.HadEtInHE, specific.HadEtInHO, specific.HadEtInHF, specific.EmEtInHF, calotower, noHF);
      update_sumEtInpHF_MExInpHF_MEyInpHF_sumEtInmHF_MExInmHF_MEyInmHF(sumEtInpHF, MExInpHF, MEyInpHF, sumEtInmHF, MExInmHF, MEyInmHF, calotower);
    }
  
  double totalHad = totalEt - totalEm;
  
  if(noHF) remove_HF_from_MET(met, sumEtInpHF, MExInpHF, MEyInpHF, sumEtInmHF, MExInmHF, MEyInmHF);

  if(!noHF) add_MET_in_HF(specific, sumEtInpHF, MExInpHF, MEyInpHF, sumEtInmHF, MExInmHF, MEyInmHF);

  specific.EtFractionHadronic = (totalEt == 0.0)? 0.0 : totalHad/totalEt;
  specific.EtFractionEm = (totalEt == 0.0)? 0.0 : totalEm/totalEt;

  const LorentzVector p4( met.mex, met.mey, 0.0, met.met);
  const Point vtx(0.0, 0.0, 0.0);
  CaloMET caloMET(specific, met.sumet, p4, vtx);
  return caloMET;
}

//____________________________________________________________________________||
void CaloSpecificAlgo::initializeSpecificCaloMETData(SpecificCaloMETData &specific)
{
  specific.MaxEtInEmTowers = 0.0;    // Maximum energy in EM towers
  specific.MaxEtInHadTowers = 0.0;   // Maximum energy in HCAL towers
  specific.HadEtInHO = 0.0;          // Hadronic energy fraction in HO
  specific.HadEtInHB = 0.0;          // Hadronic energy in HB
  specific.HadEtInHF = 0.0;          // Hadronic energy in HF
  specific.HadEtInHE = 0.0;          // Hadronic energy in HE
  specific.EmEtInEB = 0.0;           // Em energy in EB
  specific.EmEtInEE = 0.0;           // Em energy in EE
  specific.EmEtInHF = 0.0;           // Em energy in HF
  specific.EtFractionHadronic = 0.0; // Hadronic energy fraction
  specific.EtFractionEm = 0.0;       // Em energy fraction
  specific.METSignificance = 0.0;
  specific.CaloMETInpHF = 0.0;        // CaloMET in HF+ 
  specific.CaloMETInmHF = 0.0;        // CaloMET in HF- 
  specific.CaloSETInpHF = 0.0;        // CaloSET in HF+ 
  specific.CaloSETInmHF = 0.0;        // CaloSET in HF- 
  specific.CaloMETPhiInpHF = 0.0;     // CaloMET-phi in HF+ 
  specific.CaloMETPhiInmHF = 0.0;     // CaloMET-phi in HF- 
}

//____________________________________________________________________________||
void CaloSpecificAlgo::update_totalEt_totalEm(double &totalEt, double& totalEm, const CaloTower* calotower, bool noHF)
{
  if( noHF )
    {
      DetId detIdHcal = find_DetId_of_HCAL_cell_in_constituent_of(calotower);
      if(!detIdHcal.null()) 
	{
	  HcalSubdetector subdet = HcalDetId(detIdHcal).subdet();
	  if( subdet == HcalForward ) return;
	}
    }

  totalEt += calotower->et();
  totalEm += calotower->emEt();
}

//____________________________________________________________________________||
void CaloSpecificAlgo::update_MaxTowerEm_MaxTowerHad(double &MaxTowerEm, double &MaxTowerHad, const CaloTower* calotower, bool noHF)
{
  DetId detIdHcal = find_DetId_of_HCAL_cell_in_constituent_of(calotower);
  DetId detIdEcal = find_DetId_of_ECAL_cell_in_constituent_of(calotower);

  if( !detIdHcal.null() )
    {
      HcalSubdetector subdet = HcalDetId(detIdHcal).subdet();
      if( subdet == HcalBarrel || subdet == HcalOuter || subdet == HcalEndcap || (!noHF && subdet == HcalForward))
	{
	  if( calotower->hadEt() > MaxTowerHad ) MaxTowerHad = calotower->hadEt();
	  if( calotower->emEt() > MaxTowerEm  ) MaxTowerEm  = calotower->emEt();
	}

    }

  if( !detIdEcal.null() )
    {
      if( calotower->emEt() > MaxTowerEm ) MaxTowerEm = calotower->emEt();
    }
}

//____________________________________________________________________________||
void CaloSpecificAlgo::update_EmEtInEB_EmEtInEE(double &EmEtInEB, double &EmEtInEE, const CaloTower* calotower)
{
  DetId detIdEcal = find_DetId_of_ECAL_cell_in_constituent_of(calotower);
  if(detIdEcal.null()) return;

  EcalSubdetector subdet = EcalSubdetector( detIdEcal.subdetId() );
  if( subdet == EcalBarrel )
    {
      EmEtInEB += calotower->emEt(); 
    }
  else if( subdet == EcalEndcap ) 
    {
      EmEtInEE += calotower->emEt();
    }
}

//____________________________________________________________________________||
void CaloSpecificAlgo::update_HadEtInHB_HadEtInHE_HadEtInHO_HadEtInHF_EmEtInHF(double &HadEtInHB, double &HadEtInHE, double &HadEtInHO, double &HadEtInHF, double &EmEtInHF, const CaloTower* calotower, bool noHF)
{
  DetId detIdHcal = find_DetId_of_HCAL_cell_in_constituent_of(calotower);
  if(detIdHcal.null()) return;

  HcalSubdetector subdet = HcalDetId(detIdHcal).subdet();
  if( subdet == HcalBarrel || subdet == HcalOuter )
    {
      HadEtInHB += calotower->hadEt();
      HadEtInHO += calotower->outerEt();
    }

  if( subdet == HcalEndcap )
    {
      HadEtInHE += calotower->hadEt();
    }

  if( subdet == HcalForward && !noHF)
    {
      HadEtInHF += calotower->hadEt();
      EmEtInHF += calotower->emEt();
    }
}

//____________________________________________________________________________||
void CaloSpecificAlgo::update_sumEtInpHF_MExInpHF_MEyInpHF_sumEtInmHF_MExInmHF_MEyInmHF(double &sumEtInpHF, double &MExInpHF, double &MEyInpHF, double &sumEtInmHF, double &MExInmHF, double &MEyInmHF, const CaloTower* calotower)
{
  DetId detIdHcal = find_DetId_of_HCAL_cell_in_constituent_of(calotower);
  if(detIdHcal.null()) return;

  HcalSubdetector subdet = HcalDetId(detIdHcal).subdet();
  if( !(subdet == HcalForward) ) return;

  if (calotower->eta() >= 0)
    {
      sumEtInpHF += calotower->et();
      MExInpHF -= (calotower->et() * cos(calotower->phi()));
      MEyInpHF -= (calotower->et() * sin(calotower->phi()));
    }
  else
    {
      sumEtInmHF += calotower->et();
      MExInmHF -= (calotower->et() * cos(calotower->phi()));
      MEyInmHF -= (calotower->et() * sin(calotower->phi()));
    }
}

//____________________________________________________________________________||
void CaloSpecificAlgo::remove_HF_from_MET(CommonMETData &met, double sumEtInpHF, double MExInpHF, double MEyInpHF, double sumEtInmHF, double MExInmHF, double MEyInmHF)
{
  met.mex -= (MExInmHF + MExInpHF);
  met.mey -= (MEyInmHF + MEyInpHF);
  met.sumet -= (sumEtInpHF + sumEtInmHF);
  met.met = sqrt(met.mex*met.mex + met.mey*met.mey);   
}

//____________________________________________________________________________||
void CaloSpecificAlgo::add_MET_in_HF(SpecificCaloMETData &specific, double sumEtInpHF, double MExInpHF, double MEyInpHF, double sumEtInmHF, double MExInmHF, double MEyInmHF)
{
  LorentzVector METpHF(MExInpHF, MEyInpHF, 0, sqrt(MExInpHF*MExInpHF + MEyInpHF*MEyInpHF));
  LorentzVector METmHF(MExInmHF, MEyInmHF, 0, sqrt(MExInmHF*MExInmHF + MEyInmHF*MEyInmHF));
  specific.CaloMETInpHF = METpHF.pt();
  specific.CaloMETInmHF = METmHF.pt();
  specific.CaloMETPhiInpHF = METpHF.Phi();
  specific.CaloMETPhiInmHF = METmHF.Phi();
  specific.CaloSETInpHF = sumEtInpHF;
  specific.CaloSETInmHF = sumEtInmHF;
}

//____________________________________________________________________________||
DetId CaloSpecificAlgo::find_DetId_of_HCAL_cell_in_constituent_of(const CaloTower* calotower)
{
  DetId ret;
  for (int cell = calotower->constituentsSize() - 1; cell >= 0; --cell)
    {
      DetId id = calotower->constituent( cell );
      if( id.det() == DetId::Hcal )
	{
	  ret = id;
	  break;
	}
    }
  return ret;
}

//____________________________________________________________________________||
DetId CaloSpecificAlgo::find_DetId_of_ECAL_cell_in_constituent_of(const CaloTower* calotower)
{
  DetId ret;
  for (int cell = calotower->constituentsSize() - 1; cell >= 0; --cell)
    {
      DetId id = calotower->constituent( cell );
      if( id.det() == DetId::Ecal )
	{
	  ret = id;
	  break;
	}
    }
  return ret;
}

//____________________________________________________________________________||
