// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      PFSpecificAlgo
// 
// Original Authors:  R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
//          Created:  October 27, 2008
//
//
//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

//____________________________________________________________________________||
reco::PFMET PFSpecificAlgo::addInfo(const edm::View<reco::Candidate>& pfCands, const CommonMETData& met)
{
  SpecificPFMETData specific = run(pfCands);

  const math::XYZTLorentzVector p4(met.mex, met.mey, 0.0, met.met);
  const math::XYZPoint vtx(0.0, 0.0, 0.0);
  reco::PFMET pfMET(specific, met.sumet, p4, vtx);

  return pfMET;
}

//____________________________________________________________________________||
void PFSpecificAlgo::initializeSpecificPFMETData(SpecificPFMETData &specific)
{
  specific.NeutralEMFraction = 0.0;
  specific.NeutralHadFraction = 0.0;
  specific.ChargedEMFraction = 0.0;
  specific.ChargedHadFraction = 0.0;
  specific.MuonFraction = 0.0;
  specific.Type6Fraction = 0.0;
  specific.Type7Fraction = 0.0;
}

//____________________________________________________________________________||
SpecificPFMETData PFSpecificAlgo::run(const edm::View<reco::Candidate>& pfCands)
{
  if(!pfCands.size())
  {
    SpecificPFMETData specific;
    initializeSpecificPFMETData(specific);
    return specific;
  } 

  double NeutralEMEt = 0.0;
  double NeutralHadEt = 0.0;
  double ChargedEMEt = 0.0;
  double ChargedHadEt = 0.0;
  double MuonEt = 0.0;
  double type6Et = 0.0;
  double type7Et = 0.0;

  for( edm::View<reco::Candidate>::const_iterator iPfCand = pfCands.begin(); iPfCand != pfCands.end(); ++iPfCand)
    {   
      const reco::PFCandidate* pfCand = dynamic_cast<const reco::PFCandidate*> (&(*iPfCand));
      if (!pfCand) continue;
      const double theta = pfCand->theta();
      const double e     = pfCand->energy();
      const double et    = e*sin(theta);

      if (pfCand->particleId() == 1) ChargedHadEt += et;
      if (pfCand->particleId() == 2) ChargedEMEt += et;
      if (pfCand->particleId() == 3) MuonEt += et;
      if (pfCand->particleId() == 4) NeutralEMEt += et;
      if (pfCand->particleId() == 5) NeutralHadEt += et;
      if (pfCand->particleId() == 6) type6Et += et;
      if (pfCand->particleId() == 7) type7Et += et;
    }

  const double Et_total = NeutralEMEt + NeutralHadEt + ChargedEMEt + ChargedHadEt + MuonEt + type6Et + type7Et;
  SpecificPFMETData specific;
  initializeSpecificPFMETData(specific);
  if (Et_total!=0.0)
  {
    specific.NeutralEMFraction = NeutralEMEt/Et_total;
    specific.NeutralHadFraction = NeutralHadEt/Et_total;
    specific.ChargedEMFraction = ChargedEMEt/Et_total;
    specific.ChargedHadFraction = ChargedHadEt/Et_total;
    specific.MuonFraction = MuonEt/Et_total;
    specific.Type6Fraction = type6Et/Et_total;
    specific.Type7Fraction = type7Et/Et_total;
  }
  return specific;
}

//____________________________________________________________________________||
