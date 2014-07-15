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

//____________________________________________________________________________||
SpecificPFMETData PFSpecificAlgo::run(const edm::View<reco::Candidate>& pfCands)
{
  if(!pfCands.size()) return SpecificPFMETData();

  double NeutralEMEt = 0.0;
  double NeutralHadEt = 0.0;
  double ChargedEMEt = 0.0;
  double ChargedHadEt = 0.0;
  double MuonEt = 0.0;
  double type6Et = 0.0;
  double type7Et = 0.0;

  for( edm::View<reco::Candidate>::const_iterator iPfCand = pfCands.begin(); iPfCand != pfCands.end(); ++iPfCand)
    {   
      const reco::Candidate* pfCand = dynamic_cast<const reco::Candidate*> (&(*iPfCand));
      if (!pfCand) continue;
      const double theta = pfCand->theta();
      const double e     = pfCand->energy();
      const double et    = e*sin(theta);
      switch (abs(pfCand->pdgId())) {
        case 211: ChargedHadEt += et; break;
        case  11: ChargedEMEt += et; break;
        case  13: MuonEt += et; break;
        case  22: NeutralEMEt += et; break;
        case 130: NeutralHadEt += et; break;
        case   1: type6Et += et; break;
        case   2: type7Et += et; break;
      }
    }

  const double Et_total = NeutralEMEt + NeutralHadEt + ChargedEMEt + ChargedHadEt + MuonEt + type6Et + type7Et;
  SpecificPFMETData specific;
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
