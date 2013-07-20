// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      GenSpecificAlgo
// 
// Original Authors:  R. Cavanaugh (taken from F.Ratnikov, UMd)
//          Created:  June 6, 2006
// $Id: GenSpecificAlgo.cc,v 1.11 2013/05/03 18:53:44 salee Exp $
//
//
//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "TMath.h"

#include <set>

//____________________________________________________________________________||
reco::GenMET GenSpecificAlgo::addInfo(edm::Handle<edm::View<reco::Candidate> > particles, CommonMETData *met, double globalThreshold, bool onlyFiducial,bool applyFiducialThresholdForFractions, bool usePt)
{ 
  fillCommonMETData(met, particles, globalThreshold, onlyFiducial, usePt);

  SpecificGenMETData specific = mkSpecificGenMETData(particles, globalThreshold, onlyFiducial, applyFiducialThresholdForFractions, usePt);

  const LorentzVector p4( met->mex, met->mey, met->mez, met->met );
  const Point vtx( 0.0, 0.0, 0.0 );

  reco::GenMET genMet(specific, met->sumet, p4, vtx );
  return genMet;
}

//____________________________________________________________________________||
void GenSpecificAlgo::fillCommonMETData(CommonMETData *met, edm::Handle<edm::View<reco::Candidate> >& particles, double globalThreshold, bool onlyFiducial, bool usePt)
{
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;

  for( edm::View<reco::Candidate>::const_iterator iParticle = (particles.product())->begin(); iParticle != (particles.product())->end(); ++iParticle)
    {
	if( (onlyFiducial && TMath::Abs(iParticle->eta()) >= 5.0)) continue;

	if( iParticle->et() <= globalThreshold  ) continue;

	if(usePt)
	  {
	    double phi = iParticle->phi();
	    double et = iParticle->pt();
	    sum_ez += iParticle->pz();
	    sum_et += et;
	    sum_ex += et*cos(phi);
	    sum_ey += et*sin(phi);
	  }
	else
	  {
	    double phi = iParticle->phi();
	    double theta = iParticle->theta();
	    double e = iParticle->energy();
	    double et = e*sin(theta);
	    sum_ez += e*cos(theta);
	    sum_et += et;
	    sum_ex += et*cos(phi);
	    sum_ey += et*sin(phi);
	  }
    }
  
  met->mex   = -sum_ex;
  met->mey   = -sum_ey;
  met->mez   = -sum_ez;
  met->met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  met->sumet = sum_et;
  met->phi   = atan2( -sum_ey, -sum_ex );
}

//____________________________________________________________________________||
SpecificGenMETData GenSpecificAlgo::mkSpecificGenMETData(edm::Handle<edm::View<reco::Candidate> >& particles,double globalThreshold, bool onlyFiducial,bool applyFiducialThresholdForFractions, bool usePt)
{
  const static int neutralEMpdgId[] = { 22 /* photon */ };
  const static std::set<int> neutralEMpdgIdSet(neutralEMpdgId, neutralEMpdgId + sizeof(neutralEMpdgId)/sizeof(int));

  const static int chargedEMpdgId[] = { 11 /* e */ };
  const static std::set<int> chargedEMpdgIdSet(chargedEMpdgId, chargedEMpdgId + sizeof(chargedEMpdgId)/sizeof(int));

  const static int muonpdgId[] = { 13 /* muon */ };
  const static std::set<int> muonpdgIdSet(muonpdgId, muonpdgId + sizeof(muonpdgId)/sizeof(int));

  const static int neutralHADpdgId[] = {
    130  /* K_long          */,
    310  /* K_short         */,
    3122 /* Lambda          */,
    2112 /* n               */,
    3222 /* Neutral Cascade */
  };
  const static std::set<int> neutralHADpdgIdSet(neutralHADpdgId, neutralHADpdgId + sizeof(neutralHADpdgId)/sizeof(int));

  const static int chargedHADpdgId[] = {
    211  /* pi        */,
    321  /* K+/K-     */,
    2212 /* p         */,
    3312 /* Cascade - */,
    3112 /* Sigma -   */,
    3322 /* Sigma +   */,
    3334 /* Omega -   */
  };
  const static std::set<int> chargedHADpdgIdSet(chargedHADpdgId, chargedHADpdgId + sizeof(chargedHADpdgId)/sizeof(int));

  const static int invisiblepdgId[] = {
    12      /* e_nu            */,
    14      /* mu_nu           */,
    16      /* tau_nu          */,
    1000022 /* Neutral ~Chi_0  */,
    1000012 /* LH ~e_nu        */,
    1000014 /* LH ~mu_nu       */,
    1000016 /* LH ~tau_nu      */,
    2000012 /* RH ~e_nu        */,
    2000014 /* RH ~mu_nu       */,
    2000016 /* RH ~tau_nu      */,
    39      /* G               */,
    1000039 /* ~G              */,
    5100039 /* KK G            */,
    4000012 /* excited e_nu    */,
    4000014 /* excited mu_nu   */,
    4000016 /* excited tau_nu  */,
    9900012 /* Maj e_nu        */,
    9900014 /* Maj mu_nu       */,
    9900016 /* Maj tau_nu      */,
  };
  const static std::set<int> invisiblepdgIdSet(invisiblepdgId, invisiblepdgId + sizeof(invisiblepdgId)/sizeof(int));
  
  SpecificGenMETData specific = SpecificGenMETData();
  specific.NeutralEMEtFraction     = 0.0;
  specific.NeutralHadEtFraction    = 0.0;
  specific.ChargedEMEtFraction     = 0.0;
  specific.ChargedHadEtFraction    = 0.0;
  specific.MuonEtFraction          = 0.0;
  specific.InvisibleEtFraction     = 0.0;
  double Et_unclassified = 0.0;
  
  for(edm::View<reco::Candidate>::const_iterator iParticle = (particles.product())->begin(); iParticle != (particles.product())->end(); ++iParticle)
    {
      if(applyFiducialThresholdForFractions) if( onlyFiducial && (TMath::Abs(iParticle->eta()) >= 5.0) ) continue;
      if(applyFiducialThresholdForFractions) if( iParticle->et() <= globalThreshold ) continue;

      int pdgId = TMath::Abs( iParticle->pdgId() ) ;
      double pt = (usePt) ? iParticle->pt() : iParticle->et();
      if(neutralEMpdgIdSet.count(pdgId))       specific.NeutralEMEtFraction  += pt;
      else if(chargedEMpdgIdSet.count(pdgId))  specific.ChargedEMEtFraction  += pt;
      else if(muonpdgIdSet.count(pdgId))       specific.MuonEtFraction       += pt;
      else if(neutralHADpdgIdSet.count(pdgId)) specific.NeutralHadEtFraction += pt;
      else if(chargedHADpdgIdSet.count(pdgId)) specific.ChargedHadEtFraction += pt;
      else if(invisiblepdgIdSet.count(pdgId))  specific.InvisibleEtFraction  += pt;
      else Et_unclassified += pt;
    }
  
  double Et_Total = specific.NeutralEMEtFraction + specific.NeutralHadEtFraction + specific.ChargedEMEtFraction + 
    specific.ChargedHadEtFraction + specific.MuonEtFraction + specific.InvisibleEtFraction + Et_unclassified;
  
  if(Et_Total) 
    {
      specific.NeutralEMEtFraction /= Et_Total;
      specific.NeutralHadEtFraction /= Et_Total;
      specific.ChargedEMEtFraction /= Et_Total;
      specific.ChargedHadEtFraction /= Et_Total;
      specific.MuonEtFraction /= Et_Total;
      specific.InvisibleEtFraction /= Et_Total;
    }

  return specific;
}

//____________________________________________________________________________||
