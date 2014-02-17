// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      PFSpecificAlgo
// 
// Original Authors:  R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
//          Created:  October 27, 2008
// $Id: PFSpecificAlgo.cc,v 1.13 2012/06/10 16:37:16 sakuma Exp $
//
//
//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//____________________________________________________________________________||
using namespace reco;

//____________________________________________________________________________||
reco::PFMET PFSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > PFCandidates, CommonMETData met)
{
  SpecificPFMETData specific = mkSpecificPFMETData(PFCandidates);

  const LorentzVector p4(met.mex , met.mey, 0.0, met.met);
  const Point vtx(0.0,0.0,0.0);
  PFMET pfMET(specific, met.sumet, p4, vtx );

  if(doSignificance) pfMET.setSignificanceMatrix(mkSignifMatrix(PFCandidates));

  return pfMET;
}

//____________________________________________________________________________||
void PFSpecificAlgo::runSignificance(metsig::SignAlgoResolutions &resolutions, edm::Handle<edm::View<reco::PFJet> > jets)
{
  doSignificance = true;
  pfsignalgo_.setResolutions( &resolutions );
  pfsignalgo_.addPFJets(jets);
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
SpecificPFMETData PFSpecificAlgo::mkSpecificPFMETData(edm::Handle<edm::View<reco::Candidate> > &PFCandidates)
{
  if(!PFCandidates->size())
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

  for( edm::View<reco::Candidate>::const_iterator iParticle = (PFCandidates.product())->begin(); iParticle != (PFCandidates.product())->end(); ++iParticle )
    {   
      const PFCandidate* pfCandidate = dynamic_cast<const PFCandidate*> (&(*iParticle));
      if (!pfCandidate) continue;
      const double theta = pfCandidate->theta();
      const double e     = pfCandidate->energy();
      const double et    = e*sin(theta);

      if (pfCandidate->particleId() == 1) ChargedHadEt += et;
      if (pfCandidate->particleId() == 2) ChargedEMEt += et;
      if (pfCandidate->particleId() == 3) MuonEt += et;
      if (pfCandidate->particleId() == 4) NeutralEMEt += et;
      if (pfCandidate->particleId() == 5) NeutralHadEt += et;
      if (pfCandidate->particleId() == 6) type6Et += et;
      if (pfCandidate->particleId() == 7) type7Et += et;


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
TMatrixD PFSpecificAlgo::mkSignifMatrix(edm::Handle<edm::View<reco::Candidate> > &PFCandidates)
{
  pfsignalgo_.useOriginalPtrs(PFCandidates.id());
  for(edm::View<reco::Candidate>::const_iterator iParticle = (PFCandidates.product())->begin(); iParticle != (PFCandidates.product())->end(); ++iParticle )
    {   
      const PFCandidate* pfCandidate = dynamic_cast<const PFCandidate*> (&(*iParticle));
      if (!pfCandidate) continue;
      reco::CandidatePtr dau(PFCandidates, iParticle - PFCandidates->begin());
      if(dau.isNull()) continue;
      if(!dau.isAvailable()) continue;
      reco::PFCandidatePtr pf(dau.id(), pfCandidate, dau.key());
      pfsignalgo_.addPFCandidate(pf);
    }
  return pfsignalgo_.getSignifMatrix();
}

//____________________________________________________________________________||
