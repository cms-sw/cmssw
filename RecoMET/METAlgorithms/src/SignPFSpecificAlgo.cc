// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignPFSpecificAlgo
//
// Authors: A. Khukhunaishvili (Cornell), L. Gibbons (Cornell)
// First Implementation: November 11, 2011
//
//

//____________________________________________________________________________||
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//____________________________________________________________________________||
metsig::SignPFSpecificAlgo::SignPFSpecificAlgo()
: resolutions_(nullptr), algo_()
{
  clusteredParticlePtrs_.clear();
}

//____________________________________________________________________________||
void metsig::SignPFSpecificAlgo::setResolutions( metsig::SignAlgoResolutions *resolutions)
{
  resolutions_ = resolutions;
}

//____________________________________________________________________________||
void metsig::SignPFSpecificAlgo::addPFJets(const edm::View<reco::PFJet>* PFJets)
{
  std::vector<metsig::SigInputObj> vobj;
  for(edm::View<reco::PFJet>::const_iterator jet = PFJets->begin(); jet != PFJets->end(); ++jet)
    {
      vobj.push_back(resolutions_->evalPFJet(&(*jet)));
      std::vector<reco::PFCandidatePtr> pfs = jet->getPFConstituents();
      for(std::vector<reco::PFCandidatePtr>::const_iterator it=pfs.begin(); it!=pfs.end(); ++it)
	{
	  reco::CandidatePtr ptr(*it);
	  clusteredParticlePtrs_.insert(ptr);
	}
    }
  algo_.addObjects(vobj);
}

//____________________________________________________________________________||
void metsig::SignPFSpecificAlgo::useOriginalPtrs(const edm::ProductID& productID)
{
  std::set<reco::CandidatePtr>::const_iterator it=clusteredParticlePtrs_.begin();
  reco::CandidatePtr ptr(*it);
  if(ptr.id()==productID) return; //If the first element is from the right product, return

  std::set<reco::CandidatePtr> temp;
  for(; it!=clusteredParticlePtrs_.end(); ++it)
    {
      reco::CandidatePtr ptr(*it);
      while(ptr.id()!=productID)
	{
	  ptr = ptr->sourceCandidatePtr(0);
	  if(ptr.isNull()) return; //if it does not get to the correct product, return
	}
      temp.insert(ptr);
    }
  clusteredParticlePtrs_.clear();
  clusteredParticlePtrs_ = temp;
}

//____________________________________________________________________________||
void metsig::SignPFSpecificAlgo::addPFCandidate(reco::PFCandidatePtr pf)
{
  if(clusteredParticlePtrs_.find(pf) != clusteredParticlePtrs_.end())
    {
      return; //pf candidate already added in jet collection
    }
  std::vector<metsig::SigInputObj> vobj;
  vobj.push_back(resolutions_->evalPF(&(*pf)));
  algo_.addObjects(vobj);
}

//____________________________________________________________________________||
reco::METCovMatrix metsig::SignPFSpecificAlgo::mkSignifMatrix(edm::Handle<edm::View<reco::Candidate> > &PFCandidates)
{
  useOriginalPtrs(PFCandidates.id());
  for(edm::View<reco::Candidate>::const_iterator iParticle = (PFCandidates.product())->begin(); iParticle != (PFCandidates.product())->end(); ++iParticle )
    {
      const reco::PFCandidate* pfCandidate = dynamic_cast<const reco::PFCandidate*> (&(*iParticle));
      if (!pfCandidate) continue;
      reco::CandidatePtr dau(PFCandidates, iParticle - PFCandidates->begin());
      if(dau.isNull()) continue;
      if(!dau.isAvailable()) continue;
      reco::PFCandidatePtr pf(dau.id(), pfCandidate, dau.key());
      addPFCandidate(pf);
    }
  return getSignifMatrix();
}

//____________________________________________________________________________||
