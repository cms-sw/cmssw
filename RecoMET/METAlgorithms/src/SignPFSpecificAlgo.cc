/*
 class: SignPFSpecificAlgo.cc
 description:  add the infrastructure to compute the jet-based significance for MET computed with particle flow
 authors: A. Khukhunaishvili (Cornell), L. Gibbons (Cornell)
 date: 11/11/10
 */

#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"

metsig::SignPFSpecificAlgo::SignPFSpecificAlgo(): 
resolutions_(0),
algo_()
{clusteredParticlePtrs_.clear();}

metsig::SignPFSpecificAlgo::~SignPFSpecificAlgo(){}

void 
metsig::SignPFSpecificAlgo::setResolutions( metsig::SignAlgoResolutions *resolutions) {
  resolutions_ = resolutions;
}


void 
metsig::SignPFSpecificAlgo::addPFJets(edm::Handle<edm::View<reco::PFJet> > PFJets){
    std::vector<metsig::SigInputObj> vobj;
    for(edm::View<reco::PFJet>::const_iterator jet = PFJets->begin(); jet != PFJets->end(); ++jet){
	vobj.push_back(resolutions_->evalPFJet(&(*jet)));
	//save jet constituents
	std::vector<reco::PFCandidatePtr> pfs = jet->getPFConstituents();
	for(unsigned int i=0; i<pfs.size(); i++) clusteredParticlePtrs_.insert(pfs[i]);
    }
    algo_.addObjects(vobj);
}


void
metsig::SignPFSpecificAlgo::addPFCandidate(reco::PFCandidatePtr pf){
    if(clusteredParticlePtrs_.find(pf) != clusteredParticlePtrs_.end()) return; //pf candidate already added in jet collection
    std::vector<metsig::SigInputObj> vobj;
    vobj.push_back(resolutions_->evalPF(&(*pf)));
    algo_.addObjects(vobj);
}

