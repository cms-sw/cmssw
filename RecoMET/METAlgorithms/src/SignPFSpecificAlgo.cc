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
	std::vector<reco::PFCandidatePtr> pfs = jet->getPFConstituents();
	for(std::vector<reco::PFCandidatePtr>::const_iterator it=pfs.begin(); it!=pfs.end(); ++it){
	    reco::CandidatePtr ptr(*it);
	    clusteredParticlePtrs_.insert(ptr);
	}
    }
    algo_.addObjects(vobj);
}

void 
metsig::SignPFSpecificAlgo::useOriginalPtrs(const edm::ProductID& productID){
    std::set<reco::CandidatePtr>::const_iterator it=clusteredParticlePtrs_.begin();
    reco::CandidatePtr ptr(*it);
    if(ptr.id()==productID) return; //If the first element is from the right product, return

    std::set<reco::CandidatePtr> temp;
    for(; it!=clusteredParticlePtrs_.end(); ++it){
	reco::CandidatePtr ptr(*it);
	while(ptr.id()!=productID){
	    ptr = ptr->sourceCandidatePtr(0);
	    if(ptr.isNull()) return; //if it does not get to the correct product, return
	}
	temp.insert(ptr);
    }
    clusteredParticlePtrs_.clear();
    clusteredParticlePtrs_ = temp;
}

void
metsig::SignPFSpecificAlgo::addPFCandidate(reco::PFCandidatePtr pf){
    if(clusteredParticlePtrs_.find(pf) != clusteredParticlePtrs_.end()) {
	return; //pf candidate already added in jet collection
    }
    std::vector<metsig::SigInputObj> vobj;
    vobj.push_back(resolutions_->evalPF(&(*pf)));
    algo_.addObjects(vobj);
}

