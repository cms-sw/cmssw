#include "RecoTauTag/TauTagTools/interface/PFCandidateStripMerger.h"
#include "Math/GenVector/VectorUtil.h"

using namespace reco;
using namespace std;
PFCandidateStripMerger::PFCandidateStripMerger():
  PFCandidateMergerBase()
{}

PFCandidateStripMerger::PFCandidateStripMerger(const edm::ParameterSet& config):
  PFCandidateMergerBase(config)
{
  inputPdgIds_ = config.getParameter<std::vector<int> >("stripCandidatesPdgIds");
  etaAssociationDistance_ = config.getParameter<double>("stripEtaAssociationDistance");
  phiAssociationDistance_ = config.getParameter<double>("stripPhiAssociationDistance");
  stripPtThreshold_ = config.getParameter<double>("stripPtThreshold");
}


PFCandidateStripMerger::~PFCandidateStripMerger()
{}


bool 
PFCandidateStripMerger::candidateMatches(const reco::PFCandidateRef& cand)
{
  bool matches = false;
  for(unsigned int i=0; i < inputPdgIds_.size(); ++i) {
    if(abs(cand->pdgId()) == inputPdgIds_.at(i)) {
      matches = true;
      continue;
    }
  }

  return matches;
}


void 
PFCandidateStripMerger::sortRefVector(reco::PFCandidateRefVector& vec)
{
  std::vector<reco::PFCandidateRefVector::iterator> iters;
  reco::PFCandidateRefVector sorted;

  do{
  double max=0;
  reco::PFCandidateRefVector::iterator sel;

  for(reco::PFCandidateRefVector::iterator i=vec.begin();i!=vec.end();++i)
    {
      if( (*i)->pt()>max)
	{
	  max = (*i)->pt();
	  sel = i;
	}
    }
  sorted.push_back(*sel);
  vec.erase(sel);
  }
  while(vec.size()>0);
  vec = sorted;
}


vector<PFCandidateRefVector> 
PFCandidateStripMerger::mergeCandidates(const PFCandidateRefVector& candidates) 
{

  //Copy the input getting the relevant candidates and sort by pt 
  PFCandidateRefVector cands;
  for(unsigned int i=0;i<candidates.size();++i)
    if(candidateMatches(candidates.at(i)))
      cands.push_back(candidates.at(i));
 
    if(cands.size()>1)
    sortRefVector(cands);

    vector<PFCandidateRefVector> strips;


  //Repeat while there are still unclusterized gammas
  while(cands.size()>0) {

    //save the non associated candidates to a different collection
    PFCandidateRefVector notAssociated;
    
    //Create a cluster from the Seed Photon
    PFCandidateRefVector strip;
    math::XYZTLorentzVector stripVector;
    strip.push_back(cands.at(0));
    stripVector+=cands.at(0)->p4();

    //Loop and associate
    for(unsigned int i=1;i<cands.size();++i) {
	if(fabs(cands.at(i)->eta()-stripVector.eta())<etaAssociationDistance_ &&  
	   fabs(ROOT::Math::VectorUtil::DeltaPhi(cands.at(i)->p4(),stripVector))<phiAssociationDistance_) {
	    strip.push_back(cands.at(i));
	    stripVector+=cands.at(i)->p4();
	  }
	  else {
	    notAssociated.push_back(cands.at(i));
	  }
    }
    //Save the strip if it is over threshold
    if(stripVector.pt()>stripPtThreshold_)
      strips.push_back(strip);

    //Swap the candidate vector with the non associated vector
    cands.swap(notAssociated);
    //Clear 
    notAssociated.clear();
  }

  return strips;
}
