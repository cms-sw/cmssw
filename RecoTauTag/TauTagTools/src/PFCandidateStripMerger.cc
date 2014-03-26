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

}


PFCandidateStripMerger::~PFCandidateStripMerger()
{}


bool 
PFCandidateStripMerger::candidateMatches(const reco::PFCandidatePtr& cand)
{
  bool matches = false;
  for(unsigned int i=0; i < inputPdgIds_.size(); ++i) {
    if(std::abs(cand->pdgId()) == inputPdgIds_.at(i)) {
      matches = true;
      continue;
    }
  }

  return matches;
}





vector<vector<PFCandidatePtr>> 
PFCandidateStripMerger::mergeCandidates(const vector<PFCandidatePtr>& candidates) 
{

  //Copy the input getting the relevant candidates and sort by pt 
  vector<PFCandidatePtr> cands;
  for(unsigned int i=0;i<candidates.size();++i)
    if(candidateMatches(candidates.at(i)))
      cands.push_back(candidates.at(i));
 
    if(cands.size()>1)
    TauTagTools::sortRefVectorByPt(cands);

    std::vector<vector<PFCandidatePtr>> strips;


  //Repeat while there are still unclusterized gammas
  while(cands.size()>0) {

    //save the non associated candidates to a different collection
    vector<PFCandidatePtr> notAssociated;
    
    //Create a cluster from the Seed Photon
    vector<PFCandidatePtr> strip;
    math::XYZTLorentzVector stripVector;
    strip.push_back(cands.at(0));
    stripVector=cands.at(0)->p4();

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
    //Save the strip 
    strips.push_back(strip);

    //Swap the candidate vector with the non associated vector
    cands.swap(notAssociated);
    //Clear 
    notAssociated.clear();
  }

  return strips;
}
