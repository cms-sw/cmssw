#include "DataFormats/Candidate/interface/Candidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

TtSemiLepEvtPartons::TtSemiLepEvtPartons(const std::vector<std::string>& partonsToIgnore)
{
  // default: use all partons
  for(unsigned int i = 0; i < 4; i++)
    ignorePartons_.push_back(false);
  // read vector of strings and flag partons to be ignored
  for(std::vector<std::string>::const_iterator str = partonsToIgnore.begin(); str != partonsToIgnore.end(); ++str) {
    if     ((*str) == "LightQ"   ) ignorePartons_[LightQ   ] = true;
    else if((*str) == "LightQBar") ignorePartons_[LightQBar] = true;
    else if((*str) == "HadB"     ) ignorePartons_[HadB     ] = true;
    else if((*str) == "LepB"     ) ignorePartons_[LepB     ] = true;
    else throw cms::Exception("Configuration")
      << "The following string in partonsToIgnore is not supported: " << (*str) << "\n";
  }
}

std::vector<const reco::Candidate*>
TtSemiLepEvtPartons::vec(const TtGenEvent& genEvt)
{
  std::vector<const reco::Candidate*> vec;

  if(genEvt.isSemiLeptonic()) {
    // fill vector with partons from genEvent
    // (use enum for positions of the partons in the vector)
    vec.resize(4);
    vec[LightQ   ] = genEvt.hadronicDecayQuark()    ? genEvt.hadronicDecayQuark()    : dummyCandidatePtr();
    vec[LightQBar] = genEvt.hadronicDecayQuarkBar() ? genEvt.hadronicDecayQuarkBar() : dummyCandidatePtr();
    vec[HadB     ] = genEvt.hadronicDecayB()        ? genEvt.hadronicDecayB()        : dummyCandidatePtr();
    vec[LepB     ] = genEvt.leptonicDecayB()        ? genEvt.leptonicDecayB()        : dummyCandidatePtr();
  }
  else {
    // fill vector with dummy objects if the event is not semi-leptonic ttbar
    for(unsigned i=0; i<4; i++)
      vec.push_back( dummyCandidatePtr() );
  }

  // erase partons from vector if they where chosen to be ignored
  prune(vec);

  return vec;
}
