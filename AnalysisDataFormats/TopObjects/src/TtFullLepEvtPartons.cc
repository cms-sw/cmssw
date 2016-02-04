#include "DataFormats/Candidate/interface/Candidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullLepEvtPartons.h"

TtFullLepEvtPartons::TtFullLepEvtPartons(const std::vector<std::string>& partonsToIgnore)
{
  // default: use all partons
  for(unsigned int i = 0; i < 2; i++)
    ignorePartons_.push_back(false);
  // read vector of strings and flag partons to be ignored
  for(std::vector<std::string>::const_iterator str = partonsToIgnore.begin(); str != partonsToIgnore.end(); ++str) {
    if     ((*str) == "B"   ) ignorePartons_[B   ] = true;
    else if((*str) == "BBar") ignorePartons_[BBar] = true;
    else throw cms::Exception("Configuration")
      << "The following string in partonsToIgnore is not supported: " << (*str) << "\n";
  }
}

std::vector<const reco::Candidate*>
TtFullLepEvtPartons::vec(const TtGenEvent& genEvt)
{
  std::vector<const reco::Candidate*> vec;

  if(genEvt.isFullLeptonic()) {
    // fill vector with partons from genEvent
    // (use enum for positions of the partons in the vector)
    vec.resize(2);
    vec[B   ] = genEvt.b()    ? genEvt.b()    : dummyCandidatePtr();
    vec[BBar] = genEvt.bBar() ? genEvt.bBar() : dummyCandidatePtr();
  }
  else {
    // fill vector with dummy objects if the event is not fully-leptonic ttbar
    for(unsigned i=0; i<2; i++)
      vec.push_back( dummyCandidatePtr() );
  }

  // erase partons from vector if they where chosen to be ignored
  prune(vec);

  return vec;
}
