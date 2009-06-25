#include "DataFormats/Candidate/interface/Candidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

TtFullHadEvtPartons::TtFullHadEvtPartons(const std::vector<std::string>& partonsToIgnore)
{
  // default: use all partons
  for(unsigned int i = 0; i < 6; i++)
    ignorePartons_.push_back(false);
  // read vector of strings and flag partons to be ignored
  for(std::vector<std::string>::const_iterator str = partonsToIgnore.begin(); str != partonsToIgnore.end(); ++str) {
    if     ((*str) == "LightQTop"      ) ignorePartons_[LightQTop      ] = true;
    else if((*str) == "LightQBarTop"   ) ignorePartons_[LightQBarTop   ] = true;
    else if((*str) == "B"              ) ignorePartons_[B              ] = true;
    else if((*str) == "LightQTopBar"   ) ignorePartons_[LightQTopBar   ] = true;
    else if((*str) == "LightQBarTopBar") ignorePartons_[LightQBarTopBar] = true;
    else if((*str) == "BBar"           ) ignorePartons_[BBar           ] = true;
    else throw cms::Exception("Configuration")
      << "The following string in partonsToIgnore is not supported: " << (*str) << "\n";
  }
}

std::vector<const reco::Candidate*>
TtFullHadEvtPartons::vec(const TtGenEvent& genEvt)
{
  std::vector<const reco::Candidate*> vec;

  if(genEvt.isFullHadronic()) {
    // fill vector with partons from genEvent
    // (use enum for positions of the partons in the vector)
    vec.resize(6);
    vec[LightQTop      ] = genEvt.lightQFromTop()       ? genEvt.lightQFromTop()       : dummyCandidatePtr();
    vec[LightQBarTop   ] = genEvt.lightQBarFromTop()    ? genEvt.lightQBarFromTop()    : dummyCandidatePtr();
    vec[B              ] = genEvt.b()                   ? genEvt.b()                   : dummyCandidatePtr();
    vec[LightQTopBar   ] = genEvt.lightQFromTopBar()    ? genEvt.lightQFromTopBar()    : dummyCandidatePtr();
    vec[LightQBarTopBar] = genEvt.lightQBarFromTopBar() ? genEvt.lightQBarFromTopBar() : dummyCandidatePtr();
    vec[BBar           ] = genEvt.bBar()                ? genEvt.bBar()                : dummyCandidatePtr();
  }
  else {
    // fill vector with dummy objects if the event is not fully-hadronic ttbar
    for(unsigned i=0; i<6; i++)
      vec.push_back( dummyCandidatePtr() );
  }

  // erase partons from vector if they where chosen to be ignored
  prune(vec);

  return vec;
}
