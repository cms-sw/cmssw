#include "CommonTools/CandAlgos/interface/GenJetParticleSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <algorithm>
using namespace std;
using namespace edm;

GenJetParticleSelector::GenJetParticleSelector(const ParameterSet& cfg, edm::ConsumesCollector & iC) :
  stableOnly_(cfg.getParameter<bool>("stableOnly")),
  partons_(false), bInclude_(false) {
  const string excludeString("excludeList");
  const string includeString("includeList");
  vpdt includeList, excludeList;
  vector<string> vPdtParams = cfg.getParameterNamesForType<vpdt>();
  bool found = std::find(vPdtParams.begin(), vPdtParams.end(), includeString) != vPdtParams.end();
  if(found) includeList = cfg.getParameter<vpdt>(includeString);
  found = find(vPdtParams.begin(), vPdtParams.end(), excludeString) != vPdtParams.end();
  if(found) excludeList = cfg.getParameter<vpdt>(excludeString);
  const string partonsString("partons");
  vector<string> vBoolParams = cfg.getParameterNamesForType<bool>();
  found = find(vBoolParams.begin(), vBoolParams.end(), partonsString) != vBoolParams.end();
  if(found) partons_ = cfg.getParameter<bool>(partonsString);
  bool bExclude = false;
  if (includeList.size() > 0) bInclude_ = true;
  if (excludeList.size() > 0) bExclude = true;

  if (bInclude_ && bExclude) {
    throw cms::Exception("ConfigError", "not allowed to use both includeList and excludeList at the same time\n");
  }
  else if (bInclude_) {
    pdtList_ = includeList;
  }
  else {
    pdtList_ = excludeList;
  }
  if(stableOnly_ && partons_) {
    throw cms::Exception("ConfigError", "not allowed to have both stableOnly and partons true at the same time\n");
  }
}

bool GenJetParticleSelector::operator()(const reco::Candidate& p) {
  int status = p.status();
  int id = abs(p.pdgId());
  if((!stableOnly_ || status == 1) && !partons_ &&
     ( (pIds_.find(id) == pIds_.end()) ^ bInclude_))
    return true;
  else if(partons_ &&
	  (p.numberOfDaughters() > 0 && (p.daughter(0)->pdgId() == 91 || p.daughter(0)->pdgId() == 92)) &&
	  ( ((pIds_.find(id) == pIds_.end()) ^ bInclude_)))
    return true;
  else
    return false;
}

void GenJetParticleSelector::init(const edm::EventSetup& es) {
  for(vpdt::iterator i = pdtList_.begin(); i != pdtList_.end(); ++i )
    i->setup(es);
}

