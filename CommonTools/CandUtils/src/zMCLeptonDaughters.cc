#include "CommonTools/CandUtils/interface/zMCLeptonDaughters.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <cassert>
using namespace std;
using namespace reco;

pair<const Candidate*, const Candidate*> zMCLeptonDaughters(const Candidate & z, int leptonPdgId) {
  if(z.numberOfDaughters()<2)
    throw cms::Exception("RuntimeError") <<
      "calling helper function reco::zMCLeptonDaughters passing a Z candidate"
      "with less than 2 daughters (" << z.numberOfDaughters() << ").\n";
  const Candidate * dau0 = z.daughter(0);
  const Candidate * dau1 = z.daughter(1);
  for(size_t i0 = 0; i0 < dau0->numberOfDaughters(); ++i0) {
    const Candidate * ddau0 = dau0->daughter(i0);
    if(abs(ddau0->pdgId())==leptonPdgId && ddau0->status()==1) {
      dau0 = ddau0; break;
    }
  }
  for(size_t i1 = 0; i1 < dau1->numberOfDaughters(); ++i1) {
    const Candidate * ddau1 = dau1->daughter(i1);
    if(abs(ddau1->pdgId())==leptonPdgId && ddau1->status()==1) {
      dau1 = ddau1; break;
    }
  }
  assert(abs(dau0->pdgId())==leptonPdgId && dau0->status()==1);
  assert(abs(dau1->pdgId())==leptonPdgId && dau1->status()==1);
  return make_pair(dau0, dau1);
}
