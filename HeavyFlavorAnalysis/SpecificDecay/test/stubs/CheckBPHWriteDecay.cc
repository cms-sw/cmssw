#include "HeavyFlavorAnalysis/SpecificDecay/test/stubs/CheckBPHWriteDecay.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "TH1.h"
#include "TFile.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

#define GET_PAR(TYPE, NAME, PSET) PSET.getParameter<TYPE>(#NAME)
// GET_PAR(string,xyz,ps);
// is equivalent to
// ps.getParameter< string >( "xyx" )
#define SET_PAR(TYPE, NAME, PSET) (NAME = GET_PAR(TYPE, NAME, PSET))
// SET_PAR(string,xyz,ps);
// is equivalent to
// ( xyz = GET_PAR(string,xyz,ps) )
// i.e. is equivalent to
// ( xyz = ps.getParameter< string >( "xyx" ) )
#define GET_UTP(TYPE, NAME, PSET) PSET.getUntrackedParameter<TYPE>(#NAME)
#define SET_UTP(TYPE, NAME, PSET) (NAME = GET_UTP(TYPE, NAME, PSET))

CheckBPHWriteDecay::CheckBPHWriteDecay(const edm::ParameterSet& ps) {
  SET_PAR(unsigned int, runNumber, ps);
  SET_PAR(unsigned int, evtNumber, ps);
  SET_PAR(vector<string>, candsLabel, ps);
  SET_UTP(bool, writePtr, ps);

  int i;
  int n = candsLabel.size();
  candsToken.resize(n);
  for (i = 0; i < n; ++i)
    consume<vector<pat::CompositeCandidate> >(candsToken[i], candsLabel[i]);

  string fileName = GET_UTP(string, fileName, ps);
  if (fileName.empty())
    osPtr = &cout;
  else
    osPtr = new ofstream(fileName.c_str());
}

void CheckBPHWriteDecay::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  vector<string> v;
  desc.add<vector<string> >("candsLabel", v);
  desc.add<unsigned int>("runNumber", 0);
  desc.add<unsigned int>("evtNumber", 0);
  desc.addUntracked<string>("fileName", "");
  desc.addUntracked<bool>("writePtr", false);
  descriptions.add("checkBPHWriteDecay", desc);
  return;
}

void CheckBPHWriteDecay::beginJob() { return; }

void CheckBPHWriteDecay::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  ostream& os = *osPtr;

  if ((runNumber != 0) && (ev.id().run() != runNumber))
    return;
  if ((evtNumber != 0) && (ev.id().event() != evtNumber))
    return;
  os << "--------- event " << ev.id().run() << " / " << ev.id().event() << " ---------" << endl;

  int il;
  int nl = candsLabel.size();
  vector<edm::Handle<vector<pat::CompositeCandidate> > > clist(nl);
  for (il = 0; il < nl; ++il) {
    edm::Handle<vector<pat::CompositeCandidate> >& cands = clist[il];
    candsToken[il].get(ev, cands);
    int ic;
    int nc = cands->size();
    vector<const pat::CompositeCandidate*> csort(nc);
    for (ic = 0; ic < nc; ++ic)
      csort[ic] = &cands->at(ic);
    sort(csort.begin(), csort.end(), [](const pat::CompositeCandidate* lc, const pat::CompositeCandidate* rc) {
      return lc->pt() < rc->pt();
    });
    for (ic = 0; ic < nc; ++ic) {
      os << "*********** " << candsLabel[il] << " " << ic << "/" << nc << " ***********" << endl;
      const pat::CompositeCandidate* cand = csort[ic];
      dump(os, *cand);
      idMap[cand] = ic;
    }
  }
  idMap.clear();
  return;
}

void CheckBPHWriteDecay::endJob() { return; }

void CheckBPHWriteDecay::dump(ostream& os, const pat::CompositeCandidate& cand) {
  float mfit = (cand.hasUserFloat("fitMass") ? cand.userFloat("fitMass") : -1);
  if (writePtr)
    os << &cand;
  os << " mass : " << cand.mass() << " " << mfit
     << (cand.hasUserInt("cowboy") ? (cand.userInt("cowboy") > 0 ? " cowboy" : " sailor") : "") << endl;
  writeCylindric(os, "cmom ", cand, false);
  writeCartesian(os, " xyz ", cand.momentum());
  if (cand.hasUserData("trackModes"))
    os << "trackModes: " << *cand.userData<string>("trackModes") << endl;
  const reco::Vertex* vptr = (cand.hasUserData("vertex") ? cand.userData<reco::Vertex>("vertex") : nullptr);
  if (vptr != nullptr) {
    writeCartesian(os, "vpos : ", *vptr, false);
    os << " --- " << vptr->chi2() << " / " << vptr->ndof() << " ( " << ChiSquaredProbability(vptr->chi2(), vptr->ndof())
       << " )" << endl;
  }
  const reco::Vertex* vfit = (cand.hasUserData("fitVertex") ? cand.userData<reco::Vertex>("fitVertex") : nullptr);
  if (vfit != nullptr) {
    writeCartesian(os, "vfit : ", *vfit, false);
    os << " --- " << vfit->chi2() << " / " << vfit->ndof() << " ( " << ChiSquaredProbability(vfit->chi2(), vfit->ndof())
       << " )" << endl;
  }
  if (cand.hasUserData("fitMomentum"))
    writeCartesian(os, "fmom : ", *cand.userData<Vector3DBase<float, GlobalTag> >("fitMomentum"));

  if (cand.hasUserData("primaryVertex")) {
    const vertex_ref* pvr = cand.userData<vertex_ref>("primaryVertex");
    if (pvr->isNonnull()) {
      const reco::Vertex* pvtx = pvr->get();
      if (pvtx != nullptr)
        writeCartesian(os, "ppos ", *pvtx);
    }
  }
  const pat::CompositeCandidate::role_collection& dNames = cand.roles();
  int i;
  int n = cand.numberOfDaughters();
  for (i = 0; i < n; ++i) {
    const reco::Candidate* dptr = cand.daughter(i);
    const string& nDau = dNames[i];
    string tDau = "trackMode_" + nDau;
    os << "daug " << i << "/" << n;
    os << ' ' << nDau;
    if (writePtr)
      os << " : " << dptr;
    writeCylindric(os, " == ", *dptr, false);
    os << " " << dptr->mass() << " " << dptr->charge();
    if (cand.hasUserData(tDau))
      os << ' ' << *cand.userData<string>(tDau);
    os << endl;
    const pat::Muon* mptr = dynamic_cast<const pat::Muon*>(dptr);
    os << "muon " << i << "/" << n << " : " << (mptr == nullptr ? 'N' : 'Y') << endl;
    const reco::Track* tptr = BPHTrackReference::getTrack(*dptr, "cfhpmnigs");
    os << "trk  " << i << "/" << n;
    if (writePtr)
      os << " : " << tptr;
    if (tptr != nullptr)
      writeCylindric(os, " == ", *tptr);
    else
      os << "no track" << endl;
  }
  const vector<string>& names = cand.userDataNames();
  map<const pat::CompositeCandidate*, int>::const_iterator iter;
  map<const pat::CompositeCandidate*, int>::const_iterator iend = idMap.end();
  int j;
  int m = names.size();
  for (j = 0; j < m; ++j) {
    const string& dname = names[j];
    if (dname.substr(0, 5) != "refTo")
      continue;
    const compcc_ref* ref = cand.userData<compcc_ref>(dname);
    const pat::CompositeCandidate* cptr = ref->get();
    os << dname << " : " << (cptr == nullptr ? -2 : ((iter = idMap.find(cptr)) == iend ? -1 : iter->second));
    if (writePtr)
      os << " : " << cptr;
    os << endl;
  }

  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CheckBPHWriteDecay);
