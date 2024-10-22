#include "HeavyFlavorAnalysis/RecoDecay/test/stubs/TestBPHRecoDecay.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "TH1.h"
#include "TFile.h"
#include "TMath.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

#define SET_LABEL(NAME, PSET) (NAME = PSET.getParameter<string>(#NAME))
// SET_LABEL(xyz,ps);
// is equivalent to
// xyz = ps.getParameter<string>( "xyx" )

TestBPHRecoDecay::TestBPHRecoDecay(const edm::ParameterSet& ps) {
  usePM = (!SET_LABEL(patMuonLabel, ps).empty());
  useCC = (!SET_LABEL(ccCandsLabel, ps).empty());
  usePF = (!SET_LABEL(pfCandsLabel, ps).empty());
  usePC = (!SET_LABEL(pcCandsLabel, ps).empty());
  useGP = (!SET_LABEL(gpCandsLabel, ps).empty());

  esConsume<TransientTrackBuilder, TransientTrackRecord>(ttBToken, "TransientTrackBuilder");
  if (usePM)
    consume<pat::MuonCollection>(patMuonToken, patMuonLabel);
  if (useCC)
    consume<vector<pat::CompositeCandidate> >(ccCandsToken, ccCandsLabel);
  if (usePF)
    consume<vector<reco::PFCandidate> >(pfCandsToken, pfCandsLabel);
  if (usePC)
    consume<vector<BPHTrackReference::candidate> >(pcCandsToken, pcCandsLabel);
  if (useGP)
    consume<vector<pat::GenericParticle> >(gpCandsToken, gpCandsLabel);
  SET_LABEL(outDump, ps);
  SET_LABEL(outHist, ps);
  if (outDump.empty())
    fPtr = &cout;
  else
    fPtr = new ofstream(outDump.c_str());
}

void TestBPHRecoDecay::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("patMuonLabel", "");
  desc.add<string>("ccCandsLabel", "");
  desc.add<string>("pfCandsLabel", "");
  desc.add<string>("pcCandsLabel", "");
  desc.add<string>("gpCandsLabel", "");
  desc.add<string>("outDump", "dump.txt");
  desc.add<string>("outHist", "hist.root");
  descriptions.add("testBPHRecoDecay", desc);
  return;
}

void TestBPHRecoDecay::beginJob() {
  *fPtr << "TestBPHRecoDecay::beginJob" << endl;
  createHisto("massJPsi", 50, 2.85, 3.35);     // JPsi mass
  createHisto("mcstJPsi", 50, 2.85, 3.35);     // JPsi mass, with constraint
  createHisto("massPhi", 50, 0.995, 1.045);    // Phi  mass
  createHisto("massBu", 20, 5.0, 5.5);         // Bu   mass
  createHisto("mcstBu", 20, 5.0, 5.5);         // Bu   mass, with constraint
  createHisto("massBs", 20, 5.1, 5.6);         // Bs   mass
  createHisto("mcstBs", 20, 5.1, 5.6);         // Bs   mass, with constraint
  createHisto("massBsPhi", 50, 0.995, 1.045);  // Phi  mass in Bs decay
  return;
}

void TestBPHRecoDecay::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  ostream& outF = *fPtr;
  outF << "--------- event " << ev.id().run() << " / " << ev.id().event() << " ---------" << endl;

  // create a "wrapper" for EventSetup
  BPHEventSetupWrapper ew(es, BPHRecoCandidate::transientTrackBuilder, &ttBToken);

  // get object collections
  // collections are got through "BPHTokenWrapper" interface to allow
  // uniform access in different CMSSW versions

  int nrc = 0;

  // get reco::PFCandidate collection (in full AOD )
  edm::Handle<vector<reco::PFCandidate> > pfCands;
  if (usePF) {
    pfCandsToken.get(ev, pfCands);
    nrc = pfCands->size();
    if (pfCands.isValid())
      outF << nrc << " pfCands found" << endl;
    else
      outF << "no pfCands" << endl;
  }

  // get pat::PackedCandidate collection (in MiniAOD)
  // pat::PackedCandidate is not defined in CMSSW_5XY, so a
  // typedef (BPHTrackReference::candidate) is used, actually referring
  // to pat::PackedCandidate only for CMSSW versions where it's defined
  edm::Handle<vector<BPHTrackReference::candidate> > pcCands;
  if (usePC) {
    pcCandsToken.get(ev, pcCands);
    nrc = pcCands->size();
    if (pcCands.isValid())
      outF << nrc << " pcCands found" << endl;
    else
      outF << "no pcCands" << endl;
  }

  // get pat::GenericParticle collection (in skimmed data)
  edm::Handle<vector<pat::GenericParticle> > gpCands;
  if (useGP) {
    gpCandsToken.get(ev, gpCands);
    nrc = gpCands->size();
    if (gpCands.isValid())
      outF << nrc << " gpCands found" << endl;
    else
      outF << "no gpCands" << endl;
  }

  // get pat::Muon collection (in full AOD and MiniAOD)
  edm::Handle<pat::MuonCollection> patMuon;
  if (usePM) {
    patMuonToken.get(ev, patMuon);
    int n = patMuon->size();
    if (patMuon.isValid())
      outF << n << " muons found" << endl;
    else
      outF << "no muons" << endl;
  }

  // get muons from pat::CompositeCandidate objects describing onia;
  // muons from all composite objects are copied to an unique std::vector
  vector<const reco::Candidate*> muDaugs;
  set<const pat::Muon*> muonSet;
  if (useCC) {
    edm::Handle<vector<pat::CompositeCandidate> > ccCands;
    ccCandsToken.get(ev, ccCands);
    int n = ccCands->size();
    if (ccCands.isValid())
      outF << n << " ccCands found" << endl;
    else
      outF << "no ccCands" << endl;
    muDaugs.clear();
    muDaugs.reserve(n);
    muonSet.clear();
    set<const pat::Muon*>::const_iterator iter;
    set<const pat::Muon*>::const_iterator iend;
    int i;
    for (i = 0; i < n; ++i) {
      const pat::CompositeCandidate& cc = ccCands->at(i);
      int j;
      int m = cc.numberOfDaughters();
      for (j = 0; j < m; ++j) {
        const reco::Candidate* dp = cc.daughter(j);
        const pat::Muon* mp = dynamic_cast<const pat::Muon*>(dp);
        iter = muonSet.begin();
        iend = muonSet.end();
        bool add = (mp != nullptr) && (muonSet.find(mp) == iend);
        while (add && (iter != iend)) {
          if (BPHRecoBuilder::sameTrack(mp, *iter++, 1.0e-5))
            add = false;
        }
        if (add)
          muonSet.insert(mp);
      }
    }
    iter = muonSet.begin();
    iend = muonSet.end();
    while (iter != iend)
      muDaugs.push_back(*iter++);
    if ((n = muDaugs.size()))
      outF << n << " muons found" << endl;
  }

  //
  // starting objects selection
  //

  // muon selection by charge
  class MuonChargeSelect : public BPHRecoSelect {
  public:
    MuonChargeSelect(int c) : charge(c) {}
    ~MuonChargeSelect() override = default;
    bool accept(const reco::Candidate& cand) const override {
      const pat::Muon* p = dynamic_cast<const pat::Muon*>(&cand);
      if (p == nullptr)
        return false;
      return ((charge * cand.charge()) > 0);
    }

  private:
    int charge;
  };

  // muon selection by Pt
  class MuonPtSelect : public BPHRecoSelect {
  public:
    MuonPtSelect(float pt) : ptCut(pt) {}
    ~MuonPtSelect() override = default;
    bool accept(const reco::Candidate& cand) const override {
      const pat::Muon* p = dynamic_cast<const pat::Muon*>(&cand);
      if (p == nullptr)
        return false;
      return (p->p4().pt() > ptCut);
    }

  private:
    float ptCut;
  };

  // muon selection by eta
  class MuonEtaSelect : public BPHRecoSelect {
  public:
    MuonEtaSelect(float eta) : etaCut(eta) {}
    ~MuonEtaSelect() override = default;
    bool accept(const reco::Candidate& cand) const override {
      const pat::Muon* p = dynamic_cast<const pat::Muon*>(&cand);
      if (p == nullptr)
        return false;
      return (fabs(p->p4().eta()) < etaCut);
    }

  private:
    float etaCut;
  };

  // kaon selection by charge
  class KaonChargeSelect : public BPHRecoSelect {
  public:
    KaonChargeSelect(int c) : charge(c) {}
    ~KaonChargeSelect() override = default;
    bool accept(const reco::Candidate& cand) const override { return ((charge * cand.charge()) > 0); }

  private:
    int charge;
  };

  class KaonNeutralVeto : public BPHRecoSelect {
  public:
    KaonNeutralVeto() {}
    ~KaonNeutralVeto() override = default;
    bool accept(const reco::Candidate& cand) const override { return lround(fabs(cand.charge())); }
  };

  // kaon selection by Pt
  class KaonPtSelect : public BPHRecoSelect {
  public:
    KaonPtSelect(float pt) : ptCut(pt) {}
    ~KaonPtSelect() override = default;
    bool accept(const reco::Candidate& cand) const override { return (cand.p4().pt() > ptCut); }

  private:
    float ptCut;
  };

  // kaon selection by eta
  class KaonEtaSelect : public BPHRecoSelect {
  public:
    KaonEtaSelect(float eta) : etaCut(eta) {}
    ~KaonEtaSelect() override = default;
    bool accept(const reco::Candidate& cand) const override { return (fabs(cand.p4().eta()) < etaCut); }

  private:
    float etaCut;
  };

  //
  // reconstructed object selection
  //

  // selection by mass
  class MassSelect : public BPHMomentumSelect {
  public:
    MassSelect(double minMass, double maxMass) : mMin(minMass), mMax(maxMass) {}
    ~MassSelect() override = default;
    bool accept(const BPHDecayMomentum& cand) const override {
      double mass = cand.composite().mass();
      return ((mass > mMin) && (mass < mMax));
    }

  private:
    double mMin;
    double mMax;
  };

  // selection by chi^2
  class Chi2Select : public BPHVertexSelect {
  public:
    Chi2Select(double minProb) : mProb(minProb) {}
    ~Chi2Select() override = default;
    bool accept(const BPHDecayVertex& cand) const override {
      const reco::Vertex& v = cand.vertex();
      if (v.isFake())
        return false;
      if (!v.isValid())
        return false;
      return (TMath::Prob(v.chi2(), lround(v.ndof())) > mProb);
    }

  private:
    double mProb;
  };

  // build and dump JPsi

  outF << "build and dump JPsi" << endl;
  MuonPtSelect muPt(4.0);
  MuonEtaSelect muEta(2.1);
  string muPos = "MuPos";
  string muNeg = "MuNeg";
  BPHRecoBuilder bJPsi(ew);
  if (usePM) {
    bJPsi.add(muPos, BPHRecoBuilder::createCollection(patMuon, "cfmig"), 0.105658);
    bJPsi.add(muNeg, BPHRecoBuilder::createCollection(patMuon, "cfmig"), 0.105658);
  } else if (useCC) {
    bJPsi.add(muPos, BPHRecoBuilder::createCollection(muDaugs, "cfmig"), 0.105658);
    bJPsi.add(muNeg, BPHRecoBuilder::createCollection(muDaugs, "cfmig"), 0.105658);
  }
  bJPsi.filter(muPos, muPt);
  bJPsi.filter(muNeg, muPt);
  bJPsi.filter(muPos, muEta);
  bJPsi.filter(muNeg, muEta);

  MassSelect massJPsi(2.5, 3.7);
  Chi2Select chi2Valid(0.0);
  bJPsi.filter(massJPsi);
  bJPsi.filter(chi2Valid);
  vector<BPHPlusMinusConstCandPtr> lJPsi = BPHPlusMinusCandidate::build(bJPsi, muPos, muNeg, 3.096916, 0.00004);
  //  //  BPHPlusMinusCandidate::build function has embedded charge selection
  //  //  alternatively simple BPHRecoCandidate::build function can be used
  //  //  as in the following
  //  MuonChargeSelect mqPos( +1 );
  //  MuonChargeSelect mqNeg( +1 );
  //  bJPsi.filter( nPos, mqPos );
  //  bJPsi.filter( nNeg, mqNeg );
  //  vector<BPHRecoConstCandPtr> lJPsi = BPHRecoCandidate::build( bJPsi,
  //                                      3.096916, 0.00004 );
  int iJPsi;
  int nJPsi = lJPsi.size();
  outF << nJPsi << " JPsi cand found" << endl;
  for (iJPsi = 0; iJPsi < nJPsi; ++iJPsi)
    dumpRecoCand("JPsi", lJPsi[iJPsi].get());

  // build and dump Phi

  outF << "build and dump Phi" << endl;
  BPHRecoBuilder bPhi(ew);
  KaonChargeSelect tkPos(+1);
  KaonChargeSelect tkNeg(-1);
  KaonPtSelect tkPt(0.7);
  string kPos = "KPos";
  string kNeg = "KNeg";
  if (usePF) {
    bPhi.add(kPos, BPHRecoBuilder::createCollection(pfCands), 0.493677);
    bPhi.add(kNeg, BPHRecoBuilder::createCollection(pfCands), 0.493677);
  } else if (usePC) {
    bPhi.add(kPos, BPHRecoBuilder::createCollection(pcCands), 0.493677);
    bPhi.add(kNeg, BPHRecoBuilder::createCollection(pcCands), 0.493677);
  } else if (useGP) {
    bPhi.add(kPos, BPHRecoBuilder::createCollection(gpCands), 0.493677);
    bPhi.add(kNeg, BPHRecoBuilder::createCollection(gpCands), 0.493677);
  }
  bPhi.filter(kPos, tkPos);
  bPhi.filter(kNeg, tkNeg);
  bPhi.filter(kPos, tkPt);
  bPhi.filter(kNeg, tkPt);

  MassSelect massPhi(1.00, 1.04);
  bPhi.filter(massPhi);
  vector<BPHRecoConstCandPtr> lPhi = BPHRecoCandidate::build(bPhi);
  //  //  BPHRecoCandidate::build function requires explicit charge selection
  //  //  alternatively BPHPlusMinusCandidate::build function can be used
  //  //  as in the following
  //  //  (filter functions with tkPos and tkNeg can be dropped)
  //  vector<const BPHPlusMinusConstCandPtr> lPhi =
  //               BPHPlusMinusCandidate::build( bPhi, kPos, kNeg );
  int iPhi;
  int nPhi = lPhi.size();
  outF << nPhi << " Phi cand found" << endl;
  for (iPhi = 0; iPhi < nPhi; ++iPhi)
    dumpRecoCand("Phi", lPhi[iPhi].get());

  // build and dump Bs

  if (nJPsi && nPhi) {
    outF << "build and dump Bs" << endl;
    BPHRecoBuilder bBs(ew);
    bBs.setMinPDiffererence(1.0e-5);
    bBs.add("JPsi", lJPsi);
    bBs.add("Phi", lPhi);
    MassSelect mJPsi(2.946916, 3.246916);
    MassSelect mPhi(1.009461, 1.029461);
    bBs.filter("JPsi", mJPsi);
    bBs.filter("Phi", mPhi);
    Chi2Select chi2Bs(0.02);
    bBs.filter(chi2Bs);
    vector<BPHRecoConstCandPtr> lBs = BPHRecoCandidate::build(bBs);
    int iBs;
    int nBs = lBs.size();
    outF << nBs << " Bs cand found" << endl;
    // apply kinematic fit
    for (iBs = 0; iBs < nBs; ++iBs) {
      // get candidate and cast constness away
      const BPHRecoCandidate* cptr = lBs[iBs].get();
      cptr->kinematicTree("JPsi", 3.096916, 0.000040);
    }
    for (iBs = 0; iBs < nBs; ++iBs)
      dumpRecoCand("Bs", lBs[iBs].get());
  }

  // build and dump Bu

  if (nJPsi && nrc) {
    outF << "build and dump Bu" << endl;
    BPHRecoBuilder bBu(ew);
    bBu.setMinPDiffererence(1.0e-5);
    bBu.add("JPsi", lJPsi);
    if (usePF) {
      bBu.add("Kaon", BPHRecoBuilder::createCollection(pfCands), 0.493677);
    } else if (usePC) {
      bBu.add("Kaon", BPHRecoBuilder::createCollection(pcCands), 0.493677);
    } else if (useGP) {
      bBu.add("Kaon", BPHRecoBuilder::createCollection(gpCands), 0.493677);
    }
    MassSelect mJPsi(2.946916, 3.246916);
    KaonNeutralVeto knv;
    bBu.filter("JPsi", mJPsi);
    bBu.filter("Kaon", tkPt);
    bBu.filter("Kaon", knv);
    Chi2Select chi2Bu(0.02);
    bBu.filter(chi2Bu);
    vector<BPHRecoConstCandPtr> lBu = BPHRecoCandidate::build(bBu);
    int iBu;
    int nBu = lBu.size();
    outF << nBu << " Bu cand found" << endl;
    // apply kinematic fit
    for (iBu = 0; iBu < nBu; ++iBu) {
      // get candidate and cast constness away
      const BPHRecoCandidate* cptr = lBu[iBu].get();
      cptr->kinematicTree("JPsi", 3.096916, 0.000040);
    }
    for (iBu = 0; iBu < nBu; ++iBu)
      dumpRecoCand("Bu", lBu[iBu].get());
  }

  return;
}

void TestBPHRecoDecay::endJob() {
  *fPtr << "TestBPHRecoDecay::endJob" << endl;
  TDirectory* currentDir = gDirectory;
  TFile file(outHist.c_str(), "RECREATE");
  map<string, TH1F*>::iterator iter = histoMap.begin();
  map<string, TH1F*>::iterator iend = histoMap.end();
  while (iter != iend)
    iter++->second->Write();
  currentDir->cd();
  return;
}

void TestBPHRecoDecay::dumpRecoCand(const string& name, const BPHRecoCandidate* cand) {
  fillHisto(name, cand);

  ostream& outF = *fPtr;

  static string cType = " cowboy";
  static string sType = " sailor";
  static string dType = "";
  string* type;
  const BPHPlusMinusCandidate* pmCand = dynamic_cast<const BPHPlusMinusCandidate*>(cand);
  if (pmCand != nullptr) {
    if (pmCand->isCowboy())
      type = &cType;
    else
      type = &sType;
  } else
    type = &dType;

  outF << "****** " << name << "   cand mass: " << cand->composite().mass() << " momentum " << cand->composite().px()
       << " " << cand->composite().py() << " " << cand->composite().pz() << *type << endl;

  bool validFit = cand->isValidFit();
  const RefCountedKinematicTree kt = cand->kinematicTree();
  const RefCountedKinematicParticle kp = cand->currentParticle();
  if (validFit) {
    outF << "****** " << name << " constr mass: " << cand->p4().mass() << " momentum " << cand->p4().px() << " "
         << cand->p4().py() << " " << cand->p4().pz() << endl;
  }

  const reco::Vertex& vx = cand->vertex();
  const reco::Vertex::Point& vp = vx.position();
  double chi2 = vx.chi2();
  int ndof = lround(vx.ndof());
  double prob = TMath::Prob(chi2, ndof);
  string tdca = "";
  if (pmCand != nullptr) {
    stringstream sstr;
    sstr << " - " << pmCand->cAppInRPhi().distance();
    tdca = sstr.str();
  }
  outF << "****** " << name << " vertex: " << vx.isFake() << " " << vx.isValid() << " - " << chi2 << " " << ndof << " "
       << prob << " - " << vp.X() << " " << vp.Y() << " " << vp.Z() << tdca << endl;

  const vector<string>& dl = cand->daugNames();
  int i;
  int n = dl.size();
  for (i = 0; i < n; ++i) {
    const string& name = dl[i];
    const reco::Candidate* dp = cand->getDaug(name);
    GlobalPoint gp(vp.X(), vp.Y(), vp.Z());
    GlobalVector dm(0.0, 0.0, 0.0);
    const reco::TransientTrack* tt = cand->getTransientTrack(dp);
    if (tt != nullptr) {
      TrajectoryStateClosestToPoint tscp = tt->trajectoryStateClosestToPoint(gp);
      dm = tscp.momentum();
      //      TrajectoryStateOnSurface tsos = tt->stateOnSurface( gp );
      //      GlobalVector gv = tsos.globalMomentum();
    }
    outF << "daughter " << i << " " << name << " " << (dp->charge() > 0 ? '+' : '-') << " momentum: " << dp->px() << " "
         << dp->py() << " " << dp->pz() << " - at vertex: " << dm.x() << " " << dm.y() << " " << dm.z() << endl;
  }
  const vector<string>& dc = cand->compNames();
  int j;
  int m = dc.size();
  for (j = 0; j < m; ++j) {
    const string& name = dc[j];
    const BPHRecoCandidate* dp = cand->getComp(name).get();
    outF << "composite daughter " << j << " " << name << " momentum: " << dp->composite().px() << " "
         << dp->composite().py() << " " << dp->composite().pz() << endl;
  }

  if (validFit) {
    const RefCountedKinematicVertex kv = cand->currentDecayVertex();
    GlobalPoint gp = kv->position();
    outF << "   kin fit vertex: " << gp.x() << " " << gp.y() << " " << gp.z() << endl;
    vector<RefCountedKinematicParticle> dk = kt->finalStateParticles();
    int k;
    int l = dk.size();
    for (k = 0; k < l; ++k) {
      const reco::TransientTrack& tt = dk[k]->refittedTransientTrack();
      TrajectoryStateClosestToPoint tscp = tt.trajectoryStateClosestToPoint(gp);
      GlobalVector dm = tscp.momentum();
      //    TrajectoryStateOnSurface tsos = tt.stateOnSurface( gp );
      //    GlobalVector dm = tsos.globalMomentum();
      outF << "daughter " << k << " refitted: " << dm.x() << " " << dm.y() << " " << dm.z() << endl;
    }
  }

  return;
}

void TestBPHRecoDecay::fillHisto(const string& name, const BPHRecoCandidate* cand) {
  string mass = "mass";
  string mcst = "mcst";
  fillHisto(mass + name, cand->composite().mass());
  if (cand->isValidFit())
    fillHisto(mcst + name, cand->p4().mass());

  const vector<string>& dc = cand->compNames();
  int i;
  int n = dc.size();
  for (i = 0; i < n; ++i) {
    const string& daug = dc[i];
    const BPHRecoCandidate* dptr = cand->getComp(daug).get();
    fillHisto(mass + name + daug, dptr->composite().mass());
  }
  return;
}

void TestBPHRecoDecay::fillHisto(const string& name, float x) {
  map<string, TH1F*>::iterator iter = histoMap.find(name);
  map<string, TH1F*>::iterator iend = histoMap.end();
  if (iter == iend)
    return;
  iter->second->Fill(x);
  return;
}

void TestBPHRecoDecay::createHisto(const string& name, int nbin, float hmin, float hmax) {
  histoMap[name] = new TH1F(name.c_str(), name.c_str(), nbin, hmin, hmax);
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TestBPHRecoDecay);
