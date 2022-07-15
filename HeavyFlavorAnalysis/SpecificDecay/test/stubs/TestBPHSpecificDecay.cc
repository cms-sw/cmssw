#include "HeavyFlavorAnalysis/SpecificDecay/test/stubs/TestBPHSpecificDecay.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonPtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleNeutralVeto.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"

#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHOniaToMuMuBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHKx0ToKPiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHPhiToKKBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBuToJPsiKBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBsToJPsiPhiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBdToJPsiKxBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "TH1.h"
#include "TFile.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

#define SET_LABEL(NAME, PSET) (NAME = PSET.getParameter<string>(#NAME))
// SET_LABEL(xyz,ps);
// is equivalent to
// xyz = ps.getParameter<string>( "xyx" )

TestBPHSpecificDecay::TestBPHSpecificDecay(const edm::ParameterSet& ps) {
  usePM = (!SET_LABEL(patMuonLabel, ps).empty());
  useCC = (!SET_LABEL(ccCandsLabel, ps).empty());
  usePF = (!SET_LABEL(pfCandsLabel, ps).empty());
  usePC = (!SET_LABEL(pcCandsLabel, ps).empty());
  useGP = (!SET_LABEL(gpCandsLabel, ps).empty());

  if (usePM)
    consume<pat::MuonCollection>(patMuonToken, patMuonLabel);
  if (useCC)
    consume<vector<pat::CompositeCandidate>>(ccCandsToken, ccCandsLabel);
  if (usePF)
    consume<vector<reco::PFCandidate>>(pfCandsToken, pfCandsLabel);
  if (usePC)
    consume<vector<BPHTrackReference::candidate>>(pcCandsToken, pcCandsLabel);
  if (useGP)
    consume<vector<pat::GenericParticle>>(gpCandsToken, gpCandsLabel);

  SET_LABEL(outDump, ps);
  SET_LABEL(outHist, ps);
  if (outDump.empty())
    fPtr = &cout;
  else
    fPtr = new ofstream(outDump.c_str());
}

void TestBPHSpecificDecay::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("patMuonLabel", "");
  desc.add<string>("ccCandsLabel", "");
  desc.add<string>("pfCandsLabel", "");
  desc.add<string>("pcCandsLabel", "");
  desc.add<string>("gpCandsLabel", "");
  desc.add<string>("outDump", "dump.txt");
  desc.add<string>("outHist", "hist.root");
  descriptions.add("testBPHSpecificDecay", desc);
  return;
}

void TestBPHSpecificDecay::beginJob() {
  *fPtr << "TestBPHSpecificDecay::beginJob" << endl;
  createHisto("massJPsi", 60, 2.95, 3.25);   // JPsi mass
  createHisto("mcstJPsi", 60, 2.95, 3.25);   // JPsi mass, with constraint
  createHisto("massKx0", 50, 0.80, 1.05);    // Kx0  mass
  createHisto("massPhi", 40, 1.01, 1.03);    // Phi  mass
  createHisto("massBu", 20, 5.00, 5.50);     // Bu   mass
  createHisto("mcstBu", 20, 5.00, 5.50);     // Bu   mass, with constraint
  createHisto("massBd", 20, 5.00, 5.50);     // Bd   mass
  createHisto("mcstBd", 20, 5.00, 5.50);     // Bd   mass, with constraint
  createHisto("massBs", 20, 5.10, 5.60);     // Bs   mass
  createHisto("mcstBs", 20, 5.10, 5.60);     // Bs   mass, with constraint
  createHisto("massBsPhi", 50, 1.01, 1.03);  // Phi  mass in Bs decay
  createHisto("massBdKx0", 50, 0.80, 1.05);  // Kx0  mass in Bd decay

  createHisto("massFull", 200, 2.00, 12.00);  // Full mass
  createHisto("massFsel", 200, 2.00, 12.00);  // Full mass
  createHisto("massPhi", 70, 0.85, 1.20);     // Psi1 mass
  createHisto("massPsi1", 60, 2.95, 3.25);    // Psi1 mass
  createHisto("massPsi2", 50, 3.55, 3.80);    // Psi2 mass
  createHisto("massUps1", 130, 9.10, 9.75);   // Ups1 mass
  createHisto("massUps2", 90, 9.75, 10.20);   // Ups2 mass
  createHisto("massUps3", 80, 10.20, 10.60);  // Ups3 mass

  return;
}

void TestBPHSpecificDecay::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  ostream& outF = *fPtr;
  outF << "--------- event " << ev.id().run() << " / " << ev.id().event() << " ---------" << endl;

  BPHEventSetupWrapper ew(es, BPHRecoCandidate::transientTrackBuilder, &ttBToken);

  // get object collections
  // collections are got through "BPHTokenWrapper" interface to allow
  // uniform access in different CMSSW versions

  int nrc = 0;

  // get reco::PFCandidate collection (in full AOD )
  edm::Handle<vector<reco::PFCandidate>> pfCands;
  if (usePF) {
    pfCandsToken.get(ev, pfCands);
    nrc = pfCands->size();
  }

  // get pat::PackedCandidate collection (in MiniAOD)
  // pat::PackedCandidate is not defined in CMSSW_5XY, so a
  // typedef (BPHTrackReference::candidate) is used, actually referring
  // to pat::PackedCandidate only for CMSSW versions where it's defined
  edm::Handle<vector<BPHTrackReference::candidate>> pcCands;
  if (usePC) {
    pcCandsToken.get(ev, pcCands);
    nrc = pcCands->size();
  }

  // get pat::GenericParticle collection (in skimmed data)
  edm::Handle<vector<pat::GenericParticle>> gpCands;
  if (useGP) {
    gpCandsToken.get(ev, gpCands);
    nrc = gpCands->size();
  }

  // get pat::Muon collection (in full AOD and MiniAOD)
  edm::Handle<pat::MuonCollection> patMuon;
  if (usePM) {
    patMuonToken.get(ev, patMuon);
  }

  // get muons from pat::CompositeCandidate objects describing onia;
  // muons from all composite objects are copied to an unique std::vector
  vector<const reco::Candidate*> muDaugs;
  set<const pat::Muon*> muonSet;
  if (useCC) {
    edm::Handle<vector<pat::CompositeCandidate>> ccCands;
    ccCandsToken.get(ev, ccCands);
    int n = ccCands->size();
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
  }

  // reconstruct resonances

  outF << "build and dump full onia" << endl;
  BPHOniaToMuMuBuilder* onia = nullptr;
  if (usePM)
    onia = new BPHOniaToMuMuBuilder(
        ew, BPHRecoBuilder::createCollection(patMuon, "cfmig"), BPHRecoBuilder::createCollection(patMuon, "cfmig"));
  else if (useCC)
    onia = new BPHOniaToMuMuBuilder(
        ew, BPHRecoBuilder::createCollection(muDaugs, "cfmig"), BPHRecoBuilder::createCollection(muDaugs, "cfmig"));

  vector<BPHPlusMinusConstCandPtr> lFull = onia->build();
  int iFull;
  int nFull = lFull.size();
  outF << nFull << " onia cand found" << endl;
  for (iFull = 0; iFull < nFull; ++iFull)
    dumpRecoCand("Full", lFull[iFull].get());

  BPHMuonPtSelect ptSel1(3.0);
  BPHMuonPtSelect ptSel2(3.5);
  BPHMuonPtSelect ptSel3(4.5);
  BPHMuonEtaSelect etaSel1(1.6);
  BPHMuonEtaSelect etaSel2(1.4);
  BPHMuonEtaSelect etaSel3(1.2);
  BPHMultiSelect<BPHSlimSelect<BPHRecoSelect>> select1(BPHSelectOperation::and_mode);
  select1.include(ptSel1);
  select1.include(etaSel1);
  select1.include(etaSel2, false);
  BPHMultiSelect<BPHSlimSelect<BPHRecoSelect>> select2(BPHSelectOperation::and_mode);
  select2.include(ptSel2);
  select2.include(etaSel2);
  select2.include(etaSel3, false);
  BPHMultiSelect<BPHSlimSelect<BPHRecoSelect>> select3(BPHSelectOperation::and_mode);
  select3.include(ptSel3);
  select3.include(etaSel3);
  BPHMultiSelect<BPHSlimSelect<BPHRecoSelect>> muoSel(BPHSelectOperation::or_mode);
  muoSel.include(select1);
  muoSel.include(select2);
  muoSel.include(select3);

  BPHMassSelect massPh(0.85, 1.20);
  BPHMassSelect massP1(2.95, 3.25);
  BPHMassSelect massP2(3.55, 3.80);
  BPHMassSelect massU1(9.10, 9.75);
  BPHMassSelect massU2(9.75, 10.20);
  BPHMassSelect massU3(10.20, 10.60);

  outF << "extract and dump JPsi" << endl;
  vector<BPHPlusMinusConstCandPtr> lJPsi = onia->getList(BPHOniaToMuMuBuilder::Psi1);
  int iJPsi;
  int nJPsi = lJPsi.size();
  outF << nJPsi << " JPsi cand found" << endl;
  for (iJPsi = 0; iJPsi < nJPsi; ++iJPsi)
    dumpRecoCand("JPsi", lJPsi[iJPsi].get());

  outF << "extract and dump specific onia" << endl;
  vector<BPHPlusMinusConstCandPtr> lPmm = onia->getList(BPHOniaToMuMuBuilder::Phi, &muoSel, &massPh);
  vector<BPHPlusMinusConstCandPtr> lPsi1 = onia->getList(BPHOniaToMuMuBuilder::Psi1, &muoSel, &massP1);
  vector<BPHPlusMinusConstCandPtr> lPsi2 = onia->getList(BPHOniaToMuMuBuilder::Psi2, &muoSel, &massP2);
  vector<BPHPlusMinusConstCandPtr> lUps1 = onia->getList(BPHOniaToMuMuBuilder::Ups1, &muoSel, &massU1);
  vector<BPHPlusMinusConstCandPtr> lUps2 = onia->getList(BPHOniaToMuMuBuilder::Ups2, &muoSel, &massU2);
  vector<BPHPlusMinusConstCandPtr> lUps3 = onia->getList(BPHOniaToMuMuBuilder::Ups3, &muoSel, &massU3);
  int iPhi;
  int nPhi = lPmm.size();
  outF << nPhi << " PhiMuMu cand found" << endl;
  for (iPhi = 0; iPhi < nPhi; ++iPhi)
    dumpRecoCand("PhiMuMu", lPmm[iPhi].get());
  int iPsi1;
  int nPsi1 = lPsi1.size();
  outF << nPsi1 << " Psi1 cand found" << endl;
  for (iPsi1 = 0; iPsi1 < nPsi1; ++iPsi1)
    dumpRecoCand("Psi1", lPsi1[iPsi1].get());
  int iPsi2;
  int nPsi2 = lPsi2.size();
  outF << nPsi2 << " Psi2 cand found" << endl;
  for (iPsi2 = 0; iPsi2 < nPsi2; ++iPsi2)
    dumpRecoCand("Psi2", lPsi2[iPsi2].get());
  int iUps1;
  int nUps1 = lUps1.size();
  outF << nUps1 << " Ups1 cand found" << endl;
  for (iUps1 = 0; iUps1 < nUps1; ++iUps1)
    dumpRecoCand("Ups1", lUps1[iUps1].get());
  int iUps2;
  int nUps2 = lUps2.size();
  outF << nUps2 << " Ups2 cand found" << endl;
  for (iUps2 = 0; iUps2 < nUps2; ++iUps2)
    dumpRecoCand("Ups2", lUps2[iUps2].get());
  int iUps3;
  int nUps3 = lUps3.size();
  outF << nUps3 << " Ups3 cand found" << endl;
  for (iUps3 = 0; iUps3 < nUps3; ++iUps3)
    dumpRecoCand("Ups3", lUps3[iUps3].get());
  delete onia;

  if (!nPsi1)
    return;
  if (!nrc)
    return;

  BPHParticlePtSelect tkPt(0.7);
  BPHMassSelect mJPsi(3.00, 3.17);
  BPHChi2Select chi2Bs(0.02);

  // build and dump Bu

  outF << "build and dump Bu" << endl;
  BPHBuToJPsiKBuilder* bu = nullptr;
  if (usePF)
    bu = new BPHBuToJPsiKBuilder(ew, lJPsi, BPHRecoBuilder::createCollection(pfCands));
  else if (usePC)
    bu = new BPHBuToJPsiKBuilder(ew, lJPsi, BPHRecoBuilder::createCollection(pcCands));
  else if (useGP)
    bu = new BPHBuToJPsiKBuilder(ew, lJPsi, BPHRecoBuilder::createCollection(gpCands));

  vector<BPHRecoConstCandPtr> lBu = bu->build();

  int iBu;
  int nBu = lBu.size();
  outF << nBu << " Bu cand found" << endl;
  for (iBu = 0; iBu < nBu; ++iBu)
    dumpRecoCand("Bu", lBu[iBu].get());
  // the following is an example of decay reconstruction starting from
  // specific reco::Candidates
  // here the final decay products are taken from already reconstructed B+,
  // so there's no physical sense in the operation
  for (iBu = 0; iBu < nBu; ++iBu) {
    const BPHRecoCandidate* bu = lBu[iBu].get();
    const reco::Candidate* mPos = bu->originalReco(bu->getDaug("JPsi/MuPos"));
    const reco::Candidate* mNeg = bu->originalReco(bu->getDaug("JPsi/MuNeg"));
    const reco::Candidate* kaon = bu->originalReco(bu->getDaug("Kaon"));
    BPHRecoCandidatePtr njp = BPHPlusMinusCandidateWrap::create(&ew);
    njp->add("MuPos", mPos, BPHParticleMasses::muonMass, BPHParticleMasses::muonMSigma);
    njp->add("MuNeg", mNeg, BPHParticleMasses::muonMass, BPHParticleMasses::muonMSigma);
    BPHRecoCandidate nbu(&ew);
    nbu.add("JPsi", njp);
    nbu.add("Kaon", kaon, BPHParticleMasses::kaonMass, BPHParticleMasses::kaonMSigma);
    nbu.kinematicTree("JPsi", BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth);
    dumpRecoCand("nBu", &nbu);
  }

  // build and dump Kx0

  BPHKx0ToKPiBuilder* kx0 = nullptr;
  if (usePF)
    kx0 = new BPHKx0ToKPiBuilder(
        ew, BPHRecoBuilder::createCollection(pfCands), BPHRecoBuilder::createCollection(pfCands));
  else if (usePC)
    kx0 = new BPHKx0ToKPiBuilder(
        ew, BPHRecoBuilder::createCollection(pcCands), BPHRecoBuilder::createCollection(pcCands));
  else if (useGP)
    kx0 = new BPHKx0ToKPiBuilder(
        ew, BPHRecoBuilder::createCollection(gpCands), BPHRecoBuilder::createCollection(gpCands));

  vector<BPHPlusMinusConstCandPtr> lKx0 = kx0->build();

  int iKx0;
  int nKx0 = lKx0.size();
  outF << nKx0 << " Kx0 cand found" << endl;
  for (iKx0 = 0; iKx0 < nKx0; ++iKx0)
    dumpRecoCand("Kx0", lKx0[iKx0].get());

  delete kx0;

  // build and dump Bd

  outF << "build and dump Bd" << endl;
  if (nKx0) {
    BPHBdToJPsiKxBuilder* bd = new BPHBdToJPsiKxBuilder(ew, lJPsi, lKx0);
    vector<BPHRecoConstCandPtr> lBd = bd->build();
    int iBd;
    int nBd = lBd.size();
    outF << nBd << " Bd cand found" << endl;
    for (iBd = 0; iBd < nBd; ++iBd)
      dumpRecoCand("Bd", lBd[iBd].get());
  }

  // build and dump Phi

  BPHPhiToKKBuilder* phi = nullptr;
  if (usePF)
    phi =
        new BPHPhiToKKBuilder(ew, BPHRecoBuilder::createCollection(pfCands), BPHRecoBuilder::createCollection(pfCands));
  else if (usePC)
    phi =
        new BPHPhiToKKBuilder(ew, BPHRecoBuilder::createCollection(pcCands), BPHRecoBuilder::createCollection(pcCands));
  else if (useGP)
    phi =
        new BPHPhiToKKBuilder(ew, BPHRecoBuilder::createCollection(gpCands), BPHRecoBuilder::createCollection(gpCands));

  vector<BPHPlusMinusConstCandPtr> lPkk = phi->build();

  int iPkk;
  int nPkk = lPkk.size();
  outF << nPkk << " PhiKK cand found" << endl;
  for (iPkk = 0; iPkk < nPkk; ++iPkk)
    dumpRecoCand("PhiKK", lPkk[iPkk].get());

  delete phi;

  // build and dump Bs

  outF << "build and dump Bs" << endl;
  if (nPkk) {
    BPHBsToJPsiPhiBuilder* bs = new BPHBsToJPsiPhiBuilder(ew, lJPsi, lPkk);
    vector<BPHRecoConstCandPtr> lBs = bs->build();
    int iBs;
    int nBs = lBs.size();
    outF << nBs << " Bs cand found" << endl;
    for (iBs = 0; iBs < nBs; ++iBs)
      dumpRecoCand("Bs", lBs[iBs].get());
  }

  return;
}

void TestBPHSpecificDecay::endJob() {
  *fPtr << "TestBPHSpecificDecay::endJob" << endl;
  TDirectory* currentDir = gDirectory;
  TFile file(outHist.c_str(), "RECREATE");
  map<string, TH1F*>::iterator iter = histoMap.begin();
  map<string, TH1F*>::iterator iend = histoMap.end();
  while (iter != iend)
    iter++->second->Write();
  currentDir->cd();
  return;
}

void TestBPHSpecificDecay::dumpRecoCand(const string& name, const BPHRecoCandidate* cand) {
  fillHisto(name, cand);
  if ((name == "PhiMuMu") || (name == "Psi1") || (name == "Psi2") || (name == "Ups1") || (name == "Ups2") ||
      (name == "Ups3"))
    fillHisto("Fsel", cand);

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

void TestBPHSpecificDecay::fillHisto(const string& name, const BPHRecoCandidate* cand) {
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

void TestBPHSpecificDecay::fillHisto(const string& name, float x) {
  map<string, TH1F*>::iterator iter = histoMap.find(name);
  map<string, TH1F*>::iterator iend = histoMap.end();
  if (iter == iend)
    return;
  iter->second->Fill(x);
  return;
}

void TestBPHSpecificDecay::createHisto(const string& name, int nbin, float hmin, float hmax) {
  histoMap[name] = new TH1F(name.c_str(), name.c_str(), nbin, hmin, hmax);
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TestBPHSpecificDecay);
