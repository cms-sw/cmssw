#include "HeavyFlavorAnalysis/SpecificDecay/plugins/BPHWriteSpecificDecay.h"

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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBuToPsi2SKBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBsToJPsiPhiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBdToJPsiKxBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHK0sToPiPiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHLambda0ToPPiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBdToJPsiKsBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHLbToJPsiL0Builder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHBcToJPsiPiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHPsi2SToJPsiPiPiBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHX3872ToJPsiPiPiBuilder.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
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

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <set>
#include <string>
#include <iostream>
using namespace std;

#define SET_PAR(TYPE, NAME, PSET) (NAME = PSET.getParameter<TYPE>(#NAME))
// SET_PAR(string,xyz,ps);
// is equivalent to
// ( xyz = ps.getParameter< string >( "xyx" ) )

BPHWriteSpecificDecay::BPHWriteSpecificDecay(const edm::ParameterSet& ps) {
  usePV = (!SET_PAR(string, pVertexLabel, ps).empty());
  usePM = (!SET_PAR(string, patMuonLabel, ps).empty());
  useCC = (!SET_PAR(string, ccCandsLabel, ps).empty());
  usePF = (!SET_PAR(string, pfCandsLabel, ps).empty());
  usePC = (!SET_PAR(string, pcCandsLabel, ps).empty());
  useGP = (!SET_PAR(string, gpCandsLabel, ps).empty());
  useK0 = (!SET_PAR(string, k0CandsLabel, ps).empty());
  useL0 = (!SET_PAR(string, l0CandsLabel, ps).empty());
  useKS = (!SET_PAR(string, kSCandsLabel, ps).empty());
  useLS = (!SET_PAR(string, lSCandsLabel, ps).empty());
  SET_PAR(string, oniaName, ps);
  SET_PAR(string, sdName, ps);
  SET_PAR(string, ssName, ps);
  SET_PAR(string, buName, ps);
  SET_PAR(string, bpName, ps);
  SET_PAR(string, bdName, ps);
  SET_PAR(string, bsName, ps);
  SET_PAR(string, k0Name, ps);
  SET_PAR(string, l0Name, ps);
  SET_PAR(string, b0Name, ps);
  SET_PAR(string, lbName, ps);
  SET_PAR(string, bcName, ps);
  SET_PAR(string, psi2SName, ps);
  SET_PAR(string, x3872Name, ps);

  SET_PAR(bool, writeMomentum, ps);
  SET_PAR(bool, writeVertex, ps);

  rMap["Onia"] = Onia;
  rMap["PhiMuMu"] = Pmm;
  rMap["Psi1"] = Psi1;
  rMap["Psi2"] = Psi2;
  rMap["Ups"] = Ups;
  rMap["Ups1"] = Ups1;
  rMap["Ups2"] = Ups2;
  rMap["Ups3"] = Ups3;
  rMap["Kx0"] = Kx0;
  rMap["PhiKK"] = Pkk;
  rMap["Bu"] = Bu;
  rMap["Bp"] = Bp;
  rMap["Bd"] = Bd;
  rMap["Bs"] = Bs;
  rMap["K0s"] = K0s;
  rMap["Lambda0"] = Lambda0;
  rMap["B0"] = B0;
  rMap["Lambdab"] = Lambdab;
  rMap["Bc"] = Bc;
  rMap["Psi2S"] = Psi2S;
  rMap["X3872"] = X3872;

  pMap["ptMin"] = ptMin;
  pMap["etaMax"] = etaMax;
  pMap["mJPsiMin"] = mPsiMin;
  pMap["mJPsiMax"] = mPsiMax;
  pMap["mKx0Min"] = mKx0Min;
  pMap["mKx0Max"] = mKx0Max;
  pMap["mPhiMin"] = mPhiMin;
  pMap["mPhiMax"] = mPhiMax;
  pMap["mK0sMin"] = mK0sMin;
  pMap["mK0sMax"] = mK0sMax;
  pMap["mLambda0Min"] = mLambda0Min;
  pMap["mLambda0Max"] = mLambda0Max;
  pMap["massMin"] = massMin;
  pMap["massMax"] = massMax;
  pMap["probMin"] = probMin;
  pMap["massFitMin"] = mFitMin;
  pMap["massFitMax"] = mFitMax;
  pMap["constrMass"] = constrMass;
  pMap["constrSigma"] = constrSigma;

  fMap["requireJPsi"] = requireJPsi;
  fMap["constrMJPsi"] = constrMJPsi;
  fMap["constrMPsi2"] = constrMPsi2;

  fMap["writeCandidate"] = writeCandidate;

  recoOnia = recoKx0 = writeKx0 = recoPkk = writePkk = recoBu = writeBu = recoBp = writeBp = recoBd = writeBd = recoBs =
      writeBs = recoK0s = writeK0s = recoLambda0 = writeLambda0 = recoB0 = writeB0 = recoLambdab = writeLambdab =
          recoBc = writeBc = recoPsi2S = writePsi2S = recoX3872 = writeX3872 = false;

  writeOnia = true;
  const vector<edm::ParameterSet> recoSelect = ps.getParameter<vector<edm::ParameterSet> >("recoSelect");
  int iSel;
  int nSel = recoSelect.size();
  for (iSel = 0; iSel < nSel; ++iSel)
    setRecoParameters(recoSelect[iSel]);
  if (!recoOnia)
    writeOnia = false;

  if (recoBu)
    recoOnia = true;
  if (recoBd)
    recoOnia = recoKx0 = true;
  if (recoBs)
    recoOnia = recoPkk = true;
  if (recoB0)
    recoOnia = recoK0s = true;
  if (recoLambdab)
    recoOnia = recoLambda0 = true;
  if (recoBc)
    recoOnia = true;
  if (recoPsi2S)
    recoOnia = true;
  if (recoX3872)
    recoOnia = true;
  if (writeBu)
    writeOnia = true;
  if (writeBd)
    writeOnia = writeKx0 = true;
  if (writeBs)
    writeOnia = writePkk = true;
  if (writeB0)
    writeOnia = writeK0s = true;
  if (writeLambdab)
    writeOnia = writeLambda0 = true;
  if (writeBc)
    writeOnia = true;
  if (writePsi2S)
    writeOnia = true;
  if (writeX3872)
    writeOnia = true;
  if (recoBp && !recoPsi2S && !recoX3872)
    recoPsi2S = true;
  if (writeBp && !writePsi2S && !writeX3872)
    writePsi2S = true;
  allKx0 = (parMap[Kx0][requireJPsi] < 0);
  allPkk = (parMap[Pkk][requireJPsi] < 0);
  allK0s = (parMap[K0s][requireJPsi] < 0);
  allLambda0 = (parMap[Lambda0][requireJPsi] < 0);

  esConsume<MagneticField, IdealMagneticFieldRecord>(magFieldToken);
  esConsume<TransientTrackBuilder, TransientTrackRecord>(ttBToken, "TransientTrackBuilder");
  if (usePV)
    consume<vector<reco::Vertex> >(pVertexToken, pVertexLabel);
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
  if (useK0)
    consume<vector<reco::VertexCompositeCandidate> >(k0CandsToken, k0CandsLabel);
  if (useL0)
    consume<vector<reco::VertexCompositeCandidate> >(l0CandsToken, l0CandsLabel);
  if (useKS)
    consume<vector<reco::VertexCompositePtrCandidate> >(kSCandsToken, kSCandsLabel);
  if (useLS)
    consume<vector<reco::VertexCompositePtrCandidate> >(lSCandsToken, lSCandsLabel);

  if (writeOnia)
    produces<pat::CompositeCandidateCollection>(oniaName);
  if (writeKx0)
    produces<pat::CompositeCandidateCollection>(sdName);
  if (writePkk)
    produces<pat::CompositeCandidateCollection>(ssName);
  if (writeBu)
    produces<pat::CompositeCandidateCollection>(buName);
  if (writeBp)
    produces<pat::CompositeCandidateCollection>(bpName);
  if (writeBd)
    produces<pat::CompositeCandidateCollection>(bdName);
  if (writeBs)
    produces<pat::CompositeCandidateCollection>(bsName);
  if (writeK0s)
    produces<pat::CompositeCandidateCollection>(k0Name);
  if (writeLambda0)
    produces<pat::CompositeCandidateCollection>(l0Name);
  if (writeB0)
    produces<pat::CompositeCandidateCollection>(b0Name);
  if (writeLambdab)
    produces<pat::CompositeCandidateCollection>(lbName);
  if (writeBc)
    produces<pat::CompositeCandidateCollection>(bcName);
  if (writePsi2S)
    produces<pat::CompositeCandidateCollection>(psi2SName);
  if (writeX3872)
    produces<pat::CompositeCandidateCollection>(x3872Name);
}

void BPHWriteSpecificDecay::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<string>("pVertexLabel", "");
  desc.add<string>("patMuonLabel", "");
  desc.add<string>("ccCandsLabel", "");
  desc.add<string>("pfCandsLabel", "");
  desc.add<string>("pcCandsLabel", "");
  desc.add<string>("gpCandsLabel", "");
  desc.add<string>("k0CandsLabel", "");
  desc.add<string>("l0CandsLabel", "");
  desc.add<string>("kSCandsLabel", "");
  desc.add<string>("lSCandsLabel", "");
  desc.add<string>("oniaName", "oniaCand");
  desc.add<string>("sdName", "kx0Cand");
  desc.add<string>("ssName", "phiCand");
  desc.add<string>("buName", "buFitted");
  desc.add<string>("bpName", "bpFitted");
  desc.add<string>("bdName", "bdFitted");
  desc.add<string>("bsName", "bsFitted");
  desc.add<string>("k0Name", "k0Fitted");
  desc.add<string>("l0Name", "l0Fitted");
  desc.add<string>("b0Name", "b0Fitted");
  desc.add<string>("lbName", "lbFitted");
  desc.add<string>("bcName", "bcFitted");
  desc.add<string>("psi2SName", "psi2SFitted");
  desc.add<string>("x3872Name", "x3872Fitted");
  desc.add<bool>("writeVertex", true);
  desc.add<bool>("writeMomentum", true);
  edm::ParameterSetDescription dpar;
  dpar.add<string>("name");
  dpar.add<double>("ptMin", -2.0e35);
  dpar.add<double>("etaMax", -2.0e35);
  dpar.add<double>("mJPsiMin", -2.0e35);
  dpar.add<double>("mJPsiMax", -2.0e35);
  dpar.add<double>("mKx0Min", -2.0e35);
  dpar.add<double>("mKx0Max", -2.0e35);
  dpar.add<double>("mPhiMin", -2.0e35);
  dpar.add<double>("mPhiMax", -2.0e35);
  dpar.add<double>("mK0sMin", -2.0e35);
  dpar.add<double>("mK0sMax", -2.0e35);
  dpar.add<double>("mLambda0Min", -2.0e35);
  dpar.add<double>("mLambda0Max", -2.0e35);
  dpar.add<double>("massMin", -2.0e35);
  dpar.add<double>("massMax", -2.0e35);
  dpar.add<double>("probMin", -2.0e35);
  dpar.add<double>("massFitMin", -2.0e35);
  dpar.add<double>("massFitMax", -2.0e35);
  dpar.add<double>("constrMass", -2.0e35);
  dpar.add<double>("constrSigma", -2.0e35);
  dpar.add<bool>("requireJPsi", true);
  dpar.add<bool>("constrMJPsi", true);
  dpar.add<bool>("constrMPsi2", true);
  dpar.add<bool>("writeCandidate", true);
  vector<edm::ParameterSet> rpar;
  desc.addVPSet("recoSelect", dpar, rpar);
  descriptions.add("bphWriteSpecificDecay", desc);
  return;
}

void BPHWriteSpecificDecay::produce(edm::Event& ev, const edm::EventSetup& es) {
  BPHEventSetupWrapper ew(es, BPHRecoCandidate::transientTrackBuilder, &ttBToken);
  fill(ev, ew);
  if (writeOnia)
    write(ev, lFull, oniaName);
  if (writeKx0)
    write(ev, lSd, sdName);
  if (writePkk)
    write(ev, lSs, ssName);
  if (writeBu)
    write(ev, lBu, buName);
  if (writeBp)
    write(ev, lBp, bpName);
  if (writeBd)
    write(ev, lBd, bdName);
  if (writeBs)
    write(ev, lBs, bsName);
  if (writeK0s)
    write(ev, lK0, k0Name);
  if (writeLambda0)
    write(ev, lL0, l0Name);
  if (writeB0)
    write(ev, lB0, b0Name);
  if (writeLambdab)
    write(ev, lLb, lbName);
  if (writeBc)
    write(ev, lBc, bcName);
  if (writePsi2S)
    write(ev, lPsi2S, psi2SName);
  if (writeX3872)
    write(ev, lX3872, x3872Name);
  return;
}

void BPHWriteSpecificDecay::fill(edm::Event& ev, const BPHEventSetupWrapper& es) {
  lFull.clear();
  lJPsi.clear();
  lSd.clear();
  lSs.clear();
  lBu.clear();
  lBp.clear();
  lBd.clear();
  lBs.clear();
  lK0.clear();
  lL0.clear();
  lB0.clear();
  lLb.clear();
  lBc.clear();
  lPsi2S.clear();
  lX3872.clear();
  jPsiOMap.clear();
  daughMap.clear();
  pvRefMap.clear();
  ccRefMap.clear();

  // get magnetic field
  // data are got through "BPHESTokenWrapper" interface to allow
  // uniform access in different CMSSW versions
  edm::ESHandle<MagneticField> magneticField;
  magFieldToken.get(*es.get(), magneticField);

  // get object collections
  // collections are got through "BPHTokenWrapper" interface to allow
  // uniform access in different CMSSW versions

  edm::Handle<std::vector<reco::Vertex> > pVertices;
  pVertexToken.get(ev, pVertices);
  int npv = pVertices->size();

  int nrc = 0;

  // get reco::PFCandidate collection (in full AOD )
  edm::Handle<vector<reco::PFCandidate> > pfCands;
  if (usePF) {
    pfCandsToken.get(ev, pfCands);
    nrc = pfCands->size();
  }

  // get pat::PackedCandidate collection (in MiniAOD)
  // pat::PackedCandidate is not defined in CMSSW_5XY, so a
  // typedef (BPHTrackReference::candidate) is used, actually referring
  // to pat::PackedCandidate only for CMSSW versions where it's defined
  edm::Handle<vector<BPHTrackReference::candidate> > pcCands;
  if (usePC) {
    pcCandsToken.get(ev, pcCands);
    nrc = pcCands->size();
  }

  // get pat::GenericParticle collection (in skimmed data)
  edm::Handle<vector<pat::GenericParticle> > gpCands;
  if (useGP) {
    gpCandsToken.get(ev, gpCands);
    nrc = gpCands->size();
  }

  // get pat::Muon collection (in full AOD and MiniAOD)
  edm::Handle<pat::MuonCollection> patMuon;
  if (usePM) {
    patMuonToken.get(ev, patMuon);
  }

  // get K0 reco::VertexCompositeCandidate collection (in full AOD)
  edm::Handle<std::vector<reco::VertexCompositeCandidate> > k0Cand;
  if (useK0) {
    k0CandsToken.get(ev, k0Cand);
  }

  // get Lambda0 reco::VertexCompositeCandidate collection (in full AOD)
  edm::Handle<std::vector<reco::VertexCompositeCandidate> > l0Cand;
  if (useL0) {
    l0CandsToken.get(ev, l0Cand);
  }

  // get K0 reco::VertexCompositePtrCandidate collection (in MiniAOD)
  edm::Handle<std::vector<reco::VertexCompositePtrCandidate> > kSCand;
  if (useKS) {
    kSCandsToken.get(ev, kSCand);
  }

  // get Lambda0 reco::VertexCompositePtrCandidate collection (in MiniAOD)
  edm::Handle<std::vector<reco::VertexCompositePtrCandidate> > lSCand;
  if (useLS) {
    lSCandsToken.get(ev, lSCand);
  }

  // get muons from pat::CompositeCandidate objects describing onia;
  // muons from all composite objects are copied to an unique std::vector
  vector<const reco::Candidate*> muDaugs;
  set<const pat::Muon*> muonSet;
  typedef multimap<const reco::Candidate*, const pat::CompositeCandidate*> mu_cc_map;
  mu_cc_map muCCMap;
  if (useCC) {
    edm::Handle<vector<pat::CompositeCandidate> > ccCands;
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
        // associate muon to the CompositeCandidate containing it
        muCCMap.insert(pair<const reco::Candidate*, const pat::CompositeCandidate*>(dp, &cc));
      }
    }
    iter = muonSet.begin();
    iend = muonSet.end();
    while (iter != iend)
      muDaugs.push_back(*iter++);
  }

  map<recoType, map<parType, double> >::const_iterator rIter = parMap.begin();
  map<recoType, map<parType, double> >::const_iterator rIend = parMap.end();

  // reconstruct quarkonia

  BPHOniaToMuMuBuilder* onia = nullptr;
  if (recoOnia) {
    if (usePM)
      onia = new BPHOniaToMuMuBuilder(
          es, BPHRecoBuilder::createCollection(patMuon, "ingmcf"), BPHRecoBuilder::createCollection(patMuon, "ingmcf"));
    else if (useCC)
      onia = new BPHOniaToMuMuBuilder(
          es, BPHRecoBuilder::createCollection(muDaugs, "ingmcf"), BPHRecoBuilder::createCollection(muDaugs, "ingmcf"));
  }

  if (onia != nullptr) {
    while (rIter != rIend) {
      const map<recoType, map<parType, double> >::value_type& rEntry = *rIter++;
      recoType rType = rEntry.first;
      const map<parType, double>& pMap = rEntry.second;
      BPHOniaToMuMuBuilder::oniaType type;
      switch (rType) {
        case Pmm:
          type = BPHOniaToMuMuBuilder::Phi;
          break;
        case Psi1:
          type = BPHOniaToMuMuBuilder::Psi1;
          break;
        case Psi2:
          type = BPHOniaToMuMuBuilder::Psi2;
          break;
        case Ups:
          type = BPHOniaToMuMuBuilder::Ups;
          break;
        case Ups1:
          type = BPHOniaToMuMuBuilder::Ups1;
          break;
        case Ups2:
          type = BPHOniaToMuMuBuilder::Ups2;
          break;
        case Ups3:
          type = BPHOniaToMuMuBuilder::Ups3;
          break;
        default:
          continue;
      }
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            onia->setPtMin(type, pv);
            break;
          case etaMax:
            onia->setEtaMax(type, pv);
            break;
          case massMin:
            onia->setMassMin(type, pv);
            break;
          case massMax:
            onia->setMassMax(type, pv);
            break;
          case probMin:
            onia->setProbMin(type, pv);
            break;
          case constrMass:
            onia->setConstr(type, pv, onia->getConstrSigma(type));
            break;
          case constrSigma:
            onia->setConstr(type, onia->getConstrMass(type), pv);
            break;
          default:
            break;
        }
      }
    }
    lFull = onia->build();
  }

  // associate onia to primary vertex

  int iFull;
  int nFull = lFull.size();
  map<const BPHRecoCandidate*, const reco::Vertex*> oniaVtxMap;

  typedef mu_cc_map::const_iterator mu_cc_iter;
  for (iFull = 0; iFull < nFull; ++iFull) {
    const reco::Vertex* pVtx = nullptr;
    int pvId = 0;
    const BPHPlusMinusCandidate* ptr = lFull[iFull].get();
    const std::vector<const reco::Candidate*>& daugs = ptr->daughters();

    // try to recover primary vertex association in skim data:
    // get the CompositeCandidate containing both muons
    pair<mu_cc_iter, mu_cc_iter> cc0 = muCCMap.equal_range(ptr->originalReco(daugs[0]));
    pair<mu_cc_iter, mu_cc_iter> cc1 = muCCMap.equal_range(ptr->originalReco(daugs[1]));
    mu_cc_iter iter0 = cc0.first;
    mu_cc_iter iend0 = cc0.second;
    mu_cc_iter iter1 = cc1.first;
    mu_cc_iter iend1 = cc1.second;
    while ((iter0 != iend0) && (pVtx == nullptr)) {
      const pat::CompositeCandidate* ccp = iter0++->second;
      while (iter1 != iend1) {
        if (ccp != iter1++->second)
          continue;
        pVtx = ccp->userData<reco::Vertex>("PVwithmuons");
        const reco::Vertex* sVtx = nullptr;
        const reco::Vertex::Point& pPos = pVtx->position();
        float dMin = 999999.;
        int ipv;
        for (ipv = 0; ipv < npv; ++ipv) {
          const reco::Vertex* tVtx = &pVertices->at(ipv);
          const reco::Vertex::Point& tPos = tVtx->position();
          float dist = pow(pPos.x() - tPos.x(), 2) + pow(pPos.y() - tPos.y(), 2) + pow(pPos.z() - tPos.z(), 2);
          if (dist < dMin) {
            dMin = dist;
            sVtx = tVtx;
            pvId = ipv;
          }
        }
        pVtx = sVtx;
        break;
      }
    }

    // if not found, as for other type of input data,
    // try to get the nearest primary vertex in z direction
    if (pVtx == nullptr) {
      const reco::Vertex::Point& sVtp = ptr->vertex().position();
      GlobalPoint cPos(sVtp.x(), sVtp.y(), sVtp.z());
      const pat::CompositeCandidate& sCC = ptr->composite();
      GlobalVector cDir(sCC.px(), sCC.py(), sCC.pz());
      GlobalPoint bPos(0.0, 0.0, 0.0);
      GlobalVector bDir(0.0, 0.0, 1.0);
      TwoTrackMinimumDistance ttmd;
      bool state = ttmd.calculate(GlobalTrajectoryParameters(cPos, cDir, TrackCharge(0), &(*magneticField)),
                                  GlobalTrajectoryParameters(bPos, bDir, TrackCharge(0), &(*magneticField)));
      float minDz = 999999.;
      float extrapZ = (state ? ttmd.points().first.z() : -9e20);
      int ipv;
      for (ipv = 0; ipv < npv; ++ipv) {
        const reco::Vertex& tVtx = pVertices->at(ipv);
        float deltaZ = fabs(extrapZ - tVtx.position().z());
        if (deltaZ < minDz) {
          minDz = deltaZ;
          pVtx = &tVtx;
          pvId = ipv;
        }
      }
    }

    oniaVtxMap[ptr] = pVtx;
    pvRefMap[ptr] = vertex_ref(pVertices, pvId);
  }
  pVertexToken.get(ev, pVertices);

  // get JPsi subsample and associate JPsi candidate to original
  // generic onia candidate
  if (nFull)
    lJPsi = onia->getList(BPHOniaToMuMuBuilder::Psi1);

  bool jPsiFound = !lJPsi.empty();
  delete onia;

  if (!nrc)
    return;

  int ij;
  int io;
  int nj = lJPsi.size();
  int no = lFull.size();
  for (ij = 0; ij < nj; ++ij) {
    const BPHRecoCandidate* jp = lJPsi[ij].get();
    for (io = 0; io < no; ++io) {
      const BPHRecoCandidate* oc = lFull[io].get();
      if ((jp->originalReco(jp->getDaug("MuPos")) == oc->originalReco(oc->getDaug("MuPos"))) &&
          (jp->originalReco(jp->getDaug("MuNeg")) == oc->originalReco(oc->getDaug("MuNeg")))) {
        jPsiOMap[jp] = oc;
        break;
      }
    }
  }

  // build and dump Bu

  BPHBuToJPsiKBuilder* bu = nullptr;
  if (recoBu && jPsiFound) {
    if (usePF)
      bu = new BPHBuToJPsiKBuilder(es, lJPsi, BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      bu = new BPHBuToJPsiKBuilder(es, lJPsi, BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      bu = new BPHBuToJPsiKBuilder(es, lJPsi, BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  if (bu != nullptr) {
    rIter = parMap.find(Bu);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            bu->setKPtMin(pv);
            break;
          case etaMax:
            bu->setKEtaMax(pv);
            break;
          case mPsiMin:
            bu->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            bu->setJPsiMassMax(pv);
            break;
          case massMin:
            bu->setMassMin(pv);
            break;
          case massMax:
            bu->setMassMax(pv);
            break;
          case probMin:
            bu->setProbMin(pv);
            break;
          case mFitMin:
            bu->setMassFitMin(pv);
            break;
          case mFitMax:
            bu->setMassFitMax(pv);
            break;
          case constrMJPsi:
            bu->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeBu = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lBu = bu->build();
    delete bu;
  }

  // build and dump Kx0

  vector<BPHPlusMinusConstCandPtr> lKx0;
  BPHKx0ToKPiBuilder* kx0 = nullptr;
  if (recoKx0 && (jPsiFound || allKx0)) {
    if (usePF)
      kx0 = new BPHKx0ToKPiBuilder(
          es, BPHRecoBuilder::createCollection(pfCands, "f"), BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      kx0 = new BPHKx0ToKPiBuilder(
          es, BPHRecoBuilder::createCollection(pcCands, "p"), BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      kx0 = new BPHKx0ToKPiBuilder(
          es, BPHRecoBuilder::createCollection(gpCands, "h"), BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  set<BPHRecoConstCandPtr> sKx0;

  if (kx0 != nullptr) {
    rIter = parMap.find(Kx0);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            kx0->setPtMin(pv);
            break;
          case etaMax:
            kx0->setEtaMax(pv);
            break;
          case massMin:
            kx0->setMassMin(pv);
            break;
          case massMax:
            kx0->setMassMax(pv);
            break;
          case probMin:
            kx0->setProbMin(pv);
            break;
          case writeCandidate:
            writeKx0 = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lKx0 = kx0->build();
    if (allKx0)
      sKx0.insert(lKx0.begin(), lKx0.end());
    delete kx0;
  }

  bool kx0Found = !lKx0.empty();

  // build and dump Bd -> JPsi Kx0

  if (recoBd && jPsiFound && kx0Found) {
    BPHBdToJPsiKxBuilder* bd = new BPHBdToJPsiKxBuilder(es, lJPsi, lKx0);
    rIter = parMap.find(Bd);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case mPsiMin:
            bd->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            bd->setJPsiMassMax(pv);
            break;
          case mKx0Min:
            bd->setKxMassMin(pv);
            break;
          case mKx0Max:
            bd->setKxMassMax(pv);
            break;
          case massMin:
            bd->setMassMin(pv);
            break;
          case massMax:
            bd->setMassMax(pv);
            break;
          case probMin:
            bd->setProbMin(pv);
            break;
          case mFitMin:
            bd->setMassFitMin(pv);
            break;
          case mFitMax:
            bd->setMassFitMax(pv);
            break;
          case constrMJPsi:
            bd->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeBd = (pv > 0);
            break;
          default:
            break;
        }
      }
    }

    lBd = bd->build();
    delete bd;

    int iBd;
    int nBd = lBd.size();
    for (iBd = 0; iBd < nBd; ++iBd)
      sKx0.insert(lBd[iBd]->getComp("Kx0"));
  }
  set<BPHRecoConstCandPtr>::const_iterator kx0_iter = sKx0.begin();
  set<BPHRecoConstCandPtr>::const_iterator kx0_iend = sKx0.end();
  lSd.reserve(sKx0.size());
  while (kx0_iter != kx0_iend)
    lSd.push_back(*kx0_iter++);

  // build and dump Phi

  vector<BPHPlusMinusConstCandPtr> lPhi;
  BPHPhiToKKBuilder* phi = nullptr;
  if (recoPkk && (jPsiFound || allPkk)) {
    if (usePF)
      phi = new BPHPhiToKKBuilder(
          es, BPHRecoBuilder::createCollection(pfCands, "f"), BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      phi = new BPHPhiToKKBuilder(
          es, BPHRecoBuilder::createCollection(pcCands, "p"), BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      phi = new BPHPhiToKKBuilder(
          es, BPHRecoBuilder::createCollection(gpCands, "h"), BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  set<BPHRecoConstCandPtr> sPhi;

  if (phi != nullptr) {
    rIter = parMap.find(Pkk);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            phi->setPtMin(pv);
            break;
          case etaMax:
            phi->setEtaMax(pv);
            break;
          case massMin:
            phi->setMassMin(pv);
            break;
          case massMax:
            phi->setMassMax(pv);
            break;
          case probMin:
            phi->setProbMin(pv);
            break;
          case writeCandidate:
            writePkk = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lPhi = phi->build();
    if (allPkk)
      sPhi.insert(lPhi.begin(), lPhi.end());
    delete phi;
  }

  bool phiFound = !lPhi.empty();

  // build and dump Bs

  if (recoBs && jPsiFound && phiFound) {
    BPHBsToJPsiPhiBuilder* bs = new BPHBsToJPsiPhiBuilder(es, lJPsi, lPhi);
    rIter = parMap.find(Bs);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case mPsiMin:
            bs->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            bs->setJPsiMassMax(pv);
            break;
          case mPhiMin:
            bs->setPhiMassMin(pv);
            break;
          case mPhiMax:
            bs->setPhiMassMax(pv);
            break;
          case massMin:
            bs->setMassMin(pv);
            break;
          case massMax:
            bs->setMassMax(pv);
            break;
          case probMin:
            bs->setProbMin(pv);
            break;
          case mFitMin:
            bs->setMassFitMin(pv);
            break;
          case mFitMax:
            bs->setMassFitMax(pv);
            break;
          case constrMJPsi:
            bs->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeBs = (pv > 0);
            break;
          default:
            break;
        }
      }
    }

    lBs = bs->build();
    delete bs;

    int iBs;
    int nBs = lBs.size();
    for (iBs = 0; iBs < nBs; ++iBs)
      sPhi.insert(lBs[iBs]->getComp("Phi"));
  }
  set<BPHRecoConstCandPtr>::const_iterator phi_iter = sPhi.begin();
  set<BPHRecoConstCandPtr>::const_iterator phi_iend = sPhi.end();
  lSs.reserve(sPhi.size());
  while (phi_iter != phi_iend)
    lSs.push_back(*phi_iter++);

  // build K0

  BPHK0sToPiPiBuilder* k0s = nullptr;
  if (recoK0s && (jPsiFound || allK0s)) {
    if (useK0)
      k0s = new BPHK0sToPiPiBuilder(es, k0Cand.product(), "cfp");
    else if (useKS)
      k0s = new BPHK0sToPiPiBuilder(es, kSCand.product(), "cfp");
  }
  if (k0s != nullptr) {
    rIter = parMap.find(K0s);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            k0s->setPtMin(pv);
            break;
          case etaMax:
            k0s->setEtaMax(pv);
            break;
          case massMin:
            k0s->setMassMin(pv);
            break;
          case massMax:
            k0s->setMassMax(pv);
            break;
          case probMin:
            k0s->setProbMin(pv);
            break;
          case writeCandidate:
            writeK0s = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lK0 = k0s->build();
    delete k0s;
  }

  bool k0Found = !lK0.empty();

  // build Lambda0

  BPHLambda0ToPPiBuilder* l0s = nullptr;
  if (recoLambda0 && (jPsiFound || allLambda0)) {
    if (useL0)
      l0s = new BPHLambda0ToPPiBuilder(es, l0Cand.product(), "cfp");
    else if (useLS)
      l0s = new BPHLambda0ToPPiBuilder(es, lSCand.product(), "cfp");
  }
  if (l0s != nullptr) {
    rIter = parMap.find(Lambda0);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            l0s->setPtMin(pv);
            break;
          case etaMax:
            l0s->setEtaMax(pv);
            break;
          case massMin:
            l0s->setMassMin(pv);
            break;
          case massMax:
            l0s->setMassMax(pv);
            break;
          case probMin:
            l0s->setProbMin(pv);
            break;
          case writeCandidate:
            writeLambda0 = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lL0 = l0s->build();
    delete l0s;
  }

  bool l0Found = !lL0.empty();

  // build and dump Bd -> JPsi K0s

  if (recoB0 && jPsiFound && k0Found) {
    BPHBdToJPsiKsBuilder* b0 = new BPHBdToJPsiKsBuilder(es, lJPsi, lK0);
    rIter = parMap.find(B0);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case mPsiMin:
            b0->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            b0->setJPsiMassMax(pv);
            break;
          case mK0sMin:
            b0->setK0MassMin(pv);
            break;
          case mK0sMax:
            b0->setK0MassMax(pv);
            break;
          case massMin:
            b0->setMassMin(pv);
            break;
          case massMax:
            b0->setMassMax(pv);
            break;
          case probMin:
            b0->setProbMin(pv);
            break;
          case mFitMin:
            b0->setMassFitMin(pv);
            break;
          case mFitMax:
            b0->setMassFitMax(pv);
            break;
          case constrMJPsi:
            b0->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeB0 = (pv > 0);
            break;
          default:
            break;
        }
      }
    }

    lB0 = b0->build();
    const map<const BPHRecoCandidate*, const BPHRecoCandidate*>& b0Map = b0->daughMap();
    daughMap.insert(b0Map.begin(), b0Map.end());
    delete b0;
  }

  // build and dump Lambdab -> JPsi Lambda0

  if (recoLambdab && jPsiFound && l0Found) {
    BPHLbToJPsiL0Builder* lb = new BPHLbToJPsiL0Builder(es, lJPsi, lL0);
    rIter = parMap.find(Lambdab);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case mPsiMin:
            lb->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            lb->setJPsiMassMax(pv);
            break;
          case mLambda0Min:
            lb->setLambda0MassMin(pv);
            break;
          case mLambda0Max:
            lb->setLambda0MassMax(pv);
            break;
          case massMin:
            lb->setMassMin(pv);
            break;
          case massMax:
            lb->setMassMax(pv);
            break;
          case probMin:
            lb->setProbMin(pv);
            break;
          case mFitMin:
            lb->setMassFitMin(pv);
            break;
          case mFitMax:
            lb->setMassFitMax(pv);
            break;
          case constrMJPsi:
            lb->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeLambdab = (pv > 0);
            break;
          default:
            break;
        }
      }
    }

    lLb = lb->build();
    const map<const BPHRecoCandidate*, const BPHRecoCandidate*>& ldMap = lb->daughMap();
    daughMap.insert(ldMap.begin(), ldMap.end());
    delete lb;
  }

  // build and dump Bc

  BPHBcToJPsiPiBuilder* bc = nullptr;
  if (recoBc && jPsiFound) {
    if (usePF)
      bc = new BPHBcToJPsiPiBuilder(es, lJPsi, BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      bc = new BPHBcToJPsiPiBuilder(es, lJPsi, BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      bc = new BPHBcToJPsiPiBuilder(es, lJPsi, BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  if (bc != nullptr) {
    rIter = parMap.find(Bc);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            bc->setPiPtMin(pv);
            break;
          case etaMax:
            bc->setPiEtaMax(pv);
            break;
          case mPsiMin:
            bc->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            bc->setJPsiMassMax(pv);
            break;
          case massMin:
            bc->setMassMin(pv);
            break;
          case massMax:
            bc->setMassMax(pv);
            break;
          case probMin:
            bc->setProbMin(pv);
            break;
          case mFitMin:
            bc->setMassFitMin(pv);
            break;
          case mFitMax:
            bc->setMassFitMax(pv);
            break;
          case constrMJPsi:
            bc->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeBc = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lBc = bc->build();
    delete bc;
  }

  // build and dump Psi2S

  BPHPsi2SToJPsiPiPiBuilder* psi2S = nullptr;
  if (recoPsi2S && jPsiFound) {
    if (usePF)
      psi2S = new BPHPsi2SToJPsiPiPiBuilder(
          es, lJPsi, BPHRecoBuilder::createCollection(pfCands, "f"), BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      psi2S = new BPHPsi2SToJPsiPiPiBuilder(
          es, lJPsi, BPHRecoBuilder::createCollection(pcCands, "p"), BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      psi2S = new BPHPsi2SToJPsiPiPiBuilder(
          es, lJPsi, BPHRecoBuilder::createCollection(gpCands, "h"), BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  if (psi2S != nullptr) {
    rIter = parMap.find(Psi2S);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            psi2S->setPiPtMin(pv);
            break;
          case etaMax:
            psi2S->setPiEtaMax(pv);
            break;
          case mPsiMin:
            psi2S->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            psi2S->setJPsiMassMax(pv);
            break;
          case massMin:
            psi2S->setMassMin(pv);
            break;
          case massMax:
            psi2S->setMassMax(pv);
            break;
          case probMin:
            psi2S->setProbMin(pv);
            break;
          case mFitMin:
            psi2S->setMassFitMin(pv);
            break;
          case mFitMax:
            psi2S->setMassFitMax(pv);
            break;
          case constrMJPsi:
            psi2S->setConstr(pv > 0);
            break;
          case writeCandidate:
            writePsi2S = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lPsi2S = psi2S->build();
    delete psi2S;
  }

  // build and dump X3872

  BPHX3872ToJPsiPiPiBuilder* x3872 = nullptr;
  if (recoX3872 && jPsiFound) {
    if (usePF)
      x3872 = new BPHX3872ToJPsiPiPiBuilder(
          es, lJPsi, BPHRecoBuilder::createCollection(pfCands, "f"), BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      x3872 = new BPHX3872ToJPsiPiPiBuilder(
          es, lJPsi, BPHRecoBuilder::createCollection(pcCands, "p"), BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      x3872 = new BPHX3872ToJPsiPiPiBuilder(
          es, lJPsi, BPHRecoBuilder::createCollection(gpCands, "h"), BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  if (x3872 != nullptr) {
    rIter = parMap.find(X3872);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            x3872->setPiPtMin(pv);
            break;
          case etaMax:
            x3872->setPiEtaMax(pv);
            break;
          case mPsiMin:
            x3872->setJPsiMassMin(pv);
            break;
          case mPsiMax:
            x3872->setJPsiMassMax(pv);
            break;
          case massMin:
            x3872->setMassMin(pv);
            break;
          case massMax:
            x3872->setMassMax(pv);
            break;
          case probMin:
            x3872->setProbMin(pv);
            break;
          case mFitMin:
            x3872->setMassFitMin(pv);
            break;
          case mFitMax:
            x3872->setMassFitMax(pv);
            break;
          case constrMJPsi:
            x3872->setConstr(pv > 0);
            break;
          case writeCandidate:
            writeX3872 = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    lX3872 = x3872->build();
    delete x3872;
  }

  // merge Psi2S and X3872
  class ResTrkTrkCompare {
  public:
    bool operator()(const BPHRecoConstCandPtr& l, const BPHRecoConstCandPtr& r) const {
      vector<const reco::Track*> tl = l->tracks();
      vector<const reco::Track*> tr = r->tracks();
      if (tl.size() < tr.size())
        return true;
      sort(tl.begin(), tl.end());
      sort(tr.begin(), tr.end());
      int n = tr.size();
      int i;
      for (i = 0; i < n; ++i) {
        if (tl[i] < tr[i])
          return true;
        if (tl[i] > tr[i])
          return false;
      }
      return false;
    }
  } rttc;
  set<BPHRecoConstCandPtr, ResTrkTrkCompare> sjpPiPi(rttc);
  sjpPiPi.insert(lPsi2S.begin(), lPsi2S.end());
  sjpPiPi.insert(lX3872.begin(), lX3872.end());
  vector<BPHRecoConstCandPtr> ljpPiPi;
  ljpPiPi.insert(ljpPiPi.end(), sjpPiPi.begin(), sjpPiPi.end());
  bool jpPiPiFound = !ljpPiPi.empty();

  // build and dump Bp

  BPHBuToPsi2SKBuilder* bp = nullptr;
  if (recoBp && jpPiPiFound) {
    if (usePF)
      bp = new BPHBuToPsi2SKBuilder(es, ljpPiPi, BPHRecoBuilder::createCollection(pfCands, "f"));
    else if (usePC)
      bp = new BPHBuToPsi2SKBuilder(es, ljpPiPi, BPHRecoBuilder::createCollection(pcCands, "p"));
    else if (useGP)
      bp = new BPHBuToPsi2SKBuilder(es, ljpPiPi, BPHRecoBuilder::createCollection(gpCands, "h"));
  }

  if (bp != nullptr) {
    class BPHBuToPsi2SSelect : public BPHMassFitSelect {
    public:
      BPHBuToPsi2SSelect()
          : BPHMassFitSelect("Psi2S", BPHParticleMasses::psi2Mass, BPHParticleMasses::psi2MWidth, 5.0, 6.0) {}
      ~BPHBuToPsi2SSelect() override = default;
      bool accept(const BPHKinematicFit& cand) const override {
        const_cast<BPHRecoCandidate*>(cand.getComp("Psi2S").get())
            ->setIndependentFit("JPsi", true, BPHParticleMasses::jPsiMass, BPHParticleMasses::jPsiMWidth);
        return BPHMassFitSelect::accept(cand);
      }
    };
    bool mcJPsi = false;
    bool mcPsi2 = true;
    rIter = parMap.find(Bp);
    if (rIter != rIend) {
      const map<parType, double>& pMap = rIter->second;
      map<parType, double>::const_iterator pIter = pMap.begin();
      map<parType, double>::const_iterator pIend = pMap.end();
      while (pIter != pIend) {
        const map<parType, double>::value_type& pEntry = *pIter++;
        parType id = pEntry.first;
        double pv = pEntry.second;
        switch (id) {
          case ptMin:
            bp->setKPtMin(pv);
            break;
          case etaMax:
            bp->setKEtaMax(pv);
            break;
          case mPsiMin:
            bp->setPsi2SMassMin(pv);
            break;
          case mPsiMax:
            bp->setPsi2SMassMax(pv);
            break;
          case massMin:
            bp->setMassMin(pv);
            break;
          case massMax:
            bp->setMassMax(pv);
            break;
          case probMin:
            bp->setProbMin(pv);
            break;
          case mFitMin:
            bp->setMassFitMin(pv);
            break;
          case mFitMax:
            bp->setMassFitMax(pv);
            break;
          case constrMJPsi:
            mcJPsi = (pv > 0);
            break;
          case constrMPsi2:
            mcPsi2 = (pv > 0);
            break;
          case writeCandidate:
            writeBp = (pv > 0);
            break;
          default:
            break;
        }
      }
    }
    if (mcJPsi)
      bp->setMassFitSelect(mcPsi2 ? new BPHBuToPsi2SSelect
                                  : new BPHMassFitSelect("Psi2S/JPsi",
                                                         BPHParticleMasses::jPsiMass,
                                                         BPHParticleMasses::jPsiMWidth,
                                                         bp->getMassFitMin(),
                                                         bp->getMassFitMax()));
    else
      bp->setConstr(mcPsi2);
    lBp = bp->build();
    const map<const BPHRecoCandidate*, const BPHRecoCandidate*>& bpMap = bp->daughMap();
    daughMap.insert(bpMap.begin(), bpMap.end());
    delete bp;
  }

  return;
}

void BPHWriteSpecificDecay::setRecoParameters(const edm::ParameterSet& ps) {
  const string& name = ps.getParameter<string>("name");
  bool writeCandidate = ps.getParameter<bool>("writeCandidate");
  switch (rMap[name]) {
    case Onia:
      recoOnia = true;
      writeOnia = writeCandidate;
      break;
    case Pmm:
    case Psi1:
    case Psi2:
    case Ups:
    case Ups1:
    case Ups2:
    case Ups3:
      recoOnia = true;
      break;
    case Kx0:
      recoKx0 = true;
      allKx0 = false;
      writeKx0 = writeCandidate;
      break;
    case Pkk:
      recoPkk = true;
      allPkk = false;
      writePkk = writeCandidate;
      break;
    case Bu:
      recoBu = true;
      writeBu = writeCandidate;
      break;
    case Bp:
      recoBp = true;
      writeBp = writeCandidate;
      break;
    case Bd:
      recoBd = true;
      writeBd = writeCandidate;
      break;
    case Bs:
      recoBs = true;
      writeBs = writeCandidate;
      break;
    case K0s:
      recoK0s = true;
      allK0s = false;
      writeK0s = writeCandidate;
      break;
    case Lambda0:
      recoLambda0 = true;
      allLambda0 = false;
      writeLambda0 = writeCandidate;
      break;
    case B0:
      recoB0 = true;
      writeB0 = writeCandidate;
      break;
    case Lambdab:
      recoLambdab = true;
      writeLambdab = writeCandidate;
      break;
    case Bc:
      recoBc = true;
      writeBc = writeCandidate;
      break;
    case Psi2S:
      recoPsi2S = true;
      writePsi2S = writeCandidate;
      break;
    case X3872:
      recoX3872 = true;
      writeX3872 = writeCandidate;
      break;
  }

  map<string, parType>::const_iterator pIter = pMap.begin();
  map<string, parType>::const_iterator pIend = pMap.end();
  while (pIter != pIend) {
    const map<string, parType>::value_type& entry = *pIter++;
    const string& pn = entry.first;
    parType id = entry.second;
    double pv = ps.getParameter<double>(pn);
    if (pv > -1.0e35)
      edm::LogVerbatim("Configuration") << "BPHWriteSpecificDecay::setRecoParameters: set " << pn << " for " << name
                                        << " : " << (parMap[rMap[name]][id] = pv);
  }

  map<string, parType>::const_iterator fIter = fMap.begin();
  map<string, parType>::const_iterator fIend = fMap.end();
  while (fIter != fIend) {
    const map<string, parType>::value_type& entry = *fIter++;
    const string& fn = entry.first;
    parType id = entry.second;
    double pv = (ps.getParameter<bool>(fn) ? 1 : -1);
    if (pv > -1.0e35)
      edm::LogVerbatim("Configuration") << "BPHWriteSpecificDecay::setRecoParameters: set " << fn << " for " << name
                                        << " : " << (parMap[rMap[name]][id] = pv);
  }

  return;
}

void BPHWriteSpecificDecay::addTrackModes(const std::string& name,
                                          const BPHRecoCandidate& cand,
                                          std::string& modes,
                                          bool& count) {
  for (const std::map<std::string, const reco::Candidate*>::value_type& entry : cand.daugMap()) {
    if (count)
      modes += "#";
    modes += (name + entry.first + ":" + cand.getTMode(entry.second));
    count = true;
  }
  for (const std::map<std::string, BPHRecoConstCandPtr>::value_type& entry : cand.compMap()) {
    addTrackModes(entry.first + "/", *entry.second, modes, count);
  }
  return;
}

void BPHWriteSpecificDecay::addTrackModes(const std::string& name,
                                          const BPHRecoCandidate& cand,
                                          pat::CompositeCandidate& cc) {
  for (const std::map<std::string, const reco::Candidate*>::value_type& entry : cand.daugMap())
    cc.addUserData(name + entry.first, string(1, cand.getTMode(entry.second)), true);
  for (const std::map<std::string, BPHRecoConstCandPtr>::value_type& entry : cand.compMap())
    addTrackModes(name + entry.first + "/", *entry.second, cc);
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(BPHWriteSpecificDecay);
