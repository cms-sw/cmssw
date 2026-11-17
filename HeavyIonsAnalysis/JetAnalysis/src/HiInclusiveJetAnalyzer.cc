#include <algorithm>
#include <utility>
#include "FWCore/Utilities/interface/Exception.h"

/*
  Based on the jet response analyzer
  Modified by Matt Nguyen, November 2010
*/

#include "HeavyIonsAnalysis/JetAnalysis/interface/HiInclusiveJetAnalyzer.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "fastjet/contrib/Njettiness.hh"
#include "fastjet/AreaDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/contrib/SoftDrop.hh"

using namespace std;
using namespace edm;
using namespace reco;

HiInclusiveJetAnalyzer::HiInclusiveJetAnalyzer(const edm::ParameterSet& iConfig) {
  branchesToKeep_ = iConfig.getUntrackedParameter<std::vector<std::string>>("branchesToKeep", std::vector<std::string>{});
  doMatch_ = iConfig.getUntrackedParameter<bool>("matchJets", false);
  jetTag_ = consumes<pat::JetCollection>(iConfig.getParameter<InputTag>("jetTag"));
  caloJetTag_ = consumes<reco::CaloJetCollection>(iConfig.getParameter<InputTag>("caloJetTag"));
  matchTag_ = consumes<pat::JetCollection>(iConfig.getUntrackedParameter<InputTag>("matchTag"));

  useQuality_ = iConfig.getUntrackedParameter<bool>("useQuality", true);
  trackQuality_ = iConfig.getUntrackedParameter<string>("trackQuality", "highPurity");

  jetName_ = iConfig.getUntrackedParameter<string>("jetName");
  doGenTaus_ = iConfig.getUntrackedParameter<bool>("doGenTaus", false);
  doGenSym_ = iConfig.getUntrackedParameter<bool>("doGenSym", false);
  doSubJets_ = iConfig.getUntrackedParameter<bool>("doSubJets", false);
  doJetConstituents_ = iConfig.getUntrackedParameter<bool>("doJetConstituents", false);
  doCaloJets_ = iConfig.getUntrackedParameter<bool>("doCaloJets", true);
  doGenSubJets_ = iConfig.getUntrackedParameter<bool>("doGenSubJets", false);
  if (doGenSubJets_)
    subjetGenTag_ = consumes<reco::JetView>(iConfig.getUntrackedParameter<InputTag>("subjetGenTag"));

  //reWTA reclustering
  doWTARecluster_ = iConfig.getUntrackedParameter<bool>("doWTARecluster", false);

  if (doGenTaus_) {
    tokenGenTau1_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("genTau1"));
    tokenGenTau2_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("genTau2"));
    tokenGenTau3_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("genTau3"));
  }

  if (doGenSym_) {
    tokenGenSym_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("genSym"));
    tokenGenDroppedBranches_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("genDroppedBranches"));
  }

  isMC_ = iConfig.getUntrackedParameter<bool>("isMC", false);
  useHepMC_ = iConfig.getUntrackedParameter<bool>("useHepMC", false);
  fillGenJets_ = iConfig.getUntrackedParameter<bool>("fillGenJets", false);

  doHiJetID_ = iConfig.getUntrackedParameter<bool>("doHiJetID", false);
  doStandardJetID_ = iConfig.getUntrackedParameter<bool>("doStandardJetID", false);

  r2Param = iConfig.getParameter<double>("rParam");
  r2Param *= r2Param;

  hardPtMin_ = iConfig.getUntrackedParameter<double>("hardPtMin", 4);
  jetPtMin_ = iConfig.getParameter<double>("jetPtMin");
  jetAbsEtaMax_ = iConfig.getUntrackedParameter<double>("jetAbsEtaMax", 5.1);

  if (isMC_) {
    genjetTag_ = consumes<edm::View<reco::GenJet>>(iConfig.getParameter<InputTag>("genjetTag"));
    if (useHepMC_)
      eventInfoTag_ = consumes<HepMCProduct>(iConfig.getParameter<InputTag>("eventInfoTag"));
    eventGenInfoTag_ = consumes<GenEventInfoProduct>(iConfig.getParameter<InputTag>("eventInfoTag"));
    jetFlavourInfosToken_ =
        consumes<reco::JetFlavourInfoMatchingCollection>(iConfig.getParameter<edm::InputTag>("jetFlavourInfos"));
  }
  useRawPt_ = iConfig.getUntrackedParameter<bool>("useRawPt", true);

  doBtagging_ = iConfig.getUntrackedParameter<bool>("doBtagging", false);

  pfCandidateLabel_ =
      consumes<edm::View<pat::PackedCandidate>>(iConfig.getUntrackedParameter<edm::InputTag>("pfCandidateLabel"));

  if (isMC_)
    genParticleSrc_ =
        consumes<reco::GenParticleCollection>(iConfig.getUntrackedParameter<edm::InputTag>("genParticles"));

  if (doBtagging_) {
    // this takes b-tagging default from miniAOD
    //particleTransformerJetTagsTkn_ = consumes<JetTagCollection> (iConfig.getUntrackedParameter<string>("pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour",("pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour"))+std::string(":probb"));
    //particleTransformerJetTagsBBTkn_ = consumes<JetTagCollection> (iConfig.getUntrackedParameter<string>("pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour",("pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour")) +std::string(":probbb"));
    //particleTransformerJetTagsLepBTkn_ = consumes<JetTagCollection> (iConfig.getUntrackedParameter<string>("pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour",("pfParticleNetFromMiniAODAK4CHSCentralJetTagsSlimmedDeepFlavour"))+std::string(":problepb"));
    // this uses the PbPb version
    for (const std::string label : {"pfJetProbabilityBJetTag",
                                    "pfDeepCSVJetTags",
                                    "pfDeepFlavourJetTags",
                                    "pfParticleTransformerAK4JetTags",
                                    "pfUnifiedParticleTransformerAK4JetTags"}) {
      const auto& tag = iConfig.getUntrackedParameter<string>(label, "");
      if (tag.empty())
        continue;
      else if (label == "pfJetProbabilityBJetTag")
        jetTaggers_["pfJP"].emplace("probb", consumes<JetTagCollection>(tag));
      else if (label == "pfDeepCSVJetTags")
        for (const auto& cat : {"probb", "probbb"})
          jetTaggers_["deepCSV"].emplace(cat, consumes<JetTagCollection>(tag + ":" + cat));
      else if (label == "pfDeepFlavourJetTags")
        for (const auto& cat : {"probb", "probbb", "problepb"})
          jetTaggers_["deepFlavour"].emplace(cat, consumes<JetTagCollection>(tag + ":" + cat));
      else if (label == "pfParticleTransformerAK4JetTags")
        for (const auto& cat : {"probb", "probbb", "problepb"})
          jetTaggers_["particleTransformer"].emplace(cat, consumes<JetTagCollection>(tag + ":" + cat));
      else if (label == "pfUnifiedParticleTransformerAK4JetTags")
        for (const auto& cat :
             {"probb",        "probbb",       "problepb",     "probc",        "probg",        "probu",
              "probd",        "probs",        "probtaup1h0p", "probtaup1h1p", "probtaup1h2p", "probtaup3h0p",
              "probtaup3h1p", "probtaum1h0p", "probtaum1h1p", "probtaum1h2p", "probtaum3h0p", "probtaum3h1p",
              "probele",      "probmu",       "ptcorr",       "ptnu"})
          jetTaggers_["unifiedParticleTransformer"].emplace(cat, consumes<JetTagCollection>(tag + ":" + cat));
    }
  }
  doSubEvent_ = false;

  if (isMC_) {
    genPtMin_ = iConfig.getUntrackedParameter<double>("genPtMin", 10);
    doSubEvent_ = iConfig.getUntrackedParameter<bool>("doSubEvent", false);
  }
}

HiInclusiveJetAnalyzer::~HiInclusiveJetAnalyzer() {}

void HiInclusiveJetAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& es) {}
void HiInclusiveJetAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& es) {}

void HiInclusiveJetAnalyzer::beginJob() {
  string jetTagTitle = jetTagLabel_.label() + " Jet Analysis Tree";
  t = fs1->make<TTree>("t", jetTagTitle.c_str());

  // Filter branch booking, not branch reading: omitted payloads are never written.
  const auto bookBranch = [this](const char* name, auto&&... args) {
    if (branchesToKeep_.empty() ||
        std::find(branchesToKeep_.begin(), branchesToKeep_.end(), name) != branchesToKeep_.end())
      t->Branch(name, std::forward<decltype(args)>(args)...);
  };

  bookBranch("run", &jets_.run, "run/I");
  bookBranch("evt", &jets_.evt, "evt/I");
  bookBranch("lumi", &jets_.lumi, "lumi/I");
  bookBranch("nref", &jets_.nref, "nref/I");
  bookBranch("ncalo", &jets_.ncalo, "ncalo/I");
  bookBranch("rawpt", jets_.rawpt, "rawpt[nref]/F");
  bookBranch("jtpt", jets_.jtpt, "jtpt[nref]/F");
  bookBranch("jteta", jets_.jteta, "jteta[nref]/F");
  bookBranch("jty", jets_.jty, "jty[nref]/F");
  bookBranch("jtphi", jets_.jtphi, "jtphi[nref]/F");
  bookBranch("jtpu", jets_.jtpu, "jtpu[nref]/F");
  bookBranch("jtm", jets_.jtm, "jtm[nref]/F");
  bookBranch("jtarea", jets_.jtarea, "jtarea[nref]/F");

  if (doCaloJets_) {
    bookBranch("ncalo", &jets_.ncalo, "ncalo/I");
    bookBranch("calopt", jets_.calopt, "calopt[ncalo]/F");
    bookBranch("caloeta", jets_.caloeta, "caloeta[ncalo]/F");
    bookBranch("calophi", jets_.calophi, "calophi[ncalo]/F");
  }

  //for reWTA reclustering
  if (doWTARecluster_) {
    bookBranch("WTAeta", jets_.WTAeta, "WTAeta[nref]/F");
    bookBranch("WTAphi", jets_.WTAphi, "WTAphi[nref]/F");
  }

  bookBranch("jtPfCHF", jets_.jtPfCHF, "jtPfCHF[nref]/F");
  bookBranch("jtPfNHF", jets_.jtPfNHF, "jtPfNHF[nref]/F");
  bookBranch("jtPfCEF", jets_.jtPfCEF, "jtPfCEF[nref]/F");
  bookBranch("jtPfNEF", jets_.jtPfNEF, "jtPfNEF[nref]/F");
  bookBranch("jtPfMUF", jets_.jtPfMUF, "jtPfMUF[nref]/F");

  bookBranch("jtPfCHM", jets_.jtPfCHM, "jtPfCHM[nref]/I");
  bookBranch("jtPfNHM", jets_.jtPfNHM, "jtPfNHM[nref]/I");
  bookBranch("jtPfCEM", jets_.jtPfCEM, "jtPfCEM[nref]/I");
  bookBranch("jtPfNEM", jets_.jtPfNEM, "jtPfNEM[nref]/I");
  bookBranch("jtPfMUM", jets_.jtPfMUM, "jtPfMUM[nref]/I");
  bookBranch("jtPfCM", jets_.jtPfCM, "jtPfCM[nref]/I");
  bookBranch("jtPfNM", jets_.jtPfNM, "jtPfNM[nref]/I");

  bookBranch("jttau1", jets_.jttau1, "jttau1[nref]/F");
  bookBranch("jttau2", jets_.jttau2, "jttau2[nref]/F");
  bookBranch("jttau3", jets_.jttau3, "jttau3[nref]/F");

  if (doSubJets_) {
    bookBranch("jtSubJetPt", &jets_.jtSubJetPt);
    bookBranch("jtSubJetEta", &jets_.jtSubJetEta);
    bookBranch("jtSubJetPhi", &jets_.jtSubJetPhi);
    bookBranch("jtSubJetM", &jets_.jtSubJetM);
    bookBranch("jtsym", jets_.jtsym, "jtsym[nref]/F");
    bookBranch("jtdroppedBranches", jets_.jtdroppedBranches, "jtdroppedBranches[nref]/I");
  }

  if (doJetConstituents_) {
    bookBranch("jtConstituentsId", &jets_.jtConstituentsId);
    bookBranch("jtConstituentsE", &jets_.jtConstituentsE);
    bookBranch("jtConstituentsPt", &jets_.jtConstituentsPt);
    bookBranch("jtConstituentsEta", &jets_.jtConstituentsEta);
    bookBranch("jtConstituentsPhi", &jets_.jtConstituentsPhi);
    bookBranch("jtConstituentsM", &jets_.jtConstituentsM);
    bookBranch("jtSDConstituentsId", &jets_.jtSDConstituentsId);
    bookBranch("jtSDConstituentsE", &jets_.jtSDConstituentsE);
    bookBranch("jtSDConstituentsPt", &jets_.jtSDConstituentsPt);
    bookBranch("jtSDConstituentsEta", &jets_.jtSDConstituentsEta);
    bookBranch("jtSDConstituentsPhi", &jets_.jtSDConstituentsPhi);
    bookBranch("jtSDConstituentsM", &jets_.jtSDConstituentsM);
  }
  // jet ID information, jet composition
  if (doHiJetID_) {
    bookBranch("trackMax", jets_.trackMax, "trackMax[nref]/F");
    bookBranch("trackSum", jets_.trackSum, "trackSum[nref]/F");
    bookBranch("trackN", jets_.trackN, "trackN[nref]/I");
    bookBranch("trackHardSum", jets_.trackHardSum, "trackHardSum[nref]/F");
    bookBranch("trackHardN", jets_.trackHardN, "trackHardN[nref]/I");

    bookBranch("chargedMax", jets_.chargedMax, "chargedMax[nref]/F");
    bookBranch("chargedSum", jets_.chargedSum, "chargedSum[nref]/F");
    bookBranch("chargedN", jets_.chargedN, "chargedN[nref]/I");
    bookBranch("chargedHardSum", jets_.chargedHardSum, "chargedHardSum[nref]/F");
    bookBranch("chargedHardN", jets_.chargedHardN, "chargedHardN[nref]/I");

    bookBranch("photonMax", jets_.photonMax, "photonMax[nref]/F");
    bookBranch("photonSum", jets_.photonSum, "photonSum[nref]/F");
    bookBranch("photonN", jets_.photonN, "photonN[nref]/I");
    bookBranch("photonHardSum", jets_.photonHardSum, "photonHardSum[nref]/F");
    bookBranch("photonHardN", jets_.photonHardN, "photonHardN[nref]/I");

    bookBranch("neutralMax", jets_.neutralMax, "neutralMax[nref]/F");
    bookBranch("neutralSum", jets_.neutralSum, "neutralSum[nref]/F");
    bookBranch("neutralN", jets_.neutralN, "neutralN[nref]/I");

    bookBranch("eMax", jets_.eMax, "eMax[nref]/F");
    bookBranch("eSum", jets_.eSum, "eSum[nref]/F");
    bookBranch("eN", jets_.eN, "eN[nref]/I");

    bookBranch("muMax", jets_.muMax, "muMax[nref]/F");
    bookBranch("muSum", jets_.muSum, "muSum[nref]/F");
    bookBranch("muN", jets_.muN, "muN[nref]/I");
  }

  if (doStandardJetID_) {
    bookBranch("fHPD", jets_.fHPD, "fHPD[nref]/F");
    bookBranch("fRBX", jets_.fRBX, "fRBX[nref]/F");
    bookBranch("n90", jets_.n90, "n90[nref]/I");
    bookBranch("fSubDet1", jets_.fSubDet1, "fSubDet1[nref]/F");
    bookBranch("fSubDet2", jets_.fSubDet2, "fSubDet2[nref]/F");
    bookBranch("fSubDet3", jets_.fSubDet3, "fSubDet3[nref]/F");
    bookBranch("fSubDet4", jets_.fSubDet4, "fSubDet4[nref]/F");
    bookBranch("restrictedEMF", jets_.restrictedEMF, "restrictedEMF[nref]/F");
    bookBranch("nHCAL", jets_.nHCAL, "nHCAL[nref]/I");
    bookBranch("nECAL", jets_.nECAL, "nECAL[nref]/I");
    bookBranch("apprHPD", jets_.apprHPD, "apprHPD[nref]/F");
    bookBranch("apprRBX", jets_.apprRBX, "apprRBX[nref]/F");
    bookBranch("n2RPC", jets_.n2RPC, "n2RPC[nref]/I");
    bookBranch("n3RPC", jets_.n3RPC, "n3RPC[nref]/I");
    bookBranch("nRPC", jets_.nRPC, "nRPC[nref]/I");

    bookBranch("fEB", jets_.fEB, "fEB[nref]/F");
    bookBranch("fEE", jets_.fEE, "fEE[nref]/F");
    bookBranch("fHB", jets_.fHB, "fHB[nref]/F");
    bookBranch("fHE", jets_.fHE, "fHE[nref]/F");
    bookBranch("fHO", jets_.fHO, "fHO[nref]/F");
    bookBranch("fLong", jets_.fLong, "fLong[nref]/F");
    bookBranch("fShort", jets_.fShort, "fShort[nref]/F");
    bookBranch("fLS", jets_.fLS, "fLS[nref]/F");
    bookBranch("fHFOOT", jets_.fHFOOT, "fHFOOT[nref]/F");
  }

  // Jet ID
  if (doMatch_) {
    bookBranch("matchedPt", jets_.matchedPt, "matchedPt[nref]/F");
    bookBranch("matchedRawPt", jets_.matchedRawPt, "matchedRawPt[nref]/F");
    bookBranch("matchedPu", jets_.matchedPu, "matchedPu[nref]/F");
    bookBranch("matchedR", jets_.matchedR, "matchedR[nref]/F");
    if (isMC_) {
      bookBranch("matchedHadronFlavor", jets_.matchedHadronFlavor, "matchedHadronFlavor[nref]/I");
      bookBranch("matchedPartonFlavor", jets_.matchedPartonFlavor, "matchedPartonFlavor[nref]/I");
    }
  }

  if (doBtagging_) {
    for (const auto& tg : jetTaggers_) {
      auto& discr = jets_discr_[tg.first];
      bookBranch(("discr_" + tg.first).c_str(), discr["b"].data(), ("discr_" + tg.first + "[nref]/F").c_str());
      if (tg.first == "unifiedParticleTransformer") {
        bookBranch(("discr_" + tg.first + "_probtau").c_str(),
                  discr["probtau"].data(),
                  ("discr_" + tg.first + "_probtau[nref]/F").c_str());
        for (const auto& c : tg.second)
          if (c.first.rfind("probtau", 0) != 0)
            bookBranch(("discr_" + tg.first + "_" + c.first).c_str(),
                      discr[c.first].data(),
                      ("discr_" + tg.first + "_" + c.first + "[nref]/F").c_str());
      }
    }
  }
  if (isMC_) {
    if (useHepMC_) {
      bookBranch("beamId1", &jets_.beamId1, "beamId1/I");
      bookBranch("beamId2", &jets_.beamId2, "beamId2/I");
    }

    bookBranch("pthat", &jets_.pthat, "pthat/F");

    // Only matched gen jets
    bookBranch("refpt", jets_.refpt, "refpt[nref]/F");
    bookBranch("refeta", jets_.refeta, "refeta[nref]/F");
    bookBranch("refy", jets_.refy, "refy[nref]/F");
    bookBranch("refphi", jets_.refphi, "refphi[nref]/F");
    bookBranch("refm", jets_.refm, "refm[nref]/F");
    bookBranch("refarea", jets_.refarea, "refarea[nref]/F");

    if (doGenTaus_) {
      bookBranch("reftau1", jets_.reftau1, "reftau1[nref]/F");
      bookBranch("reftau2", jets_.reftau2, "reftau2[nref]/F");
      bookBranch("reftau3", jets_.reftau3, "reftau3[nref]/F");
    }
    bookBranch("refdphijt", jets_.refdphijt, "refdphijt[nref]/F");
    bookBranch("refdrjt", jets_.refdrjt, "refdrjt[nref]/F");
    // matched parton
    bookBranch("refparton_pt", jets_.refparton_pt, "refparton_pt[nref]/F");
    bookBranch("refparton_flavor", jets_.refparton_flavor, "refparton_flavor[nref]/I");
    bookBranch("refparton_flavorForB", jets_.refparton_flavorForB, "refparton_flavorForB[nref]/I");

    if (doGenSubJets_) {
      bookBranch("refptG", jets_.refptG, "refptG[nref]/F");
      bookBranch("refetaG", jets_.refetaG, "refetaG[nref]/F");
      bookBranch("refphiG", jets_.refphiG, "refphiG[nref]/F");
      bookBranch("refmG", jets_.refmG, "refmG[nref]/F");
      bookBranch("refSubJetPt", &jets_.refSubJetPt);
      bookBranch("refSubJetEta", &jets_.refSubJetEta);
      bookBranch("refSubJetPhi", &jets_.refSubJetPhi);
      bookBranch("refSubJetM", &jets_.refSubJetM);
      bookBranch("refsym", jets_.refsym, "refsym[nref]/F");
      bookBranch("refdroppedBranches", jets_.refdroppedBranches, "refdroppedBranches[nref]/I");
    }

    if (doJetConstituents_) {
      bookBranch("refConstituentsId", &jets_.refConstituentsId);
      bookBranch("refConstituentsE", &jets_.refConstituentsE);
      bookBranch("refConstituentsPt", &jets_.refConstituentsPt);
      bookBranch("refConstituentsEta", &jets_.refConstituentsEta);
      bookBranch("refConstituentsPhi", &jets_.refConstituentsPhi);
      bookBranch("refConstituentsM", &jets_.refConstituentsM);
      bookBranch("refSDConstituentsId", &jets_.refSDConstituentsId);
      bookBranch("refSDConstituentsE", &jets_.refSDConstituentsE);
      bookBranch("refSDConstituentsPt", &jets_.refSDConstituentsPt);
      bookBranch("refSDConstituentsEta", &jets_.refSDConstituentsEta);
      bookBranch("refSDConstituentsPhi", &jets_.refSDConstituentsPhi);
      bookBranch("refSDConstituentsM", &jets_.refSDConstituentsM);
    }

    bookBranch("genChargedSum", jets_.genChargedSum, "genChargedSum[nref]/F");
    bookBranch("genHardSum", jets_.genHardSum, "genHardSum[nref]/F");
    bookBranch("signalChargedSum", jets_.signalChargedSum, "signalChargedSum[nref]/F");
    bookBranch("signalHardSum", jets_.signalHardSum, "signalHardSum[nref]/F");

    if (doSubEvent_) {
      bookBranch("subid", jets_.subid, "subid[nref]/I");
    }

    if (fillGenJets_) {
      // For all gen jets, matched or unmatched
      bookBranch("ngen", &jets_.ngen, "ngen/I");
      bookBranch("genmatchindex", jets_.genmatchindex, "genmatchindex[ngen]/I");
      bookBranch("genpt", jets_.genpt, "genpt[ngen]/F");
      bookBranch("geneta", jets_.geneta, "geneta[ngen]/F");
      bookBranch("geny", jets_.geny, "geny[ngen]/F");
      if (doGenTaus_) {
        bookBranch("gentau1", jets_.gentau1, "gentau1[ngen]/F");
        bookBranch("gentau2", jets_.gentau2, "gentau2[ngen]/F");
        bookBranch("gentau3", jets_.gentau3, "gentau3[ngen]/F");
      }
      bookBranch("genphi", jets_.genphi, "genphi[ngen]/F");
      bookBranch("genm", jets_.genm, "genm[ngen]/F");
      bookBranch("gendphijt", jets_.gendphijt, "gendphijt[ngen]/F");
      bookBranch("gendrjt", jets_.gendrjt, "gendrjt[ngen]/F");

      //for reWTA reclustering
      if (doWTARecluster_) {
        bookBranch("WTAgeneta", jets_.WTAgeneta, "WTAgeneta[ngen]/F");
        bookBranch("WTAgenphi", jets_.WTAgenphi, "WTAgenphi[ngen]/F");
      }

      if (doGenSubJets_) {
        bookBranch("genptG", jets_.genptG, "genptG[ngen]/F");
        bookBranch("genetaG", jets_.genetaG, "genetaG[ngen]/F");
        bookBranch("genphiG", jets_.genphiG, "genphiG[ngen]/F");
        bookBranch("genmG", jets_.genmG, "genmG[ngen]/F");
        bookBranch("genSubJetPt", &jets_.genSubJetPt);
        bookBranch("genSubJetEta", &jets_.genSubJetEta);
        bookBranch("genSubJetPhi", &jets_.genSubJetPhi);
        bookBranch("genSubJetM", &jets_.genSubJetM);
        bookBranch("gensym", jets_.gensym, "gensym[ngen]/F");
        bookBranch("gendroppedBranches", jets_.gendroppedBranches, "gendroppedBranches[ngen]/I");
      }

      if (doJetConstituents_) {
        bookBranch("genConstituentsId", &jets_.genConstituentsId);
        bookBranch("genConstituentsE", &jets_.genConstituentsE);
        bookBranch("genConstituentsPt", &jets_.genConstituentsPt);
        bookBranch("genConstituentsEta", &jets_.genConstituentsEta);
        bookBranch("genConstituentsPhi", &jets_.genConstituentsPhi);
        bookBranch("genConstituentsM", &jets_.genConstituentsM);
        bookBranch("genSDConstituentsId", &jets_.genSDConstituentsId);
        bookBranch("genSDConstituentsE", &jets_.genSDConstituentsE);
        bookBranch("genSDConstituentsPt", &jets_.genSDConstituentsPt);
        bookBranch("genSDConstituentsEta", &jets_.genSDConstituentsEta);
        bookBranch("genSDConstituentsPhi", &jets_.genSDConstituentsPhi);
        bookBranch("genSDConstituentsM", &jets_.genSDConstituentsM);
      }

      if (doSubEvent_) {
        bookBranch("gensubid", jets_.gensubid, "gensubid[ngen]/I");
      }
    }
  }

  if (doBtagging_) {
    for (auto& t : jets_discr_)
      for (auto& c : t.second)
        memset(c.second.data(), 0, MAXJETS * sizeof(float));
  }

  for (const auto& name : branchesToKeep_)
    if (!t->GetBranch(name.c_str()))
      throw cms::Exception("Configuration") << "branchesToKeep requested an unavailable branch: " << name;
}

void HiInclusiveJetAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
  int event = iEvent.id().event();
  int run = iEvent.id().run();
  int lumi = iEvent.id().luminosityBlock();

  jets_.run = run;
  jets_.evt = event;
  jets_.lumi = lumi;

  LogDebug("HiInclusiveJetAnalyzer") << "START event: " << event << " in run " << run << endl;

  // loop the events
  edm::Handle<pat::JetCollection> jets;
  iEvent.getByToken(jetTag_, jets);

  edm::Handle<reco::CaloJetCollection> calojets;
  if (doCaloJets_)
    iEvent.getByToken(caloJetTag_, calojets);

  edm::Handle<pat::JetCollection> matchedjets;
  iEvent.getByToken(matchTag_, matchedjets);

  if (doGenSubJets_)
    iEvent.getByToken(subjetGenTag_, gensubjets_);
  if (doGenSym_) {
    iEvent.getByToken(tokenGenSym_, genSymVM_);
    iEvent.getByToken(tokenGenDroppedBranches_, genDroppedBranchesVM_);
  }

  edm::Handle<edm::View<pat::PackedCandidate>> pfCandidates;
  iEvent.getByToken(pfCandidateLabel_, pfCandidates);
  edm::Handle<reco::JetFlavourInfoMatchingCollection> jetFlavourInfos;

  if (isMC_) {
    edm::Handle<reco::GenParticleCollection> genparts;
    iEvent.getByToken(genParticleSrc_, genparts);
    iEvent.getByToken(jetFlavourInfosToken_, jetFlavourInfos);
  }
  /*
  edm::Handle<JetTagCollection> bTags_partTransf, bTags_partTransfBB, bTags_partTransfLepB;
  if (doBtagging_) {
    bTags_partTransf = iEvent.getHandle(particleTransformerJetTagsTkn_);
    bTags_partTransfBB = iEvent.getHandle(particleTransformerJetTagsBBTkn_);
    bTags_partTransfLepB = iEvent.getHandle(particleTransformerJetTagsLepBTkn_);
  }
  */
  std::map<std::string, std::map<std::string, edm::Handle<JetTagCollection>>> jetTaggers;
  if (doBtagging_) {
    for (const auto& t : jetTaggers_)
      for (const auto& c : t.second)
        jetTaggers[t.first].emplace(c.first, iEvent.getHandle(c.second));
  }

  // FILL JRA TREE
  jets_.nref = 0;
  jets_.ncalo = 0;

  if (doJetConstituents_) {
    jets_.jtConstituentsId.clear();
    jets_.jtConstituentsE.clear();
    jets_.jtConstituentsPt.clear();
    jets_.jtConstituentsEta.clear();
    jets_.jtConstituentsPhi.clear();
    jets_.jtConstituentsM.clear();
    jets_.jtSDConstituentsE.clear();
    jets_.jtSDConstituentsPt.clear();
    jets_.jtSDConstituentsEta.clear();
    jets_.jtSDConstituentsPhi.clear();
    jets_.jtSDConstituentsM.clear();

    jets_.refConstituentsId.clear();
    jets_.refConstituentsE.clear();
    jets_.refConstituentsPt.clear();
    jets_.refConstituentsEta.clear();
    jets_.refConstituentsPhi.clear();
    jets_.refConstituentsM.clear();
    jets_.refSDConstituentsE.clear();
    jets_.refSDConstituentsPt.clear();
    jets_.refSDConstituentsEta.clear();
    jets_.refSDConstituentsPhi.clear();
    jets_.refSDConstituentsM.clear();

    jets_.genConstituentsId.clear();
    jets_.genConstituentsE.clear();
    jets_.genConstituentsPt.clear();
    jets_.genConstituentsEta.clear();
    jets_.genConstituentsPhi.clear();
    jets_.genConstituentsM.clear();
    jets_.genSDConstituentsE.clear();
    jets_.genSDConstituentsPt.clear();
    jets_.genSDConstituentsEta.clear();
    jets_.genSDConstituentsPhi.clear();
    jets_.genSDConstituentsM.clear();
  }

  auto getTag = [](const edm::Handle<reco::JetTagCollection>& bTags, const pat::Jet& jet) {
    float tagValue(-999), maxDR2(9.8690);  // = 3.1415^2
    for (const auto& t : *bTags) {
      auto const dR2 = deltaR2(jet, *(t.first));
      if (dR2 > maxDR2)
        continue;
      maxDR2 = dR2;
      tagValue = t.second;
    }
    if (maxDR2 > 0.16)
      tagValue = -999;  // 0.16 = 0.4^2
    return tagValue;
  };

  for (unsigned int j = 0; j < jets->size(); ++j) {
    const pat::Jet& jet = (*jets)[j];

    auto pt = useRawPt_ ? jet.correctedJet("Uncorrected").pt() : jet.pt();
    if (pt < jetPtMin_)
      continue;
    if (std::abs(jet.eta()) > jetAbsEtaMax_)
      continue;

    if (doBtagging_) {
      for (const auto& t : jetTaggers) {
        auto& discr = jets_discr_.at(t.first);
        if (t.first == "pfJP")
          discr["b"][jets_.nref] = getTag(t.second.at("probb"), jet);
        else if (t.first == "deepCSV")
          discr["b"][jets_.nref] = getTag(t.second.at("probb"), jet) + getTag(t.second.at("probbb"), jet);
        else if (t.first == "deepFlavour" || t.first == "particleTransformer" ||
                 t.first == "unifiedParticleTransformer")
          discr["b"][jets_.nref] = getTag(t.second.at("probb"), jet) + getTag(t.second.at("probbb"), jet) +
                                   getTag(t.second.at("problepb"), jet);
        if (t.first == "unifiedParticleTransformer") {
          float tag(0.0);
          for (const auto& n : {"probtaup1h0p",
                                "probtaup1h1p",
                                "probtaup1h2p",
                                "probtaup3h0p",
                                "probtaup3h1p",
                                "probtaum1h0p",
                                "probtaum1h1p",
                                "probtaum1h2p",
                                "probtaum3h0p",
                                "probtaum3h1p"})
            tag += getTag(t.second.at(n), jet);
          discr["probtau"][jets_.nref] = tag;
          for (const auto& c : t.second)
            if (c.first.rfind("probtau", 0) != 0)
              discr[c.first][jets_.nref] = getTag(c.second, jet);
        }
      }
    }
    /*
    if (doBtagging_) {
      std::string pNetLabel = "pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags";
      std::string BvsAllLabel_ = pNetLabel+":BvsAll";
      jets_.discr_BvsAll[jets_.nref] = jet.bDiscriminator(BvsAllLabel_);
      std::string CvsBLabel_ = pNetLabel+":CvsB";
      jets_.discr_CvsB[jets_.nref] = jet.bDiscriminator(CvsBLabel_);
      std::string CvsLLabel_ = pNetLabel+":CvsL";
      jets_.discr_CvsL[jets_.nref] = jet.bDiscriminator(CvsLLabel_);
    }
    */
    if (doHiJetID_) {
      // Jet ID variables

      jets_.muMax[jets_.nref] = 0;
      jets_.muSum[jets_.nref] = 0;
      jets_.muN[jets_.nref] = 0;

      jets_.eMax[jets_.nref] = 0;
      jets_.eSum[jets_.nref] = 0;
      jets_.eN[jets_.nref] = 0;

      jets_.neutralMax[jets_.nref] = 0;
      jets_.neutralSum[jets_.nref] = 0;
      jets_.neutralN[jets_.nref] = 0;

      jets_.photonMax[jets_.nref] = 0;
      jets_.photonSum[jets_.nref] = 0;
      jets_.photonN[jets_.nref] = 0;
      jets_.photonHardSum[jets_.nref] = 0;
      jets_.photonHardN[jets_.nref] = 0;

      jets_.chargedMax[jets_.nref] = 0;
      jets_.chargedSum[jets_.nref] = 0;
      jets_.chargedN[jets_.nref] = 0;
      jets_.chargedHardSum[jets_.nref] = 0;
      jets_.chargedHardN[jets_.nref] = 0;

      jets_.trackMax[jets_.nref] = 0;
      jets_.trackSum[jets_.nref] = 0;
      jets_.trackN[jets_.nref] = 0;
      jets_.trackHardSum[jets_.nref] = 0;
      jets_.trackHardN[jets_.nref] = 0;

      jets_.genChargedSum[jets_.nref] = 0;
      jets_.genHardSum[jets_.nref] = 0;

      jets_.signalChargedSum[jets_.nref] = 0;
      jets_.signalHardSum[jets_.nref] = 0;

      jets_.subid[jets_.nref] = -1;

      for (unsigned int icand = 0; icand < pfCandidates->size(); ++icand) {
        const pat::PackedCandidate& t = (*pfCandidates)[icand];

        if (!t.hasTrackDetails())
          continue;

        reco::Track const& track = t.pseudoTrack();

        if (useQuality_) {
          bool goodtrack = track.quality(reco::TrackBase::qualityByName(trackQuality_));
          if (!goodtrack)
            continue;
        }

        double dr2 = deltaR2(jet, track);
        if (dr2 < r2Param) {
          double ptcand = track.pt();
          jets_.trackSum[jets_.nref] += ptcand;
          jets_.trackN[jets_.nref] += 1;

          if (ptcand > hardPtMin_) {
            jets_.trackHardSum[jets_.nref] += ptcand;
            jets_.trackHardN[jets_.nref] += 1;
          }
          if (ptcand > jets_.trackMax[jets_.nref])
            jets_.trackMax[jets_.nref] = ptcand;
        }
      }

      reco::PFCandidate converter = reco::PFCandidate();
      for (const auto& track : *pfCandidates) {
        double dr2 = deltaR2(jet, track);
        if (dr2 < r2Param) {
          double ptcand = track.pt();
          int pfid = converter.translatePdgIdToType(track.pdgId());

          switch (pfid) {
            case 1:
              jets_.chargedSum[jets_.nref] += ptcand;
              jets_.chargedN[jets_.nref] += 1;
              if (ptcand > hardPtMin_) {
                jets_.chargedHardSum[jets_.nref] += ptcand;
                jets_.chargedHardN[jets_.nref] += 1;
              }
              if (ptcand > jets_.chargedMax[jets_.nref])
                jets_.chargedMax[jets_.nref] = ptcand;
              break;

            case 2:
              jets_.eSum[jets_.nref] += ptcand;
              jets_.eN[jets_.nref] += 1;
              if (ptcand > jets_.eMax[jets_.nref])
                jets_.eMax[jets_.nref] = ptcand;
              break;

            case 3:
              jets_.muSum[jets_.nref] += ptcand;
              jets_.muN[jets_.nref] += 1;
              if (ptcand > jets_.muMax[jets_.nref])
                jets_.muMax[jets_.nref] = ptcand;
              break;

            case 4:
              jets_.photonSum[jets_.nref] += ptcand;
              jets_.photonN[jets_.nref] += 1;
              if (ptcand > hardPtMin_) {
                jets_.photonHardSum[jets_.nref] += ptcand;
                jets_.photonHardN[jets_.nref] += 1;
              }
              if (ptcand > jets_.photonMax[jets_.nref])
                jets_.photonMax[jets_.nref] = ptcand;
              break;

            case 5:
              jets_.neutralSum[jets_.nref] += ptcand;
              jets_.neutralN[jets_.nref] += 1;
              if (ptcand > jets_.neutralMax[jets_.nref])
                jets_.neutralMax[jets_.nref] = ptcand;
              break;

            default:
              break;
          }
        }
      }
    }

    if (doMatch_) {
      // Alternative reconstruction matching (PF for calo, calo for PF)

      double dr2Min = 10000;
      for (unsigned int imatch = 0; imatch < matchedjets->size(); ++imatch) {
        const pat::Jet& mjet = (*matchedjets)[imatch];

        double dr2 = deltaR2(jet, mjet);
        if (dr2 < dr2Min) {
          jets_.matchedPt[jets_.nref] = mjet.pt();

          jets_.matchedRawPt[jets_.nref] = mjet.correctedJet("Uncorrected").pt();
          jets_.matchedPu[jets_.nref] = mjet.pileup();
          if (isMC_) {
            jets_.matchedHadronFlavor[jets_.nref] = mjet.hadronFlavour();
            jets_.matchedPartonFlavor[jets_.nref] = mjet.partonFlavour();
          }

          jets_.matchedR[jets_.nref] = std::sqrt(dr2);
          dr2Min = dr2;
        }
      }
    }

    jets_.rawpt[jets_.nref] = jet.correctedJet("Uncorrected").pt();
    jets_.jtpt[jets_.nref] = jet.pt();
    jets_.jteta[jets_.nref] = jet.eta();
    jets_.jtphi[jets_.nref] = jet.phi();
    jets_.jty[jets_.nref] = jet.eta();
    jets_.jtpu[jets_.nref] = jet.pileup();
    jets_.jtm[jets_.nref] = jet.mass();
    jets_.jtarea[jets_.nref] = jet.jetArea();

    //recluster the jet constituents in reWTA scheme-------------------------
    if (doWTARecluster_) {
      std::vector<fastjet::PseudoJet> candidates;
      auto daughters = jet.getJetConstituents();
      for (auto it = daughters.begin(); it != daughters.end(); ++it) {
        candidates.push_back(fastjet::PseudoJet((**it).px(), (**it).py(), (**it).pz(), (**it).energy()));
      }
      auto cs = new fastjet::ClusterSequence(candidates, WTAjtDef);
      std::vector<fastjet::PseudoJet> wtajt = fastjet::sorted_by_pt(cs->inclusive_jets(0));

      jets_.WTAeta[jets_.nref] = (!wtajt.empty()) ? wtajt[0].eta() : -999;
      jets_.WTAphi[jets_.nref] = (!wtajt.empty()) ? wtajt[0].phi_std() : -999;
      delete cs;
    }
    //------------------------------------------------------------------

    jets_.jttau1[jets_.nref] = -999.;
    jets_.jttau2[jets_.nref] = -999.;
    jets_.jttau3[jets_.nref] = -999.;

    jets_.jtsym[jets_.nref] = -999.;
    jets_.jtdroppedBranches[jets_.nref] = -999;

    if (doSubJets_)
      analyzeSubjets(jet);

    if (jet.hasUserFloat(jetName_ + "Njettiness:tau1"))
      jets_.jttau1[jets_.nref] = jet.userFloat(jetName_ + "Njettiness:tau1");
    if (jet.hasUserFloat(jetName_ + "Njettiness:tau2"))
      jets_.jttau2[jets_.nref] = jet.userFloat(jetName_ + "Njettiness:tau2");
    if (jet.hasUserFloat(jetName_ + "Njettiness:tau3"))
      jets_.jttau3[jets_.nref] = jet.userFloat(jetName_ + "Njettiness:tau3");

    if (jet.hasUserFloat(jetName_ + "Jets:sym"))
      jets_.jtsym[jets_.nref] = jet.userFloat(jetName_ + "Jets:sym");
    if (jet.hasUserInt(jetName_ + "Jets:droppedBranches"))
      jets_.jtdroppedBranches[jets_.nref] = jet.userInt(jetName_ + "Jets:droppedBranches");

    if (jet.isPFJet()) {
      jets_.jtPfCHF[jets_.nref] = jet.chargedHadronEnergyFraction();
      jets_.jtPfNHF[jets_.nref] = jet.neutralHadronEnergyFraction();
      jets_.jtPfCEF[jets_.nref] = jet.chargedEmEnergyFraction();
      jets_.jtPfNEF[jets_.nref] = jet.neutralEmEnergyFraction();
      jets_.jtPfMUF[jets_.nref] = jet.muonEnergyFraction();

      jets_.jtPfCHM[jets_.nref] = jet.chargedHadronMultiplicity();
      jets_.jtPfNHM[jets_.nref] = jet.neutralHadronMultiplicity();
      jets_.jtPfCEM[jets_.nref] = jet.electronMultiplicity();
      jets_.jtPfNEM[jets_.nref] = jet.photonMultiplicity();
      jets_.jtPfMUM[jets_.nref] = jet.muonMultiplicity();
      jets_.jtPfCM[jets_.nref] = jet.chargedMultiplicity();
      jets_.jtPfNM[jets_.nref] = jet.neutralMultiplicity();
    } else {
      jets_.jtPfCHF[jets_.nref] = 0;
      jets_.jtPfNHF[jets_.nref] = 0;
      jets_.jtPfCEF[jets_.nref] = 0;
      jets_.jtPfNEF[jets_.nref] = 0;
      jets_.jtPfMUF[jets_.nref] = 0;

      jets_.jtPfCHM[jets_.nref] = 0;
      jets_.jtPfNHM[jets_.nref] = 0;
      jets_.jtPfCEM[jets_.nref] = 0;
      jets_.jtPfNEM[jets_.nref] = 0;
      jets_.jtPfMUM[jets_.nref] = 0;
      jets_.jtPfCM[jets_.nref] = 0;
      jets_.jtPfNM[jets_.nref] = 0;
    }

    //    if(isMC_){

    //      for(UInt_t i = 0; i < genparts->size(); ++i){
    // const reco::GenParticle& p = (*genparts)[i];
    // if ( p.status()!=1 || p.charge()==0) continue;
    // double dr = deltaR(jet,p);
    // if(dr < rParam){
    //   double ppt = p.pt();
    //   jets_.genChargedSum[jets_.nref] += ppt;
    //   if(ppt > hardPtMin_) jets_.genHardSum[jets_.nref] += ppt;
    //   if(p.collisionId() == 0){
    //     jets_.signalChargedSum[jets_.nref] += ppt;
    //     if(ppt > hardPtMin_) jets_.signalHardSum[jets_.nref] += ppt;
    //   }
    // }
    //      }
    //    }

    if (isMC_) {
      const reco::GenJet* genjet = jet.genJet();

      if (genjet) {
        jets_.refpt[jets_.nref] = genjet->pt();
        jets_.refeta[jets_.nref] = genjet->eta();
        jets_.refphi[jets_.nref] = genjet->phi();
        jets_.refm[jets_.nref] = genjet->mass();
        jets_.refarea[jets_.nref] = genjet->jetArea();
        jets_.refy[jets_.nref] = genjet->eta();
        jets_.refdphijt[jets_.nref] = reco::deltaPhi(jet.phi(), genjet->phi());
        jets_.refdrjt[jets_.nref] = reco::deltaR(jet.eta(), jet.phi(), genjet->eta(), genjet->phi());

        if (doSubEvent_) {
          const GenParticle* gencon = genjet->getGenConstituent(0);
          jets_.subid[jets_.nref] = gencon->collisionId();
        }

        if (doGenSubJets_)
          analyzeRefSubjets(*genjet);

      } else {
        jets_.refpt[jets_.nref] = -999.;
        jets_.refeta[jets_.nref] = -999.;
        jets_.refphi[jets_.nref] = -999.;
        jets_.refm[jets_.nref] = -999.;
        jets_.refarea[jets_.nref] = -999.;
        jets_.refy[jets_.nref] = -999.;
        jets_.refdphijt[jets_.nref] = -999.;
        jets_.refdrjt[jets_.nref] = -999.;

        if (doJetConstituents_) {
          jets_.refConstituentsId.emplace_back(1, -999);
          jets_.refConstituentsE.emplace_back(1, -999);
          jets_.refConstituentsPt.emplace_back(1, -999);
          jets_.refConstituentsEta.emplace_back(1, -999);
          jets_.refConstituentsPhi.emplace_back(1, -999);
          jets_.refConstituentsM.emplace_back(1, -999);

          jets_.refSDConstituentsId.emplace_back(1, -999);
          jets_.refSDConstituentsE.emplace_back(1, -999);
          jets_.refSDConstituentsPt.emplace_back(1, -999);
          jets_.refSDConstituentsEta.emplace_back(1, -999);
          jets_.refSDConstituentsPhi.emplace_back(1, -999);
          jets_.refSDConstituentsM.emplace_back(1, -999);
        }

        if (doGenSubJets_) {
          jets_.refptG[jets_.nref] = -999.;
          jets_.refetaG[jets_.nref] = -999.;
          jets_.refphiG[jets_.nref] = -999.;
          jets_.refmG[jets_.nref] = -999.;
          jets_.refsym[jets_.nref] = -999.;
          jets_.refdroppedBranches[jets_.nref] = -999;

          jets_.refSubJetPt.emplace_back(1, -999);
          jets_.refSubJetEta.emplace_back(1, -999);
          jets_.refSubJetPhi.emplace_back(1, -999);
          jets_.refSubJetM.emplace_back(1, -999);
        }
      }
      jets_.reftau1[jets_.nref] = -999.;
      jets_.reftau2[jets_.nref] = -999.;
      jets_.reftau3[jets_.nref] = -999.;

      jets_.refparton_flavorForB[jets_.nref] = jet.partonFlavour();

      //      if(jet.genParton()){
      // // matched partons
      // const reco::GenParticle & parton = *jet.genParton();

      // jets_.refparton_pt[jets_.nref] = parton.pt();
      // jets_.refparton_flavor[jets_.nref] = parton.pdgId();

      //      } else {
      jets_.refparton_pt[jets_.nref] = -999;
      jets_.refparton_flavor[jets_.nref] = -999;
      //      }
    }
    jets_.nref++;
  }

  if (isMC_) {
    if (useHepMC_) {
      edm::Handle<HepMCProduct> hepMCProduct;
      iEvent.getByToken(eventInfoTag_, hepMCProduct);
      const HepMC::GenEvent* MCEvt = hepMCProduct->GetEvent();

      std::pair<HepMC::GenParticle*, HepMC::GenParticle*> beamParticles = MCEvt->beam_particles();
      jets_.beamId1 = (beamParticles.first != 0) ? beamParticles.first->pdg_id() : 0;
      jets_.beamId2 = (beamParticles.second != 0) ? beamParticles.second->pdg_id() : 0;
    }

    edm::Handle<GenEventInfoProduct> hEventInfo;
    iEvent.getByToken(eventGenInfoTag_, hEventInfo);

    // binning values and qscale appear to be equivalent, but binning values not always present
    jets_.pthat = hEventInfo->qScale();

    edm::Handle<edm::View<reco::GenJet>> genjets;
    iEvent.getByToken(genjetTag_, genjets);

    //get gen-level n-jettiness
    edm::Handle<edm::ValueMap<float>> genTau1s;
    edm::Handle<edm::ValueMap<float>> genTau2s;
    edm::Handle<edm::ValueMap<float>> genTau3s;
    if (doGenTaus_) {
      iEvent.getByToken(tokenGenTau1_, genTau1s);
      iEvent.getByToken(tokenGenTau2_, genTau2s);
      iEvent.getByToken(tokenGenTau3_, genTau3s);
    }

    jets_.ngen = 0;

    for (unsigned int igen = 0; igen < genjets->size(); ++igen) {
      const reco::GenJet& genjet = (*genjets)[igen];
      float genjet_pt = genjet.pt();

      float tau1 = -999.;
      float tau2 = -999.;
      float tau3 = -999.;
      Ptr<reco::GenJet> genJetPtr = genjets->ptrAt(igen);
      if (doGenTaus_) {
        tau1 = (*genTau1s)[genJetPtr];
        tau2 = (*genTau2s)[genJetPtr];
        tau3 = (*genTau3s)[genJetPtr];
      }

      // find matching patJet if there is one
      jets_.gendrjt[jets_.ngen] = -1.0;
      jets_.genmatchindex[jets_.ngen] = -1;

      for (int ijet = 0; ijet < jets_.nref; ++ijet) {
        // poor man's matching, someone fix please

        double deltaPt = std::abs(genjet.pt() - jets_.refpt[ijet]);  //Note: precision of this ~ .0001, so cut .01
        double deltaEta = std::abs(
            genjet.eta() -
            jets_.refeta
                [ijet]);  //Note: precision of this is  ~.0000001, but keep it low, .0001 is well below cone size and typical pointing resolution
        double deltaPhi = std::abs(reco::deltaPhi(
            genjet.phi(),
            jets_.refphi
                [ijet]));  //Note: precision of this is  ~.0000001, but keep it low, .0001 is well below cone size and typical pointing resolution

        if (deltaPt < 0.01 && deltaEta < .0001 && deltaPhi < .0001) {
          if (genjet_pt > genPtMin_) {
            jets_.genmatchindex[jets_.ngen] = (int)ijet;
            jets_.gendphijt[jets_.ngen] = reco::deltaPhi(jets_.refphi[ijet], genjet.phi());
            jets_.gendrjt[jets_.ngen] = sqrt(pow(jets_.gendphijt[jets_.ngen], 2) + pow(deltaEta, 2));
          }
          if (doGenTaus_) {
            jets_.reftau1[ijet] = tau1;
            jets_.reftau2[ijet] = tau2;
            jets_.reftau3[ijet] = tau3;
          }
          break;
        }
      }

      //reWTA reclustering----------------------------------
      if (doWTARecluster_) {
        if (genjet_pt > genPtMin_) {
          std::vector<fastjet::PseudoJet> candidates;
          auto daughters = genjet.getJetConstituents();
          for (auto it = daughters.begin(); it != daughters.end(); ++it) {
            candidates.push_back(fastjet::PseudoJet((**it).px(), (**it).py(), (**it).pz(), (**it).energy()));
          }
          auto cs = new fastjet::ClusterSequence(candidates, WTAjtDef);
          std::vector<fastjet::PseudoJet> wtajt = fastjet::sorted_by_pt(cs->inclusive_jets(0));

          jets_.WTAgeneta[jets_.ngen] = (!wtajt.empty()) ? wtajt[0].eta() : -999;
          jets_.WTAgenphi[jets_.ngen] = (!wtajt.empty()) ? wtajt[0].phi_std() : -999;
          delete cs;
        }
      }
      //-------------------------------------------------

      // threshold to reduce size of output in minbias PbPb
      if (genjet_pt > genPtMin_) {
        jets_.genpt[jets_.ngen] = genjet_pt;
        jets_.geneta[jets_.ngen] = genjet.eta();
        jets_.genphi[jets_.ngen] = genjet.phi();
        jets_.genm[jets_.ngen] = genjet.mass();
        jets_.geny[jets_.ngen] = genjet.eta();

        if (doGenTaus_) {
          jets_.gentau1[jets_.ngen] = tau1;
          jets_.gentau2[jets_.ngen] = tau2;
          jets_.gentau3[jets_.ngen] = tau3;
        }

        if (doGenSubJets_)
          analyzeGenSubjets(genjet);

        if (doSubEvent_) {
          const GenParticle* gencon = genjet.getGenConstituent(0);
          jets_.gensubid[jets_.ngen] = gencon->collisionId();
        }
        jets_.ngen++;
      }
    }
  }

  if (doCaloJets_) {
    for (unsigned int j = 0; j < calojets->size(); ++j) {
      const reco::Jet& jet = (*calojets)[j];
      jets_.calopt[jets_.ncalo] = jet.pt();
      jets_.caloeta[jets_.ncalo] = jet.eta();
      jets_.calophi[jets_.ncalo] = jet.phi();
      jets_.ncalo++;
    }
  }

  t->Fill();

  //memset(&jets_,0,sizeof jets_);
  jets_ = {0};
}

int HiInclusiveJetAnalyzer::getPFJetMuon(const pat::Jet& pfJet,
                                         const edm::View<pat::PackedCandidate>* pfCandidateColl) {
  int pfMuonIndex = -1;
  float ptMax = 0.;

  for (unsigned icand = 0; icand < pfCandidateColl->size(); icand++) {
    const pat::PackedCandidate& pfCandidate = pfCandidateColl->at(icand);
    int id = pfCandidate.pdgId();
    if (abs(id) != 3)
      continue;

    if (reco::deltaR2(pfJet, pfCandidate) > 0.25)  // 0.25 = 0.5^2
      continue;

    double pt = pfCandidate.pt();
    if (pt > ptMax) {
      ptMax = pt;
      pfMuonIndex = (int)icand;
    }
  }

  return pfMuonIndex;
}

double HiInclusiveJetAnalyzer::getPtRel(const pat::PackedCandidate& lep, const pat::Jet& jet) {
  float lj_x = jet.p4().px();
  float lj_y = jet.p4().py();
  float lj_z = jet.p4().pz();

  // absolute values squared
  float lj2 = lj_x * lj_x + lj_y * lj_y + lj_z * lj_z;
  float lep2 = lep.px() * lep.px() + lep.py() * lep.py() + lep.pz() * lep.pz();

  // projection vec(mu) to lepjet axis
  float lepXlj = lep.px() * lj_x + lep.py() * lj_y + lep.pz() * lj_z;

  // absolute value squared and normalized
  float pLrel2 = lepXlj * lepXlj / lj2;

  // lep2 = pTrel2 + pLrel2
  float pTrel2 = lep2 - pLrel2;

  return (pTrel2 > 0) ? std::sqrt(pTrel2) : 0.0;
}

//--------------------------------------------------------------------------------------------------
void HiInclusiveJetAnalyzer::analyzeSubjets(const reco::Jet& jet) {
  std::vector<float> sjpt;
  std::vector<float> sjeta;
  std::vector<float> sjphi;
  std::vector<float> sjm;
  if (jet.numberOfDaughters() > 0) {
    for (unsigned k = 0; k < jet.numberOfDaughters(); ++k) {
      const reco::Candidate& dp = *jet.daughter(k);
      sjpt.push_back(dp.pt());
      sjeta.push_back(dp.eta());
      sjphi.push_back(dp.phi());
      sjm.push_back(dp.mass());
    }
  } else {
    sjpt.push_back(-999.);
    sjeta.push_back(-999.);
    sjphi.push_back(-999.);
    sjm.push_back(-999.);
  }
  jets_.jtSubJetPt.push_back(sjpt);
  jets_.jtSubJetEta.push_back(sjeta);
  jets_.jtSubJetPhi.push_back(sjphi);
  jets_.jtSubJetM.push_back(sjm);
}

//--------------------------------------------------------------------------------------------------
int HiInclusiveJetAnalyzer::getGroomedGenJetIndex(const reco::GenJet& jet) const {
  //Find closest soft-dropped gen jet
  double dr2Min = 10000;
  int imatch = -1;
  for (unsigned int i = 0; i < gensubjets_->size(); ++i) {
    const reco::Jet& mjet = (*gensubjets_)[i];

    double dr2 = deltaR2(jet, mjet);
    if (dr2 < dr2Min) {
      imatch = i;
      dr2Min = dr2;
    }
  }
  return imatch;
}

//--------------------------------------------------------------------------------------------------
void HiInclusiveJetAnalyzer::analyzeRefSubjets(const reco::GenJet& jet) {
  //Find closest soft-dropped gen jet
  int imatch = getGroomedGenJetIndex(jet);
  double dr2 = 999.;
  if (imatch > -1) {
    const reco::Jet& mjet = (*gensubjets_)[imatch];
    dr2 = deltaR2(jet, mjet);
  }

  jets_.refptG[jets_.nref] = -999.;
  jets_.refetaG[jets_.nref] = -999.;
  jets_.refphiG[jets_.nref] = -999.;
  jets_.refmG[jets_.nref] = -999.;
  jets_.refsym[jets_.nref] = -999.;
  jets_.refdroppedBranches[jets_.nref] = -999;

  std::vector<float> sjpt;
  std::vector<float> sjeta;
  std::vector<float> sjphi;
  std::vector<float> sjm;
  if (imatch > -1 && dr2 < 0.16) {  // 0.16 = 0.4^2
    const reco::Jet& mjet = (*gensubjets_)[imatch];
    jets_.refptG[jets_.nref] = mjet.pt();
    jets_.refetaG[jets_.nref] = mjet.eta();
    jets_.refphiG[jets_.nref] = mjet.phi();
    jets_.refmG[jets_.nref] = mjet.mass();

    if (mjet.numberOfDaughters() > 0) {
      for (unsigned k = 0; k < mjet.numberOfDaughters(); ++k) {
        const reco::Candidate& dp = *mjet.daughter(k);
        sjpt.push_back(dp.pt());
        sjeta.push_back(dp.eta());
        sjphi.push_back(dp.phi());
        sjm.push_back(dp.mass());
      }
    }
    if (doGenSym_) {
      Ptr<reco::Jet> genJetPtr = gensubjets_->ptrAt(imatch);
      float gensym = (*genSymVM_)[genJetPtr];
      jets_.refsym[jets_.nref] = gensym;
      int db = (*genDroppedBranchesVM_)[genJetPtr];
      jets_.refdroppedBranches[jets_.nref] = db;
    }
  } else {
    jets_.refptG[jets_.nref] = -999.;
    jets_.refetaG[jets_.nref] = -999.;
    jets_.refphiG[jets_.nref] = -999.;
    jets_.refmG[jets_.nref] = -999.;

    sjpt.push_back(-999.);
    sjeta.push_back(-999.);
    sjphi.push_back(-999.);
    sjm.push_back(-999.);
  }

  jets_.refSubJetPt.push_back(sjpt);
  jets_.refSubJetEta.push_back(sjeta);
  jets_.refSubJetPhi.push_back(sjphi);
  jets_.refSubJetM.push_back(sjm);
}

//--------------------------------------------------------------------------------------------------
void HiInclusiveJetAnalyzer::analyzeGenSubjets(const reco::GenJet& jet) {
  //Find closest soft-dropped gen jet
  int imatch = getGroomedGenJetIndex(jet);
  double dr2 = 999.;
  if (imatch > -1) {
    const reco::Jet& mjet = (*gensubjets_)[imatch];
    dr2 = deltaR2(jet, mjet);
  }

  jets_.genptG[jets_.ngen] = -999.;
  jets_.genetaG[jets_.ngen] = -999.;
  jets_.genphiG[jets_.ngen] = -999.;
  jets_.genmG[jets_.ngen] = -999.;
  jets_.gensym[jets_.ngen] = -999.;
  jets_.gendroppedBranches[jets_.ngen] = -999;

  std::vector<float> sjpt;
  std::vector<float> sjeta;
  std::vector<float> sjphi;
  std::vector<float> sjm;
  std::vector<float> sjarea;
  if (imatch > -1 && dr2 < 0.16) {  // 0.16 = 0.4^2
    const reco::Jet& mjet = (*gensubjets_)[imatch];
    jets_.genptG[jets_.ngen] = mjet.pt();
    jets_.genetaG[jets_.ngen] = mjet.eta();
    jets_.genphiG[jets_.ngen] = mjet.phi();
    jets_.genmG[jets_.ngen] = mjet.mass();

    if (mjet.numberOfDaughters() > 0) {
      for (unsigned k = 0; k < mjet.numberOfDaughters(); ++k) {
        const reco::Candidate& dp = *mjet.daughter(k);
        sjpt.push_back(dp.pt());
        sjeta.push_back(dp.eta());
        sjphi.push_back(dp.phi());
        sjm.push_back(dp.mass());
        //sjarea.push_back(dp.castTo<reco::JetRef>()->jetArea());
      }
    }
    if (doGenSym_) {
      Ptr<reco::Jet> genJetPtr = gensubjets_->ptrAt(imatch);
      float gensym = (*genSymVM_)[genJetPtr];
      jets_.gensym[jets_.ngen] = gensym;
      int db = (*genDroppedBranchesVM_)[genJetPtr];
      jets_.gendroppedBranches[jets_.ngen] = db;
    }
  } else {
    jets_.genptG[jets_.ngen] = -999.;
    jets_.genetaG[jets_.ngen] = -999.;
    jets_.genphiG[jets_.ngen] = -999.;
    jets_.genmG[jets_.ngen] = -999.;

    sjpt.push_back(-999.);
    sjeta.push_back(-999.);
    sjphi.push_back(-999.);
    sjm.push_back(-999.);
    sjarea.push_back(-999.);
  }

  jets_.genSubJetPt.push_back(sjpt);
  jets_.genSubJetEta.push_back(sjeta);
  jets_.genSubJetPhi.push_back(sjphi);
  jets_.genSubJetM.push_back(sjm);
  jets_.genSubJetArea.push_back(sjarea);
}

DEFINE_FWK_MODULE(HiInclusiveJetAnalyzer);
