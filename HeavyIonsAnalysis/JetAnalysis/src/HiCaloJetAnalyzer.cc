/*
  Based on the HiCaloJetAnalyzer
  Modified by Jussi Viinikainen, September 2023
*/

#include "HeavyIonsAnalysis/JetAnalysis/interface/HiCaloJetAnalyzer.h"
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

HiCaloJetAnalyzer::HiCaloJetAnalyzer(const edm::ParameterSet& iConfig) {
  jetTag_ = consumes<reco::CaloJetCollection>(iConfig.getParameter<InputTag>("jetTag"));

  useQuality_ = iConfig.getUntrackedParameter<bool>("useQuality", true);
  trackQuality_ = iConfig.getUntrackedParameter<string>("trackQuality", "highPurity");

  jetName_ = iConfig.getUntrackedParameter<string>("jetName");

  isMC_ = iConfig.getUntrackedParameter<bool>("isMC", false);
  useHepMC_ = iConfig.getUntrackedParameter<bool>("useHepMC", false);
  fillGenJets_ = iConfig.getUntrackedParameter<bool>("fillGenJets", false);

  doHiJetID_ = iConfig.getUntrackedParameter<bool>("doHiJetID", false);
  doCaloEnergyFractions_ = iConfig.getUntrackedParameter<bool>("doCaloEnergyFractions", false);

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
  }

  pfCandidateLabel_ =
      consumes<edm::View<pat::PackedCandidate>>(iConfig.getUntrackedParameter<edm::InputTag>("pfCandidateLabel"));

  if (isMC_)
    genParticleSrc_ =
        consumes<reco::GenParticleCollection>(iConfig.getUntrackedParameter<edm::InputTag>("genParticles"));

  if (isMC_) {
    genPtMin_ = iConfig.getUntrackedParameter<double>("genPtMin", 10);
  }
}

HiCaloJetAnalyzer::~HiCaloJetAnalyzer() {}

void HiCaloJetAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& es) {}
void HiCaloJetAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& es) {}

void HiCaloJetAnalyzer::beginJob() {
  string jetTagTitle = jetTagLabel_.label() + " Jet Analysis Tree";
  caloJetTree_ = fs1->make<TTree>("caloJetTree", jetTagTitle.c_str());

  caloJetTree_->Branch("run", &jets_.run, "run/I");
  caloJetTree_->Branch("evt", &jets_.evt, "evt/I");
  caloJetTree_->Branch("lumi", &jets_.lumi, "lumi/I");
  caloJetTree_->Branch("nref", &jets_.nref, "nref/I");
  caloJetTree_->Branch("rawpt", jets_.rawpt, "rawpt[nref]/F");
  caloJetTree_->Branch("jtpt", jets_.jtpt, "jtpt[nref]/F");
  caloJetTree_->Branch("jteta", jets_.jteta, "jteta[nref]/F");
  caloJetTree_->Branch("jty", jets_.jty, "jty[nref]/F");
  caloJetTree_->Branch("jtphi", jets_.jtphi, "jtphi[nref]/F");
  caloJetTree_->Branch("jtpu", jets_.jtpu, "jtpu[nref]/F");
  caloJetTree_->Branch("jtm", jets_.jtm, "jtm[nref]/F");
  caloJetTree_->Branch("jtarea", jets_.jtarea, "jtarea[nref]/F");

  // jet ID information, jet composition
  if (doHiJetID_) {
    caloJetTree_->Branch("trackMax", jets_.trackMax, "trackMax[nref]/F");
    caloJetTree_->Branch("trackSum", jets_.trackSum, "trackSum[nref]/F");
    caloJetTree_->Branch("trackN", jets_.trackN, "trackN[nref]/I");
    caloJetTree_->Branch("trackHardSum", jets_.trackHardSum, "trackHardSum[nref]/F");
    caloJetTree_->Branch("trackHardN", jets_.trackHardN, "trackHardN[nref]/I");

    caloJetTree_->Branch("chargedMax", jets_.chargedMax, "chargedMax[nref]/F");
    caloJetTree_->Branch("chargedSum", jets_.chargedSum, "chargedSum[nref]/F");
    caloJetTree_->Branch("chargedN", jets_.chargedN, "chargedN[nref]/I");
    caloJetTree_->Branch("chargedHardSum", jets_.chargedHardSum, "chargedHardSum[nref]/F");
    caloJetTree_->Branch("chargedHardN", jets_.chargedHardN, "chargedHardN[nref]/I");

    caloJetTree_->Branch("photonMax", jets_.photonMax, "photonMax[nref]/F");
    caloJetTree_->Branch("photonSum", jets_.photonSum, "photonSum[nref]/F");
    caloJetTree_->Branch("photonN", jets_.photonN, "photonN[nref]/I");
    caloJetTree_->Branch("photonHardSum", jets_.photonHardSum, "photonHardSum[nref]/F");
    caloJetTree_->Branch("photonHardN", jets_.photonHardN, "photonHardN[nref]/I");

    caloJetTree_->Branch("neutralMax", jets_.neutralMax, "neutralMax[nref]/F");
    caloJetTree_->Branch("neutralSum", jets_.neutralSum, "neutralSum[nref]/F");
    caloJetTree_->Branch("neutralN", jets_.neutralN, "neutralN[nref]/I");

    caloJetTree_->Branch("eMax", jets_.eMax, "eMax[nref]/F");
    caloJetTree_->Branch("eSum", jets_.eSum, "eSum[nref]/F");
    caloJetTree_->Branch("eN", jets_.eN, "eN[nref]/I");

    caloJetTree_->Branch("muMax", jets_.muMax, "muMax[nref]/F");
    caloJetTree_->Branch("muSum", jets_.muSum, "muSum[nref]/F");
    caloJetTree_->Branch("muN", jets_.muN, "muN[nref]/I");
  }

  // Calorimeter energy fractions
  if(doCaloEnergyFractions_){
    caloJetTree_->Branch("emEnergyFraction", jets_.emEnergyFraction, "emEnergyFraction[nref]/F");
    caloJetTree_->Branch("hadronicEnergyFraction", jets_.hadronicEnergyFraction, "hadronicEnergyFraction[nref]/F");
  }



  if (isMC_) {
    if (useHepMC_) {
      caloJetTree_->Branch("beamId1", &jets_.beamId1, "beamId1/I");
      caloJetTree_->Branch("beamId2", &jets_.beamId2, "beamId2/I");
    }

    caloJetTree_->Branch("pthat", &jets_.pthat, "pthat/F");

    caloJetTree_->Branch("genChargedSum", jets_.genChargedSum, "genChargedSum[nref]/F");
    caloJetTree_->Branch("genHardSum", jets_.genHardSum, "genHardSum[nref]/F");
    caloJetTree_->Branch("signalChargedSum", jets_.signalChargedSum, "signalChargedSum[nref]/F");
    caloJetTree_->Branch("signalHardSum", jets_.signalHardSum, "signalHardSum[nref]/F");

    if (fillGenJets_) {
      // For all gen jets, matched or unmatched
      caloJetTree_->Branch("ngen", &jets_.ngen, "ngen/I");
      caloJetTree_->Branch("genpt", jets_.genpt, "genpt[ngen]/F");
      caloJetTree_->Branch("geneta", jets_.geneta, "geneta[ngen]/F");
      caloJetTree_->Branch("geny", jets_.geny, "geny[ngen]/F");
      caloJetTree_->Branch("genphi", jets_.genphi, "genphi[ngen]/F");
      caloJetTree_->Branch("genm", jets_.genm, "genm[ngen]/F");

    }
  }

}

void HiCaloJetAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup) {
  int event = iEvent.id().event();
  int run = iEvent.id().run();
  int lumi = iEvent.id().luminosityBlock();

  jets_.run = run;
  jets_.evt = event;
  jets_.lumi = lumi;

  LogDebug("HiCaloJetAnalyzer") << "START event: " << event << " in run " << run << endl;

  // loop the events
  edm::Handle<reco::CaloJetCollection> jets;
  iEvent.getByToken(jetTag_, jets);

  edm::Handle<edm::View<pat::PackedCandidate>> pfCandidates;
  iEvent.getByToken(pfCandidateLabel_, pfCandidates);
  if (isMC_) {
    edm::Handle<reco::GenParticleCollection> genparts;
    iEvent.getByToken(genParticleSrc_, genparts);
  }

  // FILL JRA TREE
  jets_.nref = 0;

  for (const auto& jet : *jets) {

    auto pt = jet.pt();
    if (pt < jetPtMin_)
      continue;
    if (std::abs(jet.eta()) > jetAbsEtaMax_)
      continue;

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

      for (const auto& currentCandidate : *pfCandidates) {

        if (!currentCandidate.hasTrackDetails())
          continue;

        reco::Track const& track = currentCandidate.pseudoTrack();

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

    // Calorimeter energy fractions
    if(doCaloEnergyFractions_){
      jets_.emEnergyFraction[jets_.nref] = jet.emEnergyFraction();
      jets_.hadronicEnergyFraction[jets_.nref] = jet.energyFractionHadronic();
    }

    jets_.rawpt[jets_.nref] = jet.pt();
    jets_.jtpt[jets_.nref] = jet.pt();
    jets_.jteta[jets_.nref] = jet.eta();
    jets_.jtphi[jets_.nref] = jet.phi();
    jets_.jty[jets_.nref] = jet.eta();
    jets_.jtpu[jets_.nref] = jet.pileup();
    jets_.jtm[jets_.nref] = jet.mass();
    jets_.jtarea[jets_.nref] = jet.jetArea();

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

    jets_.ngen = 0;

    for (const auto& genjet : *genjets) {

      float genjet_pt = genjet.pt();

      // threshold to reduce size of output in minbias PbPb
      if (genjet_pt > genPtMin_) {
        jets_.genpt[jets_.ngen] = genjet_pt;
        jets_.geneta[jets_.ngen] = genjet.eta();
        jets_.genphi[jets_.ngen] = genjet.phi();
        jets_.genm[jets_.ngen] = genjet.mass();
        jets_.geny[jets_.ngen] = genjet.eta();

        jets_.ngen++;
      }
    }
  }
  
  caloJetTree_->Fill();

  jets_ = {0};
}

DEFINE_FWK_MODULE(HiCaloJetAnalyzer);
