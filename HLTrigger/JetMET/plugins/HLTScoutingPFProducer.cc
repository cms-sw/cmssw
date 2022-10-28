// -*- C++ -*-
//
// Package:    HLTrigger/JetMET
// Class:      HLTScoutingPFProducer
//
/**\class HLTScoutingPFProducer HLTScoutingPFProducer.cc HLTrigger/JetMET/plugins/HLTScoutingPFProducer.cc

Description: Producer for ScoutingPFJets from reco::PFJet objects, ScoutingVertexs from reco::Vertexs and ScoutingParticles from reco::PFCandidates

*/
//
// Original Author:  Dustin James Anderson
//         Created:  Fri, 12 Jun 2015 15:49:20 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/Math/interface/libminifloat.h"

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

class HLTScoutingPFProducer : public edm::global::EDProducer<> {
public:
  explicit HLTScoutingPFProducer(const edm::ParameterSet &);
  ~HLTScoutingPFProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID sid, edm::Event &iEvent, edm::EventSetup const &setup) const final;

  const edm::EDGetTokenT<reco::PFJetCollection> pfJetCollection_;
  const edm::EDGetTokenT<reco::JetTagCollection> pfJetTagCollection_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidateCollection_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
  const edm::EDGetTokenT<reco::PFMETCollection> metCollection_;
  const edm::EDGetTokenT<double> rho_;

  const double pfJetPtCut_;
  const double pfJetEtaCut_;
  const double pfCandidatePtCut_;
  const double pfCandidateEtaCut_;
  const int mantissaPrecision_;

  const bool doJetTags_;
  const bool doCandidates_;
  const bool doMet_;
  const bool doTrackVars_;
  const bool relativeTrackVars_;
  const bool doCandIndsForJets_;
};

//
// constructors and destructor
//
HLTScoutingPFProducer::HLTScoutingPFProducer(const edm::ParameterSet &iConfig)
    : pfJetCollection_(consumes(iConfig.getParameter<edm::InputTag>("pfJetCollection"))),
      pfJetTagCollection_(consumes(iConfig.getParameter<edm::InputTag>("pfJetTagCollection"))),
      pfCandidateCollection_(consumes(iConfig.getParameter<edm::InputTag>("pfCandidateCollection"))),
      vertexCollection_(consumes(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
      metCollection_(consumes(iConfig.getParameter<edm::InputTag>("metCollection"))),
      rho_(consumes(iConfig.getParameter<edm::InputTag>("rho"))),
      pfJetPtCut_(iConfig.getParameter<double>("pfJetPtCut")),
      pfJetEtaCut_(iConfig.getParameter<double>("pfJetEtaCut")),
      pfCandidatePtCut_(iConfig.getParameter<double>("pfCandidatePtCut")),
      pfCandidateEtaCut_(iConfig.getParameter<double>("pfCandidateEtaCut")),
      mantissaPrecision_(iConfig.getParameter<int>("mantissaPrecision")),
      doJetTags_(iConfig.getParameter<bool>("doJetTags")),
      doCandidates_(iConfig.getParameter<bool>("doCandidates")),
      doMet_(iConfig.getParameter<bool>("doMet")),
      doTrackVars_(iConfig.getParameter<bool>("doTrackVars")),
      relativeTrackVars_(iConfig.getParameter<bool>("relativeTrackVars")),
      doCandIndsForJets_(iConfig.getParameter<bool>("doCandIndsForJets")) {
  //register products
  produces<Run3ScoutingPFJetCollection>();
  produces<Run3ScoutingParticleCollection>();
  produces<double>("rho");
  produces<double>("pfMetPt");
  produces<double>("pfMetPhi");
}

HLTScoutingPFProducer::~HLTScoutingPFProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingPFProducer::produce(edm::StreamID sid, edm::Event &iEvent, edm::EventSetup const &setup) const {
  using namespace edm;

  //get vertices
  Handle<reco::VertexCollection> vertexCollection;
  auto outVertices = std::make_unique<Run3ScoutingVertexCollection>();
  if (iEvent.getByToken(vertexCollection_, vertexCollection)) {
    for (auto const &vtx : *vertexCollection) {
      outVertices->emplace_back(MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.x(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.y(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.z(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.zError(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.xError(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.yError(), mantissaPrecision_),
                                vtx.tracksSize(),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.chi2(), mantissaPrecision_),
                                vtx.ndof(),
                                vtx.isValid());
    }
  }

  //get rho
  Handle<double> rho;
  auto outRho = std::make_unique<double>(-999);
  if (iEvent.getByToken(rho_, rho)) {
    *outRho = *rho;
  }

  //get MET
  Handle<reco::PFMETCollection> metCollection;
  auto outMetPt = std::make_unique<double>(-999);
  auto outMetPhi = std::make_unique<double>(-999);
  if (doMet_ && iEvent.getByToken(metCollection_, metCollection)) {
    outMetPt = std::make_unique<double>(metCollection->front().pt());
    outMetPhi = std::make_unique<double>(metCollection->front().phi());
  }

  //get PF candidates
  Handle<reco::PFCandidateCollection> pfCandidateCollection;
  auto outPFCandidates = std::make_unique<Run3ScoutingParticleCollection>();
  if (doCandidates_ && iEvent.getByToken(pfCandidateCollection_, pfCandidateCollection)) {
    for (auto const &cand : *pfCandidateCollection) {
      if (cand.pt() > pfCandidatePtCut_ && std::abs(cand.eta()) < pfCandidateEtaCut_) {
        int vertex_index = -1;
        int index_counter = 0;
        double dr2 = 0.0001;
        for (auto const &vtx : *outVertices) {
          double tmp_dr2 = pow(vtx.x() - cand.vx(), 2) + pow(vtx.y() - cand.vy(), 2) + pow(vtx.z() - cand.vz(), 2);
          if (tmp_dr2 < dr2) {
            dr2 = tmp_dr2;
            vertex_index = index_counter;
          }
          if (dr2 == 0.0)
            break;
          ++index_counter;
        }
        float normchi2{0}, dz{0}, dxy{0}, dzError{0}, dxyError{0}, trk_pt{0}, trk_eta{0}, trk_phi{0};
        uint8_t lostInnerHits{0}, quality{0};
        if (doTrackVars_) {
          const auto *trk = cand.bestTrack();
          if (trk != nullptr) {
            normchi2 = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->normalizedChi2(), mantissaPrecision_);
            lostInnerHits = btagbtvdeep::lost_inner_hits_from_pfcand(cand);
            quality = btagbtvdeep::quality_from_pfcand(cand);
            if (relativeTrackVars_) {
              trk_pt = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->pt() - cand.pt(), mantissaPrecision_);
              trk_eta = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->eta() - cand.eta(), mantissaPrecision_);
              trk_phi = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->phi() - cand.phi(), mantissaPrecision_);
            } else {
              trk_pt = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->pt(), mantissaPrecision_);
              trk_eta = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->eta(), mantissaPrecision_);
              trk_phi = MiniFloatConverter::reduceMantissaToNbitsRounding(trk->phi(), mantissaPrecision_);
            }
            if (not vertexCollection->empty()) {
              const reco::Vertex &pv = (*vertexCollection)[0];

              dz = trk->dz(pv.position());
              dzError = MiniFloatConverter::reduceMantissaToNbitsRounding(dz / trk->dzError(), mantissaPrecision_);
              dz = MiniFloatConverter::reduceMantissaToNbitsRounding(dz, mantissaPrecision_);

              dxy = trk->dxy(pv.position());
              dxyError = MiniFloatConverter::reduceMantissaToNbitsRounding(dxy / trk->dxyError(), mantissaPrecision_);
              dxy = MiniFloatConverter::reduceMantissaToNbitsRounding(dxy, mantissaPrecision_);
            }
          } else {
            normchi2 = MiniFloatConverter::reduceMantissaToNbitsRounding(999, mantissaPrecision_);
          }
        }
        outPFCandidates->emplace_back(MiniFloatConverter::reduceMantissaToNbitsRounding(cand.pt(), mantissaPrecision_),
                                      MiniFloatConverter::reduceMantissaToNbitsRounding(cand.eta(), mantissaPrecision_),
                                      MiniFloatConverter::reduceMantissaToNbitsRounding(cand.phi(), mantissaPrecision_),
                                      cand.pdgId(),
                                      vertex_index,
                                      normchi2,
                                      dz,
                                      dxy,
                                      dzError,
                                      dxyError,
                                      lostInnerHits,
                                      quality,
                                      trk_pt,
                                      trk_eta,
                                      trk_phi,
                                      relativeTrackVars_);
      }
    }
  }

  //get PF jets
  Handle<reco::PFJetCollection> pfJetCollection;
  auto outPFJets = std::make_unique<Run3ScoutingPFJetCollection>();
  if (iEvent.getByToken(pfJetCollection_, pfJetCollection)) {
    //get PF jet tags
    Handle<reco::JetTagCollection> pfJetTagCollection;
    bool haveJetTags = false;
    if (doJetTags_ && iEvent.getByToken(pfJetTagCollection_, pfJetTagCollection)) {
      haveJetTags = true;
    }

    for (auto const &jet : *pfJetCollection) {
      if (jet.pt() < pfJetPtCut_ || std::abs(jet.eta()) > pfJetEtaCut_)
        continue;
      //find the jet tag corresponding to the jet
      float tagValue = -20;
      float minDR2 = 0.01;
      if (haveJetTags) {
        for (auto const &tag : *pfJetTagCollection) {
          float dR2 = reco::deltaR2(jet, *(tag.first));
          if (dR2 < minDR2) {
            minDR2 = dR2;
            tagValue = tag.second;
          }
        }
      }
      //get the PF constituents of the jet
      std::vector<int> candIndices;
      if (doCandidates_ && doCandIndsForJets_) {
        for (auto const &cand : jet.getPFConstituents()) {
          if (not(cand.isNonnull() and cand.isAvailable())) {
            throw cms::Exception("HLTScoutingPFProducer")
                << "invalid reference to reco::PFCandidate from reco::PFJet::getPFConstituents()";
          }
          if (cand->pt() > pfCandidatePtCut_ && std::abs(cand->eta()) < pfCandidateEtaCut_) {
            //search for the candidate in the collection
            float minDR2 = 0.0001;
            int matchIndex = -1;
            int outIndex = 0;
            for (auto &outCand : *outPFCandidates) {
              auto const dR2 = reco::deltaR2(*cand, outCand);
              if (dR2 < minDR2) {
                minDR2 = dR2;
                matchIndex = outIndex;
              }
              if (minDR2 == 0) {
                break;
              }
              outIndex++;
            }
            candIndices.push_back(matchIndex);
          }
        }
      }
      outPFJets->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.pt(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.eta(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.phi(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.mass(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.jetArea(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.chargedHadronEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.neutralHadronEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.photonEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.electronEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.muonEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.HFHadronEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.HFEMEnergy(), mantissaPrecision_),
          jet.chargedHadronMultiplicity(),
          jet.neutralHadronMultiplicity(),
          jet.photonMultiplicity(),
          jet.electronMultiplicity(),
          jet.muonMultiplicity(),
          jet.HFHadronMultiplicity(),
          jet.HFEMMultiplicity(),
          MiniFloatConverter::reduceMantissaToNbitsRounding(jet.hoEnergy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(tagValue, mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(0.0, mantissaPrecision_),
          candIndices);
    }
  }

  //put output
  iEvent.put(std::move(outPFCandidates));
  iEvent.put(std::move(outPFJets));
  iEvent.put(std::move(outRho), "rho");
  iEvent.put(std::move(outMetPt), "pfMetPt");
  iEvent.put(std::move(outMetPhi), "pfMetPhi");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingPFProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pfJetCollection", edm::InputTag("hltAK4PFJets"));
  desc.add<edm::InputTag>("pfJetTagCollection", edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsPF"));
  desc.add<edm::InputTag>("pfCandidateCollection", edm::InputTag("hltParticleFlow"));
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"));
  desc.add<edm::InputTag>("metCollection", edm::InputTag("hltPFMETProducer"));
  desc.add<edm::InputTag>("rho", edm::InputTag("hltFixedGridRhoFastjetAll"));
  desc.add<double>("pfJetPtCut", 20.0);
  desc.add<double>("pfJetEtaCut", 3.0);
  desc.add<double>("pfCandidatePtCut", 0.6);
  desc.add<double>("pfCandidateEtaCut", 5.0);
  desc.add<int>("mantissaPrecision", 10)->setComment("default float16, change to 23 for float32");
  desc.add<bool>("doJetTags", true);
  desc.add<bool>("doCandidates", true);
  desc.add<bool>("doMet", true);
  desc.add<bool>("doTrackVars", true);
  desc.add<bool>("relativeTrackVars", true);
  desc.add<bool>("doCandIndsForJets", false);
  descriptions.addWithDefaultLabel(desc);
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTScoutingPFProducer);
