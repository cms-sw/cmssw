#include "JetPlusTrackCorrector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace jpt;

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::JetPlusTrackCorrector(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
    : verbose_(pset.getParameter<bool>("Verbose")),
      usePAT_(pset.getParameter<bool>("UsePAT")),
      vectorial_(pset.getParameter<bool>("VectorialCorrection")),
      vecResponse_(pset.getParameter<bool>("UseResponseInVecCorr")),
      useInConeTracks_(pset.getParameter<bool>("UseInConeTracks")),
      useOutOfConeTracks_(pset.getParameter<bool>("UseOutOfConeTracks")),
      useOutOfVertexTracks_(pset.getParameter<bool>("UseOutOfVertexTracks")),
      usePions_(pset.getParameter<bool>("UsePions")),
      useEff_(pset.getParameter<bool>("UseEfficiency")),
      useMuons_(pset.getParameter<bool>("UseMuons")),
      useElecs_(pset.getParameter<bool>("UseElectrons")),
      useTrackQuality_(pset.getParameter<bool>("UseTrackQuality")),
      jetTracksAtVertex_(pset.getParameter<edm::InputTag>("JetTracksAssociationAtVertex")),
      jetTracksAtCalo_(pset.getParameter<edm::InputTag>("JetTracksAssociationAtCaloFace")),
      jetSplitMerge_(pset.getParameter<int>("JetSplitMerge")),
      srcPVs_(pset.getParameter<edm::InputTag>("srcPVs")),
      ptErrorQuality_(pset.getParameter<double>("PtErrorQuality")),
      dzVertexCut_(pset.getParameter<double>("DzVertexCut")),
      muons_(pset.getParameter<edm::InputTag>("Muons")),
      electrons_(pset.getParameter<edm::InputTag>("Electrons")),
      electronIds_(pset.getParameter<edm::InputTag>("ElectronIds")),
      patmuons_(pset.getParameter<edm::InputTag>("PatMuons")),
      patelectrons_(pset.getParameter<edm::InputTag>("PatElectrons")),
      trackQuality_(reco::TrackBase::qualityByName(pset.getParameter<std::string>("TrackQuality"))),
      response_(Map(pset.getParameter<std::string>("ResponseMap"), verbose_)),
      efficiency_(Map(pset.getParameter<std::string>("EfficiencyMap"), verbose_)),
      leakage_(Map(pset.getParameter<std::string>("LeakageMap"), verbose_)),
      muonPtmatch_(pset.getParameter<double>("muonPtmatch")),
      muonEtamatch_(pset.getParameter<double>("muonEtamatch")),
      muonPhimatch_(pset.getParameter<double>("muonPhimatch")),
      electronDRmatch_(pset.getParameter<double>("electronDRmatch")),
      pionMass_(0.140),
      muonMass_(0.105),
      elecMass_(0.000511),
      maxEta_(pset.getParameter<double>("MaxJetEta")) {
  if (verbose_) {
    std::stringstream ss;
    ss << "[JetPlusTrackCorrector::" << __func__ << "] Configuration for JPT corrector: " << std::endl
       << " Particles" << std::endl
       << "  UsePions             : " << (usePions_ ? "true" : "false") << std::endl
       << "  UseMuons             : " << (useMuons_ ? "true" : "false") << std::endl
       << "  UseElecs             : " << (useElecs_ ? "true" : "false") << std::endl
       << " Corrections" << std::endl
       << "  UseInConeTracks      : " << (useInConeTracks_ ? "true" : "false") << std::endl
       << "  UseOutOfConeTracks   : " << (useOutOfConeTracks_ ? "true" : "false") << std::endl
       << "  UseOutOfVertexTracks : " << (useOutOfVertexTracks_ ? "true" : "false") << std::endl
       << "  ResponseMap          : " << pset.getParameter<std::string>("ResponseMap") << std::endl
       << " Efficiency" << std::endl
       << "  UsePionEfficiency    : " << (useEff_ ? "true" : "false") << std::endl
       << "  EfficiencyMap        : " << pset.getParameter<std::string>("EfficiencyMap") << std::endl
       << "  LeakageMap           : " << pset.getParameter<std::string>("LeakageMap") << std::endl
       << " Tracks" << std::endl
       << "  JetTracksAtVertex    : " << jetTracksAtVertex_ << std::endl
       << "  JetTracksAtCalo      : " << jetTracksAtCalo_ << std::endl
       << "  JetSplitMerge        : " << jetSplitMerge_ << std::endl
       << "  UseTrackQuality      : " << (useTrackQuality_ ? "true" : "false") << std::endl
       << " Collections" << std::endl
       << "  Muons                : " << muons_ << std::endl
       << "  Electrons            : " << electrons_ << std::endl
       << " Vectorial" << std::endl
       << "  UseTracksAndResponse : " << ((vectorial_ && vecResponse_) ? "true" : "false") << std::endl
       << "  UseTracksOnly        : " << ((vectorial_ && !vecResponse_) ? "true" : "false");
    edm::LogInfo("JetPlusTrackCorrector") << ss.str();
  }

  if (!useInConeTracks_ || !useOutOfConeTracks_ || !useOutOfVertexTracks_) {
    std::stringstream ss;
    ss << "[JetPlusTrackCorrector::" << __func__ << "]"
       << " You are using JPT algorithm in a non-standard way!" << std::endl
       << " UseInConeTracks      : " << (useInConeTracks_ ? "true" : "false") << std::endl
       << " UseOutOfConeTracks   : " << (useOutOfConeTracks_ ? "true" : "false") << std::endl
       << " UseOutOfVertexTracks : " << (useOutOfVertexTracks_ ? "true" : "false");
    edm::LogWarning("JetPlusTrackCorrector") << ss.str();
  }

  input_jetTracksAtVertex_token_ = iC.consumes<reco::JetTracksAssociation::Container>(jetTracksAtVertex_);
  input_jetTracksAtCalo_token_ = iC.consumes<reco::JetTracksAssociation::Container>(jetTracksAtCalo_);
  input_pvCollection_token_ = iC.consumes<reco::VertexCollection>(srcPVs_);

  input_reco_muons_token_ = iC.consumes<RecoMuons>(muons_);
  input_reco_elecs_token_ = iC.consumes<RecoElectrons>(electrons_);
  input_reco_elec_ids_token_ = iC.consumes<RecoElectronIds>(electronIds_);
  if (usePAT_) {
    input_pat_muons_token_ = iC.consumes<pat::MuonCollection>(patmuons_);
    input_pat_elecs_token_ = iC.consumes<pat::ElectronCollection>(patelectrons_);
  }
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::~JetPlusTrackCorrector() { ; }
// -----------------------------------------------------------------------------
//

double JetPlusTrackCorrector::correction(const reco::Jet& fJet,
                                         const reco::Jet& fJetcalo,
                                         const edm::Event& event,
                                         const edm::EventSetup& setup,
                                         const reco::TrackRefVector& tracksinvert,
                                         const reco::TrackRefVector& tracksincalo,
                                         P4& corrected,
                                         MatchedTracks& pions,
                                         MatchedTracks& muons,
                                         MatchedTracks& elecs) {
  double scale = 1.;
  corrected = fJet.p4();
  matchTracks(fJetcalo, event, setup, tracksinvert, tracksincalo, pions, muons, elecs);
  if (!usePAT_) {
    if (usePions_) {
      corrected += pionCorrection(fJet.p4(), pions);
    }
    if (useMuons_) {
      corrected += muonCorrection(fJet.p4(), muons);
    }
    if (useElecs_) {
      corrected += elecCorrection(fJet.p4(), elecs);
    }
  } else {
    corrected += fJetcalo.p4();
  }
  scale = checkScale(fJet.p4(), corrected);
  return scale;
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction(const reco::Jet& fJet,
                                         const reco::Jet& fJetcalo,
                                         const edm::Event& event,
                                         const edm::EventSetup& setup,
                                         P4& corrected,
                                         MatchedTracks& pions,
                                         MatchedTracks& muons,
                                         MatchedTracks& elecs,
                                         bool& validMatches) {
  theResponseOfChargedWithEff = 0.;
  theResponseOfChargedWithoutEff = 0.;
  theSumPtWithEff = 0.;
  theSumPtWithoutEff = 0.;
  theSumEnergyWithEff = 0.;
  theSumEnergyWithoutEff = 0.;
  theSumPtForBeta = 0.;

  // Corrected 4-momentum for jet
  corrected = fJet.p4();

  // Match tracks to different particle types
  validMatches = matchTracks(fJetcalo, event, setup, pions, muons, elecs);
  if (!validMatches) {
    return 1.;
  }

  // Check if jet can be JPT-corrected
  if (!canCorrect(fJet)) {
    return 1.;
  }

  // Debug
  if (verbose_) {
    edm::LogInfo("JetPlusTrackCorrector") << "[JetPlusTrackCorrector::" << __func__ << "]"
                                          << " Applying JPT corrections...";
  }

  // Pion corrections (both scalar and vectorial)
  if (usePions_) {
    corrected += pionCorrection(fJet.p4(), pions);
  }

  // Muon corrections (both scalar and vectorial)
  if (useMuons_) {
    corrected += muonCorrection(fJet.p4(), muons);
  }

  // Electrons corrections (both scalar and vectorial)
  if (useElecs_) {
    corrected += elecCorrection(fJet.p4(), elecs);
  }

  // Define jet direction using total 3-momentum of tracks (overrides above)
  if (vectorial_ && !vecResponse_) {
    if (fabs(corrected.eta()) < 2.) {
      corrected = jetDirFromTracks(corrected, pions, muons, elecs);
    }
  }

  // Check if corrected 4-momentum gives negative scale
  double scale = checkScale(fJet.p4(), corrected);

  // Debug
  if (verbose_) {
    std::stringstream ss;
    ss << "Total correction:" << std::endl
       << std::fixed << std::setprecision(6) << " Uncorrected (Px,Py,Pz,E)   : "
       << "(" << fJet.px() << "," << fJet.py() << "," << fJet.pz() << "," << fJet.energy() << ")" << std::endl
       << " Corrected (Px,Py,Pz,E)     : "
       << "(" << corrected.px() << "," << corrected.py() << "," << corrected.pz() << "," << corrected.energy() << ")"
       << std::endl
       << " Uncorrected (Pt,Eta,Phi,M) : "
       << "(" << fJet.pt() << "," << fJet.eta() << "," << fJet.phi() << "," << fJet.mass() << ")" << std::endl
       << " Corrected (Pt,Eta,Phi,M)   : "
       << "(" << corrected.pt() << "," << corrected.eta() << "," << corrected.phi() << "," << corrected.mass() << ")"
       << std::endl
       << " Scalar correction to E     : " << scale << std::endl
       << " Scalar correction to Et    : " << (fJet.et() > 0. ? corrected.Et() / fJet.et() : 1.);  // << std::endl
    edm::LogVerbatim("JetPlusTrackCorrector") << ss.str();
  }

  // Return energy correction
  return scale;
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction(const reco::Jet& jet) const {
  edm::LogError("JetPlusTrackCorrector") << "JetPlusTrackCorrector can be run on entire event only";
  return 1.;
}

// -----------------------------------------------------------------------------
//
double JetPlusTrackCorrector::correction(const reco::Particle::LorentzVector& jet) const {
  edm::LogError("JetPlusTrackCorrector") << "JetPlusTrackCorrector can be run on entire event only";
  return 1.;
}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::matchTracks(const reco::Jet& fJet,
                                        const edm::Event& event,
                                        const edm::EventSetup& setup,
                                        const reco::TrackRefVector& tracksinvert,
                                        const reco::TrackRefVector& tracksincalo,
                                        jpt::MatchedTracks& pions,
                                        jpt::MatchedTracks& muons,
                                        jpt::MatchedTracks& elecs) {
  JetTracks jet_tracks;
  jet_tracks.vertex_ = tracksinvert;
  jet_tracks.caloFace_ = tracksincalo;
  matchTracks(jet_tracks, event, pions, muons, elecs);

  return;
}

bool JetPlusTrackCorrector::matchTracks(const reco::Jet& fJet,
                                        const edm::Event& event,
                                        const edm::EventSetup& setup,
                                        jpt::MatchedTracks& pions,
                                        jpt::MatchedTracks& muons,
                                        jpt::MatchedTracks& elecs) {
  // Associate tracks to jet at both the Vertex and CaloFace
  JetTracks jet_tracks;
  bool ok = jetTrackAssociation(fJet, event, setup, jet_tracks);

  if (!ok) {
    return false;
  }

  // Track collections propagated to Vertex and CaloFace for "pions", muons and electrons
  matchTracks(jet_tracks, event, pions, muons, elecs);

  // Debug
  if (verbose_) {
    std::stringstream ss;
    ss << "Number of tracks:" << std::endl
       << " In-cone at Vertex   : " << jet_tracks.vertex_.size() << std::endl
       << " In-cone at CaloFace : " << jet_tracks.caloFace_.size();
    edm::LogVerbatim("JetPlusTrackCorrector") << ss.str();
  }

  return true;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::jetTrackAssociation(const reco::Jet& fJet,
                                                const edm::Event& event,
                                                const edm::EventSetup& setup,
                                                JetTracks& trks) const {
  // Some init
  trks.clear();

  // Check if labels are given
  if (!jetTracksAtVertex_.label().empty() && !jetTracksAtCalo_.label().empty()) {
    return jtaUsingEventData(fJet, event, trks);
  } else {
    edm::LogWarning("PatJPTCorrector") << "[JetPlusTrackCorrector::" << __func__ << "]"
                                       << " Empty label for the reco::JetTracksAssociation::Containers" << std::endl
                                       << " InputTag for JTA \"at vertex\"    (label:instance:process) \""
                                       << jetTracksAtVertex_.label() << ":" << jetTracksAtVertex_.instance() << ":"
                                       << jetTracksAtVertex_.process() << "\"" << std::endl
                                       << " InputTag for JTA \"at calo face\" (label:instance:process) \""
                                       << jetTracksAtCalo_.label() << ":" << jetTracksAtCalo_.instance() << ":"
                                       << jetTracksAtCalo_.process() << "\"";
    return false;
  }
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::jtaUsingEventData(const reco::Jet& fJet, const edm::Event& event, JetTracks& trks) const {
  // Get Jet-track association at Vertex
  edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtVertex;
  event.getByToken(input_jetTracksAtVertex_token_, jetTracksAtVertex);

  if (!jetTracksAtVertex.isValid() || jetTracksAtVertex.failedToGet()) {
    if (verbose_ && edm::isDebugEnabled()) {
      edm::LogWarning("JetPlusTrackCorrector")
          << "[JetPlusTrackCorrector::" << __func__ << "]"
          << " Invalid handle to reco::JetTracksAssociation::Container (for Vertex)"
          << " with InputTag (label:instance:process) \"" << jetTracksAtVertex_.label() << ":"
          << jetTracksAtVertex_.instance() << ":" << jetTracksAtVertex_.process() << "\"";
    }
    return false;
  }

  // Retrieve jet-tracks association for given jet
  const reco::JetTracksAssociation::Container jtV = *(jetTracksAtVertex.product());
  TrackRefs excluded;
  TrackRefs relocate;
  if (jetSplitMerge_ < 0) {
    trks.vertex_ = reco::JetTracksAssociation::getValue(jtV, fJet);
  } else {
    rebuildJta(fJet, jtV, trks.vertex_, excluded);
    rebuildJta(fJet, jtV, relocate, excluded);
    trks.vertex_ = relocate;
  }

  // Check if any tracks are associated to jet at vertex
  if (trks.vertex_.empty()) {
    return false;
  }

  // Get Jet-track association at Calo
  edm::Handle<reco::JetTracksAssociation::Container> jetTracksAtCalo;
  event.getByToken(input_jetTracksAtCalo_token_, jetTracksAtCalo);

  if (!jetTracksAtCalo.isValid() || jetTracksAtCalo.failedToGet()) {
    if (verbose_ && edm::isDebugEnabled()) {
      edm::LogWarning("JetPlusTrackCorrector")
          << "[JetPlusTrackCorrector::" << __func__ << "]"
          << " Invalid handle to reco::JetTracksAssociation::Container (for CaloFace)"
          << " with InputTag (label:instance:process) \"" << jetTracksAtCalo_.label() << ":"
          << jetTracksAtCalo_.instance() << ":" << jetTracksAtCalo_.process() << "\"";
    }
    return false;
  }

  // Retrieve jet-tracks association for given jet
  const reco::JetTracksAssociation::Container jtC = *(jetTracksAtCalo.product());
  if (jetSplitMerge_ < 0) {
    trks.caloFace_ = reco::JetTracksAssociation::getValue(jtC, fJet);
  } else {
    excludeJta(fJet, jtC, trks.caloFace_, excluded);
  }

  return true;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::getMuons(const edm::Event& event, edm::Handle<RecoMuons>& reco_muons) const {
  event.getByToken(input_reco_muons_token_, reco_muons);
  return true;
}

bool JetPlusTrackCorrector::getMuons(const edm::Event& event, edm::Handle<pat::MuonCollection>& pat_muons) const {
  event.getByToken(input_pat_muons_token_, pat_muons);
  return true;
}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::matchTracks(const JetTracks& jet_tracks,
                                        const edm::Event& event,
                                        MatchedTracks& pions,
                                        MatchedTracks& muons,
                                        MatchedTracks& elecs) {
  // Some init
  pions.clear();
  muons.clear();
  elecs.clear();

  // Need vertex for track cleaning

  vertex_ = reco::Particle::Point(0, 0, 0);
  edm::Handle<reco::VertexCollection> pvCollection;
  event.getByToken(input_pvCollection_token_, pvCollection);
  if (pvCollection.isValid() && !pvCollection->empty())
    vertex_ = pvCollection->begin()->position();

  // Get RECO muons
  edm::Handle<RecoMuons> reco_muons;
  edm::Handle<pat::MuonCollection> pat_muons;
  bool found_reco_muons = true;
  bool found_pat_muons = true;
  if (useMuons_) {
    if (!usePAT_) {
      getMuons(event, reco_muons);
    } else {
      getMuons(event, pat_muons);
      found_reco_muons = false;
    }
  }

  // Get RECO electrons and their ids
  edm::Handle<RecoElectrons> reco_elecs;
  edm::Handle<pat::ElectronCollection> pat_elecs;
  edm::Handle<RecoElectronIds> reco_elec_ids;
  bool found_reco_elecs = true;
  bool found_pat_elecs = true;
  if (useElecs_) {
    if (!usePAT_) {
      getElectrons(event, reco_elecs, reco_elec_ids);
    } else {
      getElectrons(event, pat_elecs);
      found_reco_elecs = false;
    }
  }

  // Check RECO products found
  if (!found_reco_muons || !found_reco_elecs) {
    if (!found_pat_muons || !found_pat_elecs) {
      edm::LogError("JetPlusTrackCorrector") << "[JetPlusTrackCorrector::" << __func__ << "]"
                                             << " Unable to access RECO collections for muons and electrons";
      return;
    }
  }

  // Identify pions/muons/electrons that are "in/in" and "in/out"
  {
    TrackRefs::const_iterator itrk = jet_tracks.vertex_.begin();
    TrackRefs::const_iterator jtrk = jet_tracks.vertex_.end();

    double theSumPtForBetaOld = theSumPtForBeta;

    for (; itrk != jtrk; ++itrk) {
      if (useTrackQuality_ && (*itrk)->quality(trackQuality_) && theSumPtForBetaOld <= 0.)
        theSumPtForBeta += (**itrk).pt();
      //
      // Track either belongs to PV or do not belong to any vertex

      const reco::TrackBaseRef ttr1(*itrk);

      int numpv = 0;

      int itrack_belong = -1;

      for (reco::VertexCollection::const_iterator iv = pvCollection->begin(); iv != pvCollection->end(); iv++) {
        numpv++;
        std::vector<reco::TrackBaseRef>::const_iterator rr = find((*iv).tracks_begin(), (*iv).tracks_end(), ttr1);
        if (rr != (*iv).tracks_end()) {
          itrack_belong++;
          break;
        }

      }  // all vertices

      if (numpv > 1 && itrack_belong == 0) {
        continue;
      }

      if (failTrackQuality(itrk)) {
        continue;
      }

      TrackRefs::iterator it = jet_tracks.caloFace_.end();
      bool found = findTrack(jet_tracks, itrk, it);
      bool muaccept = false;
      bool eleaccept = false;
      if (found_reco_muons)
        muaccept = matchMuons(itrk, reco_muons);
      else if (found_pat_muons) {
        muaccept = matchMuons(itrk, pat_muons);
      }
      if (found_reco_elecs)
        eleaccept = matchElectrons(itrk, reco_elecs, reco_elec_ids);
      else if (found_pat_elecs) {
        eleaccept = matchElectrons(itrk, pat_elecs);
      }

      bool is_muon = useMuons_ && muaccept;
      bool is_ele = useElecs_ && eleaccept;

      if (found) {
        if (is_muon) {
          muons.inVertexInCalo_.push_back(*it);
        } else if (is_ele) {
          elecs.inVertexInCalo_.push_back(*it);
        } else {
          pions.inVertexInCalo_.push_back(*it);
        }
      } else {
        if (is_muon) {
          muons.inVertexOutOfCalo_.push_back(*itrk);
        } else if (is_ele) {
          elecs.inVertexOutOfCalo_.push_back(*itrk);
        } else {
          pions.inVertexOutOfCalo_.push_back(*itrk);
        }
      }
    }
  }

  // Identify pions/muons/electrons that are "out/in"
  {
    TrackRefs::iterator itrk = jet_tracks.caloFace_.begin();
    TrackRefs::iterator jtrk = jet_tracks.caloFace_.end();
    for (; itrk != jtrk; ++itrk) {
      if (failTrackQuality(itrk)) {
        continue;
      }

      if (!tracksInCalo(pions, muons, elecs)) {
        continue;
      }

      bool found = findTrack(pions, muons, elecs, itrk);

      if (!found) {
        bool muaccept = false;
        bool eleaccept = false;
        if (found_reco_muons)
          muaccept = matchMuons(itrk, reco_muons);
        else if (found_pat_muons) {
          muaccept = matchMuons(itrk, pat_muons);
        }
        if (found_reco_elecs)
          eleaccept = matchElectrons(itrk, reco_elecs, reco_elec_ids);
        else if (found_pat_elecs) {
          eleaccept = matchElectrons(itrk, pat_elecs);
        }

        bool is_muon = useMuons_ && muaccept;
        bool is_ele = useElecs_ && eleaccept;

        if (is_muon) {
          muons.outOfVertexInCalo_.push_back(*itrk);
        } else if (is_ele) {
          elecs.outOfVertexInCalo_.push_back(*itrk);
        } else {
          pions.outOfVertexInCalo_.push_back(*itrk);
        }
      }
    }
  }

  if (verbose_ && edm::isDebugEnabled()) {
    std::stringstream ss;
    ss << "[JetPlusTrackCorrector::" << __func__ << "] Number of tracks:" << std::endl
       << " In-cone at Vertex and in-cone at CaloFace:" << std::endl
       << "  Pions      : " << pions.inVertexInCalo_.size() << std::endl
       << "  Muons      : " << muons.inVertexInCalo_.size() << std::endl
       << "  Electrons  : " << elecs.inVertexInCalo_.size() << std::endl
       << " In-cone at Vertex and out-of-cone at CaloFace:" << std::endl
       << "  Pions      : " << pions.inVertexOutOfCalo_.size() << std::endl
       << "  Muons      : " << muons.inVertexOutOfCalo_.size() << std::endl
       << "  Electrons  : " << elecs.inVertexOutOfCalo_.size() << std::endl
       << " Out-of-cone at Vertex and in-cone at CaloFace:" << std::endl
       << "  Pions      : " << pions.outOfVertexInCalo_.size() << std::endl
       << "  Muons      : " << muons.outOfVertexInCalo_.size() << std::endl
       << "  Electrons  : " << elecs.outOfVertexInCalo_.size() << std::endl;
    LogTrace("JetPlusTrackCorrector") << ss.str();
  }
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::getElectrons(const edm::Event& event,
                                         edm::Handle<RecoElectrons>& reco_elecs,
                                         edm::Handle<RecoElectronIds>& reco_elec_ids) const {
  event.getByToken(input_reco_elecs_token_, reco_elecs);
  event.getByToken(input_reco_elec_ids_token_, reco_elec_ids);
  return true;
}

bool JetPlusTrackCorrector::getElectrons(const edm::Event& event,
                                         edm::Handle<pat::ElectronCollection>& pat_elecs) const {
  event.getByToken(input_pat_elecs_token_, pat_elecs);
  return true;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::failTrackQuality(TrackRefs::const_iterator& itrk) const {
  bool retcode = false;

  if (useTrackQuality_ && !(*itrk)->quality(trackQuality_)) {
    retcode = true;
    return retcode;
  }
  if (((*itrk)->ptError() / (*itrk)->pt()) > ptErrorQuality_) {
    retcode = true;
    return retcode;
  }
  if (fabs((*itrk)->dz(vertex_)) > dzVertexCut_) {
    retcode = true;
    return retcode;
  }

  return retcode;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::findTrack(const JetTracks& jet_tracks,
                                      TrackRefs::const_iterator& itrk,
                                      TrackRefs::iterator& it) const {
  it = find(jet_tracks.caloFace_.begin(), jet_tracks.caloFace_.end(), *itrk);
  if (it != jet_tracks.caloFace_.end()) {
    return true;
  } else {
    return false;
  }
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::findTrack(const MatchedTracks& pions,
                                      const MatchedTracks& muons,
                                      const MatchedTracks& elecs,
                                      TrackRefs::const_iterator& itrk) const {
  TrackRefs::iterator ip = find(pions.inVertexInCalo_.begin(), pions.inVertexInCalo_.end(), *itrk);
  TrackRefs::iterator im = find(muons.inVertexInCalo_.begin(), muons.inVertexInCalo_.end(), *itrk);
  TrackRefs::iterator ie = find(elecs.inVertexInCalo_.begin(), elecs.inVertexInCalo_.end(), *itrk);
  if (ip == pions.inVertexInCalo_.end() && im == muons.inVertexInCalo_.end() && ie == elecs.inVertexInCalo_.end()) {
    return false;
  } else {
    return true;
  }
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::tracksInCalo(const MatchedTracks& pions,
                                         const MatchedTracks& muons,
                                         const MatchedTracks& elecs) const {
  if (!pions.inVertexInCalo_.empty() || !muons.inVertexInCalo_.empty() || !elecs.inVertexInCalo_.empty()) {
    return true;
  } else {
    return false;
  }
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::P4 JetPlusTrackCorrector::pionCorrection(const P4& jet, const MatchedTracks& pions) {
  P4 corr_pions;

  // In-cone

  P4 corr_pions_in_cone;
  P4 corr_pions_eff_in_cone;
  Efficiency in_cone(responseMap(), efficiencyMap(), leakageMap());

  if (useInConeTracks_) {
    corr_pions_in_cone = pionCorrection(jet, pions.inVertexInCalo_, in_cone, true, true);
    corr_pions += corr_pions_in_cone;
    if (useEff_) {
      corr_pions_eff_in_cone = pionEfficiency(jet, in_cone, true);
      corr_pions += corr_pions_eff_in_cone;
    }
  }

  // Out-of-cone

  P4 corr_pions_out_of_cone;
  P4 corr_pions_eff_out_of_cone;
  Efficiency out_of_cone(responseMap(), efficiencyMap(), leakageMap());

  if (useOutOfConeTracks_) {
    corr_pions_out_of_cone = pionCorrection(jet, pions.inVertexOutOfCalo_, out_of_cone, true, false);
    corr_pions += corr_pions_out_of_cone;
    if (useEff_) {
      corr_pions_eff_out_of_cone = pionEfficiency(jet, out_of_cone, false);
      corr_pions += corr_pions_eff_out_of_cone;
    }
  }

  // Out-of-vertex

  P4 corr_pions_out_of_vertex;
  Efficiency not_used(responseMap(), efficiencyMap(), leakageMap());

  if (useOutOfVertexTracks_) {
    corr_pions_out_of_vertex = pionCorrection(jet, pions.outOfVertexInCalo_, not_used, false, true);
    corr_pions += corr_pions_out_of_vertex;
  }

  if (verbose_) {
    std::stringstream ss;
    ss << " Pion corrections:" << std::endl
       << "  In/In      : "
       << "(" << pions.inVertexInCalo_.size() << ") " << corr_pions_in_cone.energy() << std::endl
       << "  In/Out     : "
       << "(" << pions.inVertexOutOfCalo_.size() << ") " << corr_pions_out_of_cone.energy() << std::endl
       << "  Out/In     : "
       << "(" << pions.outOfVertexInCalo_.size() << ") " << corr_pions_out_of_vertex.energy() << std::endl;
    if (useEff_) {
      ss << " Pion efficiency corrections:" << std::endl
         << "  In/In      : "
         << "(" << pions.inVertexInCalo_.size() << ") " << corr_pions_eff_in_cone.energy() << std::endl
         << "  In/Out     : "
         << "(" << pions.inVertexOutOfCalo_.size() << ") " << corr_pions_eff_out_of_cone.energy();
    }
    edm::LogVerbatim("JetPlusTrackCorrector") << ss.str();
  }

  return corr_pions;
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::P4 JetPlusTrackCorrector::muonCorrection(const P4& jet, const MatchedTracks& muons) {
  P4 corr_muons;

  P4 corr_muons_in_cone;
  P4 corr_muons_out_of_cone;
  P4 corr_muons_out_of_vertex;

  if (useInConeTracks_) {
    corr_muons_in_cone = muonCorrection(jet, muons.inVertexInCalo_, true, true);
    corr_muons += corr_muons_in_cone;
  }

  if (useOutOfConeTracks_) {
    corr_muons_out_of_cone = muonCorrection(jet, muons.inVertexOutOfCalo_, true, false);
    corr_muons += corr_muons_out_of_cone;
  }

  if (useOutOfVertexTracks_) {
    corr_muons_out_of_vertex = muonCorrection(jet, muons.outOfVertexInCalo_, false, true);
    corr_muons += corr_muons_out_of_vertex;
  }

  if (verbose_) {
    std::stringstream ss;
    ss << " Muon corrections:" << std::endl
       << "  In/In      : "
       << "(" << muons.inVertexInCalo_.size() << ") " << corr_muons_in_cone.energy() << std::endl
       << "  In/Out     : "
       << "(" << muons.inVertexOutOfCalo_.size() << ") " << corr_muons_out_of_cone.energy() << std::endl
       << "  Out/In     : "
       << "(" << muons.outOfVertexInCalo_.size() << ") " << corr_muons_out_of_vertex.energy();
    edm::LogVerbatim("JetPlusTrackCorrector") << ss.str();
  }

  return corr_muons;
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::P4 JetPlusTrackCorrector::elecCorrection(const P4& jet, const MatchedTracks& elecs) const {
  P4 null;  //@@ null 4-momentum

  if (verbose_) {
    std::stringstream ss;
    ss << " Electron corrections:" << std::endl
       << "  In/In      : "
       << "(" << elecs.inVertexInCalo_.size() << ") " << null.energy() << std::endl
       << "  In/Out     : "
       << "(" << elecs.inVertexOutOfCalo_.size() << ") " << null.energy() << std::endl
       << "  Out/In     : "
       << "(" << elecs.outOfVertexInCalo_.size() << ") " << null.energy();
    edm::LogVerbatim("JetPlusTrackCorrector") << ss.str();
  }

  return null;  //@@ to be implemented
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::P4 JetPlusTrackCorrector::jetDirFromTracks(const P4& corrected,
                                                                  const MatchedTracks& pions,
                                                                  const MatchedTracks& muons,
                                                                  const MatchedTracks& elecs) const {
  // Correction to be applied to jet 4-momentum
  P4 corr;

  //  bool tracks_present = false;
  bool tracks_present_inin = false;

  // Correct using pions in-cone at vertex

  if (!pions.inVertexInCalo_.empty()) {
    TrackRefs::iterator itrk = pions.inVertexInCalo_.begin();
    TrackRefs::iterator jtrk = pions.inVertexInCalo_.end();
    for (; itrk != jtrk; ++itrk) {
      corr += PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), 0.);
      tracks_present_inin = true;
    }
  }

  if (!pions.inVertexOutOfCalo_.empty()) {
    TrackRefs::iterator itrk = pions.inVertexOutOfCalo_.begin();
    TrackRefs::iterator jtrk = pions.inVertexOutOfCalo_.end();
    for (; itrk != jtrk; ++itrk) {
      corr += PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), 0.);
    }
  }

  // Correct using muons in-cone at vertex

  if (!muons.inVertexInCalo_.empty()) {
    TrackRefs::iterator itrk = muons.inVertexInCalo_.begin();
    TrackRefs::iterator jtrk = muons.inVertexInCalo_.end();
    for (; itrk != jtrk; ++itrk) {
      corr += PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), 0.);
      //      tracks_present = true;
    }
  }

  if (!muons.inVertexOutOfCalo_.empty()) {
    TrackRefs::iterator itrk = muons.inVertexOutOfCalo_.begin();
    TrackRefs::iterator jtrk = muons.inVertexOutOfCalo_.end();
    for (; itrk != jtrk; ++itrk) {
      corr += PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), 0.);
      //      tracks_present = true;
    }
  }

  // Correct using electrons in-cone at vertex

  if (!elecs.inVertexInCalo_.empty()) {
    TrackRefs::iterator itrk = elecs.inVertexInCalo_.begin();
    TrackRefs::iterator jtrk = elecs.inVertexInCalo_.end();
    for (; itrk != jtrk; ++itrk) {
      corr += PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), 0.);
      //      tracks_present = true;
    }
  }

  if (!elecs.inVertexOutOfCalo_.empty()) {
    TrackRefs::iterator itrk = elecs.inVertexOutOfCalo_.begin();
    TrackRefs::iterator jtrk = elecs.inVertexOutOfCalo_.end();
    for (; itrk != jtrk; ++itrk) {
      corr += PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), 0.);
      //      tracks_present = true;
    }
  }

  // Adjust direction if in cone tracks are present

  if (!tracks_present_inin) {
    corr = corrected;
  } else {
    corr *= (corr.P() > 0. ? corrected.P() / corr.P() : 1.);
    corr = P4(corr.px(), corr.py(), corr.pz(), corrected.energy());
  }

  return corr;
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::P4 JetPlusTrackCorrector::calculateCorr(const P4& jet,
                                                               const TrackRefs& tracks,
                                                               jpt::Efficiency& eff,
                                                               bool in_cone_at_vertex,
                                                               bool in_cone_at_calo_face,
                                                               double mass,
                                                               bool is_pion,
                                                               double mip) {
  // Correction to be applied to jet 4-momentum
  P4 correction;

  // Reset efficiency container
  eff.reset();

  double theSumResp = 0;
  double theSumPt = 0;
  double theSumEnergy = 0;

  // Iterate through tracks
  if (!tracks.empty()) {
    TrackRefs::iterator itrk = tracks.begin();
    TrackRefs::iterator jtrk = tracks.end();

    for (; itrk != jtrk; ++itrk) {
      // Ignore high-pt tracks (only when in-cone and not a mip)
      if (in_cone_at_calo_face && is_pion && (*itrk)->pt() >= 50.) {
        continue;
      }

      // Inner track 4-momentum
      P4 inner;
      if (vectorial_ && vecResponse_) {
        inner = PtEtaPhiM((*itrk)->pt(), (*itrk)->eta(), (*itrk)->phi(), mass);
      } else {
        double energy = sqrt((*itrk)->px() * (*itrk)->px() + (*itrk)->py() * (*itrk)->py() +
                             (*itrk)->pz() * (*itrk)->pz() + mass * mass);
        inner = (jet.energy() > 0. ? energy / jet.energy() : 1.) * jet;
      }

      // Add track momentum (if in-cone at vertex)
      if (in_cone_at_vertex) {
        correction += inner;
      }

      // Find appropriate eta/pt bin for given track
      double eta = fabs((*itrk)->eta());
      double pt = fabs((*itrk)->pt());
      uint32_t ieta = response_.etaBin(eta);
      uint32_t ipt = response_.ptBin(pt);

      // Check bins (not for mips)
      if (is_pion && (ieta == response_.nEtaBins() || ipt == response_.nPtBins())) {
        continue;
      }

      // Outer track 4-momentum
      P4 outer;
      if (in_cone_at_calo_face) {
        if (vectorial_ && vecResponse_) {
          // Build 4-momentum from outer track (SHOULD USE IMPACT POINT?!)
          double outer_pt = (*itrk)->pt();
          double outer_eta = (*itrk)->eta();
          double outer_phi = (*itrk)->phi();
          if ((*itrk)->extra().isNonnull()) {
            outer_pt = (*itrk)->pt();
            outer_eta = (*itrk)->outerPosition().eta();  //@@ outerMomentum().eta()
            outer_phi = (*itrk)->outerPosition().phi();  //@@ outerMomentum().phi()
          }
          outer = PtEtaPhiM(outer_pt, outer_eta, outer_phi, mass);
          // Check if mip or not
          if (!is_pion) {
            outer *= (outer.energy() > 0. ? mip / outer.energy() : 1.);
          }  //@@ Scale to mip energy
          else {
            outer *= (outer.energy() > 0. ? inner.energy() / outer.energy() : 1.);
          }  //@@ Scale to inner track energy
        } else {
          // Check if mip or not
          if (!is_pion) {
            outer = (jet.energy() > 0. ? mip / jet.energy() : 1.) * jet;
          }  //@@ Jet 4-mom scaled by mip energy
          else {
            outer = inner;
          }  //@@ Set to inner track 4-momentum
        }
        if (is_pion) {
          outer *= response_.value(ieta, ipt);
        }                     //@@ Scale by pion response
        correction -= outer;  //@@ Subtract

        // Calculate the sum of responses
        theSumResp += response_.value(ieta, ipt);
      }

      // Calculate the sum of pt and energies
      theSumPt += inner.pt();
      theSumEnergy += inner.energy();

      // Record inner track energy for pion efficiency correction
      if (is_pion) {
        eff.addE(ieta, ipt, inner.energy());
      }

      // Debug
      if (verbose_ && edm::isDebugEnabled()) {
        std::stringstream temp;
        temp << " Response[" << ieta << "," << ipt << "]";
        std::stringstream ss;
        ss << "[JetPlusTrackCorrector::" << __func__ << "]" << std::endl
           << " Track eta / pt    : " << eta << " / " << pt << std::endl
           << temp.str() << std::setw(21 - temp.str().size()) << " : " << response_.value(ieta, ipt) << std::endl
           << " Track momentum added : " << inner.energy() << std::endl
           << " Response subtracted  : " << outer.energy() << std::endl
           << " Energy correction    : " << correction.energy();
        LogDebug("JetPlusTrackCorrector") << ss.str();
      }

    }  // loop through tracks
  }    // ntracks != 0

  if (in_cone_at_vertex) {
    theResponseOfChargedWithEff += theSumResp;
    theResponseOfChargedWithoutEff += theSumResp;
    theSumPtWithEff += theSumPt;
    theSumPtWithoutEff += theSumPt;
    theSumEnergyWithEff += theSumEnergy;
    theSumEnergyWithoutEff += theSumEnergy;
  }
  return correction;
}

// -----------------------------------------------------------------------------
//
JetPlusTrackCorrector::P4 JetPlusTrackCorrector::pionEfficiency(const P4& jet,
                                                                const Efficiency& eff,
                                                                bool in_cone_at_calo_face) {
  // Total correction to be applied
  P4 correction;

  double theSumResp = 0;
  double theSumPt = 0;
  double theSumEnergy = 0;

  // Iterate through eta/pt bins
  for (uint32_t ieta = 0; ieta < response_.nEtaBins() - 1; ++ieta) {
    for (uint32_t ipt = 0; ipt < response_.nPtBins() - 1; ++ipt) {
      // Check tracks are found in this eta/pt bin
      if (!eff.nTrks(ieta, ipt)) {
        continue;
      }

      for (uint16_t ii = 0; ii < 2; ++ii) {
        // Check which correction should be applied
        double corr = 0.;
        if (ii == 0) {
          corr = eff.outOfConeCorr(ieta, ipt);
        } else if (ii == 1 && in_cone_at_calo_face) {
          corr = eff.inConeCorr(ieta, ipt);
        } else {
          continue;
        }

        // Calculate correction to be applied
        P4 corr_p4;
        if (vectorial_ && vecResponse_) {
          double corr_eta = response_.binCenterEta(ieta);
          double corr_phi = jet.phi();  //@@ jet phi!
          double corr_pt = response_.binCenterPt(ipt);
          corr_p4 =
              PtEtaPhiM(corr_pt, corr_eta, corr_phi, pionMass_);  //@@ E^2 = p^2 + m_pion^2, |p| calc'ed from pt bin
          corr_p4 *= (corr_p4.energy() > 0. ? corr / corr_p4.energy() : 1.);  //@@ p4 scaled up by mean energy for bin
        } else {
          corr_p4 = (jet.energy() > 0. ? corr / jet.energy() : 1.) * jet;
        }

        // Apply correction
        if (ii == 0) {
          correction += corr_p4;
          theSumPt += corr_p4.pt();
          theSumEnergy += corr_p4.energy();
        }  //@@ Add out-of-cone
        else if (ii == 1) {
          correction -= corr_p4;
          theSumResp += corr_p4.energy();
        }  //@@ Subtract in-cone
      }
    }
  }

  theResponseOfChargedWithEff += theSumResp;
  theSumPtWithEff += theSumPt;
  theSumEnergyWithEff += theSumEnergy;
  return correction;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::matchMuons(TrackRefs::const_iterator& itrk, const edm::Handle<RecoMuons>& muons) const {
  if (muons->empty()) {
    return false;
  }

  RecoMuons::const_iterator imuon = muons->begin();
  RecoMuons::const_iterator jmuon = muons->end();
  for (; imuon != jmuon; ++imuon) {
    if (imuon->innerTrack().isNull() || !muon::isGoodMuon(*imuon, muon::TMLastStationTight) ||
        imuon->innerTrack()->pt() < 3.0) {
      continue;
    }

    if (itrk->id() != imuon->innerTrack().id()) {
      edm::LogError("JetPlusTrackCorrector") << "[JetPlusTrackCorrector::" << __func__ << "]"
                                             << "Product id of the tracks associated to the jet " << itrk->id()
                                             << " is different from the product id of the inner track used for muons "
                                             << imuon->innerTrack().id() << "!" << std::endl
                                             << "Cannot compare tracks from different collection. Configuration Error!";
      return false;
    }

    if (*itrk == imuon->innerTrack())
      return true;
  }

  return false;
}

bool JetPlusTrackCorrector::matchMuons(TrackRefs::const_iterator& itrk,
                                       const edm::Handle<pat::MuonCollection>& muons) const {
  if (muons->empty()) {
    return false;
  }

  for (auto const& muon : *muons) {
    if (muon.innerTrack().isNull() == 1)
      continue;
    if (std::abs((**itrk).pt() - muon.innerTrack()->pt()) < muonPtmatch_ &&
        std::abs((**itrk).eta() - muon.innerTrack()->eta()) < muonEtamatch_ &&
        std::abs(reco::deltaPhi((**itrk).phi(), muon.innerTrack()->phi())) < muonPhimatch_) {
      return true;
    }
  }

  return false;
}

// -----------------------------------------------------------------------------
//
bool JetPlusTrackCorrector::matchElectrons(TrackRefs::const_iterator& itrk,
                                           const edm::Handle<RecoElectrons>& elecs,
                                           const edm::Handle<RecoElectronIds>& elec_ids) const {
  if (elecs->empty()) {
    return false;
  }

  double deltaRMIN = 999.;

  uint32_t electron_index = 0;
  for (auto const& ielec : *elecs) {
    edm::Ref<RecoElectrons> electron_ref(elecs, electron_index);
    electron_index++;

    if ((*elec_ids)[electron_ref] < 1.e-6) {
      continue;
    }  //@@ Check for null value

    // DR matching b/w electron and track
    auto dR2 = deltaR2(ielec, **itrk);
    if (dR2 < deltaRMIN) {
      deltaRMIN = dR2;
    }
  }
  return deltaRMIN < electronDRmatch_ * electronDRmatch_;
}

bool JetPlusTrackCorrector::matchElectrons(TrackRefs::const_iterator& itrk,
                                           const edm::Handle<pat::ElectronCollection>& elecs) const {
  if (elecs->empty()) {
    return false;
  }

  double deltaRMIN = 999.;
  for (auto const& ielec : *elecs) {
    auto dR2 = deltaR2(ielec, **itrk);
    if (dR2 < deltaRMIN) {
      deltaRMIN = dR2;
    }
  }
  return deltaRMIN < electronDRmatch_ * electronDRmatch_;
}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::rebuildJta(const reco::Jet& fJet,
                                       const JetTracksAssociations& jtV0,
                                       TrackRefs& tracksthis,
                                       TrackRefs& Excl) const {
  tracksthis = reco::JetTracksAssociation::getValue(jtV0, fJet);

  if (jetSplitMerge_ < 0)
    return;

  typedef std::vector<reco::JetBaseRef>::iterator JetBaseRefIterator;
  std::vector<reco::JetBaseRef> theJets = reco::JetTracksAssociation::allJets(jtV0);

  TrackRefs tracks = tracksthis;
  tracksthis.clear();

  double jetEta = fJet.eta();
  double jetPhi = fJet.phi();
  double jetEtIn = 1.0 / fJet.et();

  for (TrackRefs::iterator it = tracks.begin(); it != tracks.end(); it++) {
    double trkEta = (**it).eta();
    double trkPhi = (**it).phi();
    double dR2this = deltaR2(jetEta, jetPhi, trkEta, trkPhi);
    //       double dfi = fabs(fJet.phi()-(**it).phi());
    //       if(dfi>4.*atan(1.))dfi = 8.*atan(1.)-dfi;
    //       double deta = fJet.eta() - (**it).eta();
    //       double dR2check = sqrt(dfi*dfi+deta*deta);

    double scalethis = dR2this;
    if (jetSplitMerge_ == 0)
      scalethis = 1. * jetEtIn;
    if (jetSplitMerge_ == 2)
      scalethis = dR2this * jetEtIn;
    int flag = 1;
    for (JetBaseRefIterator ii = theJets.begin(); ii != theJets.end(); ii++) {
      if (&(**ii) == &fJet) {
        continue;
      }
      double dR2 = deltaR2((*ii)->eta(), (*ii)->phi(), trkEta, trkPhi);
      double scale = dR2;
      if (jetSplitMerge_ == 0)
        scale = 1. / (**ii).et();
      if (jetSplitMerge_ == 2)
        scale = dR2 / (**ii).et();
      if (scale < scalethis)
        flag = 0;

      if (flag == 0) {
        break;
      }
    }

    if (flag == 1) {
      tracksthis.push_back(*it);
    } else {
      Excl.push_back(*it);
    }
  }

  return;
}

// -----------------------------------------------------------------------------
//
void JetPlusTrackCorrector::excludeJta(const reco::Jet& fJet,
                                       const JetTracksAssociations& jtV0,
                                       TrackRefs& tracksthis,
                                       const TrackRefs& Excl) const {
  tracksthis = reco::JetTracksAssociation::getValue(jtV0, fJet);
  if (Excl.empty())
    return;
  if (jetSplitMerge_ < 0)
    return;

  TrackRefs tracks = tracksthis;
  tracksthis.clear();

  for (TrackRefs::iterator it = tracks.begin(); it != tracks.end(); it++) {
    TrackRefs::iterator itold = find(Excl.begin(), Excl.end(), (*it));
    if (itold == Excl.end()) {
      tracksthis.push_back(*it);
    }
  }

  return;
}

//================================================================================================

double JetPlusTrackCorrector::correctAA(const reco::Jet& fJet,
                                        const reco::TrackRefVector& trBgOutOfVertex,
                                        double& mConeSize,
                                        const reco::TrackRefVector& pioninin,
                                        const reco::TrackRefVector& pioninout,
                                        double ja,
                                        const reco::TrackRefVector& trBgOutOfCalo) const {
  double mScale = 1.;
  double NewResponse = fJet.energy();

  if (trBgOutOfVertex.empty())
    return mScale;
  double EnergyOfBackgroundCharged = 0.;
  double ResponseOfBackgroundCharged = 0.;

  //
  // calculate the mean response
  //

  //================= EnergyOfBackgroundCharged ==================>
  for (reco::TrackRefVector::iterator iBgtV = trBgOutOfVertex.begin(); iBgtV != trBgOutOfVertex.end(); iBgtV++) {
    double eta = fabs((**iBgtV).eta());
    double pt = fabs((**iBgtV).pt());
    uint32_t ieta = response_.etaBin(eta);
    uint32_t ipt = response_.ptBin(pt);

    if (fabs(fJet.eta() - (**iBgtV).eta()) > mConeSize)
      continue;

    double echarBg = sqrt((**iBgtV).px() * (**iBgtV).px() + (**iBgtV).py() * (**iBgtV).py() +
                          (**iBgtV).pz() * (**iBgtV).pz() + 0.14 * 0.14);

    EnergyOfBackgroundCharged += echarBg / efficiency_.value(ieta, ipt);

  }  // Energy BG tracks

  //============= ResponseOfBackgroundCharged =======================>

  for (reco::TrackRefVector::iterator iBgtC = trBgOutOfCalo.begin(); iBgtC != trBgOutOfCalo.end(); iBgtC++) {
    double eta = fabs((**iBgtC).eta());
    double pt = fabs((**iBgtC).pt());
    uint32_t ieta = response_.etaBin(eta);
    uint32_t ipt = response_.ptBin(pt);

    if (fabs(fJet.eta() - (**iBgtC).eta()) > mConeSize)
      continue;

    // Check bins (not for mips)
    if (ieta >= response_.nEtaBins()) {
      continue;
    }
    if (ipt >= response_.nPtBins()) {
      ipt = response_.nPtBins() - 1;
    }

    double echarBg = sqrt((**iBgtC).px() * (**iBgtC).px() + (**iBgtC).py() * (**iBgtC).py() +
                          (**iBgtC).pz() * (**iBgtC).pz() + 0.14 * 0.14);

    ResponseOfBackgroundCharged += echarBg * response_.value(ieta, ipt) / efficiency_.value(ieta, ipt);

  }  // Response of BG tracks

  //=================================================================>

  /*
  //=================================================================>
  // Look for in-out tracks

  double en = 0.;
  double px = 0.;
  double py = 0.;
  double pz = 0.;

  for (reco::TrackRefVector::const_iterator it = pioninout.begin(); it != pioninout.end(); it++) {
    px += (*it)->px();
    py += (*it)->py();
    pz += (*it)->pz();
    en += sqrt((*it)->p() * (*it)->p() + 0.14 * 0.14);
  }

  // Look for in-in tracks

  double en_in = 0.;
  double px_in = 0.;
  double py_in = 0.;
  double pz_in = 0.;

  for (reco::TrackRefVector::const_iterator it = pioninin.begin(); it != pioninin.end(); it++) {
    px_in += (*it)->px();
    py_in += (*it)->py();
    pz_in += (*it)->pz();
    en_in += sqrt((*it)->p() * (*it)->p() + 0.14 * 0.14);
  }

  //===================================================================>

  //=>
*/
  double SquareEtaRingWithoutJets = ja;

  EnergyOfBackgroundCharged = EnergyOfBackgroundCharged / SquareEtaRingWithoutJets;
  ResponseOfBackgroundCharged = ResponseOfBackgroundCharged / SquareEtaRingWithoutJets;

  EnergyOfBackgroundCharged = M_PI * mConeSize * mConeSize * EnergyOfBackgroundCharged;
  ResponseOfBackgroundCharged = M_PI * mConeSize * mConeSize * ResponseOfBackgroundCharged;

  NewResponse = NewResponse - EnergyOfBackgroundCharged + ResponseOfBackgroundCharged;

  mScale = NewResponse / fJet.energy();
  if (mScale < 0.)
    mScale = 0.;
  return mScale;
}
// -----------------------------------------------------------------------------
//
Map::Map(std::string input, bool verbose) : eta_(), pt_(), data_() {
  // Some init
  clear();
  std::vector<Element> data;

  // Parse file
  std::string file = edm::FileInPath(input).fullPath();
  std::ifstream in(file.c_str());
  string line;
  uint32_t ieta_old = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    Element temp;
    ss >> temp.ieta_ >> temp.ipt_ >> temp.eta_ >> temp.pt_ >> temp.val_;
    data.push_back(temp);
    if (!ieta_old || temp.ieta_ != ieta_old) {
      if (eta_.size() < temp.ieta_ + 1) {
        eta_.resize(temp.ieta_ + 1, 0.);
      }
      eta_[temp.ieta_] = temp.eta_;
      ieta_old = temp.ieta_;
    }
    if (pt_.size() < temp.ipt_ + 1) {
      pt_.resize(temp.ipt_ + 1, 0.);
    }
    pt_[temp.ipt_] = temp.pt_;
  }

  // Populate container
  data_.resize(eta_.size(), VDouble(pt_.size(), 0.));
  std::vector<Element>::const_iterator idata = data.begin();
  std::vector<Element>::const_iterator jdata = data.end();
  for (; idata != jdata; ++idata) {
    data_[idata->ieta_][idata->ipt_] = idata->val_;
  }

  // Check
  if (data_.empty() || data_[0].empty()) {
    std::stringstream ss;
    ss << "[jpt::Map::" << __func__ << "]"
       << " Problem parsing map in location \"" << file << "\"! ";
    edm::LogError("JetPlusTrackCorrector") << ss.str();
  }

  // Check
  if (eta_.size() != data_.size() || pt_.size() != (data_.empty() ? 0 : data_[0].size())) {
    std::stringstream ss;
    ss << "[jpt::Map::" << __func__ << "]"
       << " Discrepancy b/w number of bins!";
    edm::LogError("JetPlusTrackCorrector") << ss.str();
  }

  // Debug
  if (verbose && edm::isDebugEnabled()) {
    std::stringstream ss;
    ss << "[jpt::Map::" << __func__ << "]"
       << " Parsed contents of map at location:" << std::endl
       << "\"" << file << "\"" << std::endl;
    print(ss);
    LogTrace("JetPlusTrackCorrector") << ss.str();
  }
}

// -----------------------------------------------------------------------------
//
Map::Map() : eta_(), pt_(), data_() { clear(); }

// -----------------------------------------------------------------------------
//
Map::~Map() { clear(); }

// -----------------------------------------------------------------------------
//
void Map::clear() {
  eta_.clear();
  pt_.clear();
  data_.clear();
}
// -----------------------------------------------------------------------------
//
double Map::eta(uint32_t eta_bin) const {
  if (!eta_.empty() && eta_bin < eta_.size()) {
    return eta_[eta_bin];
  } else {
    //    edm::LogWarning("JetPlusTrackCorrector")
    //      << "[jpt::Map::" << __func__ << "]"
    //      << " Trying to access element " << eta_bin
    //      << " of a vector with size " << eta_.size()
    //      << "!";
    return eta_[eta_.size() - 1];
  }
}

// -----------------------------------------------------------------------------
//
double Map::pt(uint32_t pt_bin) const {
  if (!pt_.empty() && pt_bin < pt_.size()) {
    return pt_[pt_bin];
  } else {
    //    edm::LogWarning("JetPlusTrackCorrector")
    //      << "[jpt::Map::" << __func__ << "]"
    //      << " Trying to access element " << pt_bin
    //      << " of a vector with size " << pt_.size()
    //      << "!";
    return pt_[pt_.size() - 1];
  }
}

// -----------------------------------------------------------------------------
//
double Map::binCenterEta(uint32_t eta_bin) const {
  if (!eta_.empty() && eta_bin + 1 < eta_.size()) {
    return eta_[eta_bin] + (eta_[eta_bin + 1] - eta_[eta_bin]) / 2.;
  } else {
    //    edm::LogWarning("JetPlusTrackCorrector")
    //      << "[jpt::Map::" << __func__ << "]"
    //      << " Trying to access element " << eta_bin+1
    //      << " of a vector with size " << eta_.size()
    //      << "!";
    return eta_[eta_.size() - 1];
  }
}

// -----------------------------------------------------------------------------
//
double Map::binCenterPt(uint32_t pt_bin) const {
  if (!pt_.empty() && pt_bin + 1 < pt_.size()) {
    return pt_[pt_bin] + (pt_[pt_bin + 1] - pt_[pt_bin]) / 2.;
  } else {
    //    edm::LogWarning("JetPlusTrackCorrector")
    //      << "[jpt::Map::" << __func__ << "]"
    //      << " Trying to access element " << pt_bin+1
    //      << " of a vector with size " << pt_.size()
    //      << "!";
    return pt_[pt_.size() - 1];
  }
}

// -----------------------------------------------------------------------------
//
uint32_t Map::etaBin(double val) const {
  val = fabs(val);
  for (uint32_t ieta = 0; ieta < nEtaBins() - 1; ++ieta) {  //@@ "-1" is bug?
    if (val > eta(ieta) && (ieta + 1 == nEtaBins() || val < eta(ieta + 1))) {
      return ieta;
    }
  }
  return nEtaBins();
}

// -----------------------------------------------------------------------------
//
uint32_t Map::ptBin(double val) const {
  val = fabs(val);
  for (uint32_t ipt = 0; ipt < nPtBins() - 1; ++ipt) {  //@@ "-1" is bug?
    if (val > pt(ipt) && ((ipt + 1) == nPtBins() || val < pt(ipt + 1))) {
      return ipt;
    }
  }
  return nPtBins();
}

// -----------------------------------------------------------------------------
//
double Map::value(uint32_t eta_bin, uint32_t pt_bin) const {
  if (eta_bin < data_.size() && pt_bin < (data_.empty() ? 0 : data_[0].size())) {
    return data_[eta_bin][pt_bin];
  } else {
    //    edm::LogWarning("JetPlusTrackCorrector")
    //      << "[jpt::Map::" << __func__ << "]"
    //      << " Trying to access element (" << eta_bin << "," << pt_bin << ")"
    //      << " of a vector with size (" << data_.size() << "," << ( data_.empty() ? 0 : data_[0].size() ) << ")"
    //      << "!";
    return 1.;
  }
}

// -----------------------------------------------------------------------------
//
void Map::print(std::stringstream& ss) const {
  ss << " Number of bins in eta : " << data_.size() << std::endl
     << " Number of bins in pt  : " << (data_.empty() ? 0 : data_[0].size()) << std::endl;
  VVDouble::const_iterator ieta = data_.begin();
  VVDouble::const_iterator jeta = data_.end();
  for (; ieta != jeta; ++ieta) {
    VDouble::const_iterator ipt = ieta->begin();
    VDouble::const_iterator jpt = ieta->end();
    for (; ipt != jpt; ++ipt) {
      uint32_t eta_bin = static_cast<uint32_t>(ieta - data_.begin());
      uint32_t pt_bin = static_cast<uint32_t>(ipt - ieta->begin());
      ss << " EtaBinNumber: " << eta_bin << " PtBinNumber: " << pt_bin << " EtaValue: " << eta_[eta_bin]
         << " PtValue: " << pt_[pt_bin] << " Value: " << data_[eta_bin][pt_bin] << std::endl;
    }
  }
}

// -----------------------------------------------------------------------------
//
MatchedTracks::MatchedTracks() : inVertexInCalo_(), outOfVertexInCalo_(), inVertexOutOfCalo_() { clear(); }

// -----------------------------------------------------------------------------
//
MatchedTracks::~MatchedTracks() { clear(); }

// -----------------------------------------------------------------------------
//
void MatchedTracks::clear() {
  inVertexInCalo_.clear();
  outOfVertexInCalo_.clear();
  inVertexOutOfCalo_.clear();
}

// -----------------------------------------------------------------------------
//
JetTracks::JetTracks() : vertex_(), caloFace_() { clear(); }

// -----------------------------------------------------------------------------
//
JetTracks::~JetTracks() { clear(); }

// -----------------------------------------------------------------------------
//
void JetTracks::clear() {
  vertex_.clear();
  caloFace_.clear();
}

// -----------------------------------------------------------------------------
//
Efficiency::Efficiency(const jpt::Map& response, const jpt::Map& efficiency, const jpt::Map& leakage)
    : response_(response), efficiency_(efficiency), leakage_(leakage) {
  reset();
}

// -----------------------------------------------------------------------------
//
double Efficiency::inConeCorr(uint32_t eta_bin, uint32_t pt_bin) const {
  if (check(eta_bin, pt_bin, __func__)) {
    return (outOfConeCorr(eta_bin, pt_bin) * leakage_.value(eta_bin, pt_bin) * response_.value(eta_bin, pt_bin));
  } else {
    return 0.;
  }
}

// -----------------------------------------------------------------------------
//
double Efficiency::outOfConeCorr(uint32_t eta_bin, uint32_t pt_bin) const {
  if (check(eta_bin, pt_bin, __func__)) {
    uint16_t ntrks = nTrks(eta_bin, pt_bin);
    double mean = meanE(eta_bin, pt_bin);
    double eff = (1. - efficiency_.value(eta_bin, pt_bin)) / efficiency_.value(eta_bin, pt_bin);
    if (!ntrks) {
      return 0.;
    }
    return (ntrks * eff * mean);
  } else {
    return 0.;
  }
}

// -----------------------------------------------------------------------------
//
uint16_t Efficiency::nTrks(uint32_t eta_bin, uint32_t pt_bin) const {
  if (check(eta_bin, pt_bin, __func__)) {
    return data_[eta_bin][pt_bin].first;
  } else {
    return 0;
  }
}

// -----------------------------------------------------------------------------
//
double Efficiency::sumE(uint32_t eta_bin, uint32_t pt_bin) const {
  if (check(eta_bin, pt_bin, __func__)) {
    return data_[eta_bin][pt_bin].second;
  } else {
    return 0.;
  }
}

// -----------------------------------------------------------------------------
//
double Efficiency::meanE(uint32_t eta_bin, uint32_t pt_bin) const {
  if (check(eta_bin, pt_bin, __func__)) {
    Pair tmp = data_[eta_bin][pt_bin];
    if (tmp.first) {
      return tmp.second / tmp.first;
    } else {
      return 0.;
    }
  } else {
    return 0.;
  }
}

// -----------------------------------------------------------------------------
//
void Efficiency::addE(uint32_t eta_bin, uint32_t pt_bin, double energy) {
  if (check(eta_bin, pt_bin, __func__)) {
    data_[eta_bin][pt_bin].first++;
    data_[eta_bin][pt_bin].second += energy;
  }
}

// -----------------------------------------------------------------------------
//
bool Efficiency::check(uint32_t eta_bin, uint32_t pt_bin, std::string method) const {
  if (eta_bin < data_.size() && pt_bin < (data_.empty() ? 0 : data_[0].size())) {
    return true;
  } else {
    //    edm::LogWarning("JetPlusTrackCorrector")
    //      << "[jpt::Efficiency::" << method << "]"
    //      << " Trying to access element (" << eta_bin << "," << pt_bin << ")"
    //      << " of a vector with size (" << data_.size() << "," << ( data_.empty() ? 0 : data_[0].size() ) << ")"
    //      << "!";
    return false;
  }
}

// -----------------------------------------------------------------------------
//
void Efficiency::reset() {
  data_.clear();
  data_.resize(response_.nEtaBins(), VPair(response_.nPtBins(), Pair(0, 0.)));
}

// -----------------------------------------------------------------------------
//
void Efficiency::print() const {
  if (edm::isDebugEnabled()) {
    std::stringstream ss;
    ss << "[jpt::Efficiency::" << __func__ << "]"
       << " Contents of maps:" << std::endl;
    ss << "Response map: " << std::endl;
    response_.print(ss);
    ss << "Efficiency map: " << std::endl;
    efficiency_.print(ss);
    ss << "Leakage map: " << std::endl;
    leakage_.print(ss);
    LogTrace("JetPlusTrackCorrector") << ss.str();
  }
}
