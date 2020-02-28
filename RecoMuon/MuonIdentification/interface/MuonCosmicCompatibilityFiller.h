/**
 *  \package: MuonIdentification
 *  \class: MuonCosmicCompatibilityFiller
 *
 *  Description: class for cosmic muon identification
 *
 *
 *  \author: A. Everett, Purdue University
 *  \author: A. Svyatkovskiy, Purdue University
 *  \author: H.D. Yoo, Purdue University
 *
 **/

#ifndef MuonIdentification_MuonCosmicCompatibilityFiller_h
#define MuonIdentification_MuonCosmicCompatibilityFiller_h

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm
class GlobalMuonRefitter;

class MuonCosmicCompatibilityFiller {
public:
  MuonCosmicCompatibilityFiller(const edm::ParameterSet&, edm::ConsumesCollector&);
  ~MuonCosmicCompatibilityFiller();

  /// fill cosmic compatibility variables
  reco::MuonCosmicCompatibility fillCompatibility(const reco::Muon& muon, edm::Event&, const edm::EventSetup&);

private:
  /// check muon time (DT and CSC) information: 0 == prompt-like
  float muonTiming(const edm::Event& iEvent, const reco::Muon& muon, bool isLoose) const;

  ///return cosmic-likeness based on presence of a track in opp side: 0 == no matching opp tracks
  unsigned int backToBack2LegCosmic(const edm::Event&, const reco::Muon&) const;

  /// return cosmic-likeness based on the 2D impact parameters (dxy, dz wrt to PV). 0 == cosmic-like
  unsigned int pvMatches(const edm::Event&, const reco::Muon&, bool) const;

  /// returns cosmic-likeness based on overlap with traversing cosmic muon (only muon/STA hits are used)
  bool isOverlappingMuon(const edm::Event&, const edm::EventSetup& iSetup, const reco::Muon&) const;

  /// get number of muons in the vent
  unsigned int nMuons(const edm::Event&) const;

  /// returns cosmic-likeness based on the event activity information: tracker track multiplicity and vertex quality. 0 == cosmic-like
  unsigned int eventActivity(const edm::Event&, const reco::Muon&) const;

  /// combined cosmic-likeness: 0 == not cosmic-like
  float combinedCosmicID(
      const edm::Event&, const edm::EventSetup& iSetup, const reco::Muon&, bool CheckMuonID, bool checkVertex) const;

  /// tag a muon as cosmic based on the muonID information
  bool checkMuonID(const reco::Muon&) const;

  /// tag a muon as cosmic based on segment compatibility and the number of segment matches
  bool checkMuonSegments(const reco::Muon& muon) const;

private:
  std::vector<edm::InputTag> inputMuonCollections_;
  std::vector<edm::InputTag> inputTrackCollections_;
  edm::InputTag inputCosmicMuonCollection_;
  edm::InputTag inputVertexCollection_;

  std::vector<edm::EDGetTokenT<reco::MuonCollection> > muonTokens_;
  std::vector<edm::EDGetTokenT<reco::TrackCollection> > trackTokens_;
  edm::EDGetTokenT<reco::MuonCollection> cosmicToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  double maxdxyLoose_;
  double maxdzLoose_;
  double maxdxyTight_;
  double maxdzTight_;
  double maxdxyLooseMult_;
  double maxdzLooseMult_;
  double maxdxyTightMult_;
  double maxdzTightMult_;
  double largedxyMult_;
  double largedxy_;
  double hIpTrdxy_;
  double hIpTrvProb_;
  double minvProb_;
  double maxvertZ_;
  double maxvertRho_;
  unsigned int nTrackThreshold_;
  double offTimePosLoose_;
  double offTimeNegLoose_;
  double offTimePosTight_;
  double offTimeNegTight_;
  double offTimePosLooseMult_;
  double offTimeNegLooseMult_;
  double offTimePosTightMult_;
  double offTimeNegTightMult_;
  double corrTimePos_;
  double corrTimeNeg_;
  double deltaPt_;
  double angleThreshold_;
  int sharedHits_;
  double sharedFrac_;
  double ipThreshold_;
  int nChamberMatches_;
  double segmentComp_;
};
#endif
