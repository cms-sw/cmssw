// -*- C++ -*-
//
// Package:    Skimming/DisappearingMuonsSkimming
// Class:      DisappearingMuonsSkimming
//
/**\class DisappearingMuonsSkimming DisappearingMuonsSkimming.cc Skimming/DisappearingMuonsSkimming/plugins/DisappearingMuonsSkimming.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Revering
//         Created:  Tie, 31 Jan 2023 21:22:23 GMT
//
//
#ifndef Configuration_Skimming_DisappearingMuonsSkimming_h
#define Configuration_Skimming_DisappearingMuonsSkimming_h

// system include files
#include <memory>

// user include filter
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

//
// class declaration
//

class DisappearingMuonsSkimming : public edm::one::EDFilter<> {
public:
  explicit DisappearingMuonsSkimming(const edm::ParameterSet&);
  ~DisappearingMuonsSkimming() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  bool passTriggers(const edm::Event& iEvent,
                    const edm::TriggerResults& results,
                    const std::vector<std::string>& m_muonPathsToPass);

  bool findTrackInVertices(const reco::TrackRef& tkToMatch,
                           const reco::VertexCollection& vertices,
                           unsigned int& vtxIndex,
                           unsigned int& trackIndex);

  double getTrackIsolation(const reco::TrackRef& tkToMatch, const reco::VertexCollection& vertices);
  double getECALIsolation(const edm::Event&, const edm::EventSetup&, const reco::TransientTrack& track);

  // ----------member data ---------------------------
  const edm::EDGetTokenT<reco::MuonCollection> recoMuonToken_;
  const edm::EDGetTokenT<reco::TrackCollection> standaloneMuonToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackCollectionToken_;
  const edm::EDGetTokenT<reco::VertexCollection> primaryVerticesToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
  const edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  const std::vector<std::string> muonPathsToPass_;

  //options
  const double minMuPt_;
  const double maxMuEta_;
  const double minTrackEta_;
  const double maxTrackEta_;
  const double minTrackPt_;
  const double maxTransDCA_;
  const double maxLongDCA_;
  const double maxVtxChi_;
  const double minInvMass_;
  const double maxInvMass_;
  const double trackIsoConesize_;
  const double trackIsoInnerCone_;
  const double ecalIsoConesize_;
  const double minEcalHitE_;
  const double maxTrackIso_;
  const double maxEcalIso_;
  const double minSigInvMass_;
  const double maxSigInvMass_;
  const double minStandaloneDr_;
  const double maxStandaloneDE_;
  const bool keepOffPeak_;
  const bool keepSameSign_;
  const bool keepTotalRegion_;
  const bool keepPartialRegion_;
};
#endif
