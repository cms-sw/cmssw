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
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//
// class declaration
//

class DisappearingMuonsSkimming : public edm::one::EDFilter<> {
public:
  explicit DisappearingMuonsSkimming(const edm::ParameterSet&);
  ~DisappearingMuonsSkimming() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  bool passTriggers(const edm::Event& iEvent,
                    edm::EDGetToken m_trigResultsToken,
                    std::vector<std::string> m_muonPathsToPass);
  double getTrackIsolation(const edm::Event&,
                           edm::Handle<reco::VertexCollection> vtxHandle,
                           std::vector<reco::Track>::const_iterator& iTrack);
  double getECALIsolation(const edm::Event&, const edm::EventSetup&, const reco::TransientTrack track);

  // ----------member data ---------------------------

  const edm::EDGetToken recoMuonToken_;
  const edm::EDGetToken standaloneMuonToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackCollectionToken_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> primaryVerticesToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
  const edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
  const edm::EDGetToken genParticleToken_;
  const edm::EDGetToken genInfoToken_;
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
