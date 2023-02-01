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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
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
  ~DisappearingMuonsSkimming() override;

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

  edm::EDGetToken recoMuonToken_;
  edm::EDGetToken standaloneMuonToken_;
  edm::EDGetTokenT<std::vector<reco::Track>> trackCollectionToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> primaryVerticesToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
  edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
  edm::EDGetToken genParticleToken_;
  edm::EDGetToken genInfoToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  std::vector<std::string> muonPathsToPass_;

  //options
  double minMuPt_ = 26;
  double maxMuEta_ = 2.4;
  double minTrackEta_ = 1.4;
  double maxTrackEta_ = 2.4;
  double minTrackPt_ = 20;
  double maxTransDCA_ = 0.005;
  double maxLongDCA_ = 0.05;
  double maxVtxChi_ = 3.;
  double minInvMass_ = 50;
  double maxInvMass_ = 150;
  double trackIsoConesize_ = 0.3;
  double trackIsoInnerCone_ = 0.01;
  double ecalIsoConesize_ = 0.4;
  double minEcalHitE_ = 0.3;
  double maxTrackIso_ = 0.05;
  double maxEcalIso_ = 10;
  double minSigInvMass_ = 76;
  double maxSigInvMass_ = 106;
  double minStandaloneDr_ = 1.;
  double maxStandaloneDE_ = -0.5;
  bool keepOffPeak_ = true;
  bool keepSameSign_ = true;
  bool keepTotalRegion_ = true;
  bool keepPartialRegion_ = true;

  //Event categories
  bool sameSign;
  bool totalRegion;
  bool partialRegion;
  bool offPeak;
};
#endif
