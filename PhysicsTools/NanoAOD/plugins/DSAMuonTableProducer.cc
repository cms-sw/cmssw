// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class DSAMuonTableProducer : public edm::global::EDProducer<> {
public:
  DSAMuonTableProducer(const edm::ParameterSet& iConfig)
      : name_(iConfig.getParameter<std::string>("name")),
        dsaMuonTag_(consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("dsaMuons"))),
        muonTag_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
        vtxTag_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
        bsTag_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
        transientTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
    produces<nanoaod::FlatTable>();
  }

  ~DSAMuonTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name")->setComment("name of the muon nanoaod::FlatTable");
    desc.add<edm::InputTag>("dsaMuons")->setComment("input displaced standalone muon collection");
    desc.add<edm::InputTag>("muons")->setComment("input muon collection");
    desc.add<edm::InputTag>("primaryVertex")->setComment("input primary vertex collection");
    desc.add<edm::InputTag>("beamspot")->setComment("input beamspot collection");
    descriptions.add("DSAMuonTable", desc);
  }

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  bool passesDisplacedID(const reco::Track& dsaMuon) const;
  std::tuple<int, int> getMatches(const pat::Muon& muon,
                                  const reco::Track& dsaMuon,
                                  float minPositionDiff = 1e-6) const;
  std::tuple<int, int> getSegments(const reco::Track& dsaMuon) const;

  std::string name_;
  edm::EDGetTokenT<std::vector<reco::Track>> dsaMuonTag_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muonTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxTag_;
  edm::EDGetTokenT<reco::BeamSpot> bsTag_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
};

void DSAMuonTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  float minPositionDiffForMatching = 1e-6;

  const std::vector<reco::Track>& dsaMuons = iEvent.get(dsaMuonTag_);
  const std::vector<pat::Muon>& patMuons = iEvent.get(muonTag_);

  const reco::VertexCollection& primaryVertices = iEvent.get(vtxTag_);
  const auto& pv = primaryVertices.at(0);
  GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

  const reco::BeamSpot& beamSpotInput = iEvent.get(bsTag_);
  const auto& bs = beamSpotInput.position();
  GlobalPoint beamSpot(bs.x(), bs.y(), bs.z());
  reco::Vertex beamSpotVertex(beamSpotInput.position(), beamSpotInput.covariance3D());

  const TransientTrackBuilder& builder = iSetup.getData(transientTrackBuilderToken_);

  unsigned int nDSAMuons = dsaMuons.size();
  unsigned int nMuons = patMuons.size();

  std::vector<float> idx, trkNumPlanes, trkNumHits, trkNumDTHits, trkNumCSCHits, outerEta, outerPhi;

  std::vector<float> dzPV, dzPVErr, dxyPVTraj, dxyPVTrajErr, dxyPVSigned, dxyPVSignedErr, ip3DPVSigned, ip3DPVSignedErr;

  std::vector<float> displacedId;
  std::vector<std::vector<float>> nMatchesPerMuon;
  std::vector<float> muonMatch1, muonMatch1idx, muonMatch2, muonMatch2idx, muonMatch3, muonMatch3idx, muonMatch4,
      muonMatch4idx, muonMatch5, muonMatch5idx;
  std::vector<float> muonDTMatch1, muonDTMatch1idx, muonDTMatch2, muonDTMatch2idx, muonDTMatch3, muonDTMatch3idx;
  std::vector<float> muonCSCMatch1, muonCSCMatch1idx, muonCSCMatch2, muonCSCMatch2idx, muonCSCMatch3, muonCSCMatch3idx;
  std::vector<float> nSegments, nDTSegments, nCSCSegments;

  std::vector<float> totPATmatches, totPATmatchesOS, totPATmatchesDisplID, totPATmatchesDisplIDOS,
      totLoosePATmatchesDisplID, totLoosePATmatchesDisplIDOS;
  std::vector<float> LoosePATmatchesDisplIDOSpt, LoosePATmatchesDisplIDOSdsaDetID1, LoosePATmatchesDisplIDOSdsaDetID2,
      LoosePATmatchesDisplIDOSdsaDetID3;

  for (unsigned int i = 0; i < nDSAMuons; i++) {
    const reco::Track& dsaMuon = dsaMuons[i];
    idx.push_back(i);

    trkNumPlanes.push_back(dsaMuon.hitPattern().muonStationsWithValidHits());
    trkNumHits.push_back(dsaMuon.hitPattern().numberOfValidMuonHits());
    trkNumDTHits.push_back(dsaMuon.hitPattern().numberOfValidMuonDTHits());
    trkNumCSCHits.push_back(dsaMuon.hitPattern().numberOfValidMuonCSCHits());

    outerEta.push_back(dsaMuon.extra().isNonnull() && dsaMuon.extra().isAvailable() ? dsaMuon.outerEta() : -999);
    outerPhi.push_back(dsaMuon.extra().isNonnull() && dsaMuon.extra().isAvailable() ? dsaMuon.outerPhi() : -999);

    dzPV.push_back(dsaMuon.dz(pv.position()));
    dzPVErr.push_back(std::hypot(dsaMuon.dzError(), pv.zError()));
    reco::TransientTrack transientTrack = builder.build(dsaMuon);
    TrajectoryStateClosestToPoint trajectoryPV = transientTrack.trajectoryStateClosestToPoint(primaryVertex);
    dxyPVTraj.push_back(trajectoryPV.perigeeParameters().transverseImpactParameter());
    dxyPVTrajErr.push_back(trajectoryPV.perigeeError().transverseImpactParameterError());
    GlobalVector muonRefTrackDir(dsaMuon.px(), dsaMuon.py(), dsaMuon.pz());
    dxyPVSigned.push_back(IPTools::signedTransverseImpactParameter(transientTrack, muonRefTrackDir, pv).second.value());
    dxyPVSignedErr.push_back(
        IPTools::signedTransverseImpactParameter(transientTrack, muonRefTrackDir, pv).second.error());

    ip3DPVSigned.push_back(IPTools::signedImpactParameter3D(transientTrack, muonRefTrackDir, pv).second.value());
    ip3DPVSignedErr.push_back(IPTools::signedImpactParameter3D(transientTrack, muonRefTrackDir, pv).second.error());

    float passesDisplacedId = 0;
    if (passesDisplacedID(dsaMuon))
      passesDisplacedId = 1;
    displacedId.push_back(passesDisplacedId);

    // Assigning 5 best matches and corresponding muon indices
    std::vector<std::pair<float, float>> muonMatches(5, std::make_pair(-1.0, -1.0));
    std::vector<std::pair<float, float>> muonDTMatches(5, std::make_pair(-1.0, -1.0));
    std::vector<std::pair<float, float>> muonCSCMatches(5, std::make_pair(-1.0, -1.0));
    std::vector<float> nMuonMatches;
    std::vector<float> nMuonDTMatches;
    std::vector<float> nMuonCSCMatches;
    for (unsigned int j = 0; j < nMuons; j++) {
      if (j > 4)
        break;
      const pat::Muon& muon = patMuons[j];
      // Muon-DSA Matches Table
      auto [nDTMatches, nCSCMatches] = getMatches(muon, dsaMuon, minPositionDiffForMatching);
      int nMatches = nDTMatches + nCSCMatches;
      muonMatches[j] = std::make_pair(nMatches, j);
      muonDTMatches[j] = std::make_pair(nDTMatches, j);
      muonCSCMatches[j] = std::make_pair(nCSCMatches, j);
      nMuonMatches.push_back(nMatches);
      nMuonDTMatches.push_back(nDTMatches);
      nMuonCSCMatches.push_back(nCSCMatches);
    }
    nMatchesPerMuon.push_back(nMuonMatches);
    nMatchesPerMuon.push_back(nMuonDTMatches);
    nMatchesPerMuon.push_back(nMuonCSCMatches);
    std::sort(muonMatches.rbegin(), muonMatches.rend());
    std::sort(muonDTMatches.rbegin(), muonDTMatches.rend());
    std::sort(muonCSCMatches.rbegin(), muonCSCMatches.rend());
    muonMatch1.push_back(muonMatches[0].first);
    muonMatch1idx.push_back(muonMatches[0].second);
    muonMatch2.push_back(muonMatches[1].first);
    muonMatch2idx.push_back(muonMatches[1].second);
    muonMatch3.push_back(muonMatches[2].first);
    muonMatch3idx.push_back(muonMatches[2].second);
    muonMatch4.push_back(muonMatches[3].first);
    muonMatch4idx.push_back(muonMatches[3].second);
    muonMatch5.push_back(muonMatches[4].first);
    muonMatch5idx.push_back(muonMatches[4].second);

    muonDTMatch1.push_back(muonDTMatches[0].first);
    muonDTMatch1idx.push_back(muonDTMatches[0].second);
    muonDTMatch2.push_back(muonDTMatches[1].first);
    muonDTMatch2idx.push_back(muonDTMatches[1].second);
    muonDTMatch3.push_back(muonDTMatches[2].first);
    muonDTMatch3idx.push_back(muonDTMatches[2].second);
    muonCSCMatch1.push_back(muonCSCMatches[0].first);
    muonCSCMatch1idx.push_back(muonCSCMatches[0].second);
    muonCSCMatch2.push_back(muonCSCMatches[1].first);
    muonCSCMatch2idx.push_back(muonCSCMatches[1].second);
    muonCSCMatch3.push_back(muonCSCMatches[2].first);
    muonCSCMatch3idx.push_back(muonCSCMatches[2].second);

    auto [nDTSegments_, nCSCSegments_] = getSegments(dsaMuon);
    nSegments.push_back(nDTSegments_ + nCSCSegments_);
    nDTSegments.push_back(nDTSegments_);
    nCSCSegments.push_back(nCSCSegments_);
  }

  auto dsaMuonTab = std::make_unique<nanoaod::FlatTable>(dsaMuons.size(), name_, false, true);

  dsaMuonTab->addColumn<float>("idx", idx, "");
  dsaMuonTab->addColumn<float>("trkNumPlanes", trkNumPlanes, "");
  dsaMuonTab->addColumn<float>("trkNumHits", trkNumHits, "");
  dsaMuonTab->addColumn<float>("trkNumDTHits", trkNumDTHits, "");
  dsaMuonTab->addColumn<float>("trkNumCSCHits", trkNumCSCHits, "");

  dsaMuonTab->addColumn<float>("outerEta", outerEta, "");
  dsaMuonTab->addColumn<float>("outerPhi", outerPhi, "");

  dsaMuonTab->addColumn<float>("dzPV", dzPV, "");
  dsaMuonTab->addColumn<float>("dzPVErr", dzPVErr, "");
  dsaMuonTab->addColumn<float>("dxyPVTraj", dxyPVTraj, "");
  dsaMuonTab->addColumn<float>("dxyPVTrajErr", dxyPVTrajErr, "");
  dsaMuonTab->addColumn<float>("dxyPVSigned", dxyPVSigned, "");
  dsaMuonTab->addColumn<float>("dxyPVSignedErr", dxyPVSignedErr, "");
  dsaMuonTab->addColumn<float>("ip3DPVSigned", ip3DPVSigned, "");
  dsaMuonTab->addColumn<float>("ip3DPVSignedErr", ip3DPVSignedErr, "");

  dsaMuonTab->addColumn<float>("displacedID", displacedId, "");

  dsaMuonTab->addColumn<float>("muonMatch1", muonMatch1, "");
  dsaMuonTab->addColumn<float>("muonMatch1idx", muonMatch1idx, "");
  dsaMuonTab->addColumn<float>("muonMatch2", muonMatch2, "");
  dsaMuonTab->addColumn<float>("muonMatch2idx", muonMatch2idx, "");
  dsaMuonTab->addColumn<float>("muonMatch3", muonMatch3, "");
  dsaMuonTab->addColumn<float>("muonMatch3idx", muonMatch3idx, "");
  dsaMuonTab->addColumn<float>("muonMatch4", muonMatch4, "");
  dsaMuonTab->addColumn<float>("muonMatch4idx", muonMatch4idx, "");
  dsaMuonTab->addColumn<float>("muonMatch5", muonMatch5, "");
  dsaMuonTab->addColumn<float>("muonMatch5idx", muonMatch5idx, "");

  dsaMuonTab->addColumn<float>("muonDTMatch1", muonDTMatch1, "");
  dsaMuonTab->addColumn<float>("muonDTMatch1idx", muonDTMatch1idx, "");
  dsaMuonTab->addColumn<float>("muonDTMatch2", muonDTMatch2, "");
  dsaMuonTab->addColumn<float>("muonDTMatch2idx", muonDTMatch2idx, "");
  dsaMuonTab->addColumn<float>("muonDTMatch3", muonDTMatch3, "");
  dsaMuonTab->addColumn<float>("muonDTMatch3idx", muonDTMatch3idx, "");

  dsaMuonTab->addColumn<float>("muonCSCMatch1", muonCSCMatch1, "");
  dsaMuonTab->addColumn<float>("muonCSCMatch1idx", muonCSCMatch1idx, "");
  dsaMuonTab->addColumn<float>("muonCSCMatch2", muonCSCMatch2, "");
  dsaMuonTab->addColumn<float>("muonCSCMatch2idx", muonCSCMatch2idx, "");
  dsaMuonTab->addColumn<float>("muonCSCMatch3", muonCSCMatch3, "");
  dsaMuonTab->addColumn<float>("muonCSCMatch3idx", muonCSCMatch3idx, "");

  dsaMuonTab->addColumn<float>("nSegments", nSegments, "");
  dsaMuonTab->addColumn<float>("nDTSegments", nDTSegments, "");
  dsaMuonTab->addColumn<float>("nCSCSegments", nCSCSegments, "");

  iEvent.put(std::move(dsaMuonTab));
}

bool DSAMuonTableProducer::passesDisplacedID(const reco::Track& dsaMuon) const {
  // displaced muon Id as recommended by Muon POG
  float validHits = dsaMuon.hitPattern().numberOfValidMuonCSCHits() + dsaMuon.hitPattern().numberOfValidMuonDTHits();
  if (validHits > 12) {
    if (dsaMuon.hitPattern().numberOfValidMuonCSCHits() != 0 ||
        (dsaMuon.hitPattern().numberOfValidMuonCSCHits() == 0 && dsaMuon.hitPattern().numberOfValidMuonDTHits() > 18)) {
      if (dsaMuon.normalizedChi2() < 2.5) {
        if (dsaMuon.ptError() / dsaMuon.pt() < 1) {
          return true;
        }
      }
    }
  }
  return false;
}

// Returns number of DT and CSC segments of the DSA muon are associated to the PAT muon
std::tuple<int, int> DSAMuonTableProducer::getMatches(const pat::Muon& muon,
                                                      const reco::Track& dsaMuon,
                                                      float minPositionDiff) const {
  int nMatchesDT = 0;
  int nMatchesCSC = 0;

  if (dsaMuon.extra().isNonnull() && dsaMuon.extra().isAvailable()) {
    for (auto& hit : dsaMuon.recHits()) {
      if (!hit->isValid())
        continue;
      DetId id = hit->geographicalId();
      if (id.det() != DetId::Muon)
        continue;

      if (id.subdetId() == MuonSubdetId::DT || id.subdetId() == MuonSubdetId::CSC) {
        for (auto& chamber : muon.matches()) {
          if (chamber.id.rawId() != id.rawId())
            continue;

          for (auto& segment : chamber.segmentMatches) {
            if (fabs(segment.x - hit->localPosition().x()) < minPositionDiff &&
                fabs(segment.y - hit->localPosition().y()) < minPositionDiff) {
              if (id.subdetId() == MuonSubdetId::DT)
                nMatchesDT++;
              else
                nMatchesCSC++;
              break;
            }
          }
        }
      }
    }
  }
  return {nMatchesDT, nMatchesCSC};
}

std::tuple<int, int> DSAMuonTableProducer::getSegments(const reco::Track& dsaMuon) const {
  int nHitsDT = 0;
  int nHitsCSC = 0;
  if (dsaMuon.extra().isNonnull() && dsaMuon.extra().isAvailable()) {
    for (auto& hit : dsaMuon.recHits()) {
      if (!hit->isValid())
        continue;
      DetId id = hit->geographicalId();
      if (id.det() != DetId::Muon)
        continue;
      if (id.subdetId() == MuonSubdetId::DT)
        nHitsDT++;
      if (id.subdetId() == MuonSubdetId::DT)
        nHitsCSC++;
    }
  }
  return {nHitsDT, nHitsCSC};
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DSAMuonTableProducer);
