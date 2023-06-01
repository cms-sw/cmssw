/** \class MuCSCTnPFlatTableProducer MuCSCTnPFlatTableProducer.cc DPGAnalysis/MuonTools/plugins/MuCSCTnPFlatTableProducer.cc
 *  
 * Helper class : the CSC Tag and probe segment efficiency  filler
 *
 * \author M. Herndon (UW Madison)
 *
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TString.h"
#include "TRegexp.h"

#include <iostream>

#include <numeric>
#include <vector>

#include "DPGAnalysis/MuonTools/interface/MuBaseFlatTableProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class MuonServiceProxy;

class MuCSCTnPFlatTableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuCSCTnPFlatTableProducer(const edm::ParameterSet&);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  /// Fill tree branches for a given events
  void fillTable(edm::Event&) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run&, const edm::EventSetup&) final;

  /// Get info from the ES for a given event
  void getFromES(const edm::EventSetup&) final;

private:
  static constexpr Float_t MEZ[6] = {601.3, 696.11, 696.11, 827.56, 936.44, 1025.9};

  /// Tokens
  nano_mu::EDTokenHandle<reco::MuonCollection> m_muToken;
  nano_mu::EDTokenHandle<reco::TrackCollection> m_trackToken;

  nano_mu::EDTokenHandle<CSCSegmentCollection> m_cscSegmentToken;

  nano_mu::EDTokenHandle<std::vector<reco::Vertex>> m_primaryVerticesToken;

  nano_mu::EDTokenHandle<edm::TriggerResults> m_trigResultsToken;
  nano_mu::EDTokenHandle<trigger::TriggerEvent> m_trigEventToken;

  /// Name of the triggers used by muon filler for trigger matching
  std::string m_trigName;
  std::string m_isoTrigName;

  /// Handles to geometry, detector and specialized objects
  /// CSC Geometry
  nano_mu::ESTokenHandle<CSCGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_cscGeometry;

  /// Muon service proxy
  std::unique_ptr<MuonServiceProxy> m_muonSP;

  /// Transient Track Builder
  nano_mu::ESTokenHandle<TransientTrackBuilder, TransientTrackRecord> m_transientTrackBuilder;

  // Extrapolator to cylinder
  edm::ESHandle<Propagator> propagatorAlong;
  edm::ESHandle<Propagator> propagatorOpposite;
  edm::ESHandle<MagneticField> theBField;

  /// HLT config provider
  HLTConfigProvider m_hltConfig;

  /// Indices of the triggers used by muon filler for trigger matching
  std::vector<int> m_trigIndices;
  std::vector<int> m_isoTrigIndices;

  /// Selection functions
  //bool muonTagSelection(const reco::Muon & muon,edm::Handle<std::vector<reco::Track>> tracks);
  bool trackProbeSelection(const reco::Track& track, edm::Handle<std::vector<reco::Track>>);
  bool muonTagSelection(const reco::Muon&);
  //bool trackProbeSelection(const reco::Track & track);
  bool zSelection(const reco::Muon&, const reco::Track&);

  // Calculation functions
  double zMass(const reco::Track&, const reco::Muon&);
  double calcDeltaR(double, double, double, double);
  double iso(const reco::Track&, edm::Handle<std::vector<reco::Track>>);

  // Track extrapolation and segment match functions
  TrajectoryStateOnSurface surfExtrapTrkSam(const reco::Track&, double);
  FreeTrajectoryState freeTrajStateMuon(const reco::Track&);

  UChar_t ringCandidate(Int_t iiStation, Int_t station, Float_t feta, Float_t phi);
  UChar_t thisChamberCandidate(UChar_t station, UChar_t ring, Float_t phi);

  TrajectoryStateOnSurface* matchTTwithCSCSeg(const reco::Track&,
                                              edm::Handle<CSCSegmentCollection>,
                                              CSCSegmentCollection::const_iterator&,
                                              CSCDetId&);
  Float_t TrajectoryDistToSeg(TrajectoryStateOnSurface, CSCSegmentCollection::const_iterator);
  std::vector<Float_t> GetEdgeAndDistToGap(const reco::Track&, CSCDetId&);
  Float_t YDistToHVDeadZone(Float_t, Int_t);

  /// The variables holding
  /// the T&P related information

  unsigned int m_nZCands;  // the # of digis (size of all following vectors)

  double _trackIso;
  double _muonIso;
  double _zMass;

  bool hasTrigger(std::vector<int>&,
                  const trigger::TriggerObjectCollection&,
                  edm::Handle<trigger::TriggerEvent>&,
                  const reco::Muon&);

  float computeTrkIso(const reco::MuonIsolation&, float);
  float computePFIso(const reco::MuonPFIsolation&, float);
};

MuCSCTnPFlatTableProducer::MuCSCTnPFlatTableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer(config),
      m_muToken{config, consumesCollector(), "muonSrc"},
      m_trackToken{config, consumesCollector(), "trackSrc"},
      m_cscSegmentToken{config, consumesCollector(), "cscSegmentSrc"},
      m_primaryVerticesToken{config, consumesCollector(), "primaryVerticesSrc"},
      m_trigResultsToken{config, consumesCollector(), "trigResultsSrc"},
      m_trigEventToken{config, consumesCollector(), "trigEventSrc"},
      m_trigName{config.getParameter<std::string>("trigName")},
      m_isoTrigName{config.getParameter<std::string>("isoTrigName")},
      m_cscGeometry{consumesCollector()},
      m_muonSP{std::make_unique<MuonServiceProxy>(config.getParameter<edm::ParameterSet>("ServiceParameters"),
                                                  consumesCollector())},
      m_transientTrackBuilder{consumesCollector(), "TransientTrackBuilder"} {
  produces<nanoaod::FlatTable>();
}

void MuCSCTnPFlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "cscTnP");
  desc.add<edm::InputTag>("muonSrc", edm::InputTag{"muons"});
  desc.add<edm::InputTag>("trackSrc", edm::InputTag{"generalTracks"});
  desc.add<edm::InputTag>("cscSegmentSrc", edm::InputTag{"cscSegments"});
  desc.add<edm::InputTag>("primaryVerticesSrc", edm::InputTag{"offlinePrimaryVertices"});

  desc.add<edm::InputTag>("trigEventSrc", edm::InputTag{"hltTriggerSummaryAOD::HLT"});
  desc.add<edm::InputTag>("trigResultsSrc", edm::InputTag{"TriggerResults::HLT"});

  desc.add<std::string>("trigName", "none");
  desc.add<std::string>("isoTrigName", "HLT_IsoMu2*");

  desc.setAllowAnything();

  descriptions.addWithDefaultLabel(desc);
}

void MuCSCTnPFlatTableProducer::getFromES(const edm::Run& run, const edm::EventSetup& environment) {
  m_cscGeometry.getFromES(environment);

  bool changed{true};
  m_hltConfig.init(run, environment, "HLT", changed);

  const bool enableWildcard{true};

  TString tName = TString(m_trigName);
  TRegexp tNamePattern = TRegexp(tName, enableWildcard);

  for (unsigned iPath = 0; iPath < m_hltConfig.size(); ++iPath) {
    TString pathName = TString(m_hltConfig.triggerName(iPath));
    if (pathName.Contains(tNamePattern))
      m_trigIndices.push_back(static_cast<int>(iPath));
  }

  tName = TString(m_isoTrigName);
  tNamePattern = TRegexp(tName, enableWildcard);

  for (unsigned iPath = 0; iPath < m_hltConfig.size(); ++iPath) {
    TString pathName = TString(m_hltConfig.triggerName(iPath));
    if (pathName.Contains(tNamePattern))
      m_isoTrigIndices.push_back(static_cast<int>(iPath));
  }
}

void MuCSCTnPFlatTableProducer::getFromES(const edm::EventSetup& environment) {
  m_transientTrackBuilder.getFromES(environment);
  m_muonSP->update(environment);
}

void MuCSCTnPFlatTableProducer::fillTable(edm::Event& ev) {
  unsigned int m_nZCands = 0;  // the # of digis (size of all following vectors)

  // Muon track tag variables
  std::vector<float> m_muonPt;        // muon pT [GeV/c]
  std::vector<float> m_muonPhi;       // muon phi [rad]
  std::vector<float> m_muonEta;       // muon eta
  std::vector<float> m_muonPtError;   // muon pT [GeV/c] error
  std::vector<float> m_muonPhiError;  // muon phi [rad] error
  std::vector<float> m_muonEtaError;  // muon eta error
  std::vector<int> m_muonCharge;      // muon charge
  std::vector<float> m_muonDXY;       // muon dXY
  std::vector<float> m_muonDZ;        // muon dZ
  std::vector<int> m_muonTrkHits;     // muon track Hits
  std::vector<float> m_muonChi2;      // muon Chi2
  std::vector<bool> m_muonTrigger;    // muon trigger
  std::vector<float> m_muonIso;       // track Iso

  // Track probe variabes
  std::vector<float> m_trackPt;        // track pT [GeV/c]
  std::vector<float> m_trackP;         // track P [GeV/c]
  std::vector<float> m_trackPhi;       // track phi [rad]
  std::vector<float> m_trackEta;       // track eta
  std::vector<float> m_trackPtError;   // track pT [GeV/c] error
  std::vector<float> m_trackPhiError;  // track phi [rad] error
  std::vector<float> m_trackEtaError;  // track eta error
  std::vector<int> m_trackCharge;      // track charge
  std::vector<float> m_trackDXY;       // track dXY
  std::vector<float> m_trackDZ;        // track dZ
  std::vector<int> m_trackTrkHits;     // track Hits
  std::vector<float> m_trackChi2;      // track Chi2
  std::vector<float> m_trackIso;       // track Iso

  // Z and global variables
  std::vector<float> m_zMass;                    // z mass
  std::vector<float> m_dRTrackMuon;              // dR between the track and muon
  std::vector<float> m_numberOfPrimaryVertices;  // Number of primary Vertices

  // CSC chamber information, station encoded in vector
  std::vector<int> m_chamberEndcap;                  // chamber endcap
                                                     // station encoded in array index
  std::array<std::vector<int>, 4> m_chamberRing;     // chamber Ring
  std::array<std::vector<int>, 4> m_chamberChamber;  // chamber Chamber
  std::array<std::vector<int>, 4> m_chamberLayer;    // Segment layer information

  // Track intersection variables
  std::array<std::vector<float>, 4> m_ttIntLocalX;       // track trajector intersection local X on stations 1-4
  std::array<std::vector<float>, 4> m_ttIntLocalY;       // track trajector intersection local Y on stations 1-4
  std::array<std::vector<float>, 4> m_ttIntLocalErrorX;  // track trajector intersection local X on stations 1-4
  std::array<std::vector<float>, 4> m_ttIntLocalErrorY;  // track trajector intersection local Y on stations 1-4
  std::array<std::vector<float>, 4> m_ttIntLocalW;       // track trajector intersection local Wire on stations 1-4
  std::array<std::vector<float>, 4> m_ttIntLocalS;       // track trajector intersection local Strip on stations 1-4
  std::array<std::vector<float>, 4> m_ttIntEta;          // track trajector intersection Eta stations 1-4

  // Track intersection fiducial information

  std::array<std::vector<float>, 4>
      m_ttDistToEdge;  // track trajector intersection distance to edge, neg is with chamber, on stations 1-4
  std::array<std::vector<float>, 4> m_ttDistToHVGap;  // track trajector intersection distance to HV GAP on stations 1-4

  // Segment location variables
  std::array<std::vector<float>, 4> m_segLocalX;       // segment local X on stations 1-4
  std::array<std::vector<float>, 4> m_segLocalY;       // segment local Y on stations 1-4
  std::array<std::vector<float>, 4> m_segLocalErrorX;  // segment local X error on stations 1-4
  std::array<std::vector<float>, 4> m_segLocalErrorY;  // segment local Y error on stations 1-4

  // track intersection segment residuals variables
  std::array<std::vector<float>, 4>
      m_ttIntSegResidualLocalX;  // track trajector intersection  Segment residual local X on stations 1-4
  std::array<std::vector<float>, 4>
      m_ttIntSegResidualLocalY;  // track trajector intersection  Segment residuallocal Y on stations 1-4

  auto&& propagator_along = m_muonSP->propagator("SteppingHelixPropagatorAlong");
  auto&& propagator_opposite = m_muonSP->propagator("SteppingHelixPropagatorOpposite");

  propagatorAlong = propagator_along;
  propagatorOpposite = propagator_opposite;

  theBField = m_muonSP->magneticField();

  auto muons = m_muToken.conditionalGet(ev);
  auto tracks = m_trackToken.conditionalGet(ev);
  auto segments = m_cscSegmentToken.conditionalGet(ev);
  auto primaryVertices = m_primaryVerticesToken.conditionalGet(ev);

  auto triggerResults = m_trigResultsToken.conditionalGet(ev);
  auto triggerEvent = m_trigEventToken.conditionalGet(ev);

  if (muons.isValid() && tracks.isValid() && segments.isValid() && primaryVertices.isValid() &&
      m_transientTrackBuilder.isValid()) {
    for (const auto& muon : (*muons)) {
      if (!muonTagSelection(muon))
        continue;

      bool muonTrigger = false;
      if (triggerResults.isValid() && triggerEvent.isValid()) {
        const auto& triggerObjects = triggerEvent->getObjects();
        muonTrigger = (hasTrigger(m_isoTrigIndices, triggerObjects, triggerEvent, muon) ||
                       hasTrigger(m_trigIndices, triggerObjects, triggerEvent, muon));
      }

      for (const auto& track : (*tracks)) {
        if (!trackProbeSelection(track, tracks))
          continue;
        if (!zSelection(muon, track))
          continue;
        //std::cout << "Z candidate found: " << _zMass << " track eta: " << track.eta() << std::endl;
        //std::cout.flush();
        m_nZCands++;

        m_trackPt.push_back(track.pt());
        m_trackP.push_back(track.p());
        m_trackEta.push_back(track.eta());
        m_trackPhi.push_back(track.phi());
        m_trackPtError.push_back(track.pt());
        m_trackEtaError.push_back(track.eta());
        m_trackPhiError.push_back(track.phi());
        m_trackCharge.push_back(track.charge());
        m_trackDXY.push_back(track.dxy());
        m_trackDZ.push_back(track.dz());
        m_trackTrkHits.push_back(track.hitPattern().numberOfValidTrackerHits());
        m_trackChi2.push_back(track.normalizedChi2());
        m_trackIso.push_back(_trackIso);

        m_muonPt.push_back(muon.track()->pt());
        m_muonPhi.push_back(muon.track()->phi());
        m_muonEta.push_back(muon.track()->eta());
        m_muonPtError.push_back(muon.track()->ptError());
        m_muonPhiError.push_back(muon.track()->phiError());
        m_muonEtaError.push_back(muon.track()->etaError());
        m_muonCharge.push_back(muon.charge());
        m_muonDXY.push_back(muon.track()->dxy());
        m_muonDZ.push_back(muon.track()->dz());
        m_muonTrkHits.push_back(muon.track()->hitPattern().numberOfValidTrackerHits());
        m_muonChi2.push_back(muon.track()->normalizedChi2());
        m_muonIso.push_back(computeTrkIso(muon.isolationR03(), muon.pt()));
        m_muonTrigger.push_back(muonTrigger);

        m_zMass.push_back(_zMass);
        double_t dR = calcDeltaR(track.eta(), muon.eta(), track.phi(), muon.phi());
        //double_t dR = 1.0;
        m_dRTrackMuon.push_back(dR);
        const reco::VertexCollection& vertices = *primaryVertices.product();
        m_numberOfPrimaryVertices.push_back(vertices.size());

        bool ec = (track.eta() > 0);
        UChar_t endcapCSC = ec ? 0 : 1;
        m_chamberEndcap.push_back(endcapCSC * 1);

        Int_t iiStationFail = 0;
        for (int iiStationZ = 0; iiStationZ < 6; iiStationZ++) {
          UChar_t stationCSC = iiStationZ > 2 ? iiStationZ - 2 : 0;
          UChar_t ringCSC = 0;
          TrajectoryStateOnSurface tsos = surfExtrapTrkSam(track, ec ? MEZ[iiStationZ] : -MEZ[iiStationZ]);

          if (tsos.isValid()) {
            Float_t trkEta = tsos.globalPosition().eta(), trkPhi = tsos.globalPosition().phi();
            ringCSC = ringCandidate(iiStationZ, stationCSC + 1, trkEta, trkPhi);

            if (ringCSC) {
              UChar_t chamberCSC = thisChamberCandidate(stationCSC + 1, ringCSC, track.phi()) - 1;
              CSCDetId Layer3id = CSCDetId(endcapCSC + 1, stationCSC + 1, ringCSC, chamberCSC + 1, 3);
              CSCDetId Layer0Id = CSCDetId(endcapCSC + 1,
                                           stationCSC + 1,
                                           ringCSC,
                                           chamberCSC + 1,
                                           0);  //layer 0 is the mid point of the chamber. It is not a real layer.
              // !!!!! need to fix Layer0Id problem with ME1/1 here

              const BoundPlane& Layer3Surface = m_cscGeometry->idToDet(Layer3id)->surface();

              tsos = surfExtrapTrkSam(track, Layer3Surface.position().z());

              if (tsos.isValid()) {
                // Fill track intersection denominator information
                LocalPoint localTTIntPoint = Layer3Surface.toLocal(tsos.freeState()->position());
                const CSCLayerGeometry* layerGeoma = m_cscGeometry->chamber(Layer0Id)->layer(3)->geometry();
                const CSCLayerGeometry* layerGeomb = m_cscGeometry->chamber(Layer0Id)->layer(4)->geometry();

                m_chamberRing[stationCSC].push_back(ringCSC);
                m_chamberChamber[stationCSC].push_back(chamberCSC);
                m_ttIntLocalX[stationCSC].push_back(localTTIntPoint.x());
                m_ttIntLocalY[stationCSC].push_back(localTTIntPoint.y());
                m_ttIntLocalW[stationCSC].push_back(
                    (layerGeoma->nearestWire(localTTIntPoint) + layerGeomb->nearestWire(localTTIntPoint)) / 2.0);
                m_ttIntLocalS[stationCSC].push_back(
                    (layerGeoma->strip(localTTIntPoint) + layerGeomb->strip(localTTIntPoint)) / 2.0);
                m_ttIntEta[stationCSC].push_back(trkEta);

                // Errors are those of the track intersection, chosing the plane and exact geomentry is performed in the function
                Float_t CSCProjEdgeDist = -9999.0;
                Float_t ttIntLocalErrorX = -9999.0;
                Float_t CSCDyProjHVGap = 9999.0;
                Float_t ttIntLocalErrorY = -9999.0;
                for (Int_t ly = 1; ly < 7; ly++) {
                  CSCDetId Layerid = CSCDetId(endcapCSC + 1, stationCSC + 1, ringCSC, chamberCSC + 1, ly);
                  std::vector<Float_t> EdgeAndDistToGap(GetEdgeAndDistToGap(
                      track, Layerid));  //values: 1-edge;2-err of edge;3-disttogap;4-err of dist to gap
                  if (EdgeAndDistToGap[0] > CSCProjEdgeDist) {
                    CSCProjEdgeDist = EdgeAndDistToGap[0];
                    ttIntLocalErrorX = EdgeAndDistToGap[1];
                  }
                  if (EdgeAndDistToGap[2] < CSCDyProjHVGap) {
                    CSCDyProjHVGap = EdgeAndDistToGap[2];
                    ttIntLocalErrorY = EdgeAndDistToGap[3];
                  }
                }
                m_ttDistToEdge[stationCSC].push_back(CSCProjEdgeDist);
                m_ttDistToHVGap[stationCSC].push_back(CSCDyProjHVGap);
                m_ttIntLocalErrorX[stationCSC].push_back(ttIntLocalErrorX);
                m_ttIntLocalErrorY[stationCSC].push_back(ttIntLocalErrorY);

                // now we have a local point for comparison to segments
                CSCSegmentCollection::const_iterator cscSegOut;
                TrajectoryStateOnSurface* TrajToSeg = matchTTwithCSCSeg(track, segments, cscSegOut, Layer3id);

                if (TrajToSeg == nullptr) {
                  // fill Null Num
                  m_segLocalX[stationCSC].push_back(-9999.0);
                  m_segLocalY[stationCSC].push_back(-9999.0);
                  m_segLocalErrorX[stationCSC].push_back(0.0);
                  m_segLocalErrorY[stationCSC].push_back(0.0);

                  m_ttIntSegResidualLocalX[stationCSC].push_back(-9990.0);
                  m_ttIntSegResidualLocalY[stationCSC].push_back(-9990.0);

                  m_chamberLayer[stationCSC].push_back(-9);

                  continue;
                }

                LocalPoint localSegmentPoint = (*cscSegOut).localPosition();
                LocalError localSegErr = (*cscSegOut).localPositionError();

                m_segLocalX[stationCSC].push_back(localSegmentPoint.x());
                m_segLocalY[stationCSC].push_back(localSegmentPoint.y());
                m_segLocalErrorX[stationCSC].push_back(sqrt(localSegErr.xx()));
                m_segLocalErrorY[stationCSC].push_back(sqrt(localSegErr.yy()));

                m_ttIntSegResidualLocalX[stationCSC].push_back(localTTIntPoint.x() - localSegmentPoint.x());
                m_ttIntSegResidualLocalY[stationCSC].push_back(localTTIntPoint.y() - localSegmentPoint.y());
                /* Extract layers for hits */
                int layers = 0;
                for (std::vector<CSCRecHit2D>::const_iterator itRH = cscSegOut->specificRecHits().begin();
                     itRH != cscSegOut->specificRecHits().end();
                     ++itRH) {
                  const CSCRecHit2D* recHit = &(*itRH);
                  int layer = recHit->cscDetId().layer();
                  layers |= 1 << (layer - 1);
                }
                m_chamberLayer[stationCSC].push_back(layers);

              }  // end preliminary tsos is valid

            }  // end found ring CSC

          }  // end refined tsos is valid

          if ((!tsos.isValid()) || (ringCSC == 0)) {
            // only fill Null denominator once for station 1, iiStation Z = 0,1,2
            if (iiStationZ <= 2)
              iiStationFail++;
            if (iiStationZ > 2 || iiStationFail == 3) {
              // fill Null Den Num
              m_chamberRing[stationCSC].push_back(-9);
              m_chamberChamber[stationCSC].push_back(-9);
              m_ttIntLocalX[stationCSC].push_back(-9999.0);
              m_ttIntLocalY[stationCSC].push_back(-9999.0);
              m_ttIntLocalErrorX[stationCSC].push_back(0.0);
              m_ttIntLocalErrorY[stationCSC].push_back(0.0);
              m_ttIntLocalW[stationCSC].push_back(-9999.0);
              m_ttIntLocalS[stationCSC].push_back(-9999.0);
              m_ttIntEta[stationCSC].push_back(-9999.0);

              m_ttDistToEdge[stationCSC].push_back(-9999.0);
              m_ttDistToHVGap[stationCSC].push_back(-9999.9);

              m_segLocalX[stationCSC].push_back(-9999.0);
              m_segLocalY[stationCSC].push_back(-9999.0);
              m_segLocalErrorX[stationCSC].push_back(0.0);
              m_segLocalErrorY[stationCSC].push_back(0.0);

              m_ttIntSegResidualLocalX[stationCSC].push_back(-9990.0);
              m_ttIntSegResidualLocalY[stationCSC].push_back(-9990.0);

              m_chamberLayer[stationCSC].push_back(-9);
            }
          }

        }  // end loop over CSC Z planes
      }    // endl loop over tracks
    }      // end loop over muons

  }  // End if good physics objects

  //  if (m_nZCands>0) {
  auto table = std::make_unique<nanoaod::FlatTable>(m_nZCands, m_name, false, false);

  table->setDoc("CSC Tag & Probe segment efficiency  information");

  addColumn(table, "muonPt", m_muonPt, "muon pt [GeV/c]");
  addColumn(table, "muonPhi", m_muonPhi, "muon phi [rad]");
  addColumn(table, "muonEta", m_muonEta, "muon eta");
  addColumn(table, "muonPtError", m_muonPtError, "muon pt error [GeV/c]");
  addColumn(table, "muonPhiError", m_muonPhiError, "muon phi error [rad]");
  addColumn(table, "muonEtaError", m_muonEtaError, "muon eta error");
  addColumn(table, "muonCharge", m_muonCharge, "muon charge");
  addColumn(table, "muonDXY", m_muonDXY, "muon dXY [cm]");
  addColumn(table, "muonDZ", m_muonDZ, "muon dZ [cm]");
  addColumn(table, "muonTrkHits", m_muonTrkHits, "muon track hits");
  addColumn(table, "muonChi2", m_muonChi2, "muon chi2");
  addColumn(table, "muonIso", m_trackIso, "muon relative iso");
  addColumn(table, "muonTrigger", m_muonTrigger, "muon has trigger bool");

  addColumn(table, "trackPt", m_trackPt, "track pt [GeV/c]");
  addColumn(table, "trackP", m_trackPt, "track p [GeV/c]");
  addColumn(table, "trackPhi", m_trackPhi, "track phi [rad]");
  addColumn(table, "trackEta", m_trackEta, "track eta");
  addColumn(table, "trackPtError", m_trackPtError, "track pt error [GeV/c]");
  addColumn(table, "trackPhiError", m_trackPhiError, "track phi error [rad]");
  addColumn(table, "trackEtaError", m_trackEtaError, "track eta error");
  addColumn(table, "trackCharge", m_trackCharge, "track charge");
  addColumn(table, "trackDXY", m_trackDXY, "track dXY [cm]");
  addColumn(table, "trackDZ", m_trackDZ, "track dZ [cm]");
  addColumn(table, "trackTrkHits", m_trackTrkHits, "track track hits");
  addColumn(table, "trackChi2", m_trackChi2, "track chi2");
  addColumn(table, "trackIso", m_trackIso, "track relative iso");

  addColumn(table, "zMass", m_zMass, "Z mass [GeV/c^2]");
  addColumn(table, "dRTrackMuon", m_dRTrackMuon, "dR between track and muon");
  addColumn(table, "numberOfPrimaryVertidies", m_numberOfPrimaryVertices, "Number of PVs");

  addColumn(table, "chamberEndcap", m_chamberEndcap, "");
  addColumn(table, "chamberRing1", m_chamberRing[0], "");
  addColumn(table, "chamberRing2", m_chamberRing[1], "");
  addColumn(table, "chamberRing3", m_chamberRing[2], "");
  addColumn(table, "chamberRing4", m_chamberRing[3], "");
  addColumn(table, "chamberChamber1", m_chamberChamber[0], "");
  addColumn(table, "chamberChamber2", m_chamberChamber[1], "");
  addColumn(table, "chamberChamber3", m_chamberChamber[2], "");
  addColumn(table, "chamberChamber4", m_chamberChamber[3], "");
  addColumn(table, "chamberLayer1", m_chamberLayer[0], "");
  addColumn(table, "chamberLayer2", m_chamberLayer[1], "");
  addColumn(table, "chamberLayer3", m_chamberLayer[2], "");
  addColumn(table, "chamberLayer4", m_chamberLayer[3], "");

  addColumn(table, "ttIntLocalX1", m_ttIntLocalX[0], "");
  addColumn(table, "ttIntLocalX2", m_ttIntLocalX[1], "");
  addColumn(table, "ttIntLocalX3", m_ttIntLocalX[2], "");
  addColumn(table, "ttIntLocalX4", m_ttIntLocalX[3], "");
  addColumn(table, "ttIntLocalY1", m_ttIntLocalY[0], "");
  addColumn(table, "ttIntLocalY2", m_ttIntLocalY[1], "");
  addColumn(table, "ttIntLocalY3", m_ttIntLocalY[2], "");
  addColumn(table, "ttIntLocalY4", m_ttIntLocalY[3], "");
  addColumn(table, "ttIntLocalErrorX1", m_ttIntLocalErrorX[0], "");
  addColumn(table, "ttIntLocalErrorX2", m_ttIntLocalErrorX[1], "");
  addColumn(table, "ttIntLocalErrorX3", m_ttIntLocalErrorX[2], "");
  addColumn(table, "ttIntLocalErrorX4", m_ttIntLocalErrorX[3], "");
  addColumn(table, "ttIntLocalErrorY1", m_ttIntLocalErrorY[0], "");
  addColumn(table, "ttIntLocalErrorY2", m_ttIntLocalErrorY[1], "");
  addColumn(table, "ttIntLocalErrorY3", m_ttIntLocalErrorY[2], "");
  addColumn(table, "ttIntLocalErrorY4", m_ttIntLocalErrorY[3], "");
  addColumn(table, "ttIntLocalW1", m_ttIntLocalW[0], "");
  addColumn(table, "ttIntLocalW2", m_ttIntLocalW[1], "");
  addColumn(table, "ttIntLocalW3", m_ttIntLocalW[2], "");
  addColumn(table, "ttIntLocalW4", m_ttIntLocalW[3], "");
  addColumn(table, "ttIntLocalS1", m_ttIntLocalS[0], "");
  addColumn(table, "ttIntLocalS2", m_ttIntLocalS[1], "");
  addColumn(table, "ttIntLocalS3", m_ttIntLocalS[2], "");
  addColumn(table, "ttIntLocalS4", m_ttIntLocalS[3], "");
  addColumn(table, "ttIntEta1", m_ttIntEta[0], "");
  addColumn(table, "ttIntEta2", m_ttIntEta[1], "");
  addColumn(table, "ttIntEta3", m_ttIntEta[2], "");
  addColumn(table, "ttIntEta4", m_ttIntEta[3], "");

  addColumn(table, "ttDistToEdge1", m_ttDistToEdge[0], "");
  addColumn(table, "ttDistToEdge2", m_ttDistToEdge[1], "");
  addColumn(table, "ttDistToEdge3", m_ttDistToEdge[2], "");
  addColumn(table, "ttDistToEdge4", m_ttDistToEdge[3], "");
  addColumn(table, "ttDistToHVGap1", m_ttDistToHVGap[0], "");
  addColumn(table, "ttDistToHVGap2", m_ttDistToHVGap[1], "");
  addColumn(table, "ttDistToHVGap3", m_ttDistToHVGap[2], "");
  addColumn(table, "ttDistToHVGap4", m_ttDistToHVGap[3], "");

  addColumn(table, "segLocalX1", m_segLocalX[0], "");
  addColumn(table, "segLocalX2", m_segLocalX[1], "");
  addColumn(table, "segLocalX3", m_segLocalX[2], "");
  addColumn(table, "segLocalX4", m_segLocalX[3], "");
  addColumn(table, "segLocalY1", m_segLocalY[0], "");
  addColumn(table, "segLocalY2", m_segLocalY[1], "");
  addColumn(table, "segLocalY3", m_segLocalY[2], "");
  addColumn(table, "segLocalY4", m_segLocalY[3], "");
  addColumn(table, "segLocalErrorX1", m_segLocalErrorX[0], "");
  addColumn(table, "segLocalErrorX2", m_segLocalErrorX[1], "");
  addColumn(table, "segLocalErrorX3", m_segLocalErrorX[2], "");
  addColumn(table, "segLocalErrorX4", m_segLocalErrorX[3], "");
  addColumn(table, "segLocalErrorY1", m_segLocalErrorY[0], "");
  addColumn(table, "segLocalErrorY2", m_segLocalErrorY[1], "");
  addColumn(table, "segLocalErrorY3", m_segLocalErrorY[2], "");
  addColumn(table, "segLocalErrorY4", m_segLocalErrorY[3], "");

  addColumn(table, "ttIntSegResidualLocalX1", m_ttIntSegResidualLocalX[0], "");
  addColumn(table, "ttIntSegResidualLocalX2", m_ttIntSegResidualLocalX[1], "");
  addColumn(table, "ttIntSegResidualLocalX3", m_ttIntSegResidualLocalX[2], "");
  addColumn(table, "ttIntSegResidualLocalX4", m_ttIntSegResidualLocalX[3], "");
  addColumn(table, "ttIntSegResidualLocalY1", m_ttIntSegResidualLocalY[0], "");
  addColumn(table, "ttIntSegResidualLocalY2", m_ttIntSegResidualLocalY[1], "");
  addColumn(table, "ttIntSegResidualLocalY3", m_ttIntSegResidualLocalY[2], "");
  addColumn(table, "ttIntSegResidualLocalY4", m_ttIntSegResidualLocalY[3], "");

  ev.put(std::move(table));
}

float MuCSCTnPFlatTableProducer::computeTrkIso(const reco::MuonIsolation& isolation, float muonPt) {
  return isolation.sumPt / muonPt;
}

float MuCSCTnPFlatTableProducer::computePFIso(const reco::MuonPFIsolation& pfIsolation, float muonPt) {
  return (pfIsolation.sumChargedHadronPt +
          std::max(0., pfIsolation.sumNeutralHadronEt + pfIsolation.sumPhotonEt - 0.5 * pfIsolation.sumPUPt)) /
         muonPt;
}

bool MuCSCTnPFlatTableProducer::hasTrigger(std::vector<int>& trigIndices,
                                           const trigger::TriggerObjectCollection& trigObjs,
                                           edm::Handle<trigger::TriggerEvent>& trigEvent,
                                           const reco::Muon& muon) {
  float dRMatch = 999.;
  for (int trigIdx : trigIndices) {
    const std::vector<std::string> trigModuleLabels = m_hltConfig.moduleLabels(trigIdx);

    const unsigned trigModuleIndex =
        std::find(trigModuleLabels.begin(), trigModuleLabels.end(), "hltBoolEnd") - trigModuleLabels.begin() - 1;
    const unsigned hltFilterIndex = trigEvent->filterIndex(edm::InputTag(trigModuleLabels[trigModuleIndex], "", "HLT"));
    if (hltFilterIndex < trigEvent->sizeFilters()) {
      const trigger::Keys keys = trigEvent->filterKeys(hltFilterIndex);
      const trigger::Vids vids = trigEvent->filterIds(hltFilterIndex);
      const unsigned nTriggers = vids.size();

      for (unsigned iTrig = 0; iTrig < nTriggers; ++iTrig) {
        trigger::TriggerObject trigObj = trigObjs[keys[iTrig]];
        float dR = deltaR(muon, trigObj);
        if (dR < dRMatch)
          dRMatch = dR;
      }
    }
  }

  return dRMatch < 0.1;  //CB should get it programmable
}

//bool MuCSCTnPFlatTableProducer::muonTagSelection(const reco::Muon & muon,edm::Handle<std::vector<reco::Track>> tracks)
bool MuCSCTnPFlatTableProducer::muonTagSelection(const reco::Muon& muon) {
  float ptCut = 10.0;
  int trackerHitsCut = 8;
  float dxyCut = 2.0;
  float dzCut = 24.0;
  float chi2Cut = 4.0;

  bool selected = false;
  //_muonIso = iso(*muon.track(),tracks);
  _muonIso = computePFIso(muon.pfIsolationR04(), muon.pt());

  if (!muon.isTrackerMuon())
    return false;
  if (!muon.track().isNonnull())
    return false;
  selected =
      ((muon.track()->pt() > ptCut) && (muon.track()->hitPattern().numberOfValidTrackerHits() >= trackerHitsCut) &&
       (muon.track()->dxy() < dxyCut) && (std::abs(muon.track()->dz()) < dzCut) &&
       (muon.track()->normalizedChi2() < chi2Cut) && _muonIso < 0.1);

  return selected;
}

bool MuCSCTnPFlatTableProducer::trackProbeSelection(const reco::Track& track,
                                                    edm::Handle<std::vector<reco::Track>> tracks) {
  float ptCut = 10.0;
  int trackerHitsCut = 8;
  float dxyCut = 2.0;
  float dzCut = 24.0;
  float chi2Cut = 4.0;

  bool selected = false;
  _trackIso = iso(track, tracks);

  selected =
      ((track.pt() > ptCut) && (std::abs(track.eta()) > 0.75) && (std::abs(track.eta()) < 2.55) &&
       (track.numberOfValidHits() >= trackerHitsCut) && (track.dxy() < dxyCut) && (std::abs(track.dz()) < dzCut) &&
       (track.normalizedChi2() > 0.0) && (track.normalizedChi2() < chi2Cut) && _trackIso < 0.1);

  return selected;
}

bool MuCSCTnPFlatTableProducer::zSelection(const reco::Muon& muon, const reco::Track& track) {
  bool selected = false;

  _zMass = zMass(track, muon);
  selected = (track.charge() * muon.charge() == -1 && (_zMass > 75.0) && (_zMass < 120.0));

  return selected;
}

// get track position at a particular (xy) plane given its z
TrajectoryStateOnSurface MuCSCTnPFlatTableProducer::surfExtrapTrkSam(const reco::Track& track, double z) {
  Plane::PositionType pos(0, 0, z);
  Plane::RotationType rot;
  Plane::PlanePointer myPlane = Plane::build(pos, rot);

  FreeTrajectoryState recoStart = freeTrajStateMuon(track);
  TrajectoryStateOnSurface recoProp = propagatorAlong->propagate(recoStart, *myPlane);

  if (!recoProp.isValid())
    recoProp = propagatorOpposite->propagate(recoStart, *myPlane);

  return recoProp;
}

FreeTrajectoryState MuCSCTnPFlatTableProducer::freeTrajStateMuon(const reco::Track& track) {
  //no track extras in nanoaod so directly use vx and p
  GlobalPoint innerPoint(track.vx(), track.vy(), track.vz());
  GlobalVector innerVec(track.px(), track.py(), track.pz());

  GlobalTrajectoryParameters gtPars(innerPoint, innerVec, track.charge(), &*theBField);

  AlgebraicSymMatrix66 cov;
  cov *= 1e-20;

  CartesianTrajectoryError tCov(cov);

  return (cov.kRows == 6 ? FreeTrajectoryState(gtPars, tCov) : FreeTrajectoryState(gtPars));
}

UChar_t MuCSCTnPFlatTableProducer::ringCandidate(Int_t iiStation, Int_t station, Float_t feta, Float_t phi) {
  UChar_t ring = 0;

  switch (station) {
    case 1:
      if (std::abs(feta) >= 0.85 && std::abs(feta) < 1.18) {  //ME13
        if (iiStation == 2)
          ring = 3;
        return ring;
      }
      if (std::abs(feta) >= 1.18 &&
          std::abs(feta) <= 1.5) {  //ME12 if(std::abs(feta)>1.18 && std::abs(feta)<1.7){//ME12
        if (iiStation == 1)
          ring = 2;
        return ring;
      }
      if (std::abs(feta) > 1.5 && std::abs(feta) < 2.1) {  //ME11
        if (iiStation == 0)
          ring = 1;
        return ring;
      }
      if (std::abs(feta) >= 2.1 && std::abs(feta) < 2.45) {  //ME11
        if (iiStation == 0)
          ring = 4;
        return ring;
      }
      break;
    case 2:
      if (std::abs(feta) > 0.95 && std::abs(feta) < 1.6) {  //ME22
        ring = 2;
        return ring;
      }
      if (std::abs(feta) > 1.55 && std::abs(feta) < 2.45) {  //ME21
        ring = 1;
        return ring;
      }
      break;
    case 3:
      if (std::abs(feta) > 1.08 && std::abs(feta) < 1.72) {  //ME32
        ring = 2;
        return ring;
      }
      if (std::abs(feta) > 1.69 && std::abs(feta) < 2.45) {  //ME31
        ring = 1;
        return ring;
      }
      break;
    case 4:
      if (std::abs(feta) > 1.78 && std::abs(feta) < 2.45) {  //ME41
        ring = 1;
        return ring;
      }
      if (std::abs(feta) > 1.15 && std::abs(feta) <= 1.78) {  //ME42
        ring = 2;
        return ring;
      }
      break;
    default:
      edm::LogError("") << "Invalid station: " << station << std::endl;
      break;
  }
  return 0;
}

UChar_t MuCSCTnPFlatTableProducer::thisChamberCandidate(UChar_t station, UChar_t ring, Float_t phi) {
  //    cout <<"\t\t TPTrackMuonSys::thisChamberCandidate..."<<endl;

  //search for chamber candidate based on CMS IN-2007/024
  //10 deg chambers are ME1,ME22,ME32,ME42 chambers; 20 deg chambers are ME21,31,41 chambers
  //Chambers one always starts from approx -5 deg.
  const UChar_t nVal = (station > 1 && ring == 1) ? 18 : 36;
  const Float_t ChamberSpan = 2 * M_PI / nVal;
  Float_t dphi = phi + M_PI / 36;
  while (dphi >= 2 * M_PI)
    dphi -= 2 * M_PI;
  while (dphi < 0)
    dphi += 2 * M_PI;
  UChar_t ChCand = floor(dphi / ChamberSpan) + 1;
  return ChCand > nVal ? nVal : ChCand;
}

Float_t MuCSCTnPFlatTableProducer::TrajectoryDistToSeg(TrajectoryStateOnSurface TrajSuf,
                                                       CSCSegmentCollection::const_iterator segIt) {
  if (!TrajSuf.isValid())
    return 9999.;
  const GeomDet* gdet = m_cscGeometry->idToDet((CSCDetId)(*segIt).cscDetId());
  LocalPoint localTTPos = gdet->surface().toLocal(TrajSuf.freeState()->position());
  LocalPoint localSegPos = (*segIt).localPosition();
  Float_t CSCdeltaX = localSegPos.x() - localTTPos.x();
  Float_t CSCdeltaY = localSegPos.y() - localTTPos.y();
  return sqrt(pow(CSCdeltaX, 2) + pow(CSCdeltaY, 2));
}

TrajectoryStateOnSurface* MuCSCTnPFlatTableProducer::matchTTwithCSCSeg(const reco::Track& track,
                                                                       edm::Handle<CSCSegmentCollection> cscSegments,
                                                                       CSCSegmentCollection::const_iterator& cscSegOut,
                                                                       CSCDetId& idCSC) {
  TrajectoryStateOnSurface* TrajSuf = nullptr;
  Float_t deltaCSCR = 9999.;
  for (CSCSegmentCollection::const_iterator segIt = cscSegments->begin(); segIt != cscSegments->end(); segIt++) {
    CSCDetId id = (CSCDetId)(*segIt).cscDetId();

    if (idCSC.endcap() != id.endcap())
      continue;
    if (idCSC.station() != id.station())
      continue;
    if (idCSC.chamber() != id.chamber())
      continue;

    Bool_t ed1 =
        (idCSC.station() == 1) && ((idCSC.ring() == 1 || idCSC.ring() == 4) && (id.ring() == 1 || id.ring() == 4));
    Bool_t ed2 =
        (idCSC.station() == 1) && ((idCSC.ring() == 2 && id.ring() == 2) || (idCSC.ring() == 3 && id.ring() == 3));
    Bool_t ed3 = (idCSC.station() != 1) && (idCSC.ring() == id.ring());
    Bool_t TMCSCMatch = (ed1 || ed2 || ed3);

    if (!TMCSCMatch)
      continue;

    const CSCChamber* cscchamber = m_cscGeometry->chamber(id);

    if (!cscchamber)
      continue;

    TrajectoryStateOnSurface TrajSuf_ = surfExtrapTrkSam(track, cscchamber->toGlobal((*segIt).localPosition()).z());
    Float_t dR_ = std::abs(TrajectoryDistToSeg(TrajSuf_, segIt));
    if (dR_ < deltaCSCR) {
      delete TrajSuf;
      TrajSuf = new TrajectoryStateOnSurface(TrajSuf_);
      deltaCSCR = dR_;
      cscSegOut = segIt;
    }
  }  //loop over segments

  return TrajSuf;
}

std::vector<Float_t> MuCSCTnPFlatTableProducer::GetEdgeAndDistToGap(const reco::Track& track, CSCDetId& detid) {
  std::vector<Float_t> result(4, 9999.);
  result[3] = -9999;
  const GeomDet* gdet = m_cscGeometry->idToDet(detid);
  TrajectoryStateOnSurface tsos = surfExtrapTrkSam(track, gdet->surface().position().z());
  if (!tsos.isValid())
    return result;
  LocalPoint localTTPos = gdet->surface().toLocal(tsos.freeState()->position());
  const CSCWireTopology* wireTopology = m_cscGeometry->layer(detid)->geometry()->wireTopology();
  Float_t wideWidth = wireTopology->wideWidthOfPlane();
  Float_t narrowWidth = wireTopology->narrowWidthOfPlane();
  Float_t length = wireTopology->lengthOfPlane();
  // If slanted, there is no y offset between local origin and symmetry center of wire plane
  Float_t yOfFirstWire = std::abs(wireTopology->wireAngle()) > 1.E-06 ? -0.5 * length : wireTopology->yOfWire(1);
  // y offset between local origin and symmetry center of wire plane
  Float_t yCOWPOffset = yOfFirstWire + 0.5 * length;
  // tangent of the incline angle from inside the trapezoid
  Float_t tangent = (wideWidth - narrowWidth) / (2. * length);
  // y position wrt bottom of trapezoid
  Float_t yPrime = localTTPos.y() + std::abs(yOfFirstWire);
  // half trapezoid width at y' is 0.5 * narrowWidth + x side of triangle with the above tangent and side y'
  Float_t halfWidthAtYPrime = 0.5 * narrowWidth + yPrime * tangent;
  // x offset between local origin and symmetry center of wire plane is zero
  // x offset of ME11s is also zero. x center of wire groups is not at zero, because it is not parallel to x. The wire groups of ME11s have a complex geometry, see the code in m_debug.
  Float_t edgex = std::abs(localTTPos.x()) - halfWidthAtYPrime;
  Float_t edgey = std::abs(localTTPos.y() - yCOWPOffset) - 0.5 * length;
  LocalError localTTErr = tsos.localError().positionError();
  if (edgex > edgey) {
    result[0] = edgex;
    result[1] = sqrt(localTTErr.xx());
    //result[1] = sqrt(tsos.cartesianError().position().cxx());
  } else {
    result[0] = edgey;
    result[1] = sqrt(localTTErr.yy());
    //result[1] = sqrt(tsos.cartesianError().position().cyy());
  }
  result[2] = YDistToHVDeadZone(localTTPos.y(), detid.station() * 10 + detid.ring());
  result[3] = sqrt(localTTErr.yy());
  return result;  //return values: 1-edge;2-err of edge;3-disttogap;4-err of dist to gap
}

//deadzone center is according to http://cmssdt.cern.ch/SDT/lxr/source/RecoLocalMuon/CSCEfficiency/src/CSCEfficiency.cc#605
//wire spacing is according to CSCTDR
Float_t MuCSCTnPFlatTableProducer::YDistToHVDeadZone(Float_t yLocal, Int_t StationAndRing) {
  //the ME11 wires are not parallel to x, but no gap
  //chamber edges are not included.
  const Float_t deadZoneCenterME1_2[2] = {-32.88305, 32.867423};
  const Float_t deadZoneCenterME1_3[2] = {-22.7401, 27.86665};
  const Float_t deadZoneCenterME2_1[2] = {-27.47, 33.67};
  const Float_t deadZoneCenterME3_1[2] = {-36.21, 23.68};
  const Float_t deadZoneCenterME4_1[2] = {-26.14, 23.85};
  const Float_t deadZoneCenterME234_2[4] = {-81.8744, -21.18165, 39.51105, 100.2939};
  const Float_t* deadZoneCenter;
  Float_t deadZoneHeightHalf = 0.32 * 7 / 2;  // wire spacing * (wires missing + 1)/2
  Float_t minY = 999999.;
  UChar_t nGaps = 2;
  switch (std::abs(StationAndRing)) {
    case 11:
    case 14:
      return 162;  //the height of ME11
      break;
    case 12:
      deadZoneCenter = deadZoneCenterME1_2;
      break;
    case 13:
      deadZoneCenter = deadZoneCenterME1_3;
      break;
    case 21:
      deadZoneCenter = deadZoneCenterME2_1;
      break;
    case 31:
      deadZoneCenter = deadZoneCenterME3_1;
      break;
    case 41:
      deadZoneCenter = deadZoneCenterME4_1;
      break;
    default:
      deadZoneCenter = deadZoneCenterME234_2;
      nGaps = 4;
  }
  for (UChar_t iGap = 0; iGap < nGaps; iGap++) {
    Float_t newMinY = yLocal < deadZoneCenter[iGap] ? deadZoneCenter[iGap] - deadZoneHeightHalf - yLocal
                                                    : yLocal - (deadZoneCenter[iGap] + deadZoneHeightHalf);
    if (newMinY < minY)
      minY = newMinY;
  }
  return minY;
}

double MuCSCTnPFlatTableProducer::iso(const reco::Track& track, edm::Handle<std::vector<reco::Track>> tracks) {
  double isoSum = 0.0;
  for (const auto& track2 : (*tracks)) {
    double dR = calcDeltaR(track.eta(), track2.eta(), track.phi(), track2.phi());
    if (track2.pt() > 1.0 && dR > 0.001 && dR < 0.3)
      isoSum += track2.pt();
  }
  return isoSum / track.pt();
}

double MuCSCTnPFlatTableProducer::calcDeltaR(double eta1, double eta2, double phi1, double phi2) {
  double deta = eta1 - eta2;
  if (phi1 < 0)
    phi1 += 2.0 * M_PI;
  if (phi2 < 0)
    phi2 += 2.0 * M_PI;
  double dphi = phi1 - phi2;
  if (dphi > M_PI)
    dphi -= 2. * M_PI;
  else if (dphi < -M_PI)
    dphi += 2. * M_PI;
  return std::sqrt(deta * deta + dphi * dphi);
}

double MuCSCTnPFlatTableProducer::zMass(const reco::Track& track, const reco::Muon& muon) {
  double zMass = -99.0;
  double mMu = 0.1134289256;

  zMass = std::pow((std::sqrt(std::pow(muon.p(), 2) + mMu * mMu) + std::sqrt(std::pow(track.p(), 2) + mMu * mMu)), 2) -
          (std::pow((muon.px() + track.px()), 2) + std::pow((muon.py() + track.py()), 2) +
           std::pow((muon.pz() + track.pz()), 2));

  return std::sqrt(zMass);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuCSCTnPFlatTableProducer);
