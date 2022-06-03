#ifndef MuNtuple_MuCSCTnPFlatTableProducer_h
#define MuNtuple_MuCSCTnPFlatTableProducer_h

/** \class MuCSCTnPFlatTableProducer MuCSCTnPFlatTableProducer.h MuDPGAnalysis/MuonDPGNtuples/plugins/MuCSCTnPFlatTableProducer.h
 *  
 * Helper class : CSC Tag and Probe Segement Efficiency Filler
 *
 * \author M. Herndon (UW Madison)
 *
 *
 */

#include "DPGAnalysis/MuonTools/src/MuBaseFlatTableProducer.h"

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
  MuCSCTnPFlatTableProducer(const edm::ParameterSet &);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  /// Fill tree branches for a given events
  void fillTable(edm::Event &) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run &, const edm::EventSetup &) final;

  /// Get info from the ES for a given event
  void getFromES(const edm::EventSetup &) final;

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
  bool trackProbeSelection(const reco::Track &track, edm::Handle<std::vector<reco::Track>>);
  bool muonTagSelection(const reco::Muon &);
  //bool trackProbeSelection(const reco::Track & track);
  bool zSelection(const reco::Muon &, const reco::Track &);

  // Calculation functions
  double zMass(const reco::Track &, const reco::Muon &);
  double calcDeltaR(double, double, double, double);
  double iso(const reco::Track &, edm::Handle<std::vector<reco::Track>>);

  // Track extrapolation and segment match functions
  TrajectoryStateOnSurface surfExtrapTrkSam(const reco::Track &, double);
  FreeTrajectoryState freeTrajStateMuon(const reco::Track &);

  UChar_t ringCandidate(Int_t iiStation, Int_t station, Float_t feta, Float_t phi);
  UChar_t thisChamberCandidate(UChar_t station, UChar_t ring, Float_t phi);

  TrajectoryStateOnSurface *matchTTwithCSCSeg(const reco::Track &,
                                              edm::Handle<CSCSegmentCollection>,
                                              CSCSegmentCollection::const_iterator &,
                                              CSCDetId &);
  Float_t TrajectoryDistToSeg(TrajectoryStateOnSurface, CSCSegmentCollection::const_iterator);
  std::vector<Float_t> GetEdgeAndDistToGap(const reco::Track &, CSCDetId &);
  Float_t YDistToHVDeadZone(Float_t, Int_t);

  /// The variables holding
  /// the T&P related information

  unsigned int m_nZCands;  // the # of digis (size of all following vectors)

  double _trackIso;
  double _muonIso;
  double _zMass;

  bool hasTrigger(std::vector<int> &,
                  const trigger::TriggerObjectCollection &,
                  edm::Handle<trigger::TriggerEvent> &,
                  const reco::Muon &);

  float computeTrkIso(const reco::MuonIsolation &, float);
  float computePFIso(const reco::MuonPFIsolation &, float);
};

#endif
