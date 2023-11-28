// system includes
#include <map>
#include <set>
#include <string>
#include <vector>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackCandidate/interface/TrajectoryStopReasons.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

// ROOT includes
#include "TFile.h"
#include "TH1.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TPRegexp.h"

using MVACollection = std::vector<float>;
using QualityMaskCollection = std::vector<unsigned char>;

class StandaloneTrackMonitor : public DQMEDAnalyzer {
public:
  StandaloneTrackMonitor(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void addClusterToMap(uint32_t detid, const SiStripCluster* cluster);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  void processClusters(edm::Event const& iEvent,
                       edm::EventSetup const& iSetup,
                       const TrackerGeometry& tkGeom,
                       double wfac = 1);
  void processHit(const TrackingRecHit& recHit,
                  edm::EventSetup const& iSetup,
                  const TrackerGeometry& tkGeom,
                  double wfac = 1);

private:
  const reco::Vertex* findClosestVertex(const TransientVertex aTransVtx, const reco::VertexCollection* vertices)
      const;  //same as in /DQMOffline/Alignment/src/DiMuonVertexSelector.cc
  const std::string moduleName_;
  const std::string folderName_;

  const bool isRECO_;

  SiStripClusterInfo siStripClusterInfo_;

  const edm::InputTag trackTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag vertexTag_;
  const edm::InputTag puSummaryTag_;
  const edm::InputTag clusterTag_;
  const edm::InputTag jetsTag_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > puSummaryToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken_;
  const edm::EDGetTokenT<std::vector<reco::PFJet> > jetsToken_;
  // track MVA
  const std::string trackQuality_;
  const bool doPUCorrection_;
  const bool doTrackCorrection_;
  const bool isMC_;
  const bool haveAllHistograms_;
  const std::string puScaleFactorFile_;
  const std::string trackScaleFactorFile_;
  const std::vector<std::string> mvaProducers_;
  const edm::InputTag mvaTrackTag_;
  const edm::EDGetTokenT<edm::View<reco::Track> > mvaTrackToken_;
  const edm::InputTag tcProducer_;
  const std::string algoName_;

  int nevt = 0;
  int chi2it = 0, chi2itGt = 0, chi2itLt = 0;
  const bool verbose_;
  std::vector<std::tuple<edm::EDGetTokenT<MVACollection>, edm::EDGetTokenT<QualityMaskCollection> > > mvaQualityTokens_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transTrackToken_;
  const edm::ParameterSet TrackEtaHistoPar_;
  const edm::ParameterSet TrackPhiHistoPar_;
  const edm::ParameterSet TrackPtHistoPar_;
  const TrackerGeometry* tkGeom_ = nullptr;

  MonitorElement* trackEtaH_;
  MonitorElement* trackEtaerrH_;
  MonitorElement* trackCosThetaH_;
  MonitorElement* trackThetaerrH_;
  MonitorElement* trackPhiH_;
  MonitorElement* trackPhierrH_;
  MonitorElement* trackPH_;
  MonitorElement* trackPtH_;
  MonitorElement* trackPt_ZoomH_;
  MonitorElement* trackPtUpto2GeVH_;
  MonitorElement* trackPtOver10GeVH_;
  MonitorElement* trackPterrH_;
  MonitorElement* trackqOverpH_;
  MonitorElement* trackqOverperrH_;
  MonitorElement* trackChargeH_;
  MonitorElement* trackChi2H_;
  MonitorElement* tracknDOFH_;
  MonitorElement* trackChi2ProbH_;
  MonitorElement* trackChi2oNDFH_;
  MonitorElement* trackd0H_;
  MonitorElement* trackChi2bynDOFH_;
  MonitorElement* trackalgoH_;
  MonitorElement* trackorigalgoH_;
  MonitorElement* trackStoppingSourceH_;

  MonitorElement* DistanceOfClosestApproachToPVH_;
  MonitorElement* DistanceOfClosestApproachToPVZoomedH_;
  MonitorElement* DistanceOfClosestApproachToPVVsPhiH_;
  MonitorElement* xPointOfClosestApproachVsZ0wrtPVH_;
  MonitorElement* yPointOfClosestApproachVsZ0wrtPVH_;
  MonitorElement* trackDeltaRwrtClosestTrack_;

  MonitorElement* ip2dToPVH_;
  MonitorElement* iperr2dToPVH_;

  MonitorElement* ip3dToBSH_;
  MonitorElement* iperr3dToBSH_;
  MonitorElement* iperr3dToBSWtH_;
  MonitorElement* iperr2dToBSH_;
  MonitorElement* iperr2dToPVWtH_;

  MonitorElement* ip2dToBSH_;
  MonitorElement* sip2dToBSH_;

  MonitorElement* ip3dToPVH_;
  MonitorElement* iperr3dToPVH_;
  MonitorElement* iperr3dToPVWtH_;
  MonitorElement* sip3dToPVH_;
  MonitorElement* sip3dToBSH_;
  MonitorElement* sip2dToPVH_;
  MonitorElement* sip2dToPVWtH_;
  MonitorElement* sipDxyToPVH_;
  MonitorElement* sipDzToPVH_;

  MonitorElement* ip3dToPV2validpixelhitsH_;
  MonitorElement* ip3dToBS2validpixelhitsH_;
  MonitorElement* iperr3dToPV2validpixelhitsH_;
  MonitorElement* iperr3dToBS2validpixelhitsH_;
  MonitorElement* sip3dToPV2validpixelhitsH_;
  MonitorElement* sip3dToBS2validpixelhitsH_;

  MonitorElement* nallHitsH_;
  MonitorElement* ntrackerHitsH_;

  MonitorElement* nvalidTrackerHitsH_;
  MonitorElement* nvalidPixelHitsH_;
  MonitorElement* nvalidPixelBHitsH_;
  MonitorElement* nvalidPixelEHitsH_;
  MonitorElement* nvalidStripHitsH_;
  MonitorElement* nvalidTIBHitsH_;
  MonitorElement* nvalidTOBHitsH_;
  MonitorElement* nvalidTIDHitsH_;
  MonitorElement* nvalidTECHitsH_;

  MonitorElement* nlostTrackerHitsH_;
  MonitorElement* nlostPixelHitsH_;
  MonitorElement* nlostPixelBHitsH_;
  MonitorElement* nlostPixelEHitsH_;
  MonitorElement* nlostStripHitsH_;
  MonitorElement* nlostTIBHitsH_;
  MonitorElement* nlostTOBHitsH_;
  MonitorElement* nlostTIDHitsH_;
  MonitorElement* nlostTECHitsH_;

  MonitorElement* nMissingInnerHitBH_;
  MonitorElement* nMissingInnerHitEH_;
  MonitorElement* nMissingOuterHitBH_;
  MonitorElement* nMissingOuterHitEH_;

  MonitorElement* trkLayerwithMeasurementH_;
  MonitorElement* pixelLayerwithMeasurementH_;
  MonitorElement* pixelBLayerwithMeasurementH_;
  MonitorElement* pixelELayerwithMeasurementH_;
  MonitorElement* stripLayerwithMeasurementH_;
  MonitorElement* stripTIBLayerwithMeasurementH_;
  MonitorElement* stripTOBLayerwithMeasurementH_;
  MonitorElement* stripTIDLayerwithMeasurementH_;
  MonitorElement* stripTECLayerwithMeasurementH_;

  MonitorElement* nlostHitsH_;
  MonitorElement* nMissingExpectedInnerHitsH_;
  MonitorElement* nMissingExpectedOuterHitsH_;

  MonitorElement* beamSpotXYposH_;
  MonitorElement* beamSpotXYposerrH_;
  MonitorElement* beamSpotZposH_;
  MonitorElement* beamSpotZposerrH_;

  MonitorElement* vertexXposH_;
  MonitorElement* vertexYposH_;
  MonitorElement* vertexZposH_;
  MonitorElement* nVertexH_;
  MonitorElement* nVtxH_;

  MonitorElement* nPixBarrelH_;
  MonitorElement* nPixEndcapH_;
  MonitorElement* nStripTIBH_;
  MonitorElement* nStripTOBH_;
  MonitorElement* nStripTECH_;
  MonitorElement* nStripTIDH_;
  MonitorElement* nTracksH_;

  // MC only
  MonitorElement* bunchCrossingH_;
  MonitorElement* nPUH_;
  MonitorElement* trueNIntH_;

  // Exclusive Quantities
  MonitorElement* nLostHitByLayerH_;
  MonitorElement* nLostHitByLayerPixH_;
  MonitorElement* nLostHitByLayerStripH_;
  MonitorElement* nLostHitsVspTH_;
  MonitorElement* nLostHitsVsEtaH_;
  MonitorElement* nLostHitsVsCosThetaH_;
  MonitorElement* nLostHitsVsPhiH_;
  MonitorElement* nLostHitsVsIterationH_;

  MonitorElement* nHitsTIBSVsEtaH_;
  MonitorElement* nHitsTOBSVsEtaH_;
  MonitorElement* nHitsTECSVsEtaH_;
  MonitorElement* nHitsTIDSVsEtaH_;
  MonitorElement* nHitsStripSVsEtaH_;

  MonitorElement* nHitsTIBDVsEtaH_;
  MonitorElement* nHitsTOBDVsEtaH_;
  MonitorElement* nHitsTECDVsEtaH_;
  MonitorElement* nHitsTIDDVsEtaH_;
  MonitorElement* nHitsStripDVsEtaH_;

  MonitorElement* nValidHitsVspTH_;
  MonitorElement* nValidHitsVsnVtxH_;
  MonitorElement* nValidHitsVsEtaH_;
  MonitorElement* nValidHitsVsCosThetaH_;
  MonitorElement* nValidHitsVsPhiH_;

  MonitorElement* nValidHitsPixVsEtaH_;
  MonitorElement* nValidHitsPixBVsEtaH_;
  MonitorElement* nValidHitsPixEVsEtaH_;
  MonitorElement* nValidHitsStripVsEtaH_;
  MonitorElement* nValidHitsTIBVsEtaH_;
  MonitorElement* nValidHitsTOBVsEtaH_;
  MonitorElement* nValidHitsTECVsEtaH_;
  MonitorElement* nValidHitsTIDVsEtaH_;

  MonitorElement* nValidHitsPixVsPhiH_;
  MonitorElement* nValidHitsPixBVsPhiH_;
  MonitorElement* nValidHitsPixEVsPhiH_;
  MonitorElement* nValidHitsStripVsPhiH_;
  MonitorElement* nValidHitsTIBVsPhiH_;
  MonitorElement* nValidHitsTOBVsPhiH_;
  MonitorElement* nValidHitsTECVsPhiH_;
  MonitorElement* nValidHitsTIDVsPhiH_;

  MonitorElement* nLostHitsPixVsEtaH_;
  MonitorElement* nLostHitsPixBVsEtaH_;
  MonitorElement* nLostHitsPixEVsEtaH_;
  MonitorElement* nLostHitsStripVsEtaH_;
  MonitorElement* nLostHitsTIBVsEtaH_;
  MonitorElement* nLostHitsTOBVsEtaH_;
  MonitorElement* nLostHitsTECVsEtaH_;
  MonitorElement* nLostHitsTIDVsEtaH_;

  MonitorElement* nLostHitsPixVsIterationH_;
  MonitorElement* nLostHitsPixBVsIterationH_;
  MonitorElement* nLostHitsPixEVsIterationH_;
  MonitorElement* nLostHitsStripVsIterationH_;
  MonitorElement* nLostHitsTIBVsIterationH_;
  MonitorElement* nLostHitsTOBVsIterationH_;
  MonitorElement* nLostHitsTECVsIterationH_;
  MonitorElement* nLostHitsTIDVsIterationH_;

  MonitorElement* nLostHitsPixVsPhiH_;
  MonitorElement* nLostHitsPixBVsPhiH_;
  MonitorElement* nLostHitsPixEVsPhiH_;
  MonitorElement* nLostHitsStripVsPhiH_;
  MonitorElement* nLostHitsTIBVsPhiH_;
  MonitorElement* nLostHitsTOBVsPhiH_;
  MonitorElement* nLostHitsTECVsPhiH_;
  MonitorElement* nLostHitsTIDVsPhiH_;

  MonitorElement* trackChi2oNDFVsEtaH_;
  MonitorElement* trackChi2oNDFVsPhiH_;
  MonitorElement* trackChi2probVsEtaH_;
  MonitorElement* trackChi2probVsPhiH_;

  MonitorElement* trackIperr3dVsEtaH_;
  MonitorElement* trackIperr3dVsChi2probH_;

  MonitorElement* trackSip2dVsEtaH_;

  MonitorElement* trackIperr3dVsEta2DH_;
  MonitorElement* trackIperr3dVsChi2prob2DH_;

  MonitorElement* trackSip2dVsEta2DH_;
  MonitorElement* trackSip2dVsChi2prob2DH_;

  MonitorElement* hOnTrkClusChargeThinH_;
  MonitorElement* hOnTrkClusWidthThinH_;
  MonitorElement* hOnTrkClusChargeThickH_;
  MonitorElement* hOnTrkClusWidthThickH_;

  MonitorElement* hOffTrkClusChargeThinH_;
  MonitorElement* hOffTrkClusWidthThinH_;
  MonitorElement* hOffTrkClusChargeThickH_;
  MonitorElement* hOffTrkClusWidthThickH_;

  std::vector<MonitorElement*> trackMVAs;
  std::vector<MonitorElement*> trackMVAsHP;
  std::vector<MonitorElement*> trackMVAsVsPtProfile;
  std::vector<MonitorElement*> trackMVAsHPVsPtProfile;
  std::vector<MonitorElement*> trackMVAsVsEtaProfile;
  std::vector<MonitorElement*> trackMVAsHPVsEtaProfile;

  MonitorElement* nJet_;
  MonitorElement* Jet_pt_;
  MonitorElement* Jet_eta_;
  MonitorElement* Jet_energy_;
  MonitorElement* Jet_chargedMultiplicity_;

  MonitorElement* Zpt_;
  MonitorElement* ZInvMass_;

  MonitorElement* cosPhi3DdileptonH_;

  std::vector<int> lumivec1;
  std::vector<int> lumivec2;
  std::vector<float> vpu_;
  std::vector<float> vtrack_;
  std::map<uint32_t, std::set<const SiStripCluster*> > clusterMap_;
};

void StandaloneTrackMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("moduleName", "StandaloneTrackMonitor");
  desc.addUntracked<std::string>("folderName", "highPurityTracks");
  desc.addUntracked<bool>("isRECO", false);
  desc.addUntracked<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"));
  desc.addUntracked<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"));
  desc.addUntracked<edm::InputTag>("puTag", edm::InputTag("addPileupInfo"));
  desc.addUntracked<edm::InputTag>("clusterTag", edm::InputTag("siStripClusters"));
  desc.addUntracked<edm::InputTag>("PFJetsCollection", edm::InputTag("ak4PFJetsCHS"));

  desc.addUntracked<std::string>("trackQuality", "highPurity");
  desc.addUntracked<bool>("doPUCorrection", false);
  desc.addUntracked<bool>("doTrackCorrection", false);
  desc.addUntracked<bool>("isMC", false);
  desc.addUntracked<bool>("haveAllHistograms", false);
  desc.addUntracked<std::string>("puScaleFactorFile", "PileupScaleFactor.root");
  desc.addUntracked<std::string>("trackScaleFactorFile", "PileupScaleFactor.root");
  desc.addUntracked<std::vector<std::string> >("MVAProducers");
  desc.addUntracked<edm::InputTag>("TrackProducerForMVA");

  desc.addUntracked<edm::InputTag>("TCProducer");
  desc.addUntracked<std::string>("AlgoName");
  desc.addUntracked<bool>("verbose", false);

  {
    edm::ParameterSetDescription TrackEtaHistoPar;
    TrackEtaHistoPar.add<int>("Xbins", 60);
    TrackEtaHistoPar.add<double>("Xmin", -3.0);
    TrackEtaHistoPar.add<double>("Xmax", 3.0);
    desc.add<edm::ParameterSetDescription>("trackEtaH", TrackEtaHistoPar);
  }

  {
    edm::ParameterSetDescription TrackPtHistoPar;
    TrackPtHistoPar.add<int>("Xbins", 60);
    TrackPtHistoPar.add<double>("Xmin", 0.0);
    TrackPtHistoPar.add<double>("Xmax", 100.0);
    desc.add<edm::ParameterSetDescription>("trackPtH", TrackPtHistoPar);
  }

  {
    edm::ParameterSetDescription TrackPhiHistoPar;
    TrackPhiHistoPar.add<int>("Xbins", 100);
    TrackPhiHistoPar.add<double>("Xmin", -M_PI);
    TrackPhiHistoPar.add<double>("Xmax", M_PI);
    desc.add<edm::ParameterSetDescription>("trackPhiH", TrackPhiHistoPar);
  }

  descriptions.add("standaloneTrackMonitorDefault", desc);
}

// -----------------------------
// constructors and destructor
// -----------------------------
StandaloneTrackMonitor::StandaloneTrackMonitor(const edm::ParameterSet& ps)
    : moduleName_(ps.getUntrackedParameter<std::string>("moduleName", "StandaloneTrackMonitor")),
      folderName_(ps.getUntrackedParameter<std::string>("folderName", "highPurityTracks")),
      isRECO_(ps.getUntrackedParameter<bool>("isRECO", false)),
      siStripClusterInfo_(consumesCollector()),
      trackTag_(ps.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      bsTag_(ps.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      vertexTag_(ps.getUntrackedParameter<edm::InputTag>("vertexTag", edm::InputTag("offlinePrimaryVertices"))),
      puSummaryTag_(ps.getUntrackedParameter<edm::InputTag>("puTag", edm::InputTag("addPileupInfo"))),
      clusterTag_(ps.getUntrackedParameter<edm::InputTag>("clusterTag", edm::InputTag("siStripClusters"))),
      jetsTag_(ps.getUntrackedParameter<edm::InputTag>("PFJetsCollection", edm::InputTag("ak4PFJetsCHS"))),
      trackToken_(consumes<reco::TrackCollection>(trackTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      vertexToken_(consumes<reco::VertexCollection>(vertexTag_)),
      puSummaryToken_(consumes<std::vector<PileupSummaryInfo> >(puSummaryTag_)),
      clusterToken_(consumes<edmNew::DetSetVector<SiStripCluster> >(clusterTag_)),
      jetsToken_(consumes<std::vector<reco::PFJet> >(jetsTag_)),
      trackQuality_(ps.getUntrackedParameter<std::string>("trackQuality", "highPurity")),
      doPUCorrection_(ps.getUntrackedParameter<bool>("doPUCorrection", false)),
      doTrackCorrection_(ps.getUntrackedParameter<bool>("doTrackCorrection", false)),
      isMC_(ps.getUntrackedParameter<bool>("isMC", false)),
      haveAllHistograms_(ps.getUntrackedParameter<bool>("haveAllHistograms", false)),
      puScaleFactorFile_(ps.getUntrackedParameter<std::string>("puScaleFactorFile", "PileupScaleFactor.root")),
      trackScaleFactorFile_(ps.getUntrackedParameter<std::string>("trackScaleFactorFile", "PileupScaleFactor.root")),
      mvaProducers_(ps.getUntrackedParameter<std::vector<std::string> >("MVAProducers")),
      mvaTrackTag_(ps.getUntrackedParameter<edm::InputTag>("TrackProducerForMVA")),
      mvaTrackToken_(consumes<edm::View<reco::Track> >(mvaTrackTag_)),
      tcProducer_(ps.getUntrackedParameter<edm::InputTag>("TCProducer")),
      algoName_(ps.getUntrackedParameter<std::string>("AlgoName")),
      verbose_(ps.getUntrackedParameter<bool>("verbose", false)),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      transTrackToken_(esConsumes<TransientTrackBuilder, TransientTrackRecord, edm::Transition::Event>(
          edm::ESInputTag{"", "TransientTrackBuilder"})),
      TrackEtaHistoPar_(ps.getParameter<edm::ParameterSet>("trackEtaH")),
      TrackPhiHistoPar_(ps.getParameter<edm::ParameterSet>("trackPhiH")),
      TrackPtHistoPar_(ps.getParameter<edm::ParameterSet>("trackPtH")) {
  for (const auto& v : mvaProducers_) {
    mvaQualityTokens_.push_back(std::make_tuple(consumes<MVACollection>(edm::InputTag(v, "MVAValues")),
                                                consumes<QualityMaskCollection>(edm::InputTag(v, "QualityMasks"))));
  }

  trackEtaH_ = nullptr;
  trackEtaerrH_ = nullptr;
  trackCosThetaH_ = nullptr;
  trackThetaerrH_ = nullptr;
  trackPhiH_ = nullptr;
  trackPhierrH_ = nullptr;
  trackPH_ = nullptr;
  trackPtH_ = nullptr;
  trackPtUpto2GeVH_ = nullptr;
  trackPtOver10GeVH_ = nullptr;
  trackPterrH_ = nullptr;
  trackqOverpH_ = nullptr;
  trackqOverperrH_ = nullptr;
  trackChargeH_ = nullptr;
  nlostHitsH_ = nullptr;
  nvalidTrackerHitsH_ = nullptr;
  nvalidPixelHitsH_ = nullptr;
  nvalidStripHitsH_ = nullptr;
  trkLayerwithMeasurementH_ = nullptr;
  pixelLayerwithMeasurementH_ = nullptr;
  stripLayerwithMeasurementH_ = nullptr;
  beamSpotXYposH_ = nullptr;
  beamSpotXYposerrH_ = nullptr;
  beamSpotZposH_ = nullptr;
  beamSpotZposerrH_ = nullptr;
  trackChi2H_ = nullptr;
  tracknDOFH_ = nullptr;
  trackd0H_ = nullptr;
  trackChi2bynDOFH_ = nullptr;
  vertexXposH_ = nullptr;
  vertexYposH_ = nullptr;
  vertexZposH_ = nullptr;

  nPixBarrelH_ = nullptr;
  nPixEndcapH_ = nullptr;
  nStripTIBH_ = nullptr;
  nStripTOBH_ = nullptr;
  nStripTECH_ = nullptr;
  nStripTIDH_ = nullptr;

  // for MC only
  nVtxH_ = nullptr;
  nVertexH_ = nullptr;
  bunchCrossingH_ = nullptr;
  nPUH_ = nullptr;
  trueNIntH_ = nullptr;

  nLostHitsVspTH_ = nullptr;
  nLostHitsVsEtaH_ = nullptr;
  nLostHitsVsCosThetaH_ = nullptr;
  nLostHitsVsPhiH_ = nullptr;
  nLostHitsVsIterationH_ = nullptr;

  nHitsTIBSVsEtaH_ = nullptr;
  nHitsTOBSVsEtaH_ = nullptr;
  nHitsTECSVsEtaH_ = nullptr;
  nHitsTIDSVsEtaH_ = nullptr;
  nHitsStripSVsEtaH_ = nullptr;

  nHitsTIBDVsEtaH_ = nullptr;
  nHitsTOBDVsEtaH_ = nullptr;
  nHitsTECDVsEtaH_ = nullptr;
  nHitsTIDDVsEtaH_ = nullptr;
  nHitsStripDVsEtaH_ = nullptr;

  nValidHitsVspTH_ = nullptr;
  nValidHitsVsEtaH_ = nullptr;
  nValidHitsVsCosThetaH_ = nullptr;
  nValidHitsVsPhiH_ = nullptr;
  nValidHitsVsnVtxH_ = nullptr;

  nValidHitsPixVsEtaH_ = nullptr;
  nValidHitsPixBVsEtaH_ = nullptr;
  nValidHitsPixEVsEtaH_ = nullptr;
  nValidHitsStripVsEtaH_ = nullptr;
  nValidHitsTIBVsEtaH_ = nullptr;
  nValidHitsTOBVsEtaH_ = nullptr;
  nValidHitsTECVsEtaH_ = nullptr;
  nValidHitsTIDVsEtaH_ = nullptr;

  nValidHitsPixVsPhiH_ = nullptr;
  nValidHitsPixBVsPhiH_ = nullptr;
  nValidHitsPixEVsPhiH_ = nullptr;
  nValidHitsStripVsPhiH_ = nullptr;
  nValidHitsTIBVsPhiH_ = nullptr;
  nValidHitsTOBVsPhiH_ = nullptr;
  nValidHitsTECVsPhiH_ = nullptr;
  nValidHitsTIDVsPhiH_ = nullptr;

  nLostHitsPixVsEtaH_ = nullptr;
  nLostHitsPixBVsEtaH_ = nullptr;
  nLostHitsPixEVsEtaH_ = nullptr;
  nLostHitsStripVsEtaH_ = nullptr;
  nLostHitsTIBVsEtaH_ = nullptr;
  nLostHitsTOBVsEtaH_ = nullptr;
  nLostHitsTECVsEtaH_ = nullptr;
  nLostHitsTIDVsEtaH_ = nullptr;

  nLostHitsPixVsPhiH_ = nullptr;
  nLostHitsPixBVsPhiH_ = nullptr;
  nLostHitsPixEVsPhiH_ = nullptr;
  nLostHitsStripVsPhiH_ = nullptr;
  nLostHitsTIBVsPhiH_ = nullptr;
  nLostHitsTOBVsPhiH_ = nullptr;
  nLostHitsTECVsPhiH_ = nullptr;
  nLostHitsTIDVsPhiH_ = nullptr;

  nLostHitsPixVsIterationH_ = nullptr;
  nLostHitsPixBVsIterationH_ = nullptr;
  nLostHitsPixEVsIterationH_ = nullptr;
  nLostHitsStripVsIterationH_ = nullptr;
  nLostHitsTIBVsIterationH_ = nullptr;
  nLostHitsTOBVsIterationH_ = nullptr;
  nLostHitsTECVsIterationH_ = nullptr;
  nLostHitsTIDVsIterationH_ = nullptr;

  hOnTrkClusChargeThinH_ = nullptr;
  hOnTrkClusWidthThinH_ = nullptr;
  hOnTrkClusChargeThickH_ = nullptr;
  hOnTrkClusWidthThickH_ = nullptr;

  hOffTrkClusChargeThinH_ = nullptr;
  hOffTrkClusWidthThinH_ = nullptr;
  hOffTrkClusChargeThickH_ = nullptr;
  hOffTrkClusWidthThickH_ = nullptr;

  cosPhi3DdileptonH_ = nullptr;
  ip3dToPV2validpixelhitsH_ = nullptr;
  ip3dToBS2validpixelhitsH_ = nullptr;
  iperr3dToPV2validpixelhitsH_ = nullptr;
  iperr3dToBS2validpixelhitsH_ = nullptr;
  sip3dToPV2validpixelhitsH_ = nullptr;
  sip3dToBS2validpixelhitsH_ = nullptr;

  // Read pileup weight factors

  if (isMC_ && doPUCorrection_ && doTrackCorrection_) {
    throw std::runtime_error("if isMC is true, only one of doPUCorrection and doTrackCorrection can be true");
  }

  if (isMC_ && doPUCorrection_) {
    vpu_.clear();
    TFile* f1 = TFile::Open(puScaleFactorFile_.c_str());
    TH1F* h1 = dynamic_cast<TH1F*>(f1->Get("pileupweight"));
    for (int i = 1; i <= h1->GetNbinsX(); ++i)
      vpu_.push_back(h1->GetBinContent(i));
    f1->Close();
  }

  if (isMC_ && doTrackCorrection_) {
    vtrack_.clear();
    TFile* f1 = TFile::Open(trackScaleFactorFile_.c_str());
    TH1F* h1 = dynamic_cast<TH1F*>(f1->Get("trackweight"));
    for (int i = 1; i <= h1->GetNbinsX(); ++i)
      vtrack_.push_back(h1->GetBinContent(i));
    f1->Close();
  }
}

void StandaloneTrackMonitor::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &(iSetup.getData(geomToken_));
}

void StandaloneTrackMonitor::bookHistograms(DQMStore::IBooker& ibook,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  std::string currentFolder = moduleName_ + "/" + folderName_;
  ibook.setCurrentFolder(currentFolder);

  // The following are common with the official tool
  if (haveAllHistograms_) {
    trackEtaH_ = ibook.book1D("trackEta",
                              "Track Eta",
                              TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                              TrackEtaHistoPar_.getParameter<double>("Xmin"),
                              TrackEtaHistoPar_.getParameter<double>("Xmax"));

    trackEtaerrH_ = ibook.book1D("trackEtaerr", "Track Eta Error", 50, 0.0, 1.0);
    trackCosThetaH_ = ibook.book1D("trackCosTheta", "Track Cos(Theta)", 50, -1.0, 1.0);
    trackThetaerrH_ = ibook.book1D("trackThetaerr", "Track Theta Error", 50, 0.0, 1.0);
    trackPhiH_ = ibook.book1D("trackPhi", "Track Phi", 70, -3.5, 3.5);
    trackPhierrH_ = ibook.book1D("trackPhierr", "Track Phi Error", 50, 0.0, 1.0);

    trackPH_ = ibook.book1D("trackP", "Track 4-momentum", 50, 0.0, 10.0);
    trackPtH_ = ibook.book1D("trackPt",
                             "Track Pt",
                             TrackPtHistoPar_.getParameter<int32_t>("Xbins"),
                             TrackPtHistoPar_.getParameter<double>("Xmin"),
                             TrackPtHistoPar_.getParameter<double>("Xmax"));
    trackPt_ZoomH_ = ibook.book1D("trackPt_Zoom", "Track Pt", 100, 60, 70);
    trackPterrH_ = ibook.book1D("trackPterr", "Track Pt Error", 100, 0.0, 100.0);
    trackqOverpH_ = ibook.book1D("trackqOverp", "q Over p", 100, -0.5, 0.5);
    trackqOverperrH_ = ibook.book1D("trackqOverperr", "q Over p Error", 50, 0.0, 0.5);
    trackChargeH_ = ibook.book1D("trackCharge", "Track Charge", 3, -1.5, 1.5);
    trackChi2H_ = ibook.book1D("trackChi2", "Chi2", 100, 0.0, 100.0);
    tracknDOFH_ = ibook.book1D("tracknDOF", "nDOF", 100, 0.0, 100.0);
    trackChi2ProbH_ = ibook.book1D("trackChi2Prob", "Chi2prob", 50, 0.0, 1.0);
    trackChi2oNDFH_ = ibook.book1D("trackChi2oNDF", "Chi2oNDF", 100, 0.0, 100.0);
    trackd0H_ = ibook.book1D("trackd0", "Track d0", 100, -1, 1);
    trackChi2bynDOFH_ = ibook.book1D("trackChi2bynDOF", "Chi2 Over nDOF", 100, 0.0, 10.0);
    trackalgoH_ = ibook.book1D("trackalgo", "Track Algo", 46, 0.0, 46.0);
    trackorigalgoH_ = ibook.book1D("trackorigalgo", "Track Original Algo", 46, 0.0, 46.0);

    for (size_t ibin = 0; ibin < reco::TrackBase::algoSize - 1; ibin++) {
      trackalgoH_->setBinLabel(ibin + 1, reco::TrackBase::algoNames[ibin], 1);
      trackorigalgoH_->setBinLabel(ibin + 1, reco::TrackBase::algoNames[ibin], 1);
    }

    trackStoppingSourceH_ = ibook.book1D("trackstoppingsource", "Track Stopping Source", 12, 0.0, 12.0);

    // DataFormats/TrackCandidate/interface/TrajectoryStopReasons.h
    size_t StopReasonNameSize = static_cast<size_t>(StopReason::SIZE);
    for (unsigned int i = 0; i < StopReasonNameSize; ++i) {
      trackStoppingSourceH_->setBinLabel(i + 1, StopReasonName::StopReasonName[i], 1);
    }

    DistanceOfClosestApproachToPVH_ =
        ibook.book1D("DistanceOfClosestApproachToPV", "DistanceOfClosestApproachToPV", 1000, -1.0, 1.0);
    DistanceOfClosestApproachToPVZoomedH_ =
        ibook.book1D("DistanceOfClosestApproachToPVZoomed", "DistanceOfClosestApproachToPV", 1000, -0.1, 0.1);
    DistanceOfClosestApproachToPVVsPhiH_ = ibook.bookProfile(
        "DistanceOfClosestApproachToPVVsPhi", "DistanceOfClosestApproachToPVVsPhi", 100, -3.5, 3.5, 0.0, 0.0, "g");
    xPointOfClosestApproachVsZ0wrtPVH_ = ibook.bookProfile(
        "xPointOfClosestApproachVsZ0wrtPV", "xPointOfClosestApproachVsZ0wrtPV", 120, -60, 60, 0.0, 0.0, "g");
    yPointOfClosestApproachVsZ0wrtPVH_ = ibook.bookProfile(
        "yPointOfClosestApproachVsZ0wrtPV", "yPointOfClosestApproachVsZ0wrtPV", 120, -60, 60, 0.0, 0.0, "g");
    trackDeltaRwrtClosestTrack_ =
        ibook.book1D("trackDeltaRwrtClosestTrack", "min#DeltaR(considered track,other tracks)", 500, 0, 10);

    ip2dToPVH_ = ibook.book1D("ip2dToPV", "IP in 2d To PV", 1000, -1.0, 1.0);
    unsigned int niperrbins = 100;
    float iperrbinning[niperrbins + 1];
    float miniperr = 0.0001, maxiperr = 5;
    iperrbinning[0] = miniperr;
    for (unsigned int i = 1; i != niperrbins + 1; i++) {
      iperrbinning[i] = iperrbinning[i - 1] * pow(maxiperr / miniperr, 1. / niperrbins);
    }
    iperr2dToPVH_ = ibook.book1D("iperr2dToPV", "IP error in 2d To PV", niperrbins, iperrbinning);
    iperr2dToPVWtH_ = ibook.book1D("iperr2dToPVWt", "IP error in 2d To PV", niperrbins, iperrbinning);

    ip3dToPVH_ = ibook.book1D("ip3dToPV", "IP in 3d To PV", 200, -20, 20);
    ip3dToBSH_ = ibook.book1D("ip3dToBS", "IP in 3d To BS", 200, -20, 20);
    iperr3dToPVH_ = ibook.book1D("iperr3dToPV", "IP error in 3d To PV", niperrbins, iperrbinning);
    iperr3dToBSH_ = ibook.book1D("iperr3dToBS", "IP error in 3d To BS", niperrbins, iperrbinning);
    sip3dToPVH_ = ibook.book1D("sip3dToPV", "IP significance in 3d To PV", 200, -10, 10);
    sip3dToBSH_ = ibook.book1D("sip3dToBS", "IP significance in 3d To BS", 200, -10, 10);

    ip3dToPV2validpixelhitsH_ =
        ibook.book1D("ip3dToPV2validpixelhits", "IP in 3d To PV (nValidPixelHits>2)", 200, -0.20, 0.20);
    ip3dToBS2validpixelhitsH_ =
        ibook.book1D("ip3dToBS2validpixelhits", "IP in 3d To BS (nValidPixelHits>2)", 200, -0.20, 0.20);
    iperr3dToPV2validpixelhitsH_ = ibook.book1D(
        "iperr3dToPV2validpixelhits", "IP error in 3d To PV (nValidPixelHits>2)", niperrbins, iperrbinning);
    iperr3dToBS2validpixelhitsH_ = ibook.book1D(
        "iperr3dToBS2validpixelhits", "IP error in 3d To BS (nValidPixelHits>2)", niperrbins, iperrbinning);
    sip3dToPV2validpixelhitsH_ =
        ibook.book1D("sip3dToPV2validpixelhits", "IP significance in 3d To PV (nValidPixelHits>2)", 200, -10, 10);
    sip3dToBS2validpixelhitsH_ =
        ibook.book1D("sip3dToBS2validpixelhits", "IP significance in 3d To BS (nValidPixelHits>2)", 200, -10, 10);

    ip2dToBSH_ = ibook.book1D("ip2dToBS", "IP in 2d To BS", 1000, -1., 1.);  //Beamspot
    iperr2dToBSH_ = ibook.book1D("iperr2dToBS", "IP error in 2d To BS", niperrbins, iperrbinning);
    sip2dToBSH_ = ibook.book1D("sip2dToBS", "IP significance in 2d To BS", 200, -10, 10);

    iperr3dToPVWtH_ = ibook.book1D("iperr3dToPVWt", "IP error in 3d To PV", niperrbins, iperrbinning);
    sip2dToPVH_ = ibook.book1D("sip2dToPV", "IP significance in 2d To PV", 200, -10, 10);

    sip2dToPVWtH_ = ibook.book1D("sip2dToPVWt", "IP significance in 2d To PV", 200, -10, 10);
    sipDxyToPVH_ = ibook.book1D("sipDxyToPV", "IP significance in dxy To PV", 100, -10, 10);
    sipDzToPVH_ = ibook.book1D("sipDzToPV", "IP significance in dz To PV", 100, -10, 10);

    nallHitsH_ = ibook.book1D("nallHits", "No. of All Hits", 60, -0.5, 59.5);
    ntrackerHitsH_ = ibook.book1D("ntrackerHits", "No. of Tracker Hits", 60, -0.5, 59.5);

    nvalidTrackerHitsH_ = ibook.book1D("nvalidTrackerhits", "No. of Valid Tracker Hits", 47, -0.5, 46.5);
    nvalidPixelHitsH_ = ibook.book1D("nvalidPixelHits", "No. of Valid Hits in Pixel", 8, -0.5, 7.5);
    nvalidPixelBHitsH_ = ibook.book1D("nvalidPixelBarrelHits", "No. of Valid Hits in Pixel Barrel", 6, -0.5, 5.5);
    nvalidPixelEHitsH_ = ibook.book1D("nvalidPixelEndcapHits", "No. of Valid Hits in Pixel Endcap", 6, -0.5, 6.5);
    nvalidStripHitsH_ = ibook.book1D("nvalidStripHits", "No. of Valid Hits in Strip", 36, -0.5, 35.5);
    nvalidTIBHitsH_ = ibook.book1D("nvalidTIBHits", "No. of Valid Hits in Strip TIB", 6, -0.5, 5.5);
    nvalidTOBHitsH_ = ibook.book1D("nvalidTOBHits", "No. of Valid Hits in Strip TOB", 11, -0.5, 10.5);
    nvalidTIDHitsH_ = ibook.book1D("nvalidTIDHits", "No. of Valid Hits in Strip TID", 6, -0.5, 5.5);
    nvalidTECHitsH_ = ibook.book1D("nvalidTECHits", "No. of Valid Hits in Strip TEC", 11, -0.5, 10.5);

    nlostTrackerHitsH_ = ibook.book1D("nlostTrackerhits", "No. of Lost Tracker Hits", 15, -0.5, 14.5);
    nlostPixelHitsH_ = ibook.book1D("nlostPixelHits", "No. of Lost Hits in Pixel", 8, -0.5, 7.5);
    nlostPixelBHitsH_ = ibook.book1D("nlostPixelBarrelHits", "No. of Lost Hits in Pixel Barrel", 5, -0.5, 4.5);
    nlostPixelEHitsH_ = ibook.book1D("nlostPixelEndcapHits", "No. of Lost Hits in Pixel Endcap", 4, -0.5, 3.5);
    nlostStripHitsH_ = ibook.book1D("nlostStripHits", "No. of Lost Hits in Strip", 10, -0.5, 9.5);
    nlostTIBHitsH_ = ibook.book1D("nlostTIBHits", "No. of Lost Hits in Strip TIB", 5, -0.5, 4.5);
    nlostTOBHitsH_ = ibook.book1D("nlostTOBHits", "No. of Lost Hits in Strip TOB", 10, -0.5, 9.5);
    nlostTIDHitsH_ = ibook.book1D("nlostTIDHits", "No. of Lost Hits in Strip TID", 5, -0.5, 4.5);
    nlostTECHitsH_ = ibook.book1D("nlostTECHits", "No. of Lost Hits in Strip TEC", 10, -0.5, 9.5);

    trkLayerwithMeasurementH_ = ibook.book1D("trkLayerwithMeasurement", "No. of Layers per Track", 20, 0.0, 20.0);
    pixelLayerwithMeasurementH_ =
        ibook.book1D("pixelLayerwithMeasurement", "No. of Pixel Layers per Track", 10, 0.0, 10.0);
    pixelBLayerwithMeasurementH_ =
        ibook.book1D("pixelBLayerwithMeasurement", "No. of Pixel Barrel Layers per Track", 5, 0.0, 5.0);
    pixelELayerwithMeasurementH_ =
        ibook.book1D("pixelELayerwithMeasurement", "No. of Pixel Endcap Layers per Track", 5, 0.0, 5.0);
    stripLayerwithMeasurementH_ =
        ibook.book1D("stripLayerwithMeasurement", "No. of Strip Layers per Track", 20, 0.0, 20.0);
    stripTIBLayerwithMeasurementH_ =
        ibook.book1D("stripTIBLayerwithMeasurement", "No. of Strip TIB Layers per Track", 10, 0.0, 10.0);
    stripTOBLayerwithMeasurementH_ =
        ibook.book1D("stripTOBLayerwithMeasurement", "No. of Strip TOB Layers per Track", 10, 0.0, 10.0);
    stripTIDLayerwithMeasurementH_ =
        ibook.book1D("stripTIDLayerwithMeasurement", "No. of Strip TID Layers per Track", 5, 0.0, 5.0);
    stripTECLayerwithMeasurementH_ =
        ibook.book1D("stripTECLayerwithMeasurement", "No. of Strip TEC Layers per Track", 15, 0.0, 15.0);

    nlostHitsH_ = ibook.book1D("nlostHits", "No. of Lost Hits", 10, -0.5, 9.5);
    nMissingExpectedInnerHitsH_ =
        ibook.book1D("nMissingExpectedInnerHits", "No. of Missing Expected Inner Hits", 10, -0.5, 9.5);
    nMissingExpectedOuterHitsH_ =
        ibook.book1D("nMissingExpectedOuterHits", "No. of Missing Expected Outer Hits", 10, -0.5, 9.5);

    beamSpotXYposH_ = ibook.book1D("beamSpotXYpos", "d_{xy} w.r.t beam spot", 100, -1.0, 1.0);
    beamSpotXYposerrH_ = ibook.book1D("beamSpotXYposerr", "error on d_{xy} w.r.t beam spot", 50, 0.0, 0.25);
    beamSpotZposH_ = ibook.book1D("beamSpotZpos", "d_{z} w.r.t beam spot", 100, -20.0, 20.0);
    beamSpotZposerrH_ = ibook.book1D("beamSpotZposerr", "error on d_{z} w.r.t. beam spot", 50, 0.0, 0.25);

    vertexXposH_ = ibook.book1D("vertexXpos", "Vertex X position", 100, -0.1, 0.1);
    vertexYposH_ = ibook.book1D("vertexYpos", "Vertex Y position", 200, -0.1, 0.1);
    vertexZposH_ = ibook.book1D("vertexZpos", "Vertex Z position", 100, -20.0, 20.0);
    nVertexH_ = ibook.book1D("nVertex", "# of vertices", 120, -0.5, 119.5);
    nVtxH_ = ibook.book1D("nVtx", "# of vtxs", 120, -0.5, 119.5);

    nMissingInnerHitBH_ = ibook.book1D("nMissingInnerHitB", "No. missing inner hit per Track in Barrel", 6, -0.5, 5.5);
    nMissingInnerHitEH_ = ibook.book1D("nMissingInnerHitE", "No. missing inner hit per Track in Endcap", 6, -0.5, 5.5);
    nMissingOuterHitBH_ =
        ibook.book1D("nMissingOuterHitB", "No. missing outer hit per Track in Barrel", 11, -0.5, 10.5);
    nMissingOuterHitEH_ =
        ibook.book1D("nMissingOuterHitE", "No. missing outer hit per Track in Endcap", 11, -0.5, 10.5);
    nPixBarrelH_ = ibook.book1D("nHitPixelBarrel", "No. of hits in Pixel Barrel per Track", 20, 0, 20.0);
    nPixEndcapH_ = ibook.book1D("nHitPixelEndcap", "No. of hits in Pixel Endcap per Track", 20, 0, 20.0);
    nStripTIBH_ = ibook.book1D("nHitStripTIB", "No. of hits in Strip TIB per Track", 30, 0, 30.0);
    nStripTOBH_ = ibook.book1D("nHitStripTOB", "No. of hits in Strip TOB per Track", 30, 0, 30.0);
    nStripTECH_ = ibook.book1D("nHitStripTEC", "No. of hits in Strip TEC per Track", 30, 0, 30.0);
    nStripTIDH_ = ibook.book1D("nHitStripTID", "No. of hits in Strip TID per Tracks", 30, 0, 30.0);
    nTracksH_ = ibook.book1D("nTracks", "No. of Tracks", 1200, -0.5, 1199.5);
    nJet_ = ibook.book1D("nJet", "Number of Jets", 101, -0.5, 100.5);
    Jet_pt_ = ibook.book1D("Jet_pt", "Jet p_{T}", 200, 0., 200.);
    Jet_eta_ = ibook.book1D("Jet_eta", "Jet #eta", 100, -5.2, 5.2);
    Jet_energy_ = ibook.book1D("Jet_energy", "Jet Energy", 200, 0., 200.);
    Jet_chargedMultiplicity_ =
        ibook.book1D("Jet_chargedMultiplicity", "Jet charged Hadron Multiplicity", 201, -0.5, 200.5);
    Zpt_ = ibook.book1D("Zpt", "Z-boson transverse momentum", 100, 0, 100);
    ZInvMass_ = ibook.book1D("ZInvMass", "m_{ll}", 120, 75, 105);
    cosPhi3DdileptonH_ = ibook.book1D("cosPhi3Ddilepton", "cos#Phi_{3D,ll}", 202, -1.01, 1.01);
  }
  if (isMC_) {
    bunchCrossingH_ = ibook.book1D("bunchCrossing", "Bunch Crossing", 60, 0, 60.0);
    nPUH_ = ibook.book1D("nPU", "No of Pileup", 100, 0, 100.0);
    trueNIntH_ = ibook.book1D("trueNInt", "True no of Interactions", 100, 0, 100.0);
  }
  // Exclusive histograms

  nLostHitByLayerH_ = ibook.book1D("nLostHitByLayer", "No. of Lost Hit per Layer", 29, 0.5, 29.5);

  nLostHitByLayerPixH_ =
      ibook.book1D("nLostHitByLayerPix", "No. of Lost Hit per Layer for Pixel detector", 7, 0.5, 7.5);

  nLostHitByLayerStripH_ =
      ibook.book1D("nLostHitByLayerStrip", "No. of Lost Hit per Layer for SiStrip detector", 22, 0.5, 22.5);

  nLostHitsVspTH_ = ibook.bookProfile("nLostHitsVspT",
                                      "Number of Lost Hits Vs pT",
                                      TrackPtHistoPar_.getParameter<int32_t>("Xbins"),
                                      TrackPtHistoPar_.getParameter<double>("Xmin"),
                                      TrackPtHistoPar_.getParameter<double>("Xmax"),
                                      0.0,
                                      0.0,
                                      "g");
  nLostHitsVsEtaH_ = ibook.bookProfile("nLostHitsVsEta",
                                       "Number of Lost Hits Vs Eta",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nLostHitsVsCosThetaH_ =
      ibook.bookProfile("nLostHitsVsCosTheta", "Number of Lost Hits Vs Cos(Theta)", 50, -1.0, 1.0, 0.0, 0.0, "g");
  nLostHitsVsPhiH_ = ibook.bookProfile("nLostHitsVsPhi", "Number of Lost Hits Vs Phi", 100, -3.5, 3.5, 0.0, 0.0, "g");
  nLostHitsVsIterationH_ =
      ibook.bookProfile("nLostHitsVsIteration", "Number of Lost Hits Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");

  nHitsTIBSVsEtaH_ = ibook.bookProfile("nHitsTIBSVsEta",
                                       "Number of Hits in TIB Vs Eta (Single-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsTOBSVsEtaH_ = ibook.bookProfile("nHitsTOBSVsEta",
                                       "Number of Hits in TOB Vs Eta (Single-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsTECSVsEtaH_ = ibook.bookProfile("nHitsTECSVsEta",
                                       "Number of Hits in TEC Vs Eta (Single-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsTIDSVsEtaH_ = ibook.bookProfile("nHitsTIDSVsEta",
                                       "Number of Hits in TID Vs Eta (Single-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");

  nHitsStripSVsEtaH_ = ibook.bookProfile("nHitsStripSVsEta",
                                         "Number of Strip Hits Vs Eta (Single-sided)",
                                         TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                         TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                         TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                         0.0,
                                         0.0,
                                         "g");

  nHitsTIBDVsEtaH_ = ibook.bookProfile("nHitsTIBDVsEta",
                                       "Number of Hits in TIB Vs Eta (Double-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsTOBDVsEtaH_ = ibook.bookProfile("nHitsTOBDVsEta",
                                       "Number of Hits in TOB Vs Eta (Double-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsTECDVsEtaH_ = ibook.bookProfile("nHitsTECDVsEta",
                                       "Number of Hits in TEC Vs Eta (Double-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsTIDDVsEtaH_ = ibook.bookProfile("nHitsTIDDVsEta",
                                       "Number of Hits in TID Vs Eta (Double-sided)",
                                       TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                       TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nHitsStripDVsEtaH_ = ibook.bookProfile("nHitsStripDVsEta",
                                         "Number of Strip Hits Vs Eta (Double-sided)",
                                         TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                         TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                         TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                         0.0,
                                         0.0,
                                         "g");

  nValidHitsVspTH_ = ibook.bookProfile("nValidHitsVspT",
                                       "Number of Valid Hits Vs pT",
                                       TrackPtHistoPar_.getParameter<int32_t>("Xbins"),
                                       TrackPtHistoPar_.getParameter<double>("Xmin"),
                                       TrackPtHistoPar_.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "g");
  nValidHitsVsnVtxH_ =
      ibook.bookProfile("nValidHitsVsnVtx", "Number of Valid Hits Vs Number of Vertex", 100, 0., 100., 0.0, 0.0, "g");
  nValidHitsVsEtaH_ = ibook.bookProfile("nValidHitsVsEta",
                                        "Number of Hits Vs Eta",
                                        TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                        TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                        TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                        0.0,
                                        0.0,
                                        "g");

  nValidHitsVsCosThetaH_ =
      ibook.bookProfile("nValidHitsVsCosTheta", "Number of Valid Hits Vs Cos(Theta)", 50, -1.0, 1.0, 0.0, 0.0, "g");
  nValidHitsVsPhiH_ =
      ibook.bookProfile("nValidHitsVsPhi", "Number of Valid Hits Vs Phi", 100, -3.5, 3.5, 0.0, 0.0, "g");

  nValidHitsPixVsEtaH_ = ibook.bookProfile("nValidHitsPixVsEta",
                                           "Number of Valid Hits in Pixel Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsPixBVsEtaH_ = ibook.bookProfile("nValidHitsPixBVsEta",
                                            "Number of Valid Hits in Pixel Barrel Vs Eta",
                                            TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                            0.0,
                                            0.0,
                                            "g");
  nValidHitsPixEVsEtaH_ = ibook.bookProfile("nValidHitsPixEVsEta",
                                            "Number of Valid Hits in Pixel Endcap Vs Eta",
                                            TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                            0.0,
                                            0.0,
                                            "g");
  nValidHitsStripVsEtaH_ = ibook.bookProfile("nValidHitsStripVsEta",
                                             "Number of Valid Hits in SiStrip Vs Eta",
                                             TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                             TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                             TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                             0.0,
                                             0.0,
                                             "g");
  nValidHitsTIBVsEtaH_ = ibook.bookProfile("nValidHitsTIBVsEta",
                                           "Number of Valid Hits in TIB Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsTOBVsEtaH_ = ibook.bookProfile("nValidHitsTOBVsEta",
                                           "Number of Valid Hits in TOB Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsTECVsEtaH_ = ibook.bookProfile("nValidHitsTECVsEta",
                                           "Number of Valid Hits in TEC Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsTIDVsEtaH_ = ibook.bookProfile("nValidHitsTIDVsEta",
                                           "Number of Valid Hits in TID Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");

  nValidHitsPixVsPhiH_ = ibook.bookProfile("nValidHitsPixVsPhi",
                                           "Number of Valid Hits in Pixel Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsPixBVsPhiH_ = ibook.bookProfile("nValidHitsPixBVsPhi",
                                            "Number of Valid Hits in Pixel Barrel Vs Phi",
                                            TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                            TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                            TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                            0.0,
                                            0.0,
                                            "g");
  nValidHitsPixEVsPhiH_ = ibook.bookProfile("nValidHitsPixEVsPhi",
                                            "Number of Valid Hits in Pixel Endcap Vs Phi",
                                            TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                            0.0,
                                            0.0,
                                            "g");
  nValidHitsStripVsPhiH_ = ibook.bookProfile("nValidHitsStripVsPhi",
                                             "Number of Valid Hits in SiStrip Vs Phi",
                                             TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                             TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                             TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                             0.0,
                                             0.0,
                                             "g");
  nValidHitsTIBVsPhiH_ = ibook.bookProfile("nValidHitsTIBVsPhi",
                                           "Number of Valid Hits in TIB Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsTOBVsPhiH_ = ibook.bookProfile("nValidHitsTOBVsPhi",
                                           "Number of Valid Hits in TOB Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsTECVsPhiH_ = ibook.bookProfile("nValidHitsTECVsPhi",
                                           "Number of Valid Hits in TEC Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nValidHitsTIDVsPhiH_ = ibook.bookProfile("nValidHitsTIDVsPhi",
                                           "Number of Valid Hits in TID Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");

  nLostHitsPixVsEtaH_ = ibook.bookProfile("nLostHitsPixVsEta",
                                          "Number of Lost Hits in Pixel Vs Eta",
                                          TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsPixBVsEtaH_ = ibook.bookProfile("nLostHitsPixBVsEta",
                                           "Number of Lost Hits in Pixel Barrel Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nLostHitsPixEVsEtaH_ = ibook.bookProfile("nLostHitsPixEVsEta",
                                           "Number of Lost Hits in Pixel Endcap Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nLostHitsStripVsEtaH_ = ibook.bookProfile("nLostHitsStripVsEta",
                                            "Number of Lost Hits in SiStrip Vs Eta",
                                            TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                            TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                            0.0,
                                            0.0,
                                            "g");
  nLostHitsTIBVsEtaH_ = ibook.bookProfile("nLostHitsTIBVsEta",
                                          "Number of Lost Hits in TIB Vs Eta",
                                          TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsTOBVsEtaH_ = ibook.bookProfile("nLostHitsTOBVsEta",
                                          "Number of Lost Hits in TOB Vs Eta",
                                          TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsTECVsEtaH_ = ibook.bookProfile("nLostHitsTECVsEta",
                                          "Number of Lost Hits in TEC Vs Eta",
                                          TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsTIDVsEtaH_ = ibook.bookProfile("nLostHitsTIDVsEta",
                                          "Number of Lost Hits in TID Vs Eta",
                                          TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                          TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");

  nLostHitsPixVsPhiH_ = ibook.bookProfile("nLostHitsPixVsPhi",
                                          "Number of Lost Hits in Pixel Vs Phi",
                                          TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsPixBVsPhiH_ = ibook.bookProfile("nLostHitsPixBVsPhi",
                                           "Number of Lost Hits in Pixel Barrel Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nLostHitsPixEVsPhiH_ = ibook.bookProfile("nLostHitsPixEVsPhi",
                                           "Number of Lost Hits in Pixel Endcap Vs Phi",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           0.0,
                                           "g");
  nLostHitsStripVsPhiH_ = ibook.bookProfile("nLostHitsStripVsPhi",
                                            "Number of Lost Hits in SiStrip Vs Phi",
                                            TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                            TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                            TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                            0.0,
                                            0.0,
                                            "g");
  nLostHitsTIBVsPhiH_ = ibook.bookProfile("nLostHitsTIBVsPhi",
                                          "Number of Lost Hits in TIB Vs Phi",
                                          TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsTOBVsPhiH_ = ibook.bookProfile("nLostHitsTOBVsPhi",
                                          "Number of Lost Hits in TOB Vs Phi",
                                          TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsTECVsPhiH_ = ibook.bookProfile("nLostHitsTECVsPhi",
                                          "Number of Lost Hits in TEC Vs Phi",
                                          TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");
  nLostHitsTIDVsPhiH_ = ibook.bookProfile("nLostHitsTIDVsPhi",
                                          "Number of Lost Hits in TID Vs Phi",
                                          TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                          TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                          0.0,
                                          0.0,
                                          "g");

  nLostHitsPixVsIterationH_ = ibook.bookProfile(
      "nLostHitsPixVsIteration", "Number of Lost Hits in Pixel Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsPixBVsIterationH_ = ibook.bookProfile(
      "nLostHitsPixBVsIteration", "Number of Lost Hits in Pixel Barrel Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsPixEVsIterationH_ = ibook.bookProfile(
      "nLostHitsPixEVsIteration", "Number of Lost Hits in Pixel Endcap Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsStripVsIterationH_ = ibook.bookProfile(
      "nLostHitsStripVsIteration", "Number of Lost Hits in SiStrip Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsTIBVsIterationH_ = ibook.bookProfile(
      "nLostHitsTIBVsIteration", "Number of Lost Hits in TIB Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsTOBVsIterationH_ = ibook.bookProfile(
      "nLostHitsTOBVsIteration", "Number of Lost Hits in TOB Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsTECVsIterationH_ = ibook.bookProfile(
      "nLostHitsTECVsIteration", "Number of Lost Hits in TEC Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");
  nLostHitsTIDVsIterationH_ = ibook.bookProfile(
      "nLostHitsTIDVsIteration", "Number of Lost Hits in TID Vs Iteration", 47, -0.5, 46.5, 0.0, 0.0, "g");

  trackChi2oNDFVsEtaH_ = ibook.bookProfile("trackChi2oNDFVsEta",
                                           "chi2/ndof of Tracks Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           10.0,
                                           "g");
  trackChi2oNDFVsPhiH_ = ibook.bookProfile("trackChi2oNDFVsPhi",
                                           "chi2/ndof of Tracks Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           10.0,
                                           "g");

  trackChi2probVsEtaH_ = ibook.bookProfile("trackChi2probVsEta",
                                           "chi2 probability of Tracks Vs Eta",
                                           TrackEtaHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmin"),
                                           TrackEtaHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           1.0,
                                           "g");
  trackChi2probVsPhiH_ = ibook.bookProfile("trackChi2probVsPhi",
                                           "chi2 probability of Tracks Vs Phi",
                                           TrackPhiHistoPar_.getParameter<int32_t>("Xbins"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmin"),
                                           TrackPhiHistoPar_.getParameter<double>("Xmax"),
                                           0.0,
                                           1.0,
                                           "g");

  trackIperr3dVsEtaH_ =
      ibook.bookProfile("trackIperr3dVsEta", "ip3d error of Tracks Vs Eta", 80, -4., 4., 0.0, 0.0, "g");

  trackSip2dVsEtaH_ = ibook.bookProfile("trackSip2dVsEta", "sip2d of Tracks Vs Eta", 80, -4., 4., 0.0, 0.0, "g");

  trackIperr3dVsEta2DH_ =
      ibook.book2D("trackIperr3dVsEta2D", "ip3d error of Tracks Vs Eta 2d", 80, -4., 4., 100, 0., 5.);
  trackIperr3dVsChi2prob2DH_ =
      ibook.book2D("trackIperr3dVsChi2prob2D", "ip3d error of Tracks Vs chi2prob 2d", 50, 0., 1., 100, 0., 5.);
  trackSip2dVsEta2DH_ = ibook.book2D("trackSip2dVsEta2D", "sip2d of Tracks Vs Eta 2d", 80, -4., 4., 200, -10., 10.);
  trackSip2dVsChi2prob2DH_ =
      ibook.book2D("trackSip2dVsChi2prob2D", "sip2d of Tracks Vs chi2prob 2d", 50, 0., 1., 200, -10., 10.);

  // On and off-track cluster properties
  hOnTrkClusChargeThinH_ = ibook.book1D("hOnTrkClusChargeThin", "On-track Cluster Charge (Thin Sensor)", 100, 0, 1000);
  hOnTrkClusWidthThinH_ = ibook.book1D("hOnTrkClusWidthThin", "On-track Cluster Width (Thin Sensor)", 20, -0.5, 19.5);
  hOnTrkClusChargeThickH_ =
      ibook.book1D("hOnTrkClusChargeThick", "On-track Cluster Charge (Thick Sensor)", 100, 0, 1000);
  hOnTrkClusWidthThickH_ =
      ibook.book1D("hOnTrkClusWidthThick", "On-track Cluster Width (Thick Sensor)", 20, -0.5, 19.5);

  hOffTrkClusChargeThinH_ =
      ibook.book1D("hOffTrkClusChargeThin", "Off-track Cluster Charge (Thin Sensor)", 100, 0, 1000);
  hOffTrkClusWidthThinH_ =
      ibook.book1D("hOffTrkClusWidthThin", "Off-track Cluster Width (Thin Sensor)", 20, -0.5, 19.5);
  hOffTrkClusChargeThickH_ =
      ibook.book1D("hOffTrkClusChargeThick", "Off-track Cluster Charge (Thick Sensor)", 100, 0, 1000);
  hOffTrkClusWidthThickH_ =
      ibook.book1D("hOffTrkClusWidthThick", "Off-track Cluster Width (Thick Sensor)", 20, -0.5, 19.5);
}
void StandaloneTrackMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  if (verbose_)
    edm::LogInfo("StandaloneTrackMonitor") << "Begin StandaloneTrackMonitor" << std::endl;

  nevt++;

  // Get event setup (to get global transformation)
  const TrackerGeometry& tkGeom = (*tkGeom_);

  // Primary vertex collection
  edm::Handle<reco::VertexCollection> vertexColl;
  iEvent.getByToken(vertexToken_, vertexColl);
  if (!vertexColl.isValid()) {
    edm::LogError("DqmTrackStudy") << "Error! Failed to get reco::Vertex Collection, " << vertexTag_;
  }
  if (vertexColl->empty()) {
    edm::LogError("StandalonaTrackMonitor") << "No good vertex in the event!!" << std::endl;
    return;
  }
  const reco::Vertex& pv = (*vertexColl)[0];

  // Beam spot
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);
  if (!beamSpot.isValid())
    edm::LogError("StandalonaTrackMonitor") << "Beamspot for input tag: " << bsTag_ << " not found!!";

  // Track collection
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackToken_, tracks);
  if (!tracks.isValid())
    edm::LogError("StandalonaTrackMonitor") << "TrackCollection for input tag: " << trackTag_ << " not found!!";

  // Access PU information
  double wfac = 1.0;  // for data
  if (!iEvent.isRealData()) {
    edm::Handle<std::vector<PileupSummaryInfo> > PupInfo;
    iEvent.getByToken(puSummaryToken_, PupInfo);

    if (verbose_)
      edm::LogInfo("StandaloneTrackMonitor") << "nPUColl = " << PupInfo->size();
    if (PupInfo.isValid()) {
      for (auto const& v : *PupInfo) {
        int bx = v.getBunchCrossing();
        if (bunchCrossingH_)
          bunchCrossingH_->Fill(bx);
        if (bx == 0) {
          if (nPUH_)
            nPUH_->Fill(v.getPU_NumInteractions());
          int ntrueInt = v.getTrueNumInteractions();
          int nVertex = (vertexColl.isValid() ? vertexColl->size() : 0);
          if (trueNIntH_)
            trueNIntH_->Fill(ntrueInt);
          if (doPUCorrection_) {
            if (nVertex > -1 && nVertex < int(vpu_.size()))
              wfac = vpu_.at(nVertex);
            else
              wfac = 0.0;
          }
        }
      }
    } else
      edm::LogError("StandalonaTrackMonitor") << "PUSummary for input tag: " << puSummaryTag_ << " not found!!";
    if (doTrackCorrection_) {
      int ntrack = 0;
      for (auto const& track : *tracks) {
        if (!track.quality(reco::Track::qualityByName(trackQuality_)))
          continue;
        ++ntrack;
      }
      if (ntrack > -1 && ntrack < int(vtrack_.size()))
        wfac = vtrack_.at(ntrack);
      else
        wfac = 0.0;
    }
  }
  if (verbose_)
    edm::LogInfo("StandaloneTrackMonitor") << "PU reweight factor = " << wfac;

  if (haveAllHistograms_) {
    int nvtx = (vertexColl.isValid() ? vertexColl->size() : 0);
    nVertexH_->Fill(nvtx, wfac);
    nVtxH_->Fill(nvtx);
  }

  // Get MVA and quality mask collections
  int ntracks = 0;
  std::vector<TLorentzVector> list;

  const TransientTrackBuilder* theB = &iSetup.getData(transTrackToken_);
  TransientVertex mumuTransientVtx;
  std::vector<reco::TransientTrack> tks;

  if (tracks.isValid()) {
    if (verbose_)
      edm::LogInfo("StandaloneTrackMonitor") << "Total # of Tracks: " << tracks->size();
    reco::Track::TrackQuality quality = reco::Track::qualityByName(trackQuality_);

    for (auto const& track : *tracks) {
      if (!track.quality(quality))
        continue;
      ++ntracks;

      reco::TransientTrack trajectory = theB->build(track);
      tks.push_back(trajectory);

      double eta = track.eta();
      double theta = track.theta();
      double phi = track.phi();
      double pt = track.pt();

      const reco::HitPattern& hitp = track.hitPattern();
      double nAllHits = hitp.numberOfAllHits(reco::HitPattern::TRACK_HITS);
      double nAllTrackerHits = hitp.numberOfAllTrackerHits(reco::HitPattern::TRACK_HITS);

      double trackdeltaR = std::numeric_limits<double>::max();
      double muonmass = 0.1056583745;
      TLorentzVector trackpmu;
      trackpmu.SetPtEtaPhiM(track.pt(), track.eta(), track.phi(), muonmass);
      for (auto const& TRACK : *tracks) {
        if (&track == &TRACK)
          continue;
        double tmpdr = reco::deltaR(track.eta(), track.phi(), TRACK.eta(), TRACK.phi());
        if (tmpdr < trackdeltaR)
          trackdeltaR = tmpdr;
      }

      list.push_back(trackpmu);

      double nValidTrackerHits = hitp.numberOfValidTrackerHits();
      double nValidPixelHits = hitp.numberOfValidPixelHits();
      double nValidPixelBHits = hitp.numberOfValidPixelBarrelHits();
      double nValidPixelEHits = hitp.numberOfValidPixelEndcapHits();
      double nValidStripHits = hitp.numberOfValidStripHits();
      double nValidTIBHits = hitp.numberOfValidStripTIBHits();
      double nValidTOBHits = hitp.numberOfValidStripTOBHits();
      double nValidTIDHits = hitp.numberOfValidStripTIDHits();
      double nValidTECHits = hitp.numberOfValidStripTECHits();

      int missingInnerHit = hitp.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS);
      int missingOuterHit = hitp.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS);

      nValidHitsVspTH_->Fill(pt, nValidTrackerHits);
      nValidHitsVsEtaH_->Fill(eta, nValidTrackerHits);
      nValidHitsVsCosThetaH_->Fill(std::cos(theta), nValidTrackerHits);
      nValidHitsVsPhiH_->Fill(phi, nValidTrackerHits);
      nValidHitsVsnVtxH_->Fill(vertexColl->size(), nValidTrackerHits);

      nValidHitsPixVsEtaH_->Fill(eta, nValidPixelHits);
      nValidHitsPixBVsEtaH_->Fill(eta, nValidPixelBHits);
      nValidHitsPixEVsEtaH_->Fill(eta, nValidPixelEHits);
      nValidHitsStripVsEtaH_->Fill(eta, nValidStripHits);
      nValidHitsTIBVsEtaH_->Fill(eta, nValidTIBHits);
      nValidHitsTOBVsEtaH_->Fill(eta, nValidTOBHits);
      nValidHitsTECVsEtaH_->Fill(eta, nValidTECHits);
      nValidHitsTIDVsEtaH_->Fill(eta, nValidTIDHits);

      nValidHitsPixVsPhiH_->Fill(phi, nValidPixelHits);
      nValidHitsPixBVsPhiH_->Fill(phi, nValidPixelBHits);
      nValidHitsPixEVsPhiH_->Fill(phi, nValidPixelEHits);
      nValidHitsStripVsPhiH_->Fill(phi, nValidStripHits);
      nValidHitsTIBVsPhiH_->Fill(phi, nValidTIBHits);
      nValidHitsTOBVsPhiH_->Fill(phi, nValidTOBHits);
      nValidHitsTECVsPhiH_->Fill(phi, nValidTECHits);
      nValidHitsTIDVsPhiH_->Fill(phi, nValidTIDHits);

      int nLostHits = track.numberOfLostHits();
      int nMissingExpectedInnerHits = hitp.numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      int nMissingExpectedOuterHits = hitp.numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
      int nLostTrackerHits = hitp.numberOfLostTrackerHits(reco::HitPattern::TRACK_HITS);
      int nLostPixHits = hitp.numberOfLostPixelHits(reco::HitPattern::TRACK_HITS);
      int nLostPixBHits = hitp.numberOfLostPixelBarrelHits(reco::HitPattern::TRACK_HITS);
      int nLostPixEHits = hitp.numberOfLostPixelEndcapHits(reco::HitPattern::TRACK_HITS);
      int nLostStripHits = hitp.numberOfLostStripHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTIBHits = hitp.numberOfLostStripTIBHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTIDHits = hitp.numberOfLostStripTIDHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTOBHits = hitp.numberOfLostStripTOBHits(reco::HitPattern::TRACK_HITS);
      int nLostStripTECHits = hitp.numberOfLostStripTECHits(reco::HitPattern::TRACK_HITS);
      int nIteration = track.originalAlgo();

      nLostHitsVspTH_->Fill(pt, nLostTrackerHits);
      nLostHitsVsEtaH_->Fill(eta, nLostTrackerHits);
      nLostHitsVsCosThetaH_->Fill(std::cos(theta), nLostTrackerHits);
      nLostHitsVsPhiH_->Fill(phi, nLostTrackerHits);
      nLostHitsVsIterationH_->Fill(nIteration, nLostTrackerHits);

      nLostHitsPixVsEtaH_->Fill(eta, nLostPixHits);
      nLostHitsPixBVsEtaH_->Fill(eta, nLostPixBHits);
      nLostHitsPixEVsEtaH_->Fill(eta, nLostPixEHits);
      nLostHitsStripVsEtaH_->Fill(eta, nLostStripHits);
      nLostHitsTIBVsEtaH_->Fill(eta, nLostStripTIBHits);
      nLostHitsTOBVsEtaH_->Fill(eta, nLostStripTOBHits);
      nLostHitsTECVsEtaH_->Fill(eta, nLostStripTECHits);
      nLostHitsTIDVsEtaH_->Fill(eta, nLostStripTIDHits);

      nLostHitsPixVsPhiH_->Fill(phi, nLostPixHits);
      nLostHitsPixBVsPhiH_->Fill(phi, nLostPixBHits);
      nLostHitsPixEVsPhiH_->Fill(phi, nLostPixEHits);
      nLostHitsStripVsPhiH_->Fill(phi, nLostStripHits);
      nLostHitsTIBVsPhiH_->Fill(phi, nLostStripTIBHits);
      nLostHitsTOBVsPhiH_->Fill(phi, nLostStripTOBHits);
      nLostHitsTECVsPhiH_->Fill(phi, nLostStripTECHits);
      nLostHitsTIDVsPhiH_->Fill(phi, nLostStripTIDHits);

      nLostHitsPixVsIterationH_->Fill(nIteration, nLostPixHits);
      nLostHitsPixBVsIterationH_->Fill(nIteration, nLostPixBHits);
      nLostHitsPixEVsIterationH_->Fill(nIteration, nLostPixEHits);
      nLostHitsStripVsIterationH_->Fill(nIteration, nLostStripHits);
      nLostHitsTIBVsIterationH_->Fill(nIteration, nLostStripTIBHits);
      nLostHitsTOBVsIterationH_->Fill(nIteration, nLostStripTOBHits);
      nLostHitsTECVsIterationH_->Fill(nIteration, nLostStripTECHits);
      nLostHitsTIDVsIterationH_->Fill(nIteration, nLostStripTIDHits);

      if (abs(eta) <= 1.4) {
        nMissingInnerHitBH_->Fill(missingInnerHit, wfac);
        nMissingOuterHitBH_->Fill(missingOuterHit, wfac);
      } else {
        nMissingInnerHitEH_->Fill(missingInnerHit, wfac);
        nMissingOuterHitEH_->Fill(missingOuterHit, wfac);
      }

      for (int i = 0; i < hitp.numberOfAllHits(reco::HitPattern::TRACK_HITS); i++) {
        uint32_t hit = hitp.getHitPattern(reco::HitPattern::TRACK_HITS, i);
        if (hitp.missingHitFilter(hit)) {
          double losthitBylayer = -1.0;
          double losthitBylayerPix = -1.0;
          double losthitBylayerStrip = -1.0;
          int layer = hitp.getLayer(hit);
          if (hitp.pixelBarrelHitFilter(hit)) {
            losthitBylayer = layer;
            losthitBylayerPix = layer;
          } else if (hitp.pixelEndcapHitFilter(hit)) {
            losthitBylayer = layer + 4;
            losthitBylayerPix = layer + 4;
          } else if (hitp.stripTIBHitFilter(hit)) {
            losthitBylayer = layer + 7;
            losthitBylayerStrip = layer;
          } else if (hitp.stripTIDHitFilter(hit)) {
            losthitBylayer = layer + 11;
            losthitBylayerStrip = layer + 4;
          } else if (hitp.stripTOBHitFilter(hit)) {
            losthitBylayer = layer + 14;
            losthitBylayerStrip = layer + 7;
          } else if (hitp.stripTECHitFilter(hit)) {
            losthitBylayer = layer + 20;
            losthitBylayerStrip = layer + 13;
          }
          if (losthitBylayer > -1)
            nLostHitByLayerH_->Fill(losthitBylayer, wfac);
          if (losthitBylayerPix > -1)
            nLostHitByLayerPixH_->Fill(losthitBylayerPix, wfac);
          if (losthitBylayerStrip > -1)
            nLostHitByLayerStripH_->Fill(losthitBylayerStrip, wfac);
        }
      }

      if (haveAllHistograms_) {
        double etaError = track.etaError();
        double thetaError = track.thetaError();
        double phiError = track.phiError();
        double p = track.p();
        double ptError = track.ptError();
        double qoverp = track.qoverp();
        double qoverpError = track.qoverpError();
        double charge = track.charge();

        double dxy = track.dxy(beamSpot->position());
        double dxyError = track.dxyError();
        double dz = track.dz(beamSpot->position());
        double dzError = track.dzError();

        double trkd0 = track.d0();
        double chi2 = track.chi2();
        double ndof = track.ndof();
        double chi2prob = TMath::Prob(track.chi2(), (int)track.ndof());
        double chi2oNDF = track.normalizedChi2();
        double vx = track.vx();
        double vy = track.vy();
        double vz = track.vz();
        //// algorithm
        unsigned int track_algo = track.algo();
        unsigned int track_origalgo = track.originalAlgo();
        // stopping source
        int ssmax = trackStoppingSourceH_->getNbinsX();
        double stop = track.stopReason() > ssmax ? double(ssmax - 1) : static_cast<double>(track.stopReason());
        double distanceOfClosestApproachToPV = track.dxy(pv.position());
        double xPointOfClosestApproachwrtPV = track.vx() - pv.position().x();
        double yPointOfClosestApproachwrtPV = track.vy() - pv.position().y();
        double positionZ0 = track.dz(pv.position());

        reco::TransientTrack transTrack = iSetup.get<TransientTrackRecord>().get(transTrackToken_).build(track);

        double ip3dToPV = 0, iperr3dToPV = 0, sip3dToPV = 0, sip2dToPV = 0;
        GlobalVector dir(track.px(), track.py(), track.pz());
        std::pair<bool, Measurement1D> ip3d = IPTools::signedImpactParameter3D(transTrack, dir, pv);
        std::pair<bool, Measurement1D> ip2d = IPTools::signedTransverseImpactParameter(transTrack, dir, pv);
        if (ip3d.first) {
          sip3dToPV = ip3d.second.value() / ip3d.second.error();
          ip3dToPV = ip3d.second.value();
          iperr3dToPV = ip3d.second.error();
        }

        double ip3dToBS = 0, iperr3dToBS = 0, sip3dToBS = 0, sip2dToBS = 0;
        reco::Vertex beamspotvertex((*beamSpot).position(), (*beamSpot).covariance3D());
        std::pair<bool, Measurement1D> ip3dbs = IPTools::signedImpactParameter3D(transTrack, dir, beamspotvertex);
        std::pair<bool, Measurement1D> ip2dbs =
            IPTools::signedTransverseImpactParameter(transTrack, dir, beamspotvertex);
        if (ip3dbs.first) {
          sip3dToBS = ip3dbs.second.value() / ip3dbs.second.error();
          ip3dToBS = ip3dbs.second.value();
          iperr3dToBS = ip3dbs.second.error();
        }

        double ip2dToPV = 0, iperr2dToPV = 0;
        if (ip2d.first) {
          ip2dToPV = ip2d.second.value();
          iperr2dToPV = ip2d.second.error();
          sip2dToPV = ip2d.second.value() / ip2d.second.error();
        }

        double ip2dToBS = 0, iperr2dToBS = 0;
        if (ip2dbs.first) {
          ip2dToBS = ip2dbs.second.value();
          iperr2dToBS = ip2dbs.second.error();
          sip2dToBS = ip2d.second.value() / ip2d.second.error();
        }

        if (ip2d.first)
          sip2dToPV = ip2d.second.value() / ip2d.second.error();
        double sipDxyToPV = track.dxy(pv.position()) / track.dxyError();
        double sipDzToPV = track.dz(pv.position()) / track.dzError();

        // Fill the histograms
        trackDeltaRwrtClosestTrack_->Fill(trackdeltaR, wfac);
        trackEtaH_->Fill(eta, wfac);
        trackEtaerrH_->Fill(etaError, wfac);
        trackCosThetaH_->Fill(std::cos(theta), wfac);
        trackThetaerrH_->Fill(thetaError, wfac);
        trackPhiH_->Fill(phi, wfac);
        trackPhierrH_->Fill(phiError, wfac);
        trackPH_->Fill(p, wfac);
        trackPtH_->Fill(pt, wfac);
        trackPt_ZoomH_->Fill(pt, wfac);
        trackPterrH_->Fill(ptError, wfac);
        trackqOverpH_->Fill(qoverp, wfac);
        trackqOverperrH_->Fill(qoverpError, wfac);
        trackChargeH_->Fill(charge, wfac);
        trackChi2H_->Fill(chi2, wfac);
        trackChi2ProbH_->Fill(chi2prob, wfac);
        trackChi2oNDFH_->Fill(chi2oNDF, wfac);
        trackd0H_->Fill(trkd0, wfac);
        tracknDOFH_->Fill(ndof, wfac);
        trackChi2bynDOFH_->Fill(chi2 / ndof, wfac);
        trackalgoH_->Fill(track_algo, wfac);
        trackorigalgoH_->Fill(track_origalgo, wfac);
        trackStoppingSourceH_->Fill(stop, wfac);
        trackChi2oNDFVsEtaH_->Fill(eta, chi2oNDF);
        trackChi2oNDFVsPhiH_->Fill(phi, chi2oNDF);
        trackChi2probVsEtaH_->Fill(eta, chi2prob);
        trackChi2probVsPhiH_->Fill(phi, chi2prob);

        nlostHitsH_->Fill(nLostHits, wfac);
        nMissingExpectedInnerHitsH_->Fill(nMissingExpectedInnerHits, wfac);
        nMissingExpectedOuterHitsH_->Fill(nMissingExpectedOuterHits, wfac);
        nlostTrackerHitsH_->Fill(nLostTrackerHits, wfac);

        beamSpotXYposH_->Fill(dxy, wfac);
        beamSpotXYposerrH_->Fill(dxyError, wfac);
        beamSpotZposH_->Fill(dz, wfac);
        beamSpotZposerrH_->Fill(dzError, wfac);

        vertexXposH_->Fill(vx, wfac);
        vertexYposH_->Fill(vy, wfac);
        vertexZposH_->Fill(vz, wfac);

        int nPixBarrel = 0, nPixEndcap = 0, nStripTIB = 0, nStripTOB = 0, nStripTEC = 0, nStripTID = 0;
        if (isRECO_) {
          for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
            const TrackingRecHit& hit = (**it);
            if (hit.isValid()) {
              if (hit.geographicalId().det() == DetId::Tracker) {
                int subdetId = hit.geographicalId().subdetId();
                if (subdetId == PixelSubdetector::PixelBarrel)
                  ++nPixBarrel;
                else if (subdetId == PixelSubdetector::PixelEndcap)
                  ++nPixEndcap;
                else if (subdetId == StripSubdetector::TIB)
                  ++nStripTIB;
                else if (subdetId == StripSubdetector::TOB)
                  ++nStripTOB;
                else if (subdetId == StripSubdetector::TEC)
                  ++nStripTEC;
                else if (subdetId == StripSubdetector::TID)
                  ++nStripTID;

                // Find on-track clusters
                processHit(hit, iSetup, tkGeom, wfac);
              }
            }
          }
        }
        if (verbose_)
          edm::LogInfo("StandaloneTrackMonitor")
              << " >>> HITs: nPixBarrel: " << nPixBarrel << " nPixEndcap: " << nPixEndcap << " nStripTIB: " << nStripTIB
              << " nStripTOB: " << nStripTOB << " nStripTEC: " << nStripTEC << " nStripTID: " << nStripTID;
        if (haveAllHistograms_) {
          nPixBarrelH_->Fill(nPixBarrel, wfac);
          nPixEndcapH_->Fill(nPixEndcap, wfac);
          nStripTIBH_->Fill(nStripTIB, wfac);
          nStripTOBH_->Fill(nStripTOB, wfac);
          nStripTECH_->Fill(nStripTEC, wfac);
          nStripTIDH_->Fill(nStripTID, wfac);
        }

        DistanceOfClosestApproachToPVH_->Fill(distanceOfClosestApproachToPV, wfac);
        DistanceOfClosestApproachToPVZoomedH_->Fill(distanceOfClosestApproachToPV, wfac);
        DistanceOfClosestApproachToPVVsPhiH_->Fill(phi, distanceOfClosestApproachToPV);
        xPointOfClosestApproachVsZ0wrtPVH_->Fill(positionZ0, xPointOfClosestApproachwrtPV);
        yPointOfClosestApproachVsZ0wrtPVH_->Fill(positionZ0, yPointOfClosestApproachwrtPV);

        ip3dToPVH_->Fill(ip3dToPV, wfac);
        iperr3dToPVH_->Fill(iperr3dToPV, wfac);
        ip3dToBSH_->Fill(ip3dToBS, wfac);
        iperr3dToBSH_->Fill(iperr3dToBS, wfac);
        ip2dToPVH_->Fill(ip2dToPV, wfac);
        iperr2dToPVH_->Fill(iperr2dToPV, wfac);
        iperr2dToPVWtH_->Fill(iperr2dToPV, wfac);
        ip2dToBSH_->Fill(ip2dToBS, wfac);
        iperr2dToBSH_->Fill(iperr2dToBS, wfac);

        iperr3dToPVWtH_->Fill(iperr3dToPV, wfac);
        sip3dToPVH_->Fill(sip3dToPV, wfac);
        sip2dToPVH_->Fill(sip2dToPV, wfac);
        sip3dToBSH_->Fill(sip3dToBS, wfac);
        sip2dToBSH_->Fill(sip2dToBS, wfac);
        sip2dToPVWtH_->Fill(sip2dToPV, wfac);
        sipDxyToPVH_->Fill(sipDxyToPV, wfac);
        sipDzToPVH_->Fill(sipDzToPV, wfac);

        if (nValidPixelHits >= 2) {
          ip3dToPV2validpixelhitsH_->Fill(ip3dToPV, wfac);
          iperr3dToPV2validpixelhitsH_->Fill(iperr3dToPV, wfac);
          ip3dToBS2validpixelhitsH_->Fill(ip3dToBS, wfac);
          iperr3dToBS2validpixelhitsH_->Fill(iperr3dToBS, wfac);
          sip3dToPV2validpixelhitsH_->Fill(sip3dToPV, wfac);
          sip3dToBS2validpixelhitsH_->Fill(sip3dToBS, wfac);
        }

        trackIperr3dVsEta2DH_->Fill(eta, iperr3dToPV, wfac);
        trackSip2dVsEta2DH_->Fill(eta, sip2dToPV, wfac);
        trackIperr3dVsChi2prob2DH_->Fill(chi2prob, iperr3dToPV, wfac);
        trackSip2dVsChi2prob2DH_->Fill(chi2prob, sip2dToPV, wfac);

        trackIperr3dVsEtaH_->Fill(eta, iperr3dToPV);
        trackSip2dVsEtaH_->Fill(eta, sip2dToPV);

        double trackerLayersWithMeasurement = hitp.trackerLayersWithMeasurement();
        double pixelLayersWithMeasurement = hitp.pixelLayersWithMeasurement();
        double pixelBLayersWithMeasurement = hitp.pixelBarrelLayersWithMeasurement();
        double pixelELayersWithMeasurement = hitp.pixelEndcapLayersWithMeasurement();
        double stripLayersWithMeasurement = hitp.stripLayersWithMeasurement();
        double stripTIBLayersWithMeasurement = hitp.stripTIBLayersWithMeasurement();
        double stripTOBLayersWithMeasurement = hitp.stripTOBLayersWithMeasurement();
        double stripTIDLayersWithMeasurement = hitp.stripTIDLayersWithMeasurement();
        double stripTECLayersWithMeasurement = hitp.stripTECLayersWithMeasurement();

        trkLayerwithMeasurementH_->Fill(trackerLayersWithMeasurement, wfac);
        pixelLayerwithMeasurementH_->Fill(pixelLayersWithMeasurement, wfac);
        pixelBLayerwithMeasurementH_->Fill(pixelBLayersWithMeasurement, wfac);
        pixelELayerwithMeasurementH_->Fill(pixelELayersWithMeasurement, wfac);
        stripLayerwithMeasurementH_->Fill(stripLayersWithMeasurement, wfac);
        stripTIBLayerwithMeasurementH_->Fill(stripTIBLayersWithMeasurement, wfac);
        stripTOBLayerwithMeasurementH_->Fill(stripTOBLayersWithMeasurement, wfac);
        stripTIDLayerwithMeasurementH_->Fill(stripTIDLayersWithMeasurement, wfac);
        stripTECLayerwithMeasurementH_->Fill(stripTECLayersWithMeasurement, wfac);

        nallHitsH_->Fill(nAllHits, wfac);
        ntrackerHitsH_->Fill(nAllTrackerHits, wfac);
        nvalidTrackerHitsH_->Fill(nValidTrackerHits, wfac);
        nvalidPixelHitsH_->Fill(nValidPixelHits, wfac);
        nvalidPixelBHitsH_->Fill(nValidPixelBHits, wfac);
        nvalidPixelEHitsH_->Fill(nValidPixelEHits, wfac);
        nvalidStripHitsH_->Fill(nValidStripHits, wfac);
        nvalidTIBHitsH_->Fill(nValidTIBHits, wfac);
        nvalidTOBHitsH_->Fill(nValidTOBHits, wfac);
        nvalidTIDHitsH_->Fill(nValidTIDHits, wfac);
        nvalidTECHitsH_->Fill(nValidTECHits, wfac);

        nlostTrackerHitsH_->Fill(nLostTrackerHits, wfac);
        nlostPixelHitsH_->Fill(nLostPixHits, wfac);
        nlostPixelBHitsH_->Fill(nLostPixBHits, wfac);
        nlostPixelEHitsH_->Fill(nLostPixEHits, wfac);
        nlostStripHitsH_->Fill(nLostStripHits, wfac);
        nlostTIBHitsH_->Fill(nLostStripTIBHits, wfac);
        nlostTOBHitsH_->Fill(nLostStripTOBHits, wfac);
        nlostTIDHitsH_->Fill(nLostStripTIDHits, wfac);
        nlostTECHitsH_->Fill(nLostStripTECHits, wfac);
      }
    }
  } else {
    edm::LogError("StandaloneTrackMonitor") << "Error! Failed to get reco::Track collection, " << trackTag_;
  }

  if (haveAllHistograms_) {
    nTracksH_->Fill(ntracks, wfac);
    edm::Handle<std::vector<reco::PFJet> > jetsColl;
    iEvent.getByToken(jetsToken_, jetsColl);
    nJet_->Fill(jetsColl->size());

    for (auto const& jet : *jetsColl) {
      Jet_pt_->Fill(jet.pt(), wfac);
      Jet_eta_->Fill(jet.eta(), wfac);
      Jet_energy_->Fill(jet.energy(), wfac);
      Jet_chargedMultiplicity_->Fill(jet.chargedHadronMultiplicity(), wfac);
    }
    if (list.size() >= 2) {
      TLorentzVector Zp = list[0] + list[1];
      Zpt_->Fill(Zp.Pt(), wfac);
      ZInvMass_->Fill(Zp.Mag(), wfac);
      KalmanVertexFitter kalman(true);
      mumuTransientVtx = kalman.vertex(tks);
      if (mumuTransientVtx.isValid()) {
        const reco::Vertex* theClosestVertex;
        edm::Handle<reco::VertexCollection> vertexHandle = iEvent.getHandle(vertexToken_);
        if (vertexHandle.isValid()) {
          const reco::VertexCollection* vertices = vertexHandle.product();
          theClosestVertex = this->findClosestVertex(mumuTransientVtx, vertices);
          reco::Vertex theMainVtx;
          if (theClosestVertex == nullptr) {
            theMainVtx = vertexHandle.product()->front();
          } else {
            theMainVtx = *theClosestVertex;
          }
          if (theMainVtx.isValid()) {
            const math::XYZPoint theMainVtxPos(
                theMainVtx.position().x(), theMainVtx.position().y(), theMainVtx.position().z());
            const math::XYZPoint myVertex(
                mumuTransientVtx.position().x(), mumuTransientVtx.position().y(), mumuTransientVtx.position().z());
            const math::XYZPoint deltaVtx(
                theMainVtxPos.x() - myVertex.x(), theMainVtxPos.y() - myVertex.y(), theMainVtxPos.z() - myVertex.z());
            double cosphi3D =
                (Zp.Px() * deltaVtx.x() + Zp.Py() * deltaVtx.y() + Zp.Pz() * deltaVtx.z()) /
                (sqrt(Zp.Px() * Zp.Px() + Zp.Py() * Zp.Py() + Zp.Pz() * Zp.Pz()) *
                 sqrt(deltaVtx.x() * deltaVtx.x() + deltaVtx.y() * deltaVtx.y() + deltaVtx.z() * deltaVtx.z()));
            cosPhi3DdileptonH_->Fill(cosphi3D, wfac);
          }
        }
      }
    }
  }

  // off track cluster properties
  processClusters(iEvent, iSetup, tkGeom, wfac);

  if (verbose_)
    edm::LogInfo("StandaloneTrackMonitor") << "Ends StandaloneTrackMonitor successfully";
}
void StandaloneTrackMonitor::processClusters(edm::Event const& iEvent,
                                             edm::EventSetup const& iSetup,
                                             const TrackerGeometry& tkGeom,
                                             double wfac) {
  // SiStripClusters
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusterHandle;
  iEvent.getByToken(clusterToken_, clusterHandle);

  if (clusterHandle.isValid()) {
    // Loop on Dets
    for (edmNew::DetSetVector<SiStripCluster>::const_iterator dsvit = clusterHandle->begin();
         dsvit != clusterHandle->end();
         ++dsvit) {
      uint32_t detId = dsvit->id();
      std::map<uint32_t, std::set<const SiStripCluster*> >::iterator jt = clusterMap_.find(detId);
      bool detid_found = (jt != clusterMap_.end()) ? true : false;

      // Loop on Clusters
      for (edmNew::DetSet<SiStripCluster>::const_iterator clusit = dsvit->begin(); clusit != dsvit->end(); ++clusit) {
        if (detid_found) {
          std::set<const SiStripCluster*>& s = jt->second;
          if (s.find(&*clusit) != s.end())
            continue;
        }

        siStripClusterInfo_.setCluster(*clusit, detId);
        float charge = siStripClusterInfo_.charge();
        float width = siStripClusterInfo_.width();

        const GeomDetUnit* detUnit = tkGeom.idToDetUnit(detId);
        float thickness = detUnit->surface().bounds().thickness();  // unit cm
        if (thickness > 0.035) {
          hOffTrkClusChargeThickH_->Fill(charge, wfac);
          hOffTrkClusWidthThickH_->Fill(width, wfac);
        } else {
          hOffTrkClusChargeThinH_->Fill(charge, wfac);
          hOffTrkClusWidthThinH_->Fill(width, wfac);
        }
      }
    }
  } else {
    edm::LogError("StandaloneTrackMonitor") << "ClusterCollection " << clusterTag_ << " not valid!!" << std::endl;
  }
}
void StandaloneTrackMonitor::processHit(const TrackingRecHit& recHit,
                                        edm::EventSetup const& iSetup,
                                        const TrackerGeometry& tkGeom,
                                        double wfac) {
  uint32_t detid = recHit.geographicalId();
  const GeomDetUnit* detUnit = tkGeom.idToDetUnit(detid);
  float thickness = detUnit->surface().bounds().thickness();  // unit cm

  auto const& thit = static_cast<BaseTrackerRecHit const&>(recHit);
  if (!thit.isValid())
    return;

  auto const& clus = thit.firstClusterRef();
  if (!clus.isValid())
    return;
  if (!clus.isStrip())
    return;

  if (thit.isMatched()) {
    const SiStripMatchedRecHit2D& matchedHit = dynamic_cast<const SiStripMatchedRecHit2D&>(recHit);

    auto& clusterM = matchedHit.monoCluster();
    siStripClusterInfo_.setCluster(clusterM, detid);

    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(siStripClusterInfo_.width(), wfac);
    } else {
      hOnTrkClusChargeThinH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(siStripClusterInfo_.width(), wfac);
    }
    addClusterToMap(detid, &clusterM);

    auto& clusterS = matchedHit.stereoCluster();

    siStripClusterInfo_.setCluster(clusterS, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(siStripClusterInfo_.width(), wfac);
    } else {
      hOnTrkClusChargeThinH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(siStripClusterInfo_.width(), wfac);
    }
    addClusterToMap(detid, &clusterS);
  } else {
    auto& cluster = clus.stripCluster();
    siStripClusterInfo_.setCluster(cluster, detid);
    if (thickness > 0.035) {
      hOnTrkClusChargeThickH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThickH_->Fill(siStripClusterInfo_.width(), wfac);
    } else {
      hOnTrkClusChargeThinH_->Fill(siStripClusterInfo_.charge(), wfac);
      hOnTrkClusWidthThinH_->Fill(siStripClusterInfo_.width(), wfac);
    }

    addClusterToMap(detid, &cluster);
  }
}
void StandaloneTrackMonitor::addClusterToMap(uint32_t detid, const SiStripCluster* cluster) {
  std::map<uint32_t, std::set<const SiStripCluster*> >::iterator it = clusterMap_.find(detid);
  if (it == clusterMap_.end()) {
    std::set<const SiStripCluster*> s;
    s.insert(cluster);
    clusterMap_.insert(std::pair<uint32_t, std::set<const SiStripCluster*> >(detid, s));
  } else {
    std::set<const SiStripCluster*>& s = it->second;
    s.insert(cluster);
  }
}

const reco::Vertex* StandaloneTrackMonitor::findClosestVertex(
    const TransientVertex aTransVtx,
    const reco::VertexCollection* vertices) const {  //same as in /DQMOffline/Alignment/src/DiMuonVertexSelector.cc
  reco::Vertex* defaultVtx = nullptr;

  if (!aTransVtx.isValid())
    return defaultVtx;

  // find the closest vertex to the secondary vertex in 3D
  VertexDistance3D vertTool3D;
  float minD = 9999.;
  int closestVtxIndex = 0;
  int counter = 0;
  for (const auto& vtx : *vertices) {
    double dist3D = vertTool3D.distance(aTransVtx, vtx).value();
    if (dist3D < minD) {
      minD = dist3D;
      closestVtxIndex = counter;
    }
    counter++;
  }

  if ((*vertices).at(closestVtxIndex).isValid()) {
    return &(vertices->at(closestVtxIndex));
  } else {
    return defaultVtx;
  }
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StandaloneTrackMonitor);
