#ifndef DQM_TrackingMonitorSource_StandaloneTrackMonitor_h
#define DQM_TrackingMonitorSource_StandaloneTrackMonitor_h

#include <string>
#include <vector>
#include <map>
#include <set>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"

class TrackingRecHit;
class SiStripCluster;
class PileupSummaryInfo;

class StandaloneTrackMonitor : public DQMEDAnalyzer {
public:
  StandaloneTrackMonitor(const edm::ParameterSet&);

protected:
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void processHit(const TrackingRecHit& recHit,
                  edm::EventSetup const& iSetup,
                  const TrackerGeometry& tkGeom,
                  double wfac = 1);
  void processClusters(edm::Event const& iEvent,
                       edm::EventSetup const& iSetup,
                       const TrackerGeometry& tkGeom,
                       double wfac = 1);
  void addClusterToMap(uint32_t detid, const SiStripCluster* cluster);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  edm::ParameterSet parameters_;

  std::string moduleName_;
  std::string folderName_;
  const edm::InputTag trackTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag vertexTag_;
  const edm::InputTag puSummaryTag_;
  const edm::InputTag clusterTag_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  const edm::EDGetTokenT<std::vector<PileupSummaryInfo> > puSummaryToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken_;

  const std::string trackQuality_;
  SiStripClusterInfo siStripClusterInfo_;
  const bool doPUCorrection_;
  const bool isMC_;
  const bool haveAllHistograms_;
  const std::string puScaleFactorFile_;
  const bool verbose_;

  MonitorElement* trackEtaH_;
  MonitorElement* trackEtaerrH_;
  MonitorElement* trackCosThetaH_;
  MonitorElement* trackThetaerrH_;
  MonitorElement* trackPhiH_;
  MonitorElement* trackPhierrH_;
  MonitorElement* trackPH_;
  MonitorElement* trackPtH_;
  MonitorElement* trackPtUpto2GeVH_;
  MonitorElement* trackPtOver10GeVH_;
  MonitorElement* trackPterrH_;
  MonitorElement* trackqOverpH_;
  MonitorElement* trackqOverperrH_;
  MonitorElement* trackChargeH_;
  MonitorElement* trackChi2H_;
  MonitorElement* tracknDOFH_;
  MonitorElement* trackd0H_;
  MonitorElement* trackChi2bynDOFH_;

  MonitorElement* nlostHitsH_;
  MonitorElement* nvalidTrackerHitsH_;
  MonitorElement* nvalidPixelHitsH_;
  MonitorElement* nvalidStripHitsH_;
  MonitorElement* trkLayerwithMeasurementH_;
  MonitorElement* pixelLayerwithMeasurementH_;
  MonitorElement* stripLayerwithMeasurementH_;

  MonitorElement* beamSpotXYposH_;
  MonitorElement* beamSpotXYposerrH_;
  MonitorElement* beamSpotZposH_;
  MonitorElement* beamSpotZposerrH_;

  MonitorElement* vertexXposH_;
  MonitorElement* vertexYposH_;
  MonitorElement* vertexZposH_;
  MonitorElement* nVertexH_;

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
  MonitorElement* nHitsVspTH_;
  MonitorElement* nHitsVsnVtxH_;
  MonitorElement* nHitsVsEtaH_;
  MonitorElement* nHitsVsCosThetaH_;
  MonitorElement* nHitsVsPhiH_;
  MonitorElement* nLostHitsVspTH_;
  MonitorElement* nLostHitsVsEtaH_;
  MonitorElement* nLostHitsVsCosThetaH_;
  MonitorElement* nLostHitsVsPhiH_;

  MonitorElement* hOnTrkClusChargeThinH_;
  MonitorElement* hOnTrkClusWidthThinH_;
  MonitorElement* hOnTrkClusChargeThickH_;
  MonitorElement* hOnTrkClusWidthThickH_;

  MonitorElement* hOffTrkClusChargeThinH_;
  MonitorElement* hOffTrkClusWidthThinH_;
  MonitorElement* hOffTrkClusChargeThickH_;
  MonitorElement* hOffTrkClusWidthThickH_;

  unsigned long long m_cacheID_;

  std::vector<float> vpu_;
  std::map<uint32_t, std::set<const SiStripCluster*> > clusterMap_;
};
#endif
