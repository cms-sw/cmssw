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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class BeamSpot;
class TrackCollection;
class VertexCollection;
class TrackingRecHit;

class StandaloneTrackMonitor : public DQMEDAnalyzer {
public:
  StandaloneTrackMonitor( const edm::ParameterSet& );

protected:

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void processHit(const TrackingRecHit& recHit, edm::EventSetup const& iSetup, const TrackerGeometry& tkGeom, double wfac=1);
  void processClusters(edm::Event const& iEvent, edm::EventSetup const& iSetup, const TrackerGeometry& tkGeom, double wfac=1);
  void addClusterToMap(uint32_t detid, const SiStripCluster* cluster);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

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
  MonitorElement* trackChi2ProbH_;
  MonitorElement* trackChi2oNDFH_;
  MonitorElement* trackd0H_;
  MonitorElement* trackChi2bynDOFH_;

  MonitorElement* DistanceOfClosestApproachToPVH_;
  MonitorElement* DistanceOfClosestApproachToPVVsPhiH_;
  MonitorElement* xPointOfClosestApproachVsZ0wrtPVH_;
  MonitorElement* yPointOfClosestApproachVsZ0wrtPVH_;

  MonitorElement* sip3dToPVH_;
  MonitorElement* sip2dToPVH_;
  MonitorElement* sipDxyToPVH_;
  MonitorElement* sipDzToPVH_;

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

  MonitorElement* residualXPBH_;
  MonitorElement* residualXPEH_;
  MonitorElement* residualXTIBH_;
  MonitorElement* residualXTOBH_;
  MonitorElement* residualXTIDH_;
  MonitorElement* residualXTECH_;
  MonitorElement* residualYPBH_;
  MonitorElement* residualYPEH_;
  MonitorElement* residualYTIBH_;
  MonitorElement* residualYTOBH_;
  MonitorElement* residualYTIDH_;
  MonitorElement* residualYTECH_;

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

  MonitorElement* beamSpotXYposH_;
  MonitorElement* beamSpotXYposerrH_;
  MonitorElement* beamSpotZposH_;
  MonitorElement* beamSpotZposerrH_;

  MonitorElement* vertexXposH_;
  MonitorElement* vertexYposH_;
  MonitorElement* vertexZposH_;
  MonitorElement* nVertexH_;
  MonitorElement* nVtxH_;

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

  MonitorElement* nLostHitsPixVsPhiH_;
  MonitorElement* nLostHitsPixBVsPhiH_;
  MonitorElement* nLostHitsPixEVsPhiH_;
  MonitorElement* nLostHitsStripVsPhiH_;
  MonitorElement* nLostHitsTIBVsPhiH_;
  MonitorElement* nLostHitsTOBVsPhiH_;
  MonitorElement* nLostHitsTECVsPhiH_;
  MonitorElement* nLostHitsTIDVsPhiH_;


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
