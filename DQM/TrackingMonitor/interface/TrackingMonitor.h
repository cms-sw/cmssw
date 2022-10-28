#ifndef TrackingMonitor_H
#define TrackingMonitor_H
// -*- C++ -*-
//
// Package:    TrackingMonitor
// Class:      TrackingMonitor
//
/**\class TrackingMonitor TrackingMonitor.cc DQM/TrackerMonitorTrack/src/TrackingMonitor.cc
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Suchandra Dutta, Giorgia Mila
//         Created:  Thu 28 22:45:30 CEST 2008

#include <memory>
#include <fstream>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

#include "DataFormats/TrackReco/interface/SeedStopInfo.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionsSeedingLayerSets.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

class TrackAnalyzer;
class TrackBuildingAnalyzer;
class VertexMonitor;
class GetLumi;
class TProfile;
class GenericTriggerEventFlag;

class TrackingMonitor : public DQMEDAnalyzer {
public:
  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  explicit TrackingMonitor(const edm::ParameterSet&);
  ~TrackingMonitor() override;

  virtual void setMaxMinBin(
      std::vector<double>&, std::vector<double>&, std::vector<int>&, double, double, int, double, double, int);
  virtual void setNclus(const edm::Event&, std::vector<int>&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  void doProfileX(TH2* th2, MonitorElement* me);
  void doProfileX(MonitorElement* th2m, MonitorElement* me);

  // ----------member data ---------------------------

  std::string MEFolderName;
  std::string histname;  //for naming the histograms according to algorithm used

  //        DQMStore * dqmStore_;

  edm::ParameterSetID confID_;

  // the track analyzer
  edm::InputTag bsSrc_;
  edm::InputTag pvSrc_;
  edm::EDGetTokenT<reco::BeamSpot> bsSrcToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvSrcToken_;

  edm::EDGetTokenT<edm::View<reco::Track> > allTrackToken_;
  edm::EDGetTokenT<edm::View<reco::Track> > trackToken_;
  edm::EDGetTokenT<TrackCandidateCollection> trackCandidateToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken_;
  edm::EDGetTokenT<std::vector<SeedStopInfo> > seedStopInfoToken_;
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;
  edm::EDGetTokenT<TrackingRegionsSeedingLayerSets> regionLayerSetsToken_;
  edm::EDGetTokenT<reco::CandidateView> regionCandidateToken_;

  edm::EDGetTokenT<LumiScalersCollection> lumiscalersToken_;
  edm::EDGetTokenT<OnlineLuminosityRecord> metaDataToken_;

  edm::InputTag stripClusterInputTag_;
  edm::InputTag pixelClusterInputTag_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;

  std::vector<std::tuple<edm::EDGetTokenT<MVACollection>, edm::EDGetTokenT<QualityMaskCollection> > > mvaQualityTokens_;
  edm::EDGetTokenT<edm::View<reco::Track> > mvaTrackToken_;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderToken_;

  std::string Quality_;
  std::string AlgoName_;

  tadqm::TrackAnalyzer* theTrackAnalyzer;
  TrackBuildingAnalyzer* theTrackBuildingAnalyzer;
  std::vector<VertexMonitor*> theVertexMonitor;
  GetLumi* theLumiDetails_;

  // Tracks
  MonitorElement* NumberOfTracks;
  MonitorElement* NumberOfTracks_PUvtx;
  MonitorElement* NumberofTracks_Hardvtx;
  MonitorElement* NumberofTracks_Hardvtx_PUvtx;
  MonitorElement* NumberOfMeanRecHitsPerTrack;
  MonitorElement* NumberOfMeanLayersPerTrack;

  // Good Tracks
  MonitorElement* FractionOfGoodTracks;

  // Tracking regions
  MonitorElement* NumberOfTrackingRegions;

  // Track Seeds
  MonitorElement* NumberOfSeeds;
  MonitorElement* NumberOfSeeds_lumiFlag;
  std::vector<MonitorElement*> SeedsVsClusters;
  std::vector<std::string> ClusterLabels;

  // Track Candidates
  MonitorElement* NumberOfTrackCandidates;
  MonitorElement* FractionCandidatesOverSeeds;

  // Cluster Properties
  std::vector<MonitorElement*> NumberOfTrkVsClusters;
  MonitorElement* NumberOfTrkVsClus;
  MonitorElement* NumberOfTrkVsStripClus;
  MonitorElement* NumberOfTrkVsPixelClus;

  // Monitoring vs LS
  MonitorElement* NumberEventsOfVsLS;
  MonitorElement* NumberOfTracksVsLS;
  MonitorElement* GoodTracksFractionVsLS;
  MonitorElement* NumberOfRecHitsPerTrackVsLS;
  MonitorElement* NumberOfGoodPVtxVsLS;
  MonitorElement* NumberOfGoodPVtxWO0VsLS;

  // Monitoring vs BX
  MonitorElement* NumberEventsOfVsBX;
  MonitorElement* NumberOfTracksVsBX;
  MonitorElement* GoodTracksFractionVsBX;
  MonitorElement* NumberOfRecHitsPerTrackVsBX;
  MonitorElement* NumberOfGoodPVtxVsBX;
  MonitorElement* NumberOfGoodPVtxWO0VsBX;

  MonitorElement* NumberOfTracksVsBXlumi;

  // Monitoring PU
  MonitorElement* NumberOfTracksVsGoodPVtx;
  MonitorElement* NumberOfTracksVsPUPVtx;
  MonitorElement* NumberEventsOfVsGoodPVtx;
  MonitorElement* GoodTracksFractionVsGoodPVtx;
  MonitorElement* NumberOfRecHitsPerTrackVsGoodPVtx;
  MonitorElement* NumberOfPVtxVsGoodPVtx;
  MonitorElement* NumberOfPixelClustersVsGoodPVtx;
  MonitorElement* NumberOfStripClustersVsGoodPVtx;

  // Monitoring vs lumi
  MonitorElement* NumberEventsOfVsLUMI;
  MonitorElement* NumberOfTracksVsLUMI;
  MonitorElement* GoodTracksFractionVsLUMI;
  MonitorElement* NumberOfRecHitsPerTrackVsLUMI;
  MonitorElement* NumberOfGoodPVtxVsLUMI;
  MonitorElement* NumberOfGoodPVtxWO0VsLUMI;
  MonitorElement* NumberOfPixelClustersVsLUMI;
  MonitorElement* NumberOfStripClustersVsLUMI;

  // add in order to deal with LS transitions
  MonitorElement* NumberOfTracks_lumiFlag;

  std::string builderName;

  bool doTrackerSpecific_;
  bool doLumiAnalysis;
  bool doProfilesVsLS_;
  bool doAllSeedPlots;
  bool doAllPlots;
  bool doDCAPlots_;
  bool doGeneralPropertiesPlots_;
  bool doHitPropertiesPlots_;
  bool doTkCandPlots;
  bool doMVAPlots;
  bool doRegionPlots;
  bool doRegionCandidatePlots;
  bool doSeedNumberPlot;
  bool doSeedLumiAnalysis_;
  bool doSeedVsClusterPlot;
  bool runTrackBuildingAnalyzerForSeed;
  // ADD by Mia in order to have GoodTrack plots only for collision
  bool doPUmonitoring_;
  bool doPlotsVsBXlumi_;
  bool doPlotsVsGoodPVtx_;
  bool doPlotsVsLUMI_;
  bool doPlotsVsBX_;
  bool doFractionPlot_;

  GenericTriggerEventFlag* genTriggerEventFlag_;

  StringCutObjectSelector<reco::Track, true> numSelection_;
  StringCutObjectSelector<reco::Track, true> denSelection_;
  int pvNDOF_;
  const bool forceSCAL_;
};

#endif  //define TrackingMonitor_H
