#ifndef TrackBuildingAnalyzer_H
#define TrackBuildingAnalyzer_H
// -*- C++ -*-
//
//
/**\class TrackBuildingAnalyzer TrackBuildingAnalyzer.cc 
Monitoring source for general quantities related to tracks.
*/
// Original Author:  Ryan Kelley
//         Created:  Sat 28 13;30:00 CEST 2009
//

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionsSeedingLayerSets.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

class TrackBuildingAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  TrackBuildingAnalyzer(const edm::ParameterSet&);
  ~TrackBuildingAnalyzer() = default;
  void initHisto(DQMStore::IBooker& ibooker, const edm::ParameterSet&);
  void analyze(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               const TrajectorySeed& seed,
               const SeedStopInfo& stopInfo,
               const reco::BeamSpot& bs,
               const MagneticField& theMF,
               const TransientTrackingRecHitBuilder& theTTRHBuilder);
  void analyze(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               const TrackCandidate& candidate,
               const reco::BeamSpot& bs,
               const MagneticField& theMF,
               const TransientTrackingRecHitBuilder& theTTRHBuilder);
  void analyze(const edm::View<reco::Track>& trackCollection,
               const std::vector<const MVACollection*>& mvaCollections,
               const std::vector<const QualityMaskCollection*>& qualityMaskCollections);
  void analyze(const reco::CandidateView& regionCandidates);
  void analyze(const edm::OwnVector<TrackingRegion>& regions);
  void analyze(const TrackingRegionsSeedingLayerSets& regions);

private:
  void fillHistos(const edm::EventSetup& iSetup, const reco::Track& track, std::string sname);
  void bookHistos(std::string sname, DQMStore::IBooker& ibooker);

  template <typename T>
  void analyzeRegions(const T& regions);

  // ----------member data ---------------------------

  // Regions covered by tracking regions
  MonitorElement* TrackingRegionEta = nullptr;
  MonitorElement* TrackingRegionPhi = nullptr;
  MonitorElement* TrackingRegionPhiVsEta = nullptr;
  double etaBinWidth = 0.;
  double phiBinWidth = 0.;
  // Candidates used for tracking regions
  MonitorElement* TrackingRegionCandidatePt = nullptr;
  MonitorElement* TrackingRegionCandidateEta = nullptr;
  MonitorElement* TrackingRegionCandidatePhi = nullptr;
  MonitorElement* TrackingRegionCandidatePhiVsEta = nullptr;

  // Track Seeds
  MonitorElement* SeedPt = nullptr;
  MonitorElement* SeedEta = nullptr;
  MonitorElement* SeedPhi = nullptr;
  MonitorElement* SeedPhiVsEta = nullptr;
  MonitorElement* SeedTheta = nullptr;
  MonitorElement* SeedQ = nullptr;
  MonitorElement* SeedDxy = nullptr;
  MonitorElement* SeedDz = nullptr;
  MonitorElement* NumberOfRecHitsPerSeed = nullptr;
  MonitorElement* NumberOfRecHitsPerSeedVsPhiProfile = nullptr;
  MonitorElement* NumberOfRecHitsPerSeedVsEtaProfile = nullptr;

  MonitorElement* seedStoppingSource = nullptr;
  MonitorElement* seedStoppingSourceVsPhi = nullptr;
  MonitorElement* seedStoppingSourceVsEta = nullptr;

  MonitorElement* numberOfTrajCandsPerSeed = nullptr;
  MonitorElement* numberOfTrajCandsPerSeedVsPhi = nullptr;
  MonitorElement* numberOfTrajCandsPerSeedVsEta = nullptr;

  MonitorElement* seedStoppingSourceVsNumberOfTrajCandsPerSeed = nullptr;

  // Track Candidate
  MonitorElement* TrackCandPt = nullptr;
  MonitorElement* TrackCandEta = nullptr;
  MonitorElement* TrackCandPhi = nullptr;
  MonitorElement* TrackCandPhiVsEta = nullptr;
  MonitorElement* TrackCandTheta = nullptr;
  MonitorElement* TrackCandQ = nullptr;
  MonitorElement* TrackCandDxy = nullptr;
  MonitorElement* TrackCandDz = nullptr;
  MonitorElement* NumberOfRecHitsPerTrackCand = nullptr;
  MonitorElement* NumberOfRecHitsPerTrackCandVsPhiProfile = nullptr;
  MonitorElement* NumberOfRecHitsPerTrackCandVsEtaProfile = nullptr;

  MonitorElement* stoppingSource = nullptr;
  MonitorElement* stoppingSourceVSeta = nullptr;
  MonitorElement* stoppingSourceVSphi = nullptr;

  std::vector<MonitorElement*> trackMVAs;
  std::vector<MonitorElement*> trackMVAsHP;
  std::vector<MonitorElement*> trackMVAsVsPtProfile;
  std::vector<MonitorElement*> trackMVAsHPVsPtProfile;
  std::vector<MonitorElement*> trackMVAsVsEtaProfile;
  std::vector<MonitorElement*> trackMVAsHPVsEtaProfile;

  std::string histname;  //for naming the histograms according to algorithm used

  //to disable some plots
  const bool doAllPlots;
  const bool doAllSeedPlots;
  const bool doTCPlots;
  const bool doAllTCPlots;
  const bool doPT;
  const bool doETA;
  const bool doPHI;
  const bool doPHIVsETA;
  const bool doTheta;
  const bool doQ;
  const bool doDxy;
  const bool doDz;
  const bool doNRecHits;
  const bool doProfPHI;
  const bool doProfETA;
  const bool doStopSource;
  const bool doMVAPlots;
  const bool doRegionPlots;
  const bool doRegionCandidatePlots;
};
#endif
