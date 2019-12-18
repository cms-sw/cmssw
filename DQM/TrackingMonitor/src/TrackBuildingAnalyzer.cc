#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackBuildingAnalyzer.h"
#include <string>
#include "TMath.h"

#include <iostream>

TrackBuildingAnalyzer::TrackBuildingAnalyzer(const edm::ParameterSet& iConfig)
    : doAllPlots(iConfig.getParameter<bool>("doAllPlots")),
      doAllSeedPlots(iConfig.getParameter<bool>("doSeedParameterHistos")),
      doTCPlots(iConfig.getParameter<bool>("doTrackCandHistos")),
      doAllTCPlots(iConfig.getParameter<bool>("doAllTrackCandHistos")),
      doPT(iConfig.getParameter<bool>("doSeedPTHisto")),
      doETA(iConfig.getParameter<bool>("doSeedETAHisto")),
      doPHI(iConfig.getParameter<bool>("doSeedPHIHisto")),
      doPHIVsETA(iConfig.getParameter<bool>("doSeedPHIVsETAHisto")),
      doTheta(iConfig.getParameter<bool>("doSeedThetaHisto")),
      doQ(iConfig.getParameter<bool>("doSeedQHisto")),
      doDxy(iConfig.getParameter<bool>("doSeedDxyHisto")),
      doDz(iConfig.getParameter<bool>("doSeedDzHisto")),
      doNRecHits(iConfig.getParameter<bool>("doSeedNRecHitsHisto")),
      doProfPHI(iConfig.getParameter<bool>("doSeedNVsPhiProf")),
      doProfETA(iConfig.getParameter<bool>("doSeedNVsEtaProf")),
      doStopSource(iConfig.getParameter<bool>("doStopSource")),
      doMVAPlots(iConfig.getParameter<bool>("doMVAPlots")),
      doRegionPlots(iConfig.getParameter<bool>("doRegionPlots")),
      doRegionCandidatePlots(iConfig.getParameter<bool>("doRegionCandidatePlots")) {}

void TrackBuildingAnalyzer::initHisto(DQMStore::IBooker& ibooker, const edm::ParameterSet& iConfig) {
  // parameters from the configuration
  std::string AlgoName = iConfig.getParameter<std::string>("AlgoName");
  std::string MEFolderName = iConfig.getParameter<std::string>("FolderName");

  //  std::cout << "[TrackBuildingAnalyzer::beginRun] AlgoName: " << AlgoName << std::endl;

  // use the AlgoName and Quality Name
  const std::string& CatagoryName = AlgoName;

  // get binning from the configuration
  int TrackPtBin = iConfig.getParameter<int>("TrackPtBin");
  double TrackPtMin = iConfig.getParameter<double>("TrackPtMin");
  double TrackPtMax = iConfig.getParameter<double>("TrackPtMax");

  int PhiBin = iConfig.getParameter<int>("PhiBin");
  double PhiMin = iConfig.getParameter<double>("PhiMin");
  double PhiMax = iConfig.getParameter<double>("PhiMax");
  phiBinWidth = PhiBin > 0 ? (PhiMax - PhiMin) / PhiBin : 0.;

  int EtaBin = iConfig.getParameter<int>("EtaBin");
  double EtaMin = iConfig.getParameter<double>("EtaMin");
  double EtaMax = iConfig.getParameter<double>("EtaMax");
  etaBinWidth = EtaBin > 0 ? (EtaMax - EtaMin) / EtaBin : 0.;

  int ThetaBin = iConfig.getParameter<int>("ThetaBin");
  double ThetaMin = iConfig.getParameter<double>("ThetaMin");
  double ThetaMax = iConfig.getParameter<double>("ThetaMax");

  int TrackQBin = iConfig.getParameter<int>("TrackQBin");
  double TrackQMin = iConfig.getParameter<double>("TrackQMin");
  double TrackQMax = iConfig.getParameter<double>("TrackQMax");

  int SeedDxyBin = iConfig.getParameter<int>("SeedDxyBin");
  double SeedDxyMin = iConfig.getParameter<double>("SeedDxyMin");
  double SeedDxyMax = iConfig.getParameter<double>("SeedDxyMax");

  int SeedDzBin = iConfig.getParameter<int>("SeedDzBin");
  double SeedDzMin = iConfig.getParameter<double>("SeedDzMin");
  double SeedDzMax = iConfig.getParameter<double>("SeedDzMax");

  int SeedHitBin = iConfig.getParameter<int>("SeedHitBin");
  double SeedHitMin = iConfig.getParameter<double>("SeedHitMin");
  double SeedHitMax = iConfig.getParameter<double>("SeedHitMax");

  int TCDxyBin = iConfig.getParameter<int>("TCDxyBin");
  double TCDxyMin = iConfig.getParameter<double>("TCDxyMin");
  double TCDxyMax = iConfig.getParameter<double>("TCDxyMax");

  int TCDzBin = iConfig.getParameter<int>("TCDzBin");
  double TCDzMin = iConfig.getParameter<double>("TCDzMin");
  double TCDzMax = iConfig.getParameter<double>("TCDzMax");

  int TCHitBin = iConfig.getParameter<int>("TCHitBin");
  double TCHitMin = iConfig.getParameter<double>("TCHitMin");
  double TCHitMax = iConfig.getParameter<double>("TCHitMax");

  int MVABin = iConfig.getParameter<int>("MVABin");
  double MVAMin = iConfig.getParameter<double>("MVAMin");
  double MVAMax = iConfig.getParameter<double>("MVAMax");

  edm::InputTag seedProducer = iConfig.getParameter<edm::InputTag>("SeedProducer");
  edm::InputTag tcProducer = iConfig.getParameter<edm::InputTag>("TCProducer");
  std::vector<std::string> mvaProducers = iConfig.getParameter<std::vector<std::string> >("MVAProducers");
  edm::InputTag regionProducer = iConfig.getParameter<edm::InputTag>("RegionProducer");

  //    if (doAllPlots){doAllSeedPlots=true; doTCPlots=true;}

  ibooker.setCurrentFolder(MEFolderName);

  // book the Seed histograms
  // ---------------------------------------------------------------------------------//
  //  std::cout << "[TrackBuildingAnalyzer::beginRun] MEFolderName: " << MEFolderName << std::endl;
  ibooker.setCurrentFolder(MEFolderName + "/TrackBuilding");

  if (doAllSeedPlots || doPT) {
    histname = "SeedPt_" + seedProducer.label() + "_";
    SeedPt = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TrackPtBin, TrackPtMin, TrackPtMax);
    SeedPt->setAxisTitle("Seed p_{T} (GeV/c)", 1);
    SeedPt->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doETA) {
    histname = "SeedEta_" + seedProducer.label() + "_";
    SeedEta = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax);
    SeedEta->setAxisTitle("Seed #eta", 1);
    SeedEta->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doPHI) {
    histname = "SeedPhi_" + seedProducer.label() + "_";
    SeedPhi = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax);
    SeedPhi->setAxisTitle("Seed #phi", 1);
    SeedPhi->setAxisTitle("Number of Seed", 2);
  }

  if (doAllSeedPlots || doPHIVsETA) {
    histname = "SeedPhiVsEta_" + seedProducer.label() + "_";
    SeedPhiVsEta = ibooker.book2D(
        histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax);
    SeedPhiVsEta->setAxisTitle("Seed #eta", 1);
    SeedPhiVsEta->setAxisTitle("Seed #phi", 2);
  }

  if (doAllSeedPlots || doTheta) {
    histname = "SeedTheta_" + seedProducer.label() + "_";
    SeedTheta = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, ThetaBin, ThetaMin, ThetaMax);
    SeedTheta->setAxisTitle("Seed #theta", 1);
    SeedTheta->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doQ) {
    histname = "SeedQ_" + seedProducer.label() + "_";
    SeedQ = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TrackQBin, TrackQMin, TrackQMax);
    SeedQ->setAxisTitle("Seed Charge", 1);
    SeedQ->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doDxy) {
    histname = "SeedDxy_" + seedProducer.label() + "_";
    SeedDxy = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, SeedDxyBin, SeedDxyMin, SeedDxyMax);
    SeedDxy->setAxisTitle("Seed d_{xy} (cm)", 1);
    SeedDxy->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doDz) {
    histname = "SeedDz_" + seedProducer.label() + "_";
    SeedDz = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, SeedDzBin, SeedDzMin, SeedDzMax);
    SeedDz->setAxisTitle("Seed d_{z} (cm)", 1);
    SeedDz->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doNRecHits) {
    histname = "NumberOfRecHitsPerSeed_" + seedProducer.label() + "_";
    NumberOfRecHitsPerSeed =
        ibooker.book1D(histname + CatagoryName, histname + CatagoryName, SeedHitBin, SeedHitMin, SeedHitMax);
    NumberOfRecHitsPerSeed->setAxisTitle("Number of RecHits per Seed", 1);
    NumberOfRecHitsPerSeed->setAxisTitle("Number of Seeds", 2);
  }

  if (doAllSeedPlots || doProfPHI) {
    histname = "NumberOfRecHitsPerSeedVsPhiProfile_" + seedProducer.label() + "_";
    NumberOfRecHitsPerSeedVsPhiProfile = ibooker.bookProfile(histname + CatagoryName,
                                                             histname + CatagoryName,
                                                             PhiBin,
                                                             PhiMin,
                                                             PhiMax,
                                                             SeedHitBin,
                                                             SeedHitMin,
                                                             SeedHitMax,
                                                             "s");
    NumberOfRecHitsPerSeedVsPhiProfile->setAxisTitle("Seed #phi", 1);
    NumberOfRecHitsPerSeedVsPhiProfile->setAxisTitle("Number of RecHits of each Seed", 2);
  }

  if (doAllSeedPlots || doProfETA) {
    histname = "NumberOfRecHitsPerSeedVsEtaProfile_" + seedProducer.label() + "_";
    NumberOfRecHitsPerSeedVsEtaProfile = ibooker.bookProfile(histname + CatagoryName,
                                                             histname + CatagoryName,
                                                             EtaBin,
                                                             EtaMin,
                                                             EtaMax,
                                                             SeedHitBin,
                                                             SeedHitMin,
                                                             SeedHitMax,
                                                             "s");
    NumberOfRecHitsPerSeedVsEtaProfile->setAxisTitle("Seed #eta", 1);
    NumberOfRecHitsPerSeedVsEtaProfile->setAxisTitle("Number of RecHits of each Seed", 2);
  }

  if (doRegionPlots) {
    if (doAllSeedPlots || doETA) {
      histname = "TrackingRegionEta_" + seedProducer.label() + "_";
      TrackingRegionEta = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax);
      TrackingRegionEta->setAxisTitle("TrackingRegion-covered #eta", 1);
      TrackingRegionEta->setAxisTitle("Number of TrackingRegions", 2);
    }

    if (doAllSeedPlots || doPHI) {
      histname = "TrackingRegionPhi_" + seedProducer.label() + "_";
      TrackingRegionPhi = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax);
      TrackingRegionPhi->setAxisTitle("TrackingRegion-covered #phi", 1);
      TrackingRegionPhi->setAxisTitle("Number of TrackingRegions", 2);
    }

    if (doAllSeedPlots || doPHIVsETA) {
      histname = "TrackingRegionPhiVsEta_" + seedProducer.label() + "_";
      TrackingRegionPhiVsEta = ibooker.book2D(
          histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax);
      TrackingRegionPhiVsEta->setAxisTitle("TrackingRegion-covered #eta", 1);
      TrackingRegionPhiVsEta->setAxisTitle("TrackingRegion-covered #phi", 2);
    }

    if (doRegionCandidatePlots) {
      if (doAllSeedPlots || doPT) {
        auto ptBin = iConfig.getParameter<int>("RegionCandidatePtBin");
        auto ptMin = iConfig.getParameter<double>("RegionCandidatePtMin");
        auto ptMax = iConfig.getParameter<double>("RegionCandidatePtMax");

        histname = "TrackingRegionCandidatePt_" + seedProducer.label() + "_";
        TrackingRegionCandidatePt =
            ibooker.book1D(histname + CatagoryName, histname + CatagoryName, ptBin, ptMin, ptMax);
        TrackingRegionCandidatePt->setAxisTitle("TrackingRegion Candidate p_{T} (GeV/c)", 1);
        TrackingRegionCandidatePt->setAxisTitle("Number of TrackingRegion Candidates", 2);
      }

      if (doAllSeedPlots || doETA) {
        histname = "TrackingRegionCandidateEta_" + seedProducer.label() + "_";
        TrackingRegionCandidateEta =
            ibooker.book1D(histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax);
        TrackingRegionCandidateEta->setAxisTitle("TrackingRegion Candidate #eta", 1);
        TrackingRegionCandidateEta->setAxisTitle("Number of TrackingRegion Candidates", 2);
      }

      if (doAllSeedPlots || doPHI) {
        histname = "TrackingRegionCandidatePhi_" + seedProducer.label() + "_";
        TrackingRegionCandidatePhi =
            ibooker.book1D(histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax);
        TrackingRegionCandidatePhi->setAxisTitle("TrackingRegion Candidate #phi", 1);
        TrackingRegionCandidatePhi->setAxisTitle("Number of TrackingRegion Candidates", 2);
      }

      if (doAllSeedPlots || doPHIVsETA) {
        histname = "TrackingRegionCandidatePhiVsEta_" + seedProducer.label() + "_";
        TrackingRegionCandidatePhiVsEta = ibooker.book2D(
            histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax);
        TrackingRegionCandidatePhiVsEta->setAxisTitle("TrackingRegion Candidate #eta", 1);
        TrackingRegionCandidatePhiVsEta->setAxisTitle("TrackingRegion Candidate #phi", 2);
      }
    }
  }

  if (doAllSeedPlots || doStopSource) {
    const auto stopReasonSize = static_cast<unsigned int>(SeedStopReason::SIZE);

    const auto candsBin = iConfig.getParameter<int>("SeedCandBin");
    const auto candsMin = iConfig.getParameter<double>("SeedCandMin");
    const auto candsMax = iConfig.getParameter<double>("SeedCandMax");

    histname = "SeedStoppingSource_" + seedProducer.label() + "_";
    seedStoppingSource =
        ibooker.book1D(histname + CatagoryName, histname + CatagoryName, stopReasonSize, 0., stopReasonSize);
    seedStoppingSource->setAxisTitle("Stopping reason", 1);
    seedStoppingSource->setAxisTitle("Number of seeds", 2);

    histname = "SeedStoppingSourceVsPhi_" + seedProducer.label() + "_";
    seedStoppingSourceVsPhi =
        ibooker.bookProfile(histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax, 2, 0., 2.);
    seedStoppingSourceVsPhi->setAxisTitle("seed #phi", 1);
    seedStoppingSourceVsPhi->setAxisTitle("fraction stopped", 2);

    histname = "SeedStoppingSourceVsEta_" + seedProducer.label() + "_";
    seedStoppingSourceVsEta =
        ibooker.bookProfile(histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, 2, 0., 2.);
    seedStoppingSourceVsEta->setAxisTitle("seed #eta", 1);
    seedStoppingSourceVsEta->setAxisTitle("fraction stopped", 2);

    histname = "NumberOfTrajCandsPerSeed_" + seedProducer.label() + "_";
    numberOfTrajCandsPerSeed =
        ibooker.book1D(histname + CatagoryName, histname + CatagoryName, candsBin, candsMin, candsMax);
    numberOfTrajCandsPerSeed->setAxisTitle("Number of Trajectory Candidate for each Seed", 1);
    numberOfTrajCandsPerSeed->setAxisTitle("Number of Seeds", 2);

    histname = "NumberOfTrajCandsPerSeedVsPhi_" + seedProducer.label() + "_";
    numberOfTrajCandsPerSeedVsPhi = ibooker.bookProfile(
        histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax, candsBin, candsMin, candsMax);
    numberOfTrajCandsPerSeedVsPhi->setAxisTitle("Seed #phi", 1);
    numberOfTrajCandsPerSeedVsPhi->setAxisTitle("Number of Trajectory Candidates for each Seed", 2);

    histname = "NumberOfTrajCandsPerSeedVsEta_" + seedProducer.label() + "_";
    numberOfTrajCandsPerSeedVsEta = ibooker.bookProfile(
        histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, candsBin, candsMin, candsMax);
    numberOfTrajCandsPerSeedVsEta->setAxisTitle("Seed #eta", 1);
    numberOfTrajCandsPerSeedVsEta->setAxisTitle("Number of Trajectory Candidates for each Seed", 2);

    histname = "SeedStoppingSourceVsNumberOfTrajCandsPerSeed_" + seedProducer.label() + "_";
    seedStoppingSourceVsNumberOfTrajCandsPerSeed =
        ibooker.bookProfile(histname + CatagoryName, histname + CatagoryName, candsBin, candsMin, candsMax, 2, 0., 2.);
    seedStoppingSourceVsNumberOfTrajCandsPerSeed->setAxisTitle("Number of Trajectory Candidates for each Seed", 1);
    seedStoppingSourceVsNumberOfTrajCandsPerSeed->setAxisTitle("fraction stopped", 2);

    for (unsigned int i = 0; i < stopReasonSize; ++i) {
      seedStoppingSource->setBinLabel(i + 1, SeedStopReasonName::SeedStopReasonName[i], 1);
    }
  }

  if (doAllTCPlots || doStopSource) {
    // DataFormats/TrackReco/interface/TrajectoryStopReasons.h
    size_t StopReasonNameSize = sizeof(StopReasonName::StopReasonName) / sizeof(std::string);

    histname = "StoppingSource_" + seedProducer.label() + "_";
    stoppingSource = ibooker.book1D(
        histname + CatagoryName, histname + CatagoryName, StopReasonNameSize, 0., double(StopReasonNameSize));
    stoppingSource->setAxisTitle("stopping reason", 1);
    stoppingSource->setAxisTitle("Number of Tracks", 2);

    histname = "StoppingSourceVSeta_" + seedProducer.label() + "_";
    stoppingSourceVSeta =
        ibooker.bookProfile(histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, 2, 0., 2.);
    stoppingSourceVSeta->setAxisTitle("track #eta", 1);
    stoppingSourceVSeta->setAxisTitle("fraction stopped", 2);

    histname = "StoppingSourceVSphi_" + seedProducer.label() + "_";
    stoppingSourceVSphi =
        ibooker.bookProfile(histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax, 2, 0., 2.);
    stoppingSourceVSphi->setAxisTitle("track #phi", 1);
    stoppingSourceVSphi->setAxisTitle("fraction stopped", 2);

    for (size_t ibin = 0; ibin < StopReasonNameSize; ibin++) {
      stoppingSource->setBinLabel(ibin + 1, StopReasonName::StopReasonName[ibin], 1);
    }
  }

  // book the TrackCandidate histograms
  // ---------------------------------------------------------------------------------//

  if (doTCPlots) {
    ibooker.setCurrentFolder(MEFolderName + "/TrackBuilding");

    histname = "TrackCandPt_" + tcProducer.label() + "_";
    TrackCandPt = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TrackPtBin, TrackPtMin, TrackPtMax);
    TrackCandPt->setAxisTitle("Track Candidate p_{T} (GeV/c)", 1);
    TrackCandPt->setAxisTitle("Number of Track Candidates", 2);

    histname = "TrackCandEta_" + tcProducer.label() + "_";
    TrackCandEta = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax);
    TrackCandEta->setAxisTitle("Track Candidate #eta", 1);
    TrackCandEta->setAxisTitle("Number of Track Candidates", 2);

    histname = "TrackCandPhi_" + tcProducer.label() + "_";
    TrackCandPhi = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax);
    TrackCandPhi->setAxisTitle("Track Candidate #phi", 1);
    TrackCandPhi->setAxisTitle("Number of Track Candidates", 2);

    if (doTheta) {
      histname = "TrackCandTheta_" + tcProducer.label() + "_";
      TrackCandTheta = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, ThetaBin, ThetaMin, ThetaMax);
      TrackCandTheta->setAxisTitle("Track Candidate #theta", 1);
      TrackCandTheta->setAxisTitle("Number of Track Candidates", 2);
    }

    if (doAllTCPlots) {
      histname = "TrackCandQ_" + tcProducer.label() + "_";
      TrackCandQ = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TrackQBin, TrackQMin, TrackQMax);
      TrackCandQ->setAxisTitle("Track Candidate Charge", 1);
      TrackCandQ->setAxisTitle("Number of Track Candidates", 2);

      histname = "TrackCandDxy_" + tcProducer.label() + "_";
      TrackCandDxy = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TCDxyBin, TCDxyMin, TCDxyMax);
      TrackCandDxy->setAxisTitle("Track Candidate d_{xy} (cm)", 1);
      TrackCandDxy->setAxisTitle("Number of Track Candidates", 2);

      histname = "TrackCandDz_" + tcProducer.label() + "_";
      TrackCandDz = ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TCDzBin, TCDzMin, TCDzMax);
      TrackCandDz->setAxisTitle("Track Candidate d_{z} (cm)", 1);
      TrackCandDz->setAxisTitle("Number of Track Candidates", 2);

      histname = "NumberOfRecHitsPerTrackCand_" + tcProducer.label() + "_";
      NumberOfRecHitsPerTrackCand =
          ibooker.book1D(histname + CatagoryName, histname + CatagoryName, TCHitBin, TCHitMin, TCHitMax);
      NumberOfRecHitsPerTrackCand->setAxisTitle("Number of RecHits per Track Candidate", 1);
      NumberOfRecHitsPerTrackCand->setAxisTitle("Number of Track Candidates", 2);

      histname = "NumberOfRecHitsPerTrackCandVsPhiProfile_" + tcProducer.label() + "_";
      NumberOfRecHitsPerTrackCandVsPhiProfile = ibooker.bookProfile(
          histname + CatagoryName, histname + CatagoryName, PhiBin, PhiMin, PhiMax, TCHitBin, TCHitMin, TCHitMax, "s");
      NumberOfRecHitsPerTrackCandVsPhiProfile->setAxisTitle("Track Candidate #phi", 1);
      NumberOfRecHitsPerTrackCandVsPhiProfile->setAxisTitle("Number of RecHits of each Track Candidate", 2);

      histname = "NumberOfRecHitsPerTrackCandVsEtaProfile_" + tcProducer.label() + "_";
      NumberOfRecHitsPerTrackCandVsEtaProfile = ibooker.bookProfile(
          histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, TCHitBin, TCHitMin, TCHitMax, "s");
      NumberOfRecHitsPerTrackCandVsEtaProfile->setAxisTitle("Track Candidate #eta", 1);
      NumberOfRecHitsPerTrackCandVsEtaProfile->setAxisTitle("Number of RecHits of each Track Candidate", 2);
    }

    histname = "TrackCandPhiVsEta_" + tcProducer.label() + "_";
    TrackCandPhiVsEta = ibooker.book2D(
        histname + CatagoryName, histname + CatagoryName, EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax);
    TrackCandPhiVsEta->setAxisTitle("Track Candidate #eta", 1);
    TrackCandPhiVsEta->setAxisTitle("Track Candidate #phi", 2);

    if (doAllTCPlots || doMVAPlots) {
      for (size_t i = 1, end = mvaProducers.size(); i <= end; ++i) {
        auto num = std::to_string(i);
        std::string pfix;

        if (i == 1) {
          trackMVAsHP.push_back(nullptr);
          trackMVAsHPVsPtProfile.push_back(nullptr);
          trackMVAsHPVsEtaProfile.push_back(nullptr);
        } else {
          pfix = " (not loose-selected)";
          std::string pfix2 = " (not HP-selected)";
          histname = "TrackMVA" + num + "HP_" + tcProducer.label() + "_";
          trackMVAsHP.push_back(
              ibooker.book1D(histname + CatagoryName, histname + CatagoryName + pfix2, MVABin, MVAMin, MVAMax));
          trackMVAsHP.back()->setAxisTitle("Track selection MVA" + num, 1);
          trackMVAsHP.back()->setAxisTitle("Number of tracks", 2);

          histname = "TrackMVA" + num + "HPVsPtProfile_" + tcProducer.label() + "_";
          trackMVAsHPVsPtProfile.push_back(ibooker.bookProfile(histname + CatagoryName,
                                                               histname + CatagoryName + pfix2,
                                                               TrackPtBin,
                                                               TrackPtMin,
                                                               TrackPtMax,
                                                               MVABin,
                                                               MVAMin,
                                                               MVAMax));
          trackMVAsHPVsPtProfile.back()->setAxisTitle("Track p_{T} (GeV/c)", 1);
          trackMVAsHPVsPtProfile.back()->setAxisTitle("Track selection MVA" + num, 2);

          histname = "TrackMVA" + num + "HPVsEtaProfile_" + tcProducer.label() + "_";
          trackMVAsHPVsEtaProfile.push_back(ibooker.bookProfile(
              histname + CatagoryName, histname + CatagoryName + pfix2, EtaBin, EtaMin, EtaMax, MVABin, MVAMin, MVAMax));
          trackMVAsHPVsEtaProfile.back()->setAxisTitle("Track #eta", 1);
          trackMVAsHPVsEtaProfile.back()->setAxisTitle("Track selection MVA" + num, 2);
        }

        histname = "TrackMVA" + num + "_" + tcProducer.label() + "_";
        trackMVAs.push_back(
            ibooker.book1D(histname + CatagoryName, histname + CatagoryName + pfix, MVABin, MVAMin, MVAMax));
        trackMVAs.back()->setAxisTitle("Track selection MVA" + num, 1);
        trackMVAs.back()->setAxisTitle("Number of tracks", 2);

        histname = "TrackMVA" + num + "VsPtProfile_" + tcProducer.label() + "_";
        trackMVAsVsPtProfile.push_back(ibooker.bookProfile(histname + CatagoryName,
                                                           histname + CatagoryName + pfix,
                                                           TrackPtBin,
                                                           TrackPtMin,
                                                           TrackPtMax,
                                                           MVABin,
                                                           MVAMin,
                                                           MVAMax));
        trackMVAsVsPtProfile.back()->setAxisTitle("Track p_{T} (GeV/c)", 1);
        trackMVAsVsPtProfile.back()->setAxisTitle("Track selection MVA" + num, 2);

        histname = "TrackMVA" + num + "VsEtaProfile_" + tcProducer.label() + "_";
        trackMVAsVsEtaProfile.push_back(ibooker.bookProfile(
            histname + CatagoryName, histname + CatagoryName + pfix, EtaBin, EtaMin, EtaMax, MVABin, MVAMin, MVAMax));
        trackMVAsVsEtaProfile.back()->setAxisTitle("Track #eta", 1);
        trackMVAsVsEtaProfile.back()->setAxisTitle("Track selection MVA" + num, 2);
      }
    }
  }
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackBuildingAnalyzer::analyze(const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    const TrajectorySeed& candidate,
                                    const SeedStopInfo& stopInfo,
                                    const reco::BeamSpot& bs,
                                    const MagneticField& theMF,
                                    const TransientTrackingRecHitBuilder& theTTRHBuilder) {
  TSCBLBuilderNoMaterial tscblBuilder;

  //get parameters and errors from the candidate state
  auto const& theG = ((TkTransientTrackingRecHitBuilder const*)(&theTTRHBuilder))->geometry();
  auto const& candSS = candidate.startingState();
  TrajectoryStateOnSurface state =
      trajectoryStateTransform::transientState(candSS, &(theG->idToDet(candSS.detId())->surface()), &theMF);
  TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed =
      tscblBuilder(*state.freeState(), bs);  //as in TrackProducerAlgorithm
  if (!(tsAtClosestApproachSeed.isValid())) {
    edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
    return;
  }
  GlobalPoint v0 = tsAtClosestApproachSeed.trackStateAtPCA().position();
  GlobalVector p = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
  GlobalPoint v(v0.x() - bs.x0(), v0.y() - bs.y0(), v0.z() - bs.z0());

  double pt = sqrt(state.globalMomentum().perp2());
  double eta = state.globalPosition().eta();
  double phi = state.globalPosition().phi();
  double theta = state.globalPosition().theta();
  //double pm           = sqrt(state.globalMomentum().mag2());
  //double pz           = state.globalMomentum().z();
  //double qoverp       = tsAtClosestApproachSeed.trackStateAtPCA().charge()/p.mag();
  //double theta        = p.theta();
  //double lambda       = M_PI/2-p.theta();
  double numberOfHits = candidate.recHits().second - candidate.recHits().first;
  double dxy = (-v.x() * sin(p.phi()) + v.y() * cos(p.phi()));
  double dz = v.z() - (v.x() * p.x() + v.y() * p.y()) / p.perp() * p.z() / p.perp();

  // fill the ME's
  if (doAllSeedPlots || doQ)
    SeedQ->Fill(state.charge());
  if (doAllSeedPlots || doPT)
    SeedPt->Fill(pt);
  if (doAllSeedPlots || doETA)
    SeedEta->Fill(eta);
  if (doAllSeedPlots || doPHI)
    SeedPhi->Fill(phi);
  if (doAllSeedPlots || doPHIVsETA)
    SeedPhiVsEta->Fill(eta, phi);
  if (doAllSeedPlots || doTheta)
    SeedTheta->Fill(theta);
  if (doAllSeedPlots || doDxy)
    SeedDxy->Fill(dxy);
  if (doAllSeedPlots || doDz)
    SeedDz->Fill(dz);
  if (doAllSeedPlots || doNRecHits)
    NumberOfRecHitsPerSeed->Fill(numberOfHits);
  if (doAllSeedPlots || doProfETA)
    NumberOfRecHitsPerSeedVsEtaProfile->Fill(eta, numberOfHits);
  if (doAllSeedPlots || doProfPHI)
    NumberOfRecHitsPerSeedVsPhiProfile->Fill(phi, numberOfHits);
  if (doAllSeedPlots || doStopSource) {
    const double stopped = stopInfo.stopReason() == SeedStopReason::NOT_STOPPED ? 0. : 1.;
    seedStoppingSource->Fill(stopInfo.stopReasonUC());
    seedStoppingSourceVsPhi->Fill(phi, stopped);
    seedStoppingSourceVsEta->Fill(eta, stopped);

    const auto ncands = stopInfo.candidatesPerSeed();
    numberOfTrajCandsPerSeed->Fill(ncands);
    numberOfTrajCandsPerSeedVsPhi->Fill(phi, ncands);
    numberOfTrajCandsPerSeedVsEta->Fill(eta, ncands);

    seedStoppingSourceVsNumberOfTrajCandsPerSeed->Fill(ncands, stopped);
  }
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackBuildingAnalyzer::analyze(const edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    const TrackCandidate& candidate,
                                    const reco::BeamSpot& bs,
                                    const MagneticField& theMF,
                                    const TransientTrackingRecHitBuilder& theTTRHBuilder) {
  TSCBLBuilderNoMaterial tscblBuilder;

  //get parameters and errors from the candidate state
  auto const& theG = ((TkTransientTrackingRecHitBuilder const*)(&theTTRHBuilder))->geometry();
  auto const& candSS = candidate.trajectoryStateOnDet();
  TrajectoryStateOnSurface state =
      trajectoryStateTransform::transientState(candSS, &(theG->idToDet(candSS.detId())->surface()), &theMF);
  TrajectoryStateClosestToBeamLine tsAtClosestApproachTrackCand =
      tscblBuilder(*state.freeState(), bs);  //as in TrackProducerAlgorithm
  if (!(tsAtClosestApproachTrackCand.isValid())) {
    edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
    return;
  }
  GlobalPoint v0 = tsAtClosestApproachTrackCand.trackStateAtPCA().position();
  GlobalVector p = tsAtClosestApproachTrackCand.trackStateAtPCA().momentum();
  GlobalPoint v(v0.x() - bs.x0(), v0.y() - bs.y0(), v0.z() - bs.z0());

  double pt = sqrt(state.globalMomentum().perp2());
  double eta = state.globalPosition().eta();
  double phi = state.globalPosition().phi();
  double theta = state.globalPosition().theta();
  //double pm           = sqrt(state.globalMomentum().mag2());
  //double pz           = state.globalMomentum().z();
  //double qoverp       = tsAtClosestApproachTrackCand.trackStateAtPCA().charge()/p.mag();
  //double theta        = p.theta();
  //double lambda       = M_PI/2-p.theta();
  double numberOfHits = candidate.recHits().second - candidate.recHits().first;
  double dxy = (-v.x() * sin(p.phi()) + v.y() * cos(p.phi()));

  double dz = v.z() - (v.x() * p.x() + v.y() * p.y()) / p.perp() * p.z() / p.perp();

  if (doAllTCPlots || doStopSource) {
    // stopping source
    int max = stoppingSource->getNbinsX();
    double stop = candidate.stopReason() > max ? double(max - 1) : static_cast<double>(candidate.stopReason());
    double stopped = int(StopReason::NOT_STOPPED) == candidate.stopReason() ? 0. : 1.;
    stoppingSource->Fill(stop);
    stoppingSourceVSeta->Fill(eta, stopped);
    stoppingSourceVSphi->Fill(phi, stopped);
  }

  if (doTCPlots) {
    // fill the ME's
    if (doAllTCPlots)
      TrackCandQ->Fill(state.charge());
    TrackCandPt->Fill(pt);
    TrackCandEta->Fill(eta);
    TrackCandPhi->Fill(phi);
    TrackCandPhiVsEta->Fill(eta, phi);
    if (doTheta)
      TrackCandTheta->Fill(theta);
    if (doAllTCPlots)
      TrackCandDxy->Fill(dxy);
    if (doAllTCPlots)
      TrackCandDz->Fill(dz);
    if (doAllTCPlots)
      NumberOfRecHitsPerTrackCand->Fill(numberOfHits);
    if (doAllTCPlots)
      NumberOfRecHitsPerTrackCandVsEtaProfile->Fill(eta, numberOfHits);
    if (doAllTCPlots)
      NumberOfRecHitsPerTrackCandVsPhiProfile->Fill(phi, numberOfHits);
  }
}

namespace {
  bool trackSelected(unsigned char mask, unsigned char qual) { return mask & 1 << qual; }
}  // namespace
void TrackBuildingAnalyzer::analyze(const edm::View<reco::Track>& trackCollection,
                                    const std::vector<const MVACollection*>& mvaCollections,
                                    const std::vector<const QualityMaskCollection*>& qualityMaskCollections) {
  if (!(doAllTCPlots || doMVAPlots))
    return;
  if (trackCollection.empty())
    return;

  const auto ntracks = trackCollection.size();
  const auto nmva = mvaCollections.size();
  for (const auto mva : mvaCollections) {
    if (mva->size() != ntracks) {
      edm::LogError("LogicError") << "TrackBuildingAnalyzer: Incompatible size of MVACollection, " << mva->size()
                                  << " differs from the size of the track collection " << ntracks;
      return;
    }
  }
  for (const auto qual : qualityMaskCollections) {
    if (qual->size() != ntracks) {
      edm::LogError("LogicError") << "TrackBuildingAnalyzer: Incompatible size of QualityMaskCollection, "
                                  << qual->size() << " differs from the size of the track collection " << ntracks;
      return;
    }
  }

  for (size_t iTrack = 0; iTrack < ntracks; ++iTrack) {
    // Fill MVA1 histos with all tracks, MVA2 histos only with tracks
    // not selected by MVA1 etc
    bool selectedLoose = false;
    bool selectedHP = false;

    const auto pt = trackCollection[iTrack].pt();
    const auto eta = trackCollection[iTrack].eta();

    for (size_t iMVA = 0; iMVA < nmva; ++iMVA) {
      const auto mva = (*(mvaCollections[iMVA]))[iTrack];
      if (!selectedLoose) {
        trackMVAs[iMVA]->Fill(mva);
        trackMVAsVsPtProfile[iMVA]->Fill(pt, mva);
        trackMVAsVsEtaProfile[iMVA]->Fill(eta, mva);
      }
      if (iMVA >= 1 && !selectedHP) {
        trackMVAsHP[iMVA]->Fill(mva);
        trackMVAsHPVsPtProfile[iMVA]->Fill(pt, mva);
        trackMVAsHPVsEtaProfile[iMVA]->Fill(eta, mva);
      }

      const auto qual = (*(qualityMaskCollections)[iMVA])[iTrack];
      selectedLoose |= trackSelected(qual, reco::TrackBase::loose);
      selectedHP |= trackSelected(qual, reco::TrackBase::highPurity);

      if (selectedLoose && selectedHP)
        break;
    }
  }
}

void TrackBuildingAnalyzer::analyze(const reco::CandidateView& regionCandidates) {
  if (!doRegionPlots || !doRegionCandidatePlots)
    return;

  for (const auto& candidate : regionCandidates) {
    const auto eta = candidate.eta();
    const auto phi = candidate.phi();
    if (doAllSeedPlots || doPT)
      TrackingRegionCandidatePt->Fill(candidate.pt());
    if (doAllSeedPlots || doETA)
      TrackingRegionCandidateEta->Fill(eta);
    if (doAllSeedPlots || doPHI)
      TrackingRegionCandidatePhi->Fill(phi);
    if (doAllSeedPlots || doPHIVsETA)
      TrackingRegionCandidatePhiVsEta->Fill(eta, phi);
  }
}

void TrackBuildingAnalyzer::analyze(const edm::OwnVector<TrackingRegion>& regions) { analyzeRegions(regions); }
void TrackBuildingAnalyzer::analyze(const TrackingRegionsSeedingLayerSets& regions) { analyzeRegions(regions); }

namespace {
  const TrackingRegion* regionPtr(const TrackingRegion& region) { return &region; }
  const TrackingRegion* regionPtr(const TrackingRegionsSeedingLayerSets::RegionLayers& regionLayers) {
    return &(regionLayers.region());
  }
}  // namespace

template <typename T>
void TrackBuildingAnalyzer::analyzeRegions(const T& regions) {
  if (!doRegionPlots && etaBinWidth <= 0. && phiBinWidth <= 0.)
    return;

  for (const auto& tmp : regions) {
    if (const auto* etaPhiRegion = dynamic_cast<const RectangularEtaPhiTrackingRegion*>(regionPtr(tmp))) {
      const auto& etaRange = etaPhiRegion->etaRange();
      const auto& phiMargin = etaPhiRegion->phiMargin();

      const auto etaMin = etaRange.min();
      const auto etaMax = etaRange.max();

      const auto phiMin = etaPhiRegion->phiDirection() - phiMargin.left();
      const auto phiMax = etaPhiRegion->phiDirection() + phiMargin.right();

      if (doAllSeedPlots || doETA) {
        for (auto eta = etaMin; eta < etaMax; eta += etaBinWidth) {
          TrackingRegionEta->Fill(eta);
        }
      }
      if (doAllSeedPlots || doPHI) {
        for (auto phi = phiMin; phi < phiMax; phi += phiBinWidth) {
          TrackingRegionPhi->Fill(reco::reduceRange(phi));
        }
      }
      if (doAllSeedPlots || doPHIVsETA) {
        for (auto phi = phiMin; phi < phiMax; phi += phiBinWidth) {
          const auto reducedPhi = reco::reduceRange(phi);
          for (auto eta = etaMin; eta < etaMax; eta += etaBinWidth) {
            TrackingRegionPhiVsEta->Fill(eta, reducedPhi);
          }
        }
      }
    }
  }
}
