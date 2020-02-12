/*
 *  See header file for a description of this class.
 *
 *  \author Anne-Catherine Le Bihan
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/TrackingMonitor/interface/TrackEfficiencyClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

//-----------------------------------------------------------------------------------
TrackEfficiencyClient::TrackEfficiencyClient(edm::ParameterSet const& iConfig)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient::Deleting TrackEfficiencyClient ";

  FolderName_ = iConfig.getParameter<std::string>("FolderName");
  algoName_ = iConfig.getParameter<std::string>("AlgoName");
  trackEfficiency_ = iConfig.getParameter<bool>("trackEfficiency");

  conf_ = iConfig;
}

//-----------------------------------------------------------------------------------
TrackEfficiencyClient::~TrackEfficiencyClient()
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient::Deleting TrackEfficiencyClient ";
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::beginJob(void)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient::beginJob done";
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::beginRun(edm::Run const& run, edm::EventSetup const& eSetup)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient:: Begining of Run";
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::bookMEs(DQMStore::IBooker& ibooker_)
//-----------------------------------------------------------------------------------
{
  ibooker_.setCurrentFolder(FolderName_);

  //
  int effXBin = conf_.getParameter<int>("effXBin");
  double effXMin = conf_.getParameter<double>("effXMin");
  double effXMax = conf_.getParameter<double>("effXMax");

  histName = "effX_";
  effX = ibooker_.book1D(histName + algoName_, histName + algoName_, effXBin, effXMin, effXMax);
  if (effX->getTH1F())
    effX->enableSumw2();
  effX->setAxisTitle("");

  //
  int effYBin = conf_.getParameter<int>("effYBin");
  double effYMin = conf_.getParameter<double>("effYMin");
  double effYMax = conf_.getParameter<double>("effYMax");

  histName = "effY_";
  effY = ibooker_.book1D(histName + algoName_, histName + algoName_, effYBin, effYMin, effYMax);
  if (effY->getTH1F())
    effY->enableSumw2();
  effY->setAxisTitle("");

  //
  int effZBin = conf_.getParameter<int>("effZBin");
  double effZMin = conf_.getParameter<double>("effZMin");
  double effZMax = conf_.getParameter<double>("effZMax");

  histName = "effZ_";
  effZ = ibooker_.book1D(histName + algoName_, histName + algoName_, effZBin, effZMin, effZMax);
  if (effZ->getTH1F())
    effZ->enableSumw2();
  effZ->setAxisTitle("");

  //
  int effEtaBin = conf_.getParameter<int>("effEtaBin");
  double effEtaMin = conf_.getParameter<double>("effEtaMin");
  double effEtaMax = conf_.getParameter<double>("effEtaMax");

  histName = "effEta_";
  effEta = ibooker_.book1D(histName + algoName_, histName + algoName_, effEtaBin, effEtaMin, effEtaMax);
  if (effEta->getTH1F())
    effEta->enableSumw2();
  effEta->setAxisTitle("");

  //
  int effPhiBin = conf_.getParameter<int>("effPhiBin");
  double effPhiMin = conf_.getParameter<double>("effPhiMin");
  double effPhiMax = conf_.getParameter<double>("effPhiMax");

  histName = "effPhi_";
  effPhi = ibooker_.book1D(histName + algoName_, histName + algoName_, effPhiBin, effPhiMin, effPhiMax);
  if (effPhi->getTH1F())
    effPhi->enableSumw2();
  effPhi->setAxisTitle("");

  //
  int effD0Bin = conf_.getParameter<int>("effD0Bin");
  double effD0Min = conf_.getParameter<double>("effD0Min");
  double effD0Max = conf_.getParameter<double>("effD0Max");

  histName = "effD0_";
  effD0 = ibooker_.book1D(histName + algoName_, histName + algoName_, effD0Bin, effD0Min, effD0Max);
  if (effD0->getTH1F())
    effD0->enableSumw2();
  effD0->setAxisTitle("");

  //
  int effCompatibleLayersBin = conf_.getParameter<int>("effCompatibleLayersBin");
  double effCompatibleLayersMin = conf_.getParameter<double>("effCompatibleLayersMin");
  double effCompatibleLayersMax = conf_.getParameter<double>("effCompatibleLayersMax");

  histName = "effCompatibleLayers_";
  effCompatibleLayers = ibooker_.book1D(histName + algoName_,
                                        histName + algoName_,
                                        effCompatibleLayersBin,
                                        effCompatibleLayersMin,
                                        effCompatibleLayersMax);
  if (effCompatibleLayers->getTH1F())
    effCompatibleLayers->enableSumw2();
  effCompatibleLayers->setAxisTitle("");

  histName = "MuonEffPtPhi_LowPt";
  effPtPhiLowPt = ibooker_.book2D(histName + algoName_, histName + algoName_, 20, -2.4, 2.4, 20, -3.25, 3.25);
  if (effPtPhiLowPt->getTH2F())
    effPtPhiLowPt->enableSumw2();
  effPtPhiLowPt->setAxisTitle("");

  histName = "MuonEffPtPhi_HighPt";
  effPtPhiHighPt = ibooker_.book2D(histName + algoName_, histName + algoName_, 20, -2.4, 2.4, 20, -3.25, 3.25);
  if (effPtPhiHighPt->getTH2F())
    effPtPhiHighPt->enableSumw2();
  effPtPhiHighPt->setAxisTitle("");
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::dqmEndJob(DQMStore::IBooker& ibooker_, DQMStore::IGetter& igetter_)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient::endLuminosityBlock";

  bookMEs(ibooker_);
  FolderName_ = "Tracking/TrackParameters/TrackEfficiency";
  std::vector<std::string> s1 = igetter_.getSubdirs();

  igetter_.cd("Tracking");

  histName = "/trackX_";
  MonitorElement* trackX = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/muonX_";
  MonitorElement* muonX = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/trackY_";
  MonitorElement* trackY = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/muonY_";
  MonitorElement* muonY = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/trackZ_";
  MonitorElement* trackZ = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/muonZ_";
  MonitorElement* muonZ = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/trackEta_";
  MonitorElement* trackEta = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/muonEta_";
  MonitorElement* muonEta = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/trackPhi_";
  MonitorElement* trackPhi = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/muonPhi_";
  MonitorElement* muonPhi = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/trackD0_";
  MonitorElement* trackD0 = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/muonD0_";
  MonitorElement* muonD0 = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/trackCompatibleLayers_";
  MonitorElement* trackCompatibleLayers = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/muonCompatibleLayers_";
  MonitorElement* muonCompatibleLayers = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/StandaloneMuonPtEtaPhi_LowPt_";
  MonitorElement* StandAloneMuonPtEtaPhiLowPt = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/GlobalMuonPtEtaPhi_LowPt_";
  MonitorElement* GlobalMuonPtEtaPhiLowPt = igetter_.get(FolderName_ + histName + algoName_);

  histName = "/StandaloneMuonPtEtaPhi_HighPt_";
  MonitorElement* StandAloneMuonPtEtaPhiHighPt = igetter_.get(FolderName_ + histName + algoName_);
  histName = "/GlobalMuonPtEtaPhi_HighPt_";
  MonitorElement* GlobalMuonPtEtaPhiHighPt = igetter_.get(FolderName_ + histName + algoName_);

  if (StandAloneMuonPtEtaPhiLowPt && GlobalMuonPtEtaPhiLowPt && effPtPhiLowPt) {
    if (StandAloneMuonPtEtaPhiLowPt->getTH2F() && GlobalMuonPtEtaPhiLowPt->getTH2F() && effPtPhiLowPt->getTH2F()) {
      effPtPhiLowPt->getTH2F()->Divide(
          GlobalMuonPtEtaPhiLowPt->getTH2F(), StandAloneMuonPtEtaPhiLowPt->getTH2F(), 1., 1., "");
    }
  }
  if (StandAloneMuonPtEtaPhiHighPt && GlobalMuonPtEtaPhiHighPt && effPtPhiHighPt) {
    if (StandAloneMuonPtEtaPhiHighPt->getTH2F() && GlobalMuonPtEtaPhiHighPt->getTH2F() && effPtPhiHighPt->getTH2F()) {
      effPtPhiHighPt->getTH2F()->Divide(
          GlobalMuonPtEtaPhiHighPt->getTH2F(), StandAloneMuonPtEtaPhiHighPt->getTH2F(), 1., 1., "");
    }
  }
  if (trackX && muonX && trackY && muonY && trackZ && muonZ && trackEta && muonEta && trackPhi && muonPhi && trackD0 &&
      muonD0 && trackCompatibleLayers && muonCompatibleLayers) {  // && StandAloneMuonPtEtaPhi && GlobalMuonPtEtaPhi){

    if (trackEfficiency_) {
      if (effX->getTH1F() && trackX->getTH1F() && muonX->getTH1F()) {
        effX->getTH1F()->Divide(trackX->getTH1F(), muonX->getTH1F(), 1., 1., "");
      }
      if (effY->getTH1F() && trackY->getTH1F() && muonY->getTH1F()) {
        effY->getTH1F()->Divide(trackY->getTH1F(), muonY->getTH1F(), 1., 1., "");
      }
      if (effZ->getTH1F() && trackZ->getTH1F() && muonZ->getTH1F()) {
        effZ->getTH1F()->Divide(trackZ->getTH1F(), muonZ->getTH1F(), 1., 1., "");
      }
      if (effEta->getTH1F() && trackEta->getTH1F() && muonEta->getTH1F()) {
        effEta->getTH1F()->Divide(trackEta->getTH1F(), muonEta->getTH1F(), 1., 1., "");
      }
      if (effPhi->getTH1F() && trackPhi->getTH1F() && muonPhi->getTH1F()) {
        effPhi->getTH1F()->Divide(trackPhi->getTH1F(), muonPhi->getTH1F(), 1., 1., "");
      }
      if (effD0->getTH1F() && trackD0->getTH1F() && muonD0->getTH1F()) {
        effD0->getTH1F()->Divide(trackD0->getTH1F(), muonD0->getTH1F(), 1., 1., "");
      }
      if (effCompatibleLayers->getTH1F() && trackCompatibleLayers->getTH1F() && muonCompatibleLayers->getTH1F()) {
        effCompatibleLayers->getTH1F()->Divide(
            trackCompatibleLayers->getTH1F(), muonCompatibleLayers->getTH1F(), 1., 1., "");
      }
    } else {
      if (effX->getTH1F() && trackX->getTH1F() && muonX->getTH1F()) {
        effX->getTH1F()->Divide(muonX->getTH1F(), trackX->getTH1F(), 1., 1., "");
      }
      if (effY->getTH1F() && trackY->getTH1F() && muonY->getTH1F()) {
        effY->getTH1F()->Divide(muonY->getTH1F(), trackY->getTH1F(), 1., 1., "");
      }
      if (effZ->getTH1F() && trackZ->getTH1F() && muonZ->getTH1F()) {
        effZ->getTH1F()->Divide(muonZ->getTH1F(), trackZ->getTH1F(), 1., 1., "");
      }
      if (effEta->getTH1F() && trackEta->getTH1F() && muonEta->getTH1F()) {
        effEta->getTH1F()->Divide(muonEta->getTH1F(), trackEta->getTH1F(), 1., 1., "");
      }
      if (effPhi->getTH1F() && trackPhi->getTH1F() && muonPhi->getTH1F()) {
        effPhi->getTH1F()->Divide(muonPhi->getTH1F(), trackPhi->getTH1F(), 1., 1., "");
      }
      if (effD0->getTH1F() && trackD0->getTH1F() && muonD0->getTH1F()) {
        effD0->getTH1F()->Divide(muonD0->getTH1F(), trackD0->getTH1F(), 1., 1., "");
      }
    }
  }
}

DEFINE_FWK_MODULE(TrackEfficiencyClient);
