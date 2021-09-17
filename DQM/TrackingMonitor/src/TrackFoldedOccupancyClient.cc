#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DQM/TrackingMonitor/interface/TrackFoldedOccupancyClient.h"
//-----------------------------------------------------------------------------------
TrackFoldedOccupancyClient::TrackFoldedOccupancyClient(edm::ParameterSet const& iConfig)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackFoldedOccupancyClient") << "TrackFoldedOccupancyClient::Deleting TrackFoldedOccupancyClient ";
  TopFolder_ = iConfig.getParameter<std::string>("FolderName");
  quality_ = iConfig.getParameter<std::string>("TrackQuality");
  algoName_ = iConfig.getParameter<std::string>("AlgoName");
  state_ = iConfig.getParameter<std::string>("MeasurementState");
  histTag_ = (state_ == "default") ? algoName_ : state_ + "_" + algoName_;
  conf_ = iConfig;
}

//-----------------------------------------------------------------------------------
TrackFoldedOccupancyClient::~TrackFoldedOccupancyClient()
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackFoldedOccupancyClient") << "TrackFoldedOccupancyClient::Deleting TrackFoldedOccupancyClient ";
}

//-----------------------------------------------------------------------------------
void TrackFoldedOccupancyClient::beginJob(void)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackFoldedOccupancyClient") << "TrackFoldedOccupancyClient::beginJob done";
}

//-----------------------------------------------------------------------------------
void TrackFoldedOccupancyClient::beginRun(edm::Run const& run, edm::EventSetup const& eSetup)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackFoldedOccupancyClient") << "TrackFoldedOccupancyClient:: Begining of Run";
}

//-----------------------------------------------------------------------------------
void TrackFoldedOccupancyClient::bookMEs(DQMStore::IBooker& ibooker)
//-----------------------------------------------------------------------------------
{
  ibooker.setCurrentFolder(TopFolder_ + "/" + quality_ + "/GeneralProperties/");
  int Phi2DBin = conf_.getParameter<int>("Phi2DBin");
  int Eta2DBin = conf_.getParameter<int>("Eta2DBin");
  double EtaMin = conf_.getParameter<double>("EtaMin");
  double EtaMax = conf_.getParameter<double>("EtaMax");
  double PhiMin = conf_.getParameter<double>("PhiMin");
  double PhiMax = conf_.getParameter<double>("PhiMax");

  // use the AlgoName and Quality Name
  std::string histname = "TkEtaPhi_RelativeDifference_byFoldingmap_" + histTag_;
  TkEtaPhi_RelativeDifference_byFoldingmap =
      ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
  TkEtaPhi_RelativeDifference_byFoldingmap->setAxisTitle("Track #eta", 1);
  TkEtaPhi_RelativeDifference_byFoldingmap->setAxisTitle("Track #phi", 2);

  histname = "TkEtaPhi_RelativeDifference_byFoldingmap_op_" + histTag_;
  TkEtaPhi_RelativeDifference_byFoldingmap_op =
      ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
  TkEtaPhi_RelativeDifference_byFoldingmap_op->setAxisTitle("Track #eta", 1);
  TkEtaPhi_RelativeDifference_byFoldingmap_op->setAxisTitle("Track #phi", 2);

  histname = "TkEtaPhi_Ratio_byFoldingmap_" + histTag_;
  TkEtaPhi_Ratio_byFoldingmap = ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
  TkEtaPhi_Ratio_byFoldingmap->setAxisTitle("Track #eta", 1);
  TkEtaPhi_Ratio_byFoldingmap->setAxisTitle("Track #phi", 2);

  histname = "TkEtaPhi_Ratio_byFoldingmap_op_" + histTag_;
  TkEtaPhi_Ratio_byFoldingmap_op =
      ibooker.book2D(histname, histname, Eta2DBin, EtaMin, EtaMax, Phi2DBin, PhiMin, PhiMax);
  TkEtaPhi_Ratio_byFoldingmap_op->setAxisTitle("Track #eta", 1);
  TkEtaPhi_Ratio_byFoldingmap_op->setAxisTitle("Track #phi", 2);
}

//-----------------------------------------------------------------------------------
void TrackFoldedOccupancyClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackFoldedOccupancyClient") << "TrackFoldedOccupancyClient::endLuminosityBlock";

  bookMEs(ibooker);
  std::string inFolder = TopFolder_ + "/" + quality_ + "/GeneralProperties/";

  std::string hname;
  hname = "TrackEtaPhi_";
  MonitorElement* TrackEtaPhi = igetter.get(inFolder + hname + histTag_);

  hname = "TrackEtaPhiInverted_";
  MonitorElement* TrackEtaPhiInverted = igetter.get(inFolder + hname + histTag_);

  hname = "TrackEtaPhiInvertedoutofphase_";
  MonitorElement* TrackEtaPhiInvertedoutofphase = igetter.get(inFolder + hname + histTag_);

  TkEtaPhi_Ratio_byFoldingmap->divide(TrackEtaPhi, TrackEtaPhiInverted, 1., 1., "");
  TkEtaPhi_Ratio_byFoldingmap_op->divide(TrackEtaPhi, TrackEtaPhiInvertedoutofphase, 1., 1., "");

  int nx = TrackEtaPhi->getNbinsX();
  int ny = TrackEtaPhi->getNbinsY();

  for (int ii = 1; ii <= nx; ii++) {
    for (int jj = 1; jj <= ny; jj++) {
      double Sum1 = TrackEtaPhi->getBinContent(ii, jj) + TrackEtaPhiInverted->getBinContent(ii, jj);
      double Sum2 = TrackEtaPhi->getBinContent(ii, jj) + TrackEtaPhiInvertedoutofphase->getBinContent(ii, jj);

      double Sub1 = TrackEtaPhi->getBinContent(ii, jj) - TrackEtaPhiInverted->getBinContent(ii, jj);
      double Sub2 = TrackEtaPhi->getBinContent(ii, jj) - TrackEtaPhiInvertedoutofphase->getBinContent(ii, jj);

      if (Sum1 == 0 || Sum2 == 0) {
        TkEtaPhi_RelativeDifference_byFoldingmap->setBinContent(ii, jj, 1);
        TkEtaPhi_RelativeDifference_byFoldingmap_op->setBinContent(ii, jj, 1);
      } else {
        double ratio1 = Sub1 / Sum1;
        double ratio2 = Sub2 / Sum2;
        TkEtaPhi_RelativeDifference_byFoldingmap->setBinContent(ii, jj, ratio1);
        TkEtaPhi_RelativeDifference_byFoldingmap_op->setBinContent(ii, jj, ratio2);
      }
    }
  }
}

DEFINE_FWK_MODULE(TrackFoldedOccupancyClient);
