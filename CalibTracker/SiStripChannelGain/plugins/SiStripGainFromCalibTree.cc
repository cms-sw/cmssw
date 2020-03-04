// Original Author:  Loic QUERTENMONT
//         Created:  Mon Nov  16 08:55:18 CET 2009

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "TFile.h"
#include "TObjString.h"
#include "TString.h"
#include "TH1F.h"
#include "TH2S.h"
#include "TProfile.h"
#include "TF1.h"
#include "TROOT.h"
#include "TTree.h"
#include "TChain.h"

// user includes
#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"
#include "CalibTracker/SiStripChannelGain/interface/APVGainHelpers.h"

#include <unordered_map>
#include <array>

using namespace edm;
using namespace reco;
using namespace std;
using namespace APVGain;

class SiStripGainFromCalibTree : public ConditionDBWriter<SiStripApvGain> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;
  explicit SiStripGainFromCalibTree(const edm::ParameterSet&);
  ~SiStripGainFromCalibTree() override;

private:
  void algoBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void algoEndRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void algoBeginJob(const edm::EventSetup& iSetup) override;
  void algoEndJob() override;
  void algoAnalyze(const edm::Event&, const edm::EventSetup&) override;

  int statCollectionFromMode(const char* tag) const;
  void bookDQMHistos(const char* dqm_dir, const char* tag);

  bool isBFieldConsistentWithMode(const edm::EventSetup& iSetup) const;
  void swapBFieldMode(void);

  void merge(TH2* A, TH2* B);  //needed to add histograms with different number of bins
  void algoAnalyzeTheTree();
  void algoComputeMPVandGain();
  void processEvent();  //what really does the job

  void getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange = 50, double HighRange = 5400);
  bool IsGoodLandauFit(double* FitResults);
  void storeOnTree(TFileService* tfs);
  void qualityMonitor();
  void MakeCalibrationMap();
  bool produceTagFilter();

  template <typename T>
  inline edm::Handle<T> connect(const T*& ptr, edm::EDGetTokenT<T> token, const edm::Event& evt) {
    edm::Handle<T> handle;
    evt.getByToken(token, handle);
    ptr = handle.product();
    return handle;  //return handle to keep alive pointer (safety first)
  }

  std::unique_ptr<SiStripApvGain> getNewObject() override;

  TFileService* tfs;
  DQMStore* dbe;
  double MagFieldCurrentTh;
  double MinNrEntries;
  double MaxMPVError;
  double MaxChi2OverNDF;
  double MinTrackMomentum;
  double MaxTrackMomentum;
  double MinTrackEta;
  double MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  double MaxTrackChiOverNdf;
  int MaxTrackingIteration;
  bool AllowSaturation;
  bool FirstSetOfConstants;
  bool Validation;
  bool OldGainRemoving;
  int CalibrationLevel;

  bool saveSummary;
  bool useCalibration;
  bool m_harvestingMode;
  bool m_splitDQMstat;
  bool doChargeMonitorPerPlane;  /*!< Charge monitor per detector plane */
  std::string m_calibrationMode; /*!< Type of statistics for the calibration */
  std::string m_calibrationPath;
  std::string m_DQMdir;                  /*!< DQM folder hosting the charge statistics and the monitor plots */
  std::vector<std::string> VChargeHisto; /*!< Charge monitor plots to be output */

  double tagCondition_NClusters;
  double tagCondition_GoodFrac;

  std::string AlgoMode;
  std::string OutputGains;
  vector<string> VInputFiles;

  //enum statistic_type {None=-1, StdBunch, StdBunch0T, FaABunch, FaABunch0T, IsoBunch, IsoBunch0T, Harvest};

  std::vector<string> dqm_tag_;
  std::string booked_dir_;

  std::vector<MonitorElement*> Charge_Vs_Index;         /*!< Charge per cm for each detector id */
  std::array<std::vector<APVGain::APVmon>, 7> Charge_1; /*!< Charge per cm per layer / wheel */
  std::array<std::vector<APVGain::APVmon>, 7> Charge_2; /*!< Charge per cm per layer / wheel without G2 */
  std::array<std::vector<APVGain::APVmon>, 7> Charge_3; /*!< Charge per cm per layer / wheel without G1 */
  std::array<std::vector<APVGain::APVmon>, 7> Charge_4; /*!< Charge per cm per layer / wheel without G1 and G1*/

  std::vector<MonitorElement*> Charge_Vs_PathlengthTIB;   /*!< Charge vs pathlength in TIB */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTOB;   /*!< Charge vs pathlength in TOB */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTIDP;  /*!< Charge vs pathlength in TIDP */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTIDM;  /*!< Charge vs pathlength in TIDM */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTECP1; /*!< Charge vs pathlength in TECP thin */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTECP2; /*!< Charge vs pathlength in TECP thick */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTECM1; /*!< Charge vs pathlength in TECP thin */
  std::vector<MonitorElement*> Charge_Vs_PathlengthTECM2; /*!< Charge vs pathlength in TECP thick */

  //std::vector<MonitorElement*>  Charge_Vs_Index_Absolute;

  //Validation histograms
  MonitorElement* MPV_Vs_EtaTIB;      /*!< MPV vs Eta for TIB planes */
  MonitorElement* MPV_Vs_EtaTID;      /*!< MPV vs Eta for TID planes */
  MonitorElement* MPV_Vs_EtaTOB;      /*!< MPV vs Eta for TOB planes */
  MonitorElement* MPV_Vs_EtaTEC;      /*!< MPV vs Eta for TEC planes */
  MonitorElement* MPV_Vs_EtaTECthin;  /*!< MPV vs Eta for TEC thin planes */
  MonitorElement* MPV_Vs_EtaTECthick; /*!< MPV vs Eta for TEC tick planes */

  MonitorElement* MPV_Vs_PhiTIB;      /*!< MPV vs Phi for TIB planes */
  MonitorElement* MPV_Vs_PhiTID;      /*!< MPV vs Phi for TID planes */
  MonitorElement* MPV_Vs_PhiTOB;      /*!< MPV vs Phi for TOB planes */
  MonitorElement* MPV_Vs_PhiTEC;      /*!< MPV vs Phi for TEC planes */
  MonitorElement* MPV_Vs_PhiTECthin;  /*!< MPV vs Phi for TEC thin planes */
  MonitorElement* MPV_Vs_PhiTECthick; /*!< MPV vs Phi for TID tick planes */

  MonitorElement* NoMPVmasked; /*!< R,Z map of missing APV calibration (masked modules)*/
  MonitorElement* NoMPVfit;    /*!< R,Z map of missing APV calibration (no fit) */

  MonitorElement* Gains;        /*!< distribution of gain factors */
  MonitorElement* MPVs;         /*!< distribution of MPVs */
  MonitorElement* MPVs320;      /*!< distribution of MPVs for thin sensors */
  MonitorElement* MPVs500;      /*!< distribution of MPVs for tick sensors */
  MonitorElement* MPVsTIB;      /*!< distribution of MPVs for TIB planes */
  MonitorElement* MPVsTID;      /*!< distribution of MPVs for TID disks */
  MonitorElement* MPVsTIDP;     /*!< distribution of MPVs for TIDP disks */
  MonitorElement* MPVsTIDM;     /*!< distribution of MPVs for TIDM disks */
  MonitorElement* MPVsTOB;      /*!< distribution of MPVs for TOB planes */
  MonitorElement* MPVsTEC;      /*!< distribution of MPVs for TEC disks */
  MonitorElement* MPVsTECP;     /*!< distribution of MPVs for TECP disks */
  MonitorElement* MPVsTECM;     /*!< distribution of MPVs for TECM disks */
  MonitorElement* MPVsTECthin;  /*!< distribution of MPVs for TEC thin sensors */
  MonitorElement* MPVsTECthick; /*!< distribution of MPVs for TEC tick sensors */
  MonitorElement* MPVsTECP1;    /*!< distribution of MPVs for TECP thin sensors */
  MonitorElement* MPVsTECP2;    /*!< distribution of MPVs for TECP tick sensors */
  MonitorElement* MPVsTECM1;    /*!< distribution of MPVs for TECM thin sensors */
  MonitorElement* MPVsTECM2;    /*!< distribution of MPVs for TECM tick sensors */

  MonitorElement* MPVError;      /*!< error of Landau fit */
  MonitorElement* MPVErrorVsMPV; /*!< error of Landau fit vs MPV */
  MonitorElement* MPVErrorVsEta; /*!< error of Landau fit vs Eta */
  MonitorElement* MPVErrorVsPhi; /*!< error of Landau fit vs Phi */
  MonitorElement* MPVErrorVsN;   /*!< error of landau fit vs number of entries */

  MonitorElement* DiffWRTPrevGainTIB; /*!< ratio Gain / PreviousGain for TIB layers */
  MonitorElement* DiffWRTPrevGainTID; /*!< ratio Gain / PreviousGain for TID disks */
  MonitorElement* DiffWRTPrevGainTOB; /*!< ratio Gain / PreviousGain for TOB layers */
  MonitorElement* DiffWRTPrevGainTEC; /*!< ratio gain / PreviousGain for TEC disks */

  MonitorElement* GainVsPrevGainTIB; /*!< Gain vs PreviousGain for TIB */
  MonitorElement* GainVsPrevGainTID; /*!< Gain vs PreviousGain for TID */
  MonitorElement* GainVsPrevGainTOB; /*!< Gain vs PreviousGain for TOB */
  MonitorElement* GainVsPrevGainTEC; /*!< Gain vs PreviousGain for TEC */

  std::vector<APVGain::APVmon> newCharge;

  unsigned int NEvent;
  unsigned int NTrack;
  unsigned int NClusterStrip;
  unsigned int NClusterPixel;
  int NStripAPVs;
  int NPixelDets;
  unsigned int SRun;
  unsigned int ERun;
  unsigned int GOOD;
  unsigned int BAD;
  unsigned int MASKED;

  //Data members for processing

  //Event data
  unsigned int eventnumber = 0;
  unsigned int runnumber = 0;
  const std::vector<bool>* TrigTech = nullptr;
  edm::EDGetTokenT<std::vector<bool>> TrigTech_token_;

  // Track data
  const std::vector<double>* trackchi2ndof = nullptr;
  edm::EDGetTokenT<std::vector<double>> trackchi2ndof_token_;
  const std::vector<float>* trackp = nullptr;
  edm::EDGetTokenT<std::vector<float>> trackp_token_;
  const std::vector<float>* trackpt = nullptr;
  edm::EDGetTokenT<std::vector<float>> trackpt_token_;
  const std::vector<double>* tracketa = nullptr;
  edm::EDGetTokenT<std::vector<double>> tracketa_token_;
  const std::vector<double>* trackphi = nullptr;
  edm::EDGetTokenT<std::vector<double>> trackphi_token_;
  const std::vector<unsigned int>* trackhitsvalid = nullptr;
  edm::EDGetTokenT<std::vector<unsigned int>> trackhitsvalid_token_;
  const std::vector<int>* trackalgo = nullptr;
  edm::EDGetTokenT<std::vector<int>> trackalgo_token_;

  // CalibTree data
  const std::vector<int>* trackindex = nullptr;
  edm::EDGetTokenT<std::vector<int>> trackindex_token_;
  const std::vector<unsigned int>* rawid = nullptr;
  edm::EDGetTokenT<std::vector<unsigned int>> rawid_token_;
  const std::vector<double>* localdirx = nullptr;
  edm::EDGetTokenT<std::vector<double>> localdirx_token_;
  const std::vector<double>* localdiry = nullptr;
  edm::EDGetTokenT<std::vector<double>> localdiry_token_;
  const std::vector<double>* localdirz = nullptr;
  edm::EDGetTokenT<std::vector<double>> localdirz_token_;
  const std::vector<unsigned short>* firststrip = nullptr;
  edm::EDGetTokenT<std::vector<unsigned short>> firststrip_token_;
  const std::vector<unsigned short>* nstrips = nullptr;
  edm::EDGetTokenT<std::vector<unsigned short>> nstrips_token_;
  const std::vector<bool>* saturation = nullptr;
  edm::EDGetTokenT<std::vector<bool>> saturation_token_;
  const std::vector<bool>* overlapping = nullptr;
  edm::EDGetTokenT<std::vector<bool>> overlapping_token_;
  const std::vector<bool>* farfromedge = nullptr;
  edm::EDGetTokenT<std::vector<bool>> farfromedge_token_;
  const std::vector<unsigned int>* charge = nullptr;
  edm::EDGetTokenT<std::vector<unsigned int>> charge_token_;
  const std::vector<double>* path = nullptr;
  edm::EDGetTokenT<std::vector<double>> path_token_;
  const std::vector<double>* chargeoverpath = nullptr;
  edm::EDGetTokenT<std::vector<double>> chargeoverpath_token_;
  const std::vector<unsigned char>* amplitude = nullptr;
  edm::EDGetTokenT<std::vector<unsigned char>> amplitude_token_;
  const std::vector<double>* gainused = nullptr;
  edm::EDGetTokenT<std::vector<double>> gainused_token_;
  const std::vector<double>* gainusedTick = nullptr;
  edm::EDGetTokenT<std::vector<double>> gainusedTick_token_;

  string EventPrefix_;  //("");
  string EventSuffix_;  //("");
  string TrackPrefix_;  //("track");
  string TrackSuffix_;  //("");
  string CalibPrefix_;  //("GainCalibration");
  string CalibSuffix_;  //("");

private:
  std::vector<stAPVGain*> APVsCollOrdered;
  std::unordered_map<unsigned int, stAPVGain*> APVsColl;
  const TrackerTopology* tTopo_;
};

inline int SiStripGainFromCalibTree::statCollectionFromMode(const char* tag) const {
  std::vector<string>::const_iterator it = dqm_tag_.begin();
  while (it != dqm_tag_.end()) {
    if (*it == std::string(tag))
      return it - dqm_tag_.begin();
    it++;
  }

  if (std::string(tag).empty())
    return 0;  // return StdBunch calibration mode for backward compatibility

  return None;
}

void SiStripGainFromCalibTree::merge(TH2* A, TH2* B) {
  if (A->GetNbinsX() == B->GetNbinsX()) {
    A->Add(B);
  } else {
    for (int x = 0; x <= B->GetNbinsX() + 1; x++) {
      for (int y = 0; y <= B->GetNbinsY() + 1; y++) {
        A->SetBinContent(x, y, A->GetBinContent(x, y) + B->GetBinContent(x, y));
      }
    }
  }
}

SiStripGainFromCalibTree::SiStripGainFromCalibTree(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiStripApvGain>(iConfig) {
  OutputGains = iConfig.getParameter<std::string>("OutputGains");

  AlgoMode = iConfig.getUntrackedParameter<std::string>("AlgoMode", "CalibTree");
  MagFieldCurrentTh = iConfig.getUntrackedParameter<double>("MagFieldCurrentTh", 2000.);
  MinNrEntries = iConfig.getUntrackedParameter<double>("minNrEntries", 20);
  MaxMPVError = iConfig.getUntrackedParameter<double>("maxMPVError", 500.0);
  MaxChi2OverNDF = iConfig.getUntrackedParameter<double>("maxChi2OverNDF", 5.0);
  MinTrackMomentum = iConfig.getUntrackedParameter<double>("minTrackMomentum", 3.0);
  MaxTrackMomentum = iConfig.getUntrackedParameter<double>("maxTrackMomentum", 99999.0);
  MinTrackEta = iConfig.getUntrackedParameter<double>("minTrackEta", -5.0);
  MaxTrackEta = iConfig.getUntrackedParameter<double>("maxTrackEta", 5.0);
  MaxNrStrips = iConfig.getUntrackedParameter<unsigned>("maxNrStrips", 2);
  MinTrackHits = iConfig.getUntrackedParameter<unsigned>("MinTrackHits", 8);
  MaxTrackChiOverNdf = iConfig.getUntrackedParameter<double>("MaxTrackChiOverNdf", 3);
  MaxTrackingIteration = iConfig.getUntrackedParameter<int>("MaxTrackingIteration", 7);
  AllowSaturation = iConfig.getUntrackedParameter<bool>("AllowSaturation", false);
  FirstSetOfConstants = iConfig.getUntrackedParameter<bool>("FirstSetOfConstants", true);
  Validation = iConfig.getUntrackedParameter<bool>("Validation", false);
  OldGainRemoving = iConfig.getUntrackedParameter<bool>("OldGainRemoving", false);

  CalibrationLevel = iConfig.getUntrackedParameter<int>("CalibrationLevel", 0);
  VInputFiles = iConfig.getUntrackedParameter<vector<string>>("InputFiles");
  VChargeHisto = iConfig.getUntrackedParameter<vector<string>>("ChargeHisto");

  useCalibration = iConfig.getUntrackedParameter<bool>("UseCalibration", false);
  m_harvestingMode = iConfig.getUntrackedParameter<bool>("harvestingMode", false);
  m_splitDQMstat = iConfig.getUntrackedParameter<bool>("splitDQMstat", false);
  m_calibrationMode = iConfig.getUntrackedParameter<string>("calibrationMode", "StdBunch");
  m_calibrationPath = iConfig.getUntrackedParameter<string>("calibrationPath");
  m_DQMdir = iConfig.getUntrackedParameter<string>("DQMdir", "AlCaReco/SiStripGains");

  tagCondition_NClusters = iConfig.getUntrackedParameter<double>("NClustersForTagProd", 2E8);
  tagCondition_GoodFrac = iConfig.getUntrackedParameter<double>("GoodFracForTagProd", 0.95);

  saveSummary = iConfig.getUntrackedParameter<bool>("saveSummary", false);

  doChargeMonitorPerPlane = iConfig.getUntrackedParameter<bool>("doChargeMonitorPerPlane", false);

  // Gather DQM Service
  dbe = edm::Service<DQMStore>().operator->();

  //Set the monitoring element tag and store
  dqm_tag_.reserve(7);
  dqm_tag_.clear();
  dqm_tag_.push_back("StdBunch");    // statistic collection from Standard Collision Bunch @ 3.8 T
  dqm_tag_.push_back("StdBunch0T");  // statistic collection from Standard Collision Bunch @ 0 T
  dqm_tag_.push_back("AagBunch");    // statistic collection from First Collision After Abort Gap @ 3.8 T
  dqm_tag_.push_back("AagBunch0T");  // statistic collection from First Collision After Abort Gap @ 0 T
  dqm_tag_.push_back("IsoMuon");     // statistic collection from Isolated Muon @ 3.8 T
  dqm_tag_.push_back("IsoMuon0T");   // statistic collection from Isolated Muon @ 0 T
  dqm_tag_.push_back("Harvest");     // statistic collection: Harvest

  Charge_Vs_Index.insert(Charge_Vs_Index.begin(), dqm_tag_.size(), nullptr);
  //Charge_Vs_Index_Absolute.insert( Charge_Vs_Index_Absolute.begin(), dqm_tag_.size(), 0);
  Charge_Vs_PathlengthTIB.insert(Charge_Vs_PathlengthTIB.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTOB.insert(Charge_Vs_PathlengthTOB.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTIDP.insert(Charge_Vs_PathlengthTIDP.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTIDM.insert(Charge_Vs_PathlengthTIDM.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTECP1.insert(Charge_Vs_PathlengthTECP1.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTECP2.insert(Charge_Vs_PathlengthTECP2.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTECM1.insert(Charge_Vs_PathlengthTECM1.begin(), dqm_tag_.size(), nullptr);
  Charge_Vs_PathlengthTECM2.insert(Charge_Vs_PathlengthTECM2.begin(), dqm_tag_.size(), nullptr);

  // configure token for gathering the ntuple variables
  edm::ParameterSet swhallowgain_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("gain");

  string label = swhallowgain_pset.getUntrackedParameter<string>("label");
  CalibPrefix_ = swhallowgain_pset.getUntrackedParameter<string>("prefix");
  CalibSuffix_ = swhallowgain_pset.getUntrackedParameter<string>("suffix");

  trackindex_token_ = consumes<std::vector<int>>(edm::InputTag(label, CalibPrefix_ + "trackindex" + CalibSuffix_));
  rawid_token_ = consumes<std::vector<unsigned int>>(edm::InputTag(label, CalibPrefix_ + "rawid" + CalibSuffix_));
  localdirx_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "localdirx" + CalibSuffix_));
  localdiry_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "localdiry" + CalibSuffix_));
  localdirz_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "localdirz" + CalibSuffix_));
  firststrip_token_ =
      consumes<std::vector<unsigned short>>(edm::InputTag(label, CalibPrefix_ + "firststrip" + CalibSuffix_));
  nstrips_token_ = consumes<std::vector<unsigned short>>(edm::InputTag(label, CalibPrefix_ + "nstrips" + CalibSuffix_));
  saturation_token_ = consumes<std::vector<bool>>(edm::InputTag(label, CalibPrefix_ + "saturation" + CalibSuffix_));
  overlapping_token_ = consumes<std::vector<bool>>(edm::InputTag(label, CalibPrefix_ + "overlapping" + CalibSuffix_));
  farfromedge_token_ = consumes<std::vector<bool>>(edm::InputTag(label, CalibPrefix_ + "farfromedge" + CalibSuffix_));
  charge_token_ = consumes<std::vector<unsigned int>>(edm::InputTag(label, CalibPrefix_ + "charge" + CalibSuffix_));
  path_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "path" + CalibSuffix_));
  chargeoverpath_token_ =
      consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "chargeoverpath" + CalibSuffix_));
  amplitude_token_ =
      consumes<std::vector<unsigned char>>(edm::InputTag(label, CalibPrefix_ + "amplitude" + CalibSuffix_));
  gainused_token_ = consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "gainused" + CalibSuffix_));
  gainusedTick_token_ =
      consumes<std::vector<double>>(edm::InputTag(label, CalibPrefix_ + "gainusedTick" + CalibSuffix_));

  edm::ParameterSet evtinfo_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("evtinfo");
  label = evtinfo_pset.getUntrackedParameter<string>("label");
  EventPrefix_ = evtinfo_pset.getUntrackedParameter<string>("prefix");
  EventSuffix_ = evtinfo_pset.getUntrackedParameter<string>("suffix");
  TrigTech_token_ = consumes<std::vector<bool>>(edm::InputTag(label, EventPrefix_ + "TrigTech" + EventSuffix_));

  edm::ParameterSet track_pset = iConfig.getUntrackedParameter<edm::ParameterSet>("tracks");
  label = track_pset.getUntrackedParameter<string>("label");
  TrackPrefix_ = track_pset.getUntrackedParameter<string>("prefix");
  TrackSuffix_ = track_pset.getUntrackedParameter<string>("suffix");

  trackchi2ndof_token_ = consumes<std::vector<double>>(edm::InputTag(label, TrackPrefix_ + "chi2ndof" + TrackSuffix_));
  trackp_token_ = consumes<std::vector<float>>(edm::InputTag(label, TrackPrefix_ + "momentum" + TrackSuffix_));
  trackpt_token_ = consumes<std::vector<float>>(edm::InputTag(label, TrackPrefix_ + "pt" + TrackSuffix_));
  tracketa_token_ = consumes<std::vector<double>>(edm::InputTag(label, TrackPrefix_ + "eta" + TrackSuffix_));
  trackphi_token_ = consumes<std::vector<double>>(edm::InputTag(label, TrackPrefix_ + "phi" + TrackSuffix_));
  trackhitsvalid_token_ =
      consumes<std::vector<unsigned int>>(edm::InputTag(label, TrackPrefix_ + "hitsvalid" + TrackSuffix_));
  trackalgo_token_ = consumes<std::vector<int>>(edm::InputTag(label, TrackPrefix_ + "algo" + TrackSuffix_));

  tTopo_ = nullptr;
}

void SiStripGainFromCalibTree::bookDQMHistos(const char* dqm_dir, const char* tag) {
  edm::LogInfo("SiStripGainFromCalibTree")
      << "Setting " << dqm_dir << "in DQM and booking histograms for tag " << tag << std::endl;

  if (strcmp(booked_dir_.c_str(), dqm_dir) != 0) {
    booked_dir_ = dqm_dir;
    dbe->setCurrentFolder(dqm_dir);
  }

  std::string stag(tag);
  if (!stag.empty() && stag[0] != '_')
    stag.insert(0, 1, '_');

  std::string cvi = std::string("Charge_Vs_Index") + stag;
  //std::string cviA     = std::string("Charge_Vs_Index_Absolute")  + stag;
  std::string cvpTIB = std::string("Charge_Vs_PathlengthTIB") + stag;
  std::string cvpTOB = std::string("Charge_Vs_PathlengthTOB") + stag;
  std::string cvpTIDP = std::string("Charge_Vs_PathlengthTIDP") + stag;
  std::string cvpTIDM = std::string("Charge_Vs_PathlengthTIDM") + stag;
  std::string cvpTECP1 = std::string("Charge_Vs_PathlengthTECP1") + stag;
  std::string cvpTECP2 = std::string("Charge_Vs_PathlengthTECP2") + stag;
  std::string cvpTECM1 = std::string("Charge_Vs_PathlengthTECM1") + stag;
  std::string cvpTECM2 = std::string("Charge_Vs_PathlengthTECM2") + stag;

  int elepos = (m_harvestingMode && AlgoMode == "PCL") ? Harvest : statCollectionFromMode(tag);

  // The cluster charge is stored by exploiting a non uniform binning in order
  // reduce the histogram memory size. The bin width is relaxed with a falling
  // exponential function and the bin boundaries are stored in the binYarray.
  // The binXarray is used to provide as many bins as the APVs.
  //
  // More details about this implementations are here:
  // https://indico.cern.ch/event/649344/contributions/2672267/attachments/1498323/2332518/OptimizeChHisto.pdf

  std::vector<float> binXarray;
  binXarray.reserve(NStripAPVs + 1);
  for (int a = 0; a <= NStripAPVs; a++) {
    binXarray.push_back((float)a);
  }

  std::array<float, 688> binYarray;
  double p0 = 5.445;
  double p1 = 0.002113;
  double p2 = 69.01576;
  double y = 0.;
  for (int b = 0; b < 687; b++) {
    binYarray[b] = y;
    if (y <= 902.)
      y = y + 2.;
    else
      y = (p0 - log(exp(p0 - p1 * y) - p2 * p1)) / p1;
  }
  binYarray[687] = 4000.;

  Charge_Vs_Index[elepos] = dbe->book2S(cvi.c_str(), cvi.c_str(), NStripAPVs, &binXarray[0], 687, binYarray.data());
  //Charge_Vs_Index_Absolute[elepos]  = dbe->book2S(cviA.c_str()    , cviA.c_str()    , 88625, 0   , 88624,1000,0,4000);
  Charge_Vs_PathlengthTIB[elepos] = dbe->book2S(cvpTIB.c_str(), cvpTIB.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTOB[elepos] = dbe->book2S(cvpTOB.c_str(), cvpTOB.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTIDP[elepos] = dbe->book2S(cvpTIDP.c_str(), cvpTIDP.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTIDM[elepos] = dbe->book2S(cvpTIDM.c_str(), cvpTIDM.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTECP1[elepos] = dbe->book2S(cvpTECP1.c_str(), cvpTECP1.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTECP2[elepos] = dbe->book2S(cvpTECP2.c_str(), cvpTECP2.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTECM1[elepos] = dbe->book2S(cvpTECM1.c_str(), cvpTECM1.c_str(), 20, 0.3, 1.3, 250, 0, 2000);
  Charge_Vs_PathlengthTECM2[elepos] = dbe->book2S(cvpTECM2.c_str(), cvpTECM2.c_str(), 20, 0.3, 1.3, 250, 0, 2000);

  //Book Charge monitoring histograms
  std::vector<std::pair<std::string, std::string>> hnames =
      APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = dbe->book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.);
    int id = APVGain::subdetectorId((hnames[i]).first);
    int side = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_1[elepos].push_back(APVGain::APVmon(id, side, plane, monitor));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG2");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = dbe->book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.);
    int id = APVGain::subdetectorId((hnames[i]).first);
    int side = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_2[elepos].push_back(APVGain::APVmon(id, side, plane, monitor));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = dbe->book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.);
    int id = APVGain::subdetectorId((hnames[i]).first);
    int side = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_3[elepos].push_back(APVGain::APVmon(id, side, plane, monitor));
  }

  hnames = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1G2");
  for (unsigned int i = 0; i < hnames.size(); i++) {
    std::string htag = (hnames[i]).first + stag;
    MonitorElement* monitor = dbe->book1DD(htag.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.);
    int id = APVGain::subdetectorId((hnames[i]).first);
    int side = APVGain::subdetectorSide((hnames[i]).first);
    int plane = APVGain::subdetectorPlane((hnames[i]).first);
    Charge_4[elepos].push_back(APVGain::APVmon(id, side, plane, monitor));
  }

  //Book validation histograms
  if (m_harvestingMode) {
    int MPVbin = 300;
    float MPVmin = 0.;
    float MPVmax = 600.;

    MPV_Vs_EtaTIB = dbe->book2DD("MPV_vs_EtaTIB", "MPV vs Eta TIB", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
    MPV_Vs_EtaTID = dbe->book2DD("MPV_vs_EtaTID", "MPV vs Eta TID", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
    MPV_Vs_EtaTOB = dbe->book2DD("MPV_vs_EtaTOB", "MPV vs Eta TOB", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
    MPV_Vs_EtaTEC = dbe->book2DD("MPV_vs_EtaTEC", "MPV vs Eta TEC", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
    MPV_Vs_EtaTECthin = dbe->book2DD("MPV_vs_EtaTEC1", "MPV vs Eta TEC-thin", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
    MPV_Vs_EtaTECthick = dbe->book2DD("MPV_vs_EtaTEC2", "MPV vs Eta TEC-thick", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);

    MPV_Vs_PhiTIB = dbe->book2DD("MPV_vs_PhiTIB", "MPV vs Phi TIB", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
    MPV_Vs_PhiTID = dbe->book2DD("MPV_vs_PhiTID", "MPV vs Phi TID", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
    MPV_Vs_PhiTOB = dbe->book2DD("MPV_vs_PhiTOB", "MPV vs Phi TOB", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
    MPV_Vs_PhiTEC = dbe->book2DD("MPV_vs_PhiTEC", "MPV vs Phi TEC", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
    MPV_Vs_PhiTECthin = dbe->book2DD("MPV_vs_PhiTEC1", "MPV vs Phi TEC-thin", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
    MPV_Vs_PhiTECthick = dbe->book2DD("MPV_vs_PhiTEC2", "MPV vs Phi TEC-thick", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);

    NoMPVfit = dbe->book2DD("NoMPVfit", "Modules with bad Landau Fit", 350, -350, 350, 240, 0, 120);
    NoMPVmasked = dbe->book2DD("NoMPVmasked", "Masked Modules", 350, -350, 350, 240, 0, 120);

    Gains = dbe->book1DD("Gains", "Gains", 300, 0, 2);
    MPVs = dbe->book1DD("MPVs", "MPVs", MPVbin, MPVmin, MPVmax);
    MPVs320 = dbe->book1DD("MPV_320", "MPV 320 thickness", MPVbin, MPVmin, MPVmax);
    MPVs500 = dbe->book1DD("MPV_500", "MPV 500 thickness", MPVbin, MPVmin, MPVmax);
    MPVsTIB = dbe->book1DD("MPV_TIB", "MPV TIB", MPVbin, MPVmin, MPVmax);
    MPVsTID = dbe->book1DD("MPV_TID", "MPV TID", MPVbin, MPVmin, MPVmax);
    MPVsTIDP = dbe->book1DD("MPV_TIDP", "MPV TIDP", MPVbin, MPVmin, MPVmax);
    MPVsTIDM = dbe->book1DD("MPV_TIDM", "MPV TIDM", MPVbin, MPVmin, MPVmax);
    MPVsTOB = dbe->book1DD("MPV_TOB", "MPV TOB", MPVbin, MPVmin, MPVmax);
    MPVsTEC = dbe->book1DD("MPV_TEC", "MPV TEC", MPVbin, MPVmin, MPVmax);
    MPVsTECP = dbe->book1DD("MPV_TECP", "MPV TECP", MPVbin, MPVmin, MPVmax);
    MPVsTECM = dbe->book1DD("MPV_TECM", "MPV TECM", MPVbin, MPVmin, MPVmax);
    MPVsTECthin = dbe->book1DD("MPV_TEC1", "MPV TEC1", MPVbin, MPVmin, MPVmax);
    MPVsTECthick = dbe->book1DD("MPV_TEC2", "MPV TEC2", MPVbin, MPVmin, MPVmax);
    MPVsTECP1 = dbe->book1DD("MPV_TECP1", "MPV TECP1", MPVbin, MPVmin, MPVmax);
    MPVsTECP2 = dbe->book1DD("MPV_TECP2", "MPV TECP2", MPVbin, MPVmin, MPVmax);
    MPVsTECM1 = dbe->book1DD("MPV_TECM1", "MPV TECM1", MPVbin, MPVmin, MPVmax);
    MPVsTECM2 = dbe->book1DD("MPV_TECM2", "MPV TECM2", MPVbin, MPVmin, MPVmax);

    MPVError = dbe->book1DD("MPVError", "MPV Error", 150, 0, 150);
    MPVErrorVsMPV = dbe->book2DD("MPVErrorVsMPV", "MPV Error vs MPV", 300, 0, 600, 150, 0, 150);
    MPVErrorVsEta = dbe->book2DD("MPVErrorVsEta", "MPV Error vs Eta", 50, -3.0, 3.0, 150, 0, 150);
    MPVErrorVsPhi = dbe->book2DD("MPVErrorVsPhi", "MPV Error vs Phi", 50, -3.4, 3.4, 150, 0, 150);
    MPVErrorVsN = dbe->book2DD("MPVErrorVsN", "MPV Error vs N", 500, 0, 1000, 150, 0, 150);

    DiffWRTPrevGainTIB = dbe->book1DD("DiffWRTPrevGainTIB", "Diff w.r.t. PrevGain TIB", 250, 0, 2);
    DiffWRTPrevGainTID = dbe->book1DD("DiffWRTPrevGainTID", "Diff w.r.t. PrevGain TID", 250, 0, 2);
    DiffWRTPrevGainTOB = dbe->book1DD("DiffWRTPrevGainTOB", "Diff w.r.t. PrevGain TOB", 250, 0, 2);
    DiffWRTPrevGainTEC = dbe->book1DD("DiffWRTPrevGainTEC", "Diff w.r.t. PrevGain TEC", 250, 0, 2);

    GainVsPrevGainTIB = dbe->book2DD("GainVsPrevGainTIB", "Gain vs PrevGain TIB", 100, 0, 2, 100, 0, 2);
    GainVsPrevGainTID = dbe->book2DD("GainVsPrevGainTID", "Gain vs PrevGain TID", 100, 0, 2, 100, 0, 2);
    GainVsPrevGainTOB = dbe->book2DD("GainVsPrevGainTOB", "Gain vs PrevGain TOB", 100, 0, 2, 100, 0, 2);
    GainVsPrevGainTEC = dbe->book2DD("GainVsPrevGainTEC", "Gain vs PrevGain TEC", 100, 0, 2, 100, 0, 2);

    std::vector<std::pair<std::string, std::string>> hnames =
        APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "newG2");
    for (unsigned int i = 0; i < hnames.size(); i++) {
      MonitorElement* monitor = dbe->book1DD((hnames[i]).first.c_str(), (hnames[i]).second.c_str(), 100, 0., 1000.);
      int id = APVGain::subdetectorId((hnames[i]).first);
      int side = APVGain::subdetectorSide((hnames[i]).first);
      int plane = APVGain::subdetectorPlane((hnames[i]).first);
      newCharge.push_back(APVGain::APVmon(id, side, plane, monitor));
    }
  }
}

void SiStripGainFromCalibTree::algoBeginJob(const edm::EventSetup& iSetup) {
  edm::LogInfo("SiStripGainFromCalibTree") << "AlgoMode        : " << AlgoMode << "\n"
                                           << "CalibrationMode : " << m_calibrationMode << "\n"
                                           << "HarvestingMode  : " << m_harvestingMode << std::endl;
  //Setup DQM histograms
  if (AlgoMode != "PCL" or m_harvestingMode) {
    const char* dqm_dir = "AlCaReco/SiStripGainsHarvesting/";
    this->bookDQMHistos(dqm_dir, dqm_tag_[statCollectionFromMode(m_calibrationMode.c_str())].c_str());
  } else {
    //Check consistency of calibration Mode and BField only for the ALCAPROMPT in the PCL workflow
    if (!isBFieldConsistentWithMode(iSetup)) {
      string prevMode = m_calibrationMode;
      swapBFieldMode();
      edm::LogInfo("SiStripGainFromCalibTree") << "Switching calibration mode for endorsing BField status: " << prevMode
                                               << " ==> " << m_calibrationMode << std::endl;
    }
    std::string dqm_dir = m_DQMdir + ((m_splitDQMstat) ? m_calibrationMode : "") + "/";
    int elem = statCollectionFromMode(m_calibrationMode.c_str());
    this->bookDQMHistos(dqm_dir.c_str(), dqm_tag_[elem].c_str());
  }

  edm::ESHandle<TrackerTopology> TopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(TopoHandle);
  tTopo_ = TopoHandle.product();

  edm::ESHandle<TrackerGeometry> tkGeom;
  iSetup.get<TrackerDigiGeometryRecord>().get(tkGeom);
  auto const& Det = tkGeom->dets();

  NPixelDets = 0;
  NStripAPVs = 0;
  unsigned int Index = 0;
  for (unsigned int i = 0; i < Det.size(); i++) {
    DetId Detid = Det[i]->geographicalId();
    int SubDet = Detid.subdetId();

    if (SubDet == StripSubdetector::TIB || SubDet == StripSubdetector::TID || SubDet == StripSubdetector::TOB ||
        SubDet == StripSubdetector::TEC) {
      auto DetUnit = dynamic_cast<const StripGeomDetUnit*>(Det[i]);
      if (!DetUnit)
        continue;

      const StripTopology& Topo = DetUnit->specificTopology();
      unsigned int NAPV = Topo.nstrips() / 128;

      for (unsigned int j = 0; j < NAPV; j++) {
        stAPVGain* APV = new stAPVGain;
        APV->Index = Index;
        APV->Bin = -1;
        APV->DetId = Detid.rawId();
        APV->APVId = j;
        APV->SubDet = SubDet;
        APV->FitMPV = -1;
        APV->FitMPVErr = -1;
        APV->FitWidth = -1;
        APV->FitWidthErr = -1;
        APV->FitChi2 = -1;
        APV->FitNorm = -1;
        APV->Gain = -1;
        APV->PreviousGain = 1;
        APV->PreviousGainTick = 1;
        APV->x = DetUnit->position().basicVector().x();
        APV->y = DetUnit->position().basicVector().y();
        APV->z = DetUnit->position().basicVector().z();
        APV->Eta = DetUnit->position().basicVector().eta();
        APV->Phi = DetUnit->position().basicVector().phi();
        APV->R = DetUnit->position().basicVector().transverse();
        APV->Thickness = DetUnit->surface().bounds().thickness();
        APV->NEntries = 0;
        APV->isMasked = false;

        APVsCollOrdered.push_back(APV);
        APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
        Index++;
        NStripAPVs++;
      }
    }
  }

  for (unsigned int i = 0; i < Det.size();
       i++) {  //Make two loop such that the Pixel information is added at the end --> make transition simpler
    DetId Detid = Det[i]->geographicalId();
    int SubDet = Detid.subdetId();
    if (SubDet == PixelSubdetector::PixelBarrel || SubDet == PixelSubdetector::PixelEndcap) {
      auto DetUnit = dynamic_cast<const PixelGeomDetUnit*>(Det[i]);
      if (!DetUnit)
        continue;

      const PixelTopology& Topo = DetUnit->specificTopology();
      unsigned int NROCRow = Topo.nrows() / (80.);
      unsigned int NROCCol = Topo.ncolumns() / (52.);

      for (unsigned int j = 0; j < NROCRow; j++) {
        for (unsigned int i = 0; i < NROCCol; i++) {
          stAPVGain* APV = new stAPVGain;
          APV->Index = Index;
          APV->Bin = -1;
          APV->DetId = Detid.rawId();
          APV->APVId = (j << 3 | i);
          APV->SubDet = SubDet;
          APV->FitMPV = -1;
          APV->FitMPVErr = -1;
          APV->FitWidth = -1;
          APV->FitWidthErr = -1;
          APV->FitChi2 = -1;
          APV->Gain = -1;
          APV->PreviousGain = 1;
          APV->PreviousGainTick = 1;
          APV->x = DetUnit->position().basicVector().x();
          APV->y = DetUnit->position().basicVector().y();
          APV->z = DetUnit->position().basicVector().z();
          APV->Eta = DetUnit->position().basicVector().eta();
          APV->Phi = DetUnit->position().basicVector().phi();
          APV->R = DetUnit->position().basicVector().transverse();
          APV->Thickness = DetUnit->surface().bounds().thickness();
          APV->isMasked = false;  //SiPixelQuality_->IsModuleBad(Detid.rawId());
          APV->NEntries = 0;

          APVsCollOrdered.push_back(APV);
          APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
          Index++;
          NPixelDets++;
        }
      }
    }
  }

  MakeCalibrationMap();

  NEvent = 0;
  NTrack = 0;
  NClusterStrip = 0;
  NClusterPixel = 0;
  SRun = 1 << 31;
  ERun = 0;
  GOOD = 0;
  BAD = 0;
  MASKED = 0;
}

bool SiStripGainFromCalibTree::isBFieldConsistentWithMode(const edm::EventSetup& iSetup) const {
  edm::ESHandle<RunInfo> runInfo;
  iSetup.get<RunInfoRcd>().get(runInfo);

  double average_current = runInfo.product()->m_avg_current;
  bool isOn = (average_current > MagFieldCurrentTh);
  bool is0T = (m_calibrationMode.substr(m_calibrationMode.length() - 2) == "0T");

  return ((isOn && !is0T) || (!isOn && is0T));
}

void SiStripGainFromCalibTree::swapBFieldMode() {
  if (m_calibrationMode.substr(m_calibrationMode.length() - 2) == "0T") {
    m_calibrationMode.erase(m_calibrationMode.length() - 2, 2);
  } else {
    m_calibrationMode.append("0T");
  }
}

void SiStripGainFromCalibTree::algoBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  if (!m_harvestingMode && AlgoMode == "PCL") {
    //Check consistency of calibration Mode and BField only for the ALCAPROMPT in the PCL workflow
    if (!isBFieldConsistentWithMode(iSetup)) {
      string prevMode = m_calibrationMode;
      swapBFieldMode();
      edm::LogInfo("SiStripGainFromCalibTree") << "Switching calibration mode for endorsing BField status: " << prevMode
                                               << " ==> " << m_calibrationMode << std::endl;
    }
  }

  edm::ESHandle<SiStripGain> gainHandle;
  iSetup.get<SiStripGainRcd>().get(gainHandle);
  if (!gainHandle.isValid()) {
    edm::LogError("SiStripGainFromCalibTree") << "gainHandle is not valid\n";
    exit(0);
  }

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);

  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    stAPVGain* APV = APVsCollOrdered[a];

    // MM. 03/03/2017 all of this makes sense only for SiStrip (i.e. get me out of here if I am on a pixel ROC)
    if (APV->SubDet == PixelSubdetector::PixelBarrel || APV->SubDet == PixelSubdetector::PixelEndcap)
      continue;

    APV->isMasked = SiStripQuality_->IsApvBad(APV->DetId, APV->APVId);
    //      if(!FirstSetOfConstants){

    if (gainHandle->getNumberOfTags() != 2) {
      edm::LogError("SiStripGainFromCalibTree") << "NUMBER OF GAIN TAG IS EXPECTED TO BE 2\n";
      fflush(stdout);
      exit(0);
    };
    float newPreviousGain = gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 1), 1);
    if (APV->PreviousGain != 1 and newPreviousGain != APV->PreviousGain)
      edm::LogWarning("SiStripGainFromCalibTree") << "WARNING: ParticleGain in the global tag changed\n";
    APV->PreviousGain = newPreviousGain;

    float newPreviousGainTick = gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 0), 0);
    if (APV->PreviousGainTick != 1 and newPreviousGainTick != APV->PreviousGainTick) {
      edm::LogWarning("SiStripGainFromCalibTree")
          << "WARNING: TickMarkGain in the global tag changed\n"
          << std::endl
          << " APV->SubDet: " << APV->SubDet << " APV->APVId:" << APV->APVId << std::endl
          << " APV->PreviousGainTick: " << APV->PreviousGainTick << " newPreviousGainTick: " << newPreviousGainTick
          << std::endl;
    }
    APV->PreviousGainTick = newPreviousGainTick;

    //printf("DETID = %7i APVID=%1i Previous Gain=%8.4f (G1) x %8.4f (G2)\n",APV->DetId,APV->APVId,APV->PreviousGainTick, APV->PreviousGain);
    //      }
  }
}

void SiStripGainFromCalibTree::algoEndRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  if (AlgoMode == "PCL" && !m_harvestingMode)
    return;  //nothing to do in that case

  if (AlgoMode == "PCL" and m_harvestingMode) {
    // Load the 2D histograms from the DQM objects
    // When running in AlCaHarvesting mode the histos are already booked and should be just retrieved from
    // DQMStore so that they can be used in the fit

    edm::LogInfo("SiStripGainFromCalibTree") << "Starting harvesting statistics" << std::endl;

    // check the required tag before adding histograms
    int elepos = statCollectionFromMode(m_calibrationMode.c_str());
    if (elepos != Harvest) {
      //collect statistics from DQM into the related monitored elements
      std::string stag = m_calibrationMode;
      if (!stag.empty() && stag[0] != '_')
        stag.insert(0, 1, '_');

      if (elepos == -1) {
        //implememt backward compatibility
        elepos = 0;
        stag = "";
      }

      std::string DQM_dir = m_DQMdir + ((m_splitDQMstat) ? m_calibrationMode : "");

      std::string cvi = DQM_dir + std::string("/Charge_Vs_Index") + stag;
      //std::string cviA     = DQM_dir + std::string("/Charge_Vs_Index_Absolute")  + stag;
      std::string cvpTIB = DQM_dir + std::string("/Charge_Vs_PathlengthTIB") + stag;
      std::string cvpTOB = DQM_dir + std::string("/Charge_Vs_PathlengthTOB") + stag;
      std::string cvpTIDP = DQM_dir + std::string("/Charge_Vs_PathlengthTIDP") + stag;
      std::string cvpTIDM = DQM_dir + std::string("/Charge_Vs_PathlengthTIDM") + stag;
      std::string cvpTECP1 = DQM_dir + std::string("/Charge_Vs_PathlengthTECP1") + stag;
      std::string cvpTECP2 = DQM_dir + std::string("/Charge_Vs_PathlengthTECP2") + stag;
      std::string cvpTECM1 = DQM_dir + std::string("/Charge_Vs_PathlengthTECM1") + stag;
      std::string cvpTECM2 = DQM_dir + std::string("/Charge_Vs_PathlengthTECM2") + stag;

      Charge_Vs_Index[elepos] = dbe->get(cvi);
      //Charge_Vs_Index_Absolute[elepos]  = dbe->get(cviA.c_str());
      Charge_Vs_PathlengthTIB[elepos] = dbe->get(cvpTIB);
      Charge_Vs_PathlengthTOB[elepos] = dbe->get(cvpTOB);
      Charge_Vs_PathlengthTIDP[elepos] = dbe->get(cvpTIDP);
      Charge_Vs_PathlengthTIDM[elepos] = dbe->get(cvpTIDM);
      Charge_Vs_PathlengthTECP1[elepos] = dbe->get(cvpTECP1);
      Charge_Vs_PathlengthTECP2[elepos] = dbe->get(cvpTECP2);
      Charge_Vs_PathlengthTECM1[elepos] = dbe->get(cvpTECM1);
      Charge_Vs_PathlengthTECM2[elepos] = dbe->get(cvpTECM2);

      if (Charge_Vs_Index[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvi.c_str() << ", statistics will not be summed!" << std::endl;
      } else {
        merge((Charge_Vs_Index[Harvest])->getTH2S(), (Charge_Vs_Index[elepos])->getTH2S());
        edm::LogInfo("SiStripGainFromCalibTree")
            << "Harvesting " << (Charge_Vs_Index[elepos])->getTH2S()->GetEntries() << " more clusters" << std::endl;
      }

      //if (Charge_Vs_Index_Absolute[elepos]==0) {
      //    edm::LogError("SiStripGainFromCalibTree") << "Harvesting: could not retrieve " << cviA.c_str()
      //                                              << ", statistics will not be summed!" << std::endl;
      //} else merge( (Charge_Vs_Index_Absolute[Harvest])->getTH2S(), (Charge_Vs_Index_Absolute[elepos])->getTH2S() );

      if (Charge_Vs_PathlengthTIB[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTIB.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTIB[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTIB[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTOB[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTOB.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTOB[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTOB[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTIDP[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTIDP.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTIDP[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTIDP[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTIDM[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTIDM.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTIDM[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTIDM[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTECP1[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTECP1.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTECP1[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTECP1[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTECP2[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTECP2.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTECP2[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTECP2[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTECM1[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTECM1.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTECM1[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTECM1[elepos])->getTH2S());

      if (Charge_Vs_PathlengthTECM2[elepos] == nullptr) {
        edm::LogError("SiStripGainFromCalibTree")
            << "Harvesting: could not retrieve " << cvpTECM2.c_str() << ", statistics will not be summed!" << std::endl;
      } else
        (Charge_Vs_PathlengthTECM2[Harvest])->getTH2S()->Add((Charge_Vs_PathlengthTECM2[elepos])->getTH2S());

      // Gather Charge monitoring histograms
      std::vector<std::pair<std::string, std::string>> tags =
          APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "");
      for (unsigned int i = 0; i < tags.size(); i++) {
        std::string tag = DQM_dir + "/" + (tags[i]).first + stag;
        Charge_1[elepos].push_back(APVGain::APVmon(0, 0, 0, dbe->get(tag)));
        if ((Charge_1[elepos][i]).getMonitor() == nullptr) {
          edm::LogError("SiStripGainFromCalibTree")
              << "Harvesting: could not retrieve " << tag.c_str() << ", statistics will not be summed!" << std::endl;
        } else
          (Charge_1[Harvest][i]).getMonitor()->getTH1D()->Add((Charge_1[elepos][i]).getMonitor()->getTH1D());
      }

      tags = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG2");
      for (unsigned int i = 0; i < tags.size(); i++) {
        std::string tag = DQM_dir + "/" + (tags[i]).first + stag;
        Charge_2[elepos].push_back(APVGain::APVmon(0, 0, 0, dbe->get(tag)));
        if ((Charge_2[elepos][i]).getMonitor() == nullptr) {
          edm::LogError("SiStripGainFromCalibTree")
              << "Harvesting: could not retrieve " << tag.c_str() << ", statistics will not be summed!" << std::endl;
        } else
          (Charge_2[Harvest][i]).getMonitor()->getTH1D()->Add((Charge_2[elepos][i]).getMonitor()->getTH1D());
      }

      tags = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1");
      for (unsigned int i = 0; i < tags.size(); i++) {
        std::string tag = DQM_dir + "/" + (tags[i]).first + stag;
        Charge_3[elepos].push_back(APVGain::APVmon(0, 0, 0, dbe->get(tag)));
        if ((Charge_3[elepos][i]).getMonitor() == nullptr) {
          edm::LogError("SiStripGainFromCalibTree")
              << "Harvesting: could not retrieve " << tag.c_str() << ", statistics will not be summed!" << std::endl;
        } else
          (Charge_3[Harvest][i]).getMonitor()->getTH1D()->Add((Charge_3[elepos][i]).getMonitor()->getTH1D());
      }

      tags = APVGain::monHnames(VChargeHisto, doChargeMonitorPerPlane, "woG1G2");
      for (unsigned int i = 0; i < tags.size(); i++) {
        std::string tag = DQM_dir + "/" + (tags[i]).first + stag;
        Charge_4[elepos].push_back(APVGain::APVmon(0, 0, 0, dbe->get(tag)));
        if ((Charge_4[elepos][i]).getMonitor() == nullptr) {
          edm::LogError("SiStripGainFromCalibTree")
              << "Harvesting: could not retrieve " << tag.c_str() << ", statistics will not be summed!" << std::endl;
        } else
          (Charge_4[Harvest][i]).getMonitor()->getTH1D()->Add((Charge_4[elepos][i]).getMonitor()->getTH1D());
      }
    }
  }
}

void SiStripGainFromCalibTree::algoEndJob() {
  if (AlgoMode == "PCL" && !m_harvestingMode)
    return;  //nothing to do in that case

  if (AlgoMode == "CalibTree") {
    edm::LogInfo("SiStripGainFromCalibTree") << "Analyzing calibration tree" << std::endl;
    // Loop on calibTrees to fill the 2D histograms
    algoAnalyzeTheTree();
  } else if (m_harvestingMode) {
    NClusterStrip = (Charge_Vs_Index[Harvest])->getTH2S()->Integral(0, NStripAPVs + 1, 0, 99999);
    //NClusterPixel = (Charge_Vs_Index[Harvest])->getTH2S()->Integral(NStripAPVs+2, NStripAPVs+NPixelDets+2, 0, 99999 );
  }

  // Now that we have the full statistics we can extract the information of the 2D histograms
  algoComputeMPVandGain();

  // Result monitoring
  qualityMonitor();

  // Force the DB object writing,
  // thus setting the IOV as the first processed run (if timeFromEndRun is set to false)
  storeOnDbNow();

  if (AlgoMode != "PCL" or saveSummary) {
    edm::LogInfo("SiStripGainFromCalibTree") << "Saving summary into root file" << std::endl;

    //also save the 2D monitor elements to this file as TH2D tfs
    tfs = edm::Service<TFileService>().operator->();

    //save only the statistics for the calibrationTag
    int elepos = statCollectionFromMode(m_calibrationMode.c_str());

    if (Charge_Vs_Index[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_Index[elepos])->getTH2S());
    //if( Charge_Vs_Index_Absolute[elepos]!=0 )  tfs->make<TH2S> ( *(Charge_Vs_Index_Absolute[elepos])->getTH2S() );
    if (Charge_Vs_PathlengthTIB[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTIB[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTOB[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTOB[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTIDP[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTIDP[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTIDM[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTIDM[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTECP1[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTECP1[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTECP2[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTECP2[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTECM1[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTECM1[elepos])->getTH2S());
    if (Charge_Vs_PathlengthTECM2[elepos] != nullptr)
      tfs->make<TH2S>(*(Charge_Vs_PathlengthTECM2[elepos])->getTH2S());

    storeOnTree(tfs);
  }
}

void SiStripGainFromCalibTree::getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange) {
  FitResults[0] = -0.5;  //MPV
  FitResults[1] = 0;     //MPV error
  FitResults[2] = -0.5;  //Width
  FitResults[3] = 0;     //Width error
  FitResults[4] = -0.5;  //Fit Chi2/NDF
  FitResults[5] = 0;     //Normalization

  if (InputHisto->GetEntries() < MinNrEntries)
    return;

  // perform fit with standard landau
  TF1* MyLandau = new TF1("MyLandau", "landau", LowRange, HighRange);
  MyLandau->SetParameter(1, 300);
  InputHisto->Fit(MyLandau, "0QR WW");

  // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
  FitResults[0] = MyLandau->GetParameter(1);                      //MPV
  FitResults[1] = MyLandau->GetParError(1);                       //MPV error
  FitResults[2] = MyLandau->GetParameter(2);                      //Width
  FitResults[3] = MyLandau->GetParError(2);                       //Width error
  FitResults[4] = MyLandau->GetChisquare() / MyLandau->GetNDF();  //Fit Chi2/NDF
  FitResults[5] = MyLandau->GetParameter(0);

  delete MyLandau;
}

bool SiStripGainFromCalibTree::IsGoodLandauFit(double* FitResults) {
  if (FitResults[0] <= 0)
    return false;
  //   if(FitResults[1] > MaxMPVError   )return false;
  //   if(FitResults[4] > MaxChi2OverNDF)return false;
  return true;
}

void SiStripGainFromCalibTree::processEvent() {
  edm::LogInfo("SiStripGainFromCalibTree") << "Processing run " << runnumber << " and event " << eventnumber << " for "
                                           << m_calibrationMode << " calibration." << std::endl;

  if (runnumber < SRun)
    SRun = runnumber;
  if (runnumber > ERun)
    ERun = runnumber;

  NEvent++;
  NTrack += (*trackp).size();

  int elepos = statCollectionFromMode(m_calibrationMode.c_str());

  unsigned int FirstAmplitude = 0;
  for (unsigned int i = 0; i < (*chargeoverpath).size(); i++) {
    FirstAmplitude += (*nstrips)[i];
    int TI = (*trackindex)[i];

    //printf("%i - %i - %i %i %i\n", (int)(*rawid)[i], (int)(*firststrip)[i]/128, (int)(*farfromedge)[i], (int)(*overlapping)[i], (int)(*saturation )[i] );
    if ((*tracketa)[TI] < MinTrackEta)
      continue;
    if ((*tracketa)[TI] > MaxTrackEta)
      continue;
    if ((*trackp)[TI] < MinTrackMomentum)
      continue;
    if ((*trackp)[TI] > MaxTrackMomentum)
      continue;
    if ((*trackhitsvalid)[TI] < MinTrackHits)
      continue;
    if ((*trackchi2ndof)[TI] > MaxTrackChiOverNdf)
      continue;
    if ((*trackalgo)[TI] > MaxTrackingIteration)
      continue;

    stAPVGain* APV =
        APVsColl[((*rawid)[i] << 4) |
                 ((*firststrip)[i] /
                  128)];  //works for both strip and pixel thanks to firstStrip encoding for pixel in the calibTree

    if (APV->SubDet > 2 && (*farfromedge)[i] == false)
      continue;
    if (APV->SubDet > 2 && (*overlapping)[i] == true)
      continue;
    if (APV->SubDet > 2 && (*saturation)[i] && !AllowSaturation)
      continue;
    if (APV->SubDet > 2 && (*nstrips)[i] > MaxNrStrips)
      continue;

    //printf("detId=%7i run=%7i event=%9i charge=%5i cs=%3i\n",(*rawid)[i],runnumber,eventnumber,(*charge)[i],(*nstrips)[i]);

    //double trans = atan2((*localdiry)[i],(*localdirx)[i])*(180/3.14159265);
    //double alpha = acos ((*localdirx)[i] / sqrt( pow((*localdirx)[i],2) +  pow((*localdirz)[i],2) ) ) * (180/3.14159265);
    //double beta  = acos ((*localdiry)[i] / sqrt( pow((*localdirx)[i],2) +  pow((*localdirz)[i],2) ) ) * (180/3.14159265);

    //printf("NStrip = %i : Charge = %i --> Path = %f  --> ChargeOverPath=%f\n",(*nstrips)[i],(*charge)[i],(*path)[i],(*chargeoverpath)[i]);
    //printf("Amplitudes: ");
    //for(unsigned int a=0;a<(*nstrips)[i];a++){printf("%i ",(*amplitude)[FirstAmplitude+a]);}
    //printf("\n");

    if (APV->SubDet > 2) {
      NClusterStrip++;
    } else {
      NClusterPixel++;
    }

    int Charge = 0;
    if (APV->SubDet > 2 && (useCalibration || !FirstSetOfConstants)) {
      bool Saturation = false;
      for (unsigned int s = 0; s < (*nstrips)[i]; s++) {
        int StripCharge = (*amplitude)[FirstAmplitude - (*nstrips)[i] + s];
        if (useCalibration && !FirstSetOfConstants) {
          StripCharge = (int)(StripCharge * (APV->PreviousGain / APV->CalibGain));
        } else if (useCalibration) {
          StripCharge = (int)(StripCharge / APV->CalibGain);
        } else if (!FirstSetOfConstants) {
          StripCharge = (int)(StripCharge * APV->PreviousGain);
        }
        if (StripCharge > 1024) {
          StripCharge = 255;
          Saturation = true;
        } else if (StripCharge > 254) {
          StripCharge = 254;
          Saturation = true;
        }
        Charge += StripCharge;
      }
      if (Saturation && !AllowSaturation)
        continue;
    } else if (APV->SubDet > 2) {
      Charge = (*charge)[i];
    } else {
      Charge = (*charge)[i] / 265.0;  //expected scale factor between pixel and strip charge
    }

    //printf("ChargeDifference = %i Vs %i with Gain = %f\n",(*charge)[i],Charge,APV->CalibGain);

    double ClusterChargeOverPath = ((double)Charge) / (*path)[i];
    if (APV->SubDet > 2) {
      if (Validation) {
        ClusterChargeOverPath /= (*gainused)[i];
      }
      if (OldGainRemoving) {
        ClusterChargeOverPath *= (*gainused)[i];
      }
    }

    // keep pixel cluster charge processing until here
    if (APV->SubDet <= 2)
      continue;

    (Charge_Vs_Index[elepos])->Fill(APV->Index, ClusterChargeOverPath);

    // Compute the charge for monitoring and fill the relative histograms
    int mCharge1 = 0;
    int mCharge2 = 0;
    int mCharge3 = 0;
    int mCharge4 = 0;
    if (APV->SubDet > 2) {
      for (unsigned int s = 0; s < (*nstrips)[i]; s++) {
        int StripCharge = (*amplitude)[FirstAmplitude - (*nstrips)[i] + s];
        if (StripCharge > 1024)
          StripCharge = 255;
        else if (StripCharge > 254)
          StripCharge = 254;
        mCharge1 += StripCharge;
        mCharge2 += StripCharge;
        mCharge3 += StripCharge;
        mCharge4 += StripCharge;
      }
      // Revome gains for monitoring
      mCharge2 *= (*gainused)[i];                         // remove G2
      mCharge3 *= (*gainusedTick)[i];                     // remove G1
      mCharge4 *= ((*gainused)[i] * (*gainusedTick)[i]);  // remove G1 and G2
    }
    std::vector<APVGain::APVmon>& v1 = Charge_1[elepos];
    std::vector<MonitorElement*> cmon1 = APVGain::FetchMonitor(v1, (*rawid)[i], tTopo_);
    for (unsigned int m = 0; m < cmon1.size(); m++)
      cmon1[m]->Fill(((double)mCharge1) / (*path)[i]);

    std::vector<APVGain::APVmon>& v2 = Charge_2[elepos];
    std::vector<MonitorElement*> cmon2 = APVGain::FetchMonitor(v2, (*rawid)[i], tTopo_);
    for (unsigned int m = 0; m < cmon2.size(); m++)
      cmon2[m]->Fill(((double)mCharge2) / (*path)[i]);

    std::vector<APVGain::APVmon>& v3 = Charge_3[elepos];
    std::vector<MonitorElement*> cmon3 = APVGain::FetchMonitor(v3, (*rawid)[i], tTopo_);
    for (unsigned int m = 0; m < cmon3.size(); m++)
      cmon3[m]->Fill(((double)mCharge3) / (*path)[i]);

    std::vector<APVGain::APVmon>& v4 = Charge_4[elepos];
    std::vector<MonitorElement*> cmon4 = APVGain::FetchMonitor(v4, (*rawid)[i], tTopo_);
    for (unsigned int m = 0; m < cmon4.size(); m++)
      cmon4[m]->Fill(((double)mCharge4) / (*path)[i]);

    // Fill Charge Vs pathLenght histograms
    if (APV->SubDet == StripSubdetector::TIB) {
      (Charge_Vs_PathlengthTIB[elepos])->Fill((*path)[i], Charge);  // TIB

    } else if (APV->SubDet == StripSubdetector::TOB) {
      (Charge_Vs_PathlengthTOB[elepos])->Fill((*path)[i], Charge);  // TOB

    } else if (APV->SubDet == StripSubdetector::TID) {
      if (APV->Eta < 0) {
        (Charge_Vs_PathlengthTIDM[elepos])->Fill((*path)[i], Charge);
      }  // TID minus
      else if (APV->Eta > 0) {
        (Charge_Vs_PathlengthTIDP[elepos])->Fill((*path)[i], Charge);
      }  // TID plus

    } else if (APV->SubDet == StripSubdetector::TEC) {
      if (APV->Eta < 0) {
        if (APV->Thickness < 0.04) {
          (Charge_Vs_PathlengthTECM1[elepos])->Fill((*path)[i], Charge);
        }  // TEC minus, type 1
        else if (APV->Thickness > 0.04) {
          (Charge_Vs_PathlengthTECM2[elepos])->Fill((*path)[i], Charge);
        }  // TEC minus, type 2
      } else if (APV->Eta > 0) {
        if (APV->Thickness < 0.04) {
          (Charge_Vs_PathlengthTECP1[elepos])->Fill((*path)[i], Charge);
        }  // TEC plus, type 1
        else if (APV->Thickness > 0.04) {
          (Charge_Vs_PathlengthTECP2[elepos])->Fill((*path)[i], Charge);
        }  // TEC plus, type 2
      }
    }

  }  // END OF ON-CLUSTER LOOP
}  //END OF processEvent()

void SiStripGainFromCalibTree::algoAnalyzeTheTree() {
  for (unsigned int i = 0; i < VInputFiles.size(); i++) {
    printf("Openning file %3i/%3i --> %s\n", i + 1, (int)VInputFiles.size(), (char*)(VInputFiles[i].c_str()));
    fflush(stdout);
    TFile* tfile = TFile::Open(VInputFiles[i].c_str());
    TString tree_path = TString::Format("gainCalibrationTree%s/tree", m_calibrationMode.c_str());
    TTree* tree = dynamic_cast<TTree*>(tfile->Get(tree_path.Data()));

    tree->SetBranchAddress((EventPrefix_ + "event" + EventSuffix_).c_str(), &eventnumber, nullptr);
    tree->SetBranchAddress((EventPrefix_ + "run" + EventSuffix_).c_str(), &runnumber, nullptr);
    tree->SetBranchAddress((EventPrefix_ + "TrigTech" + EventSuffix_).c_str(), &TrigTech, nullptr);

    tree->SetBranchAddress((TrackPrefix_ + "chi2ndof" + TrackSuffix_).c_str(), &trackchi2ndof, nullptr);
    tree->SetBranchAddress((TrackPrefix_ + "momentum" + TrackSuffix_).c_str(), &trackp, nullptr);
    tree->SetBranchAddress((TrackPrefix_ + "pt" + TrackSuffix_).c_str(), &trackpt, nullptr);
    tree->SetBranchAddress((TrackPrefix_ + "eta" + TrackSuffix_).c_str(), &tracketa, nullptr);
    tree->SetBranchAddress((TrackPrefix_ + "phi" + TrackSuffix_).c_str(), &trackphi, nullptr);
    tree->SetBranchAddress((TrackPrefix_ + "hitsvalid" + TrackSuffix_).c_str(), &trackhitsvalid, nullptr);
    tree->SetBranchAddress((TrackPrefix_ + "algo" + TrackSuffix_).c_str(), &trackalgo, nullptr);

    tree->SetBranchAddress((CalibPrefix_ + "trackindex" + CalibSuffix_).c_str(), &trackindex, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "rawid" + CalibSuffix_).c_str(), &rawid, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "localdirx" + CalibSuffix_).c_str(), &localdirx, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "localdiry" + CalibSuffix_).c_str(), &localdiry, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "localdirz" + CalibSuffix_).c_str(), &localdirz, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "firststrip" + CalibSuffix_).c_str(), &firststrip, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "nstrips" + CalibSuffix_).c_str(), &nstrips, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "saturation" + CalibSuffix_).c_str(), &saturation, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "overlapping" + CalibSuffix_).c_str(), &overlapping, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "farfromedge" + CalibSuffix_).c_str(), &farfromedge, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "charge" + CalibSuffix_).c_str(), &charge, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "path" + CalibSuffix_).c_str(), &path, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "chargeoverpath" + CalibSuffix_).c_str(), &chargeoverpath, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "amplitude" + CalibSuffix_).c_str(), &amplitude, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "gainused" + CalibSuffix_).c_str(), &gainused, nullptr);
    tree->SetBranchAddress((CalibPrefix_ + "gainusedTick" + CalibSuffix_).c_str(), &gainusedTick, nullptr);

    unsigned int nentries = tree->GetEntries();
    printf("Number of Events = %i + %i = %i\n", NEvent, nentries, (NEvent + nentries));
    printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
    printf("Looping on the Tree          :");
    int TreeStep = nentries / 50;
    if (TreeStep <= 1)
      TreeStep = 1;
    for (unsigned int ientry = 0; ientry < tree->GetEntries(); ientry++) {
      if (ientry % TreeStep == 0) {
        printf(".");
        fflush(stdout);
      }
      tree->GetEntry(ientry);
      processEvent();
    }
    printf("\n");  // END OF EVENT LOOP
  }
}

void SiStripGainFromCalibTree::algoComputeMPVandGain() {
  unsigned int I = 0;
  TH1F* Proj = nullptr;
  double FitResults[6];
  double MPVmean = 300;

  int elepos = (AlgoMode == "PCL") ? Harvest : statCollectionFromMode(m_calibrationMode.c_str());

  if (Charge_Vs_Index[elepos] == nullptr) {
    edm::LogError("SiStripGainFromCalibTree")
        << "Harvesting: could not execute algoComputeMPVandGain method because " << m_calibrationMode.c_str()
        << " statistics cannot be retrieved.\n"
        << "Please check if input contains " << m_calibrationMode.c_str() << " data." << std::endl;
    return;
  }

  TH2S* chvsidx = (Charge_Vs_Index[elepos])->getTH2S();

  printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
  printf("Fitting Charge Distribution  :");
  int TreeStep = APVsColl.size() / 50;
  for (auto it = APVsColl.begin(); it != APVsColl.end(); it++, I++) {
    if (I % TreeStep == 0) {
      printf(".");
      fflush(stdout);
    }
    stAPVGain* APV = it->second;
    if (APV->Bin < 0)
      APV->Bin = chvsidx->GetXaxis()->FindBin(APV->Index);

    if (APV->isMasked) {
      APV->Gain = APV->PreviousGain;
      MASKED++;
      continue;
    }

    Proj = (TH1F*)(chvsidx->ProjectionY(
        "", chvsidx->GetXaxis()->FindBin(APV->Index), chvsidx->GetXaxis()->FindBin(APV->Index), "e"));
    if (!Proj)
      continue;

    if (CalibrationLevel == 0) {
    } else if (CalibrationLevel == 1) {
      int SecondAPVId = APV->APVId;
      if (SecondAPVId % 2 == 0) {
        SecondAPVId = SecondAPVId + 1;
      } else {
        SecondAPVId = SecondAPVId - 1;
      }
      stAPVGain* APV2 = APVsColl[(APV->DetId << 4) | SecondAPVId];
      if (APV2->Bin < 0)
        APV2->Bin = chvsidx->GetXaxis()->FindBin(APV2->Index);
      TH1F* Proj2 = (TH1F*)(chvsidx->ProjectionY("", APV2->Bin, APV2->Bin, "e"));
      if (Proj2) {
        Proj->Add(Proj2, 1);
        delete Proj2;
      }
    } else if (CalibrationLevel == 2) {
      for (unsigned int i = 0; i < 16; i++) {  //loop up to 6APV for Strip and up to 16 for Pixels
        auto tmpit = APVsColl.find((APV->DetId << 4) | i);
        if (tmpit == APVsColl.end())
          continue;
        stAPVGain* APV2 = tmpit->second;
        if (APV2->DetId != APV->DetId || APV2->APVId == APV->APVId)
          continue;
        if (APV2->Bin < 0)
          APV2->Bin = chvsidx->GetXaxis()->FindBin(APV2->Index);
        TH1F* Proj2 = (TH1F*)(chvsidx->ProjectionY("", APV2->Bin, APV2->Bin, "e"));
        if (Proj2) {
          Proj->Add(Proj2, 1);
          delete Proj2;
        }
      }
    } else {
      CalibrationLevel = 0;
      printf("Unknown Calibration Level, will assume %i\n", CalibrationLevel);
    }

    getPeakOfLandau(Proj, FitResults);
    APV->FitMPV = FitResults[0];
    APV->FitMPVErr = FitResults[1];
    APV->FitWidth = FitResults[2];
    APV->FitWidthErr = FitResults[3];
    APV->FitChi2 = FitResults[4];
    APV->FitNorm = FitResults[5];
    APV->NEntries = Proj->GetEntries();

    if (IsGoodLandauFit(FitResults)) {
      APV->Gain = APV->FitMPV / MPVmean;
      if (APV->SubDet > 2)
        GOOD++;
    } else {
      APV->Gain = APV->PreviousGain;
      if (APV->SubDet > 2)
        BAD++;
    }
    if (APV->Gain <= 0)
      APV->Gain = 1;

    //printf("%5i/%5i:  %6i - %1i  %5E Entries --> MPV = %f +- %f\n",I,APVsColl.size(),APV->DetId, APV->APVId, Proj->GetEntries(), FitResults[0], FitResults[1]);fflush(stdout);
    delete Proj;
  }
  printf("\n");
}

void SiStripGainFromCalibTree::qualityMonitor() {
  int elepos = (AlgoMode == "PCL") ? Harvest : statCollectionFromMode(m_calibrationMode.c_str());

  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    stAPVGain* APV = APVsCollOrdered[a];
    if (APV == nullptr)
      continue;

    unsigned int Index = APV->Index;
    unsigned int SubDet = APV->SubDet;
    unsigned int DetId = APV->DetId;
    float z = APV->z;
    float Eta = APV->Eta;
    float R = APV->R;
    float Phi = APV->Phi;
    float Thickness = APV->Thickness;
    double FitMPV = APV->FitMPV;
    double FitMPVErr = APV->FitMPVErr;
    double Gain = APV->Gain;
    double NEntries = APV->NEntries;
    double PreviousGain = APV->PreviousGain;

    if (SubDet < 3)
      continue;  // avoid to loop over Pixel det id

    if (Gain != 1.) {
      std::vector<MonitorElement*> charge_histos = APVGain::FetchMonitor(newCharge, DetId, tTopo_);
      TH2S* chvsidx = (Charge_Vs_Index[elepos])->getTH2S();
      int bin = chvsidx->GetXaxis()->FindBin(Index);
      TH1D* Proj = chvsidx->ProjectionY("proj", bin, bin);
      for (int binId = 0; binId < Proj->GetXaxis()->GetNbins(); binId++) {
        double new_charge = Proj->GetXaxis()->GetBinCenter(binId) / Gain;
        if (Proj->GetBinContent(binId) != 0.) {
          for (unsigned int h = 0; h < charge_histos.size(); h++) {
            TH1D* chisto = (charge_histos[h])->getTH1D();
            for (int e = 0; e < Proj->GetBinContent(binId); e++)
              chisto->Fill(new_charge);
          }
        }
      }
    }

    if (FitMPV <= 0.) {  // No fit of MPV
      if (APV->isMasked)
        NoMPVmasked->Fill(z, R);
      else
        NoMPVfit->Fill(z, R);

    } else {  // Fit of MPV
      if (FitMPV > 0.)
        Gains->Fill(Gain);

      MPVs->Fill(FitMPV);
      if (Thickness < 0.04)
        MPVs320->Fill(FitMPV);
      if (Thickness > 0.04)
        MPVs500->Fill(FitMPV);

      MPVError->Fill(FitMPVErr);
      MPVErrorVsMPV->Fill(FitMPV, FitMPVErr);
      MPVErrorVsEta->Fill(Eta, FitMPVErr);
      MPVErrorVsPhi->Fill(Phi, FitMPVErr);
      MPVErrorVsN->Fill(NEntries, FitMPVErr);

      if (SubDet == 3) {
        MPV_Vs_EtaTIB->Fill(Eta, FitMPV);
        MPV_Vs_PhiTIB->Fill(Phi, FitMPV);
        MPVsTIB->Fill(FitMPV);

      } else if (SubDet == 4) {
        MPV_Vs_EtaTID->Fill(Eta, FitMPV);
        MPV_Vs_PhiTID->Fill(Phi, FitMPV);
        MPVsTID->Fill(FitMPV);
        if (Eta < 0.)
          MPVsTIDM->Fill(FitMPV);
        if (Eta > 0.)
          MPVsTIDP->Fill(FitMPV);

      } else if (SubDet == 5) {
        MPV_Vs_EtaTOB->Fill(Eta, FitMPV);
        MPV_Vs_PhiTOB->Fill(Phi, FitMPV);
        MPVsTOB->Fill(FitMPV);

      } else if (SubDet == 6) {
        MPV_Vs_EtaTEC->Fill(Eta, FitMPV);
        MPV_Vs_PhiTEC->Fill(Phi, FitMPV);
        MPVsTEC->Fill(FitMPV);
        if (Eta < 0.)
          MPVsTECM->Fill(FitMPV);
        if (Eta > 0.)
          MPVsTECP->Fill(FitMPV);
        if (Thickness < 0.04) {
          MPV_Vs_EtaTECthin->Fill(Eta, FitMPV);
          MPV_Vs_PhiTECthin->Fill(Phi, FitMPV);
          MPVsTECthin->Fill(FitMPV);
          if (Eta > 0.)
            MPVsTECP1->Fill(FitMPV);
          if (Eta < 0.)
            MPVsTECM1->Fill(FitMPV);
        }
        if (Thickness > 0.04) {
          MPV_Vs_EtaTECthick->Fill(Eta, FitMPV);
          MPV_Vs_PhiTECthick->Fill(Phi, FitMPV);
          MPVsTECthick->Fill(FitMPV);
          if (Eta > 0.)
            MPVsTECP2->Fill(FitMPV);
          if (Eta < 0.)
            MPVsTECM2->Fill(FitMPV);
        }
      }
    }

    if (SubDet == 3 && PreviousGain != 0.)
      DiffWRTPrevGainTIB->Fill(Gain / PreviousGain);
    else if (SubDet == 4 && PreviousGain != 0.)
      DiffWRTPrevGainTID->Fill(Gain / PreviousGain);
    else if (SubDet == 5 && PreviousGain != 0.)
      DiffWRTPrevGainTOB->Fill(Gain / PreviousGain);
    else if (SubDet == 6 && PreviousGain != 0.)
      DiffWRTPrevGainTEC->Fill(Gain / PreviousGain);

    if (SubDet == 3)
      GainVsPrevGainTIB->Fill(PreviousGain, Gain);
    else if (SubDet == 4)
      GainVsPrevGainTID->Fill(PreviousGain, Gain);
    else if (SubDet == 5)
      GainVsPrevGainTOB->Fill(PreviousGain, Gain);
    else if (SubDet == 6)
      GainVsPrevGainTEC->Fill(PreviousGain, Gain);
  }
}

void SiStripGainFromCalibTree::storeOnTree(TFileService* tfs) {
  unsigned int tree_Index;
  unsigned int tree_Bin;
  unsigned int tree_DetId;
  unsigned char tree_APVId;
  unsigned char tree_SubDet;
  float tree_x;
  float tree_y;
  float tree_z;
  float tree_Eta;
  float tree_R;
  float tree_Phi;
  float tree_Thickness;
  float tree_FitMPV;
  float tree_FitMPVErr;
  float tree_FitWidth;
  float tree_FitWidthErr;
  float tree_FitChi2NDF;
  float tree_FitNorm;
  double tree_Gain;
  double tree_PrevGain;
  double tree_PrevGainTick;
  double tree_NEntries;
  bool tree_isMasked;

  TTree* MyTree;
  MyTree = tfs->make<TTree>("APVGain", "APVGain");
  MyTree->Branch("Index", &tree_Index, "Index/i");
  MyTree->Branch("Bin", &tree_Bin, "Bin/i");
  MyTree->Branch("DetId", &tree_DetId, "DetId/i");
  MyTree->Branch("APVId", &tree_APVId, "APVId/b");
  MyTree->Branch("SubDet", &tree_SubDet, "SubDet/b");
  MyTree->Branch("x", &tree_x, "x/F");
  MyTree->Branch("y", &tree_y, "y/F");
  MyTree->Branch("z", &tree_z, "z/F");
  MyTree->Branch("Eta", &tree_Eta, "Eta/F");
  MyTree->Branch("R", &tree_R, "R/F");
  MyTree->Branch("Phi", &tree_Phi, "Phi/F");
  MyTree->Branch("Thickness", &tree_Thickness, "Thickness/F");
  MyTree->Branch("FitMPV", &tree_FitMPV, "FitMPV/F");
  MyTree->Branch("FitMPVErr", &tree_FitMPVErr, "FitMPVErr/F");
  MyTree->Branch("FitWidth", &tree_FitWidth, "FitWidth/F");
  MyTree->Branch("FitWidthErr", &tree_FitWidthErr, "FitWidthErr/F");
  MyTree->Branch("FitChi2NDF", &tree_FitChi2NDF, "FitChi2NDF/F");
  MyTree->Branch("FitNorm", &tree_FitNorm, "FitNorm/F");
  MyTree->Branch("Gain", &tree_Gain, "Gain/D");
  MyTree->Branch("PrevGain", &tree_PrevGain, "PrevGain/D");
  MyTree->Branch("PrevGainTick", &tree_PrevGainTick, "PrevGainTick/D");
  MyTree->Branch("NEntries", &tree_NEntries, "NEntries/D");
  MyTree->Branch("isMasked", &tree_isMasked, "isMasked/O");

  FILE* Gains = stdout;
  fprintf(Gains, "NEvents   = %i\n", NEvent);
  fprintf(Gains, "NTracks   = %i\n", NTrack);
  //fprintf(Gains,"NClustersPixel = %i\n",NClusterPixel);
  fprintf(Gains, "NClustersStrip = %i\n", NClusterStrip);
  //fprintf(Gains,"Number of Pixel Dets = %lu\n",static_cast<unsigned long>(NPixelDets));
  fprintf(Gains, "Number of Strip APVs = %lu\n", static_cast<unsigned long>(NStripAPVs));
  fprintf(Gains,
          "GoodFits = %i BadFits = %i ratio = %f%%   (MASKED=%i)\n",
          GOOD,
          BAD,
          (100.0 * GOOD) / (GOOD + BAD),
          MASKED);

  Gains = fopen(OutputGains.c_str(), "w");
  fprintf(Gains, "NEvents   = %i\n", NEvent);
  fprintf(Gains, "NTracks   = %i\n", NTrack);
  //fprintf(Gains,"NClustersPixel = %i\n",NClusterPixel);
  fprintf(Gains, "NClustersStrip = %i\n", NClusterStrip);
  fprintf(Gains, "Number of Strip APVs = %lu\n", static_cast<unsigned long>(NStripAPVs));
  //fprintf(Gains,"Number of Pixel Dets = %lu\n",static_cast<unsigned long>(NPixelDets));
  fprintf(Gains,
          "GoodFits = %i BadFits = %i ratio = %f%%   (MASKED=%i)\n",
          GOOD,
          BAD,
          (100.0 * GOOD) / (GOOD + BAD),
          MASKED);

  int elepos = statCollectionFromMode(m_calibrationMode.c_str());

  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    stAPVGain* APV = APVsCollOrdered[a];
    if (APV == nullptr)
      continue;
    //     printf(      "%i | %i | PreviousGain = %7.5f NewGain = %7.5f (#clusters=%8.0f)\n", APV->DetId,APV->APVId,APV->PreviousGain,APV->Gain, APV->NEntries);
    fprintf(Gains,
            "%i | %i | PreviousGain = %7.5f(tick) x %7.5f(particle) NewGain (particle) = %7.5f (#clusters=%8.0f)\n",
            APV->DetId,
            APV->APVId,
            APV->PreviousGainTick,
            APV->PreviousGain,
            APV->Gain,
            APV->NEntries);

    tree_Index = APV->Index;
    tree_Bin = (Charge_Vs_Index[elepos])->getTH2S()->GetXaxis()->FindBin(APV->Index);
    tree_DetId = APV->DetId;
    tree_APVId = APV->APVId;
    tree_SubDet = APV->SubDet;
    tree_x = APV->x;
    tree_y = APV->y;
    tree_z = APV->z;
    tree_Eta = APV->Eta;
    tree_R = APV->R;
    tree_Phi = APV->Phi;
    tree_Thickness = APV->Thickness;
    tree_FitMPV = APV->FitMPV;
    tree_FitMPVErr = APV->FitMPVErr;
    tree_FitWidth = APV->FitWidth;
    tree_FitWidthErr = APV->FitWidthErr;
    tree_FitChi2NDF = APV->FitChi2;
    tree_FitNorm = APV->FitNorm;
    tree_Gain = APV->Gain;
    tree_PrevGain = APV->PreviousGain;
    tree_PrevGainTick = APV->PreviousGainTick;
    tree_NEntries = APV->NEntries;
    tree_isMasked = APV->isMasked;

    if (tree_DetId == 402673324) {
      printf("%i | %i : %f --> %f  (%f)\n", tree_DetId, tree_APVId, tree_PrevGain, tree_Gain, tree_NEntries);
    }

    MyTree->Fill();
  }
  if (Gains)
    fclose(Gains);
}

bool SiStripGainFromCalibTree::produceTagFilter() {
  // The goal of this function is to check wether or not there is enough statistics
  // to produce a meaningful tag for the DB
  int elepos = (AlgoMode == "PCL") ? Harvest : statCollectionFromMode(m_calibrationMode.c_str());
  if (Charge_Vs_Index[elepos] == nullptr) {
    edm::LogError("SiStripGainFromCalibTree")
        << "produceTagFilter -> Return false: could not retrieve the " << m_calibrationMode.c_str() << " statistics.\n"
        << "Please check if input contains " << m_calibrationMode.c_str() << " data." << std::endl;
    return false;
  }

  float integral = (Charge_Vs_Index[elepos])->getTH2S()->Integral();
  if ((Charge_Vs_Index[elepos])->getTH2S()->Integral(0, NStripAPVs + 1, 0, 99999) < tagCondition_NClusters) {
    edm::LogWarning("SiStripGainFromCalibTree")
        << "calibrationMode  -> " << m_calibrationMode << "\n"
        << "produceTagFilter -> Return false: Statistics is too low : " << integral << endl;
    return false;
  }
  if ((1.0 * GOOD) / (GOOD + BAD) < tagCondition_GoodFrac) {
    edm::LogWarning("SiStripGainFromCalibTree")
        << "calibrationMode  -> " << m_calibrationMode << "\n"
        << "produceTagFilter ->  Return false: ratio of GOOD/TOTAL is too low: " << (1.0 * GOOD) / (GOOD + BAD) << endl;
    return false;
  }
  return true;
}

std::unique_ptr<SiStripApvGain> SiStripGainFromCalibTree::getNewObject() {
  auto obj = std::make_unique<SiStripApvGain>();
  if (!m_harvestingMode)
    return obj;

  if (!produceTagFilter()) {
    edm::LogWarning("SiStripGainFromCalibTree")
        << "getNewObject -> will not produce a paylaod because produceTagFilter returned false " << endl;
    setDoStore(false);
    return obj;
  }

  std::vector<float>* theSiStripVector = nullptr;
  unsigned int PreviousDetId = 0;
  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    stAPVGain* APV = APVsCollOrdered[a];
    if (APV == nullptr) {
      printf("Bug\n");
      continue;
    }
    if (APV->SubDet <= 2)
      continue;
    if (APV->DetId != PreviousDetId) {
      if (theSiStripVector != nullptr) {
        SiStripApvGain::Range range(theSiStripVector->begin(), theSiStripVector->end());
        if (!obj->put(PreviousDetId, range))
          printf("Bug to put detId = %i\n", PreviousDetId);
      }
      theSiStripVector = new std::vector<float>;
      PreviousDetId = APV->DetId;
    }
    theSiStripVector->push_back(APV->Gain);
  }
  if (theSiStripVector != nullptr) {
    SiStripApvGain::Range range(theSiStripVector->begin(), theSiStripVector->end());
    if (!obj->put(PreviousDetId, range))
      printf("Bug to put detId = %i\n", PreviousDetId);
  }

  if (theSiStripVector != nullptr)
    delete theSiStripVector;

  return obj;
}

SiStripGainFromCalibTree::~SiStripGainFromCalibTree() {
  APVsColl.clear();
  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    stAPVGain* APV = APVsCollOrdered[a];
    if (APV != nullptr)
      delete APV;
  }
  APVsCollOrdered.clear();
}

void SiStripGainFromCalibTree::MakeCalibrationMap() {
  if (!useCalibration)
    return;

  TChain* t1 = new TChain("SiStripCalib/APVGain");
  t1->Add(m_calibrationPath.c_str());

  unsigned int tree_DetId;
  unsigned char tree_APVId;
  double tree_Gain;

  t1->SetBranchAddress("DetId", &tree_DetId);
  t1->SetBranchAddress("APVId", &tree_APVId);
  t1->SetBranchAddress("Gain", &tree_Gain);

  for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
    t1->GetEntry(ientry);
    stAPVGain* APV = APVsColl[(tree_DetId << 4) | (unsigned int)tree_APVId];
    APV->CalibGain = tree_Gain;
  }

  delete t1;
}

void SiStripGainFromCalibTree::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // in AlCaHarvesting mode we just need to run the logic in the endJob step
  if (m_harvestingMode)
    return;

  if (AlgoMode == "CalibTree")
    return;

  eventnumber = iEvent.id().event();
  runnumber = iEvent.id().run();
  auto handle01 = connect(TrigTech, TrigTech_token_, iEvent);
  auto handle02 = connect(trackchi2ndof, trackchi2ndof_token_, iEvent);
  auto handle03 = connect(trackp, trackp_token_, iEvent);
  auto handle04 = connect(trackpt, trackpt_token_, iEvent);
  auto handle05 = connect(tracketa, tracketa_token_, iEvent);
  auto handle06 = connect(trackphi, trackphi_token_, iEvent);
  auto handle07 = connect(trackhitsvalid, trackhitsvalid_token_, iEvent);
  auto handle08 = connect(trackindex, trackindex_token_, iEvent);
  auto handle09 = connect(rawid, rawid_token_, iEvent);
  auto handle11 = connect(localdirx, localdirx_token_, iEvent);
  auto handle12 = connect(localdiry, localdiry_token_, iEvent);
  auto handle13 = connect(localdirz, localdirz_token_, iEvent);
  auto handle14 = connect(firststrip, firststrip_token_, iEvent);
  auto handle15 = connect(nstrips, nstrips_token_, iEvent);
  auto handle16 = connect(saturation, saturation_token_, iEvent);
  auto handle17 = connect(overlapping, overlapping_token_, iEvent);
  auto handle18 = connect(farfromedge, farfromedge_token_, iEvent);
  auto handle19 = connect(charge, charge_token_, iEvent);
  auto handle21 = connect(path, path_token_, iEvent);
  auto handle22 = connect(chargeoverpath, chargeoverpath_token_, iEvent);
  auto handle23 = connect(amplitude, amplitude_token_, iEvent);
  auto handle24 = connect(gainused, gainused_token_, iEvent);
  auto handle25 = connect(gainusedTick, gainusedTick_token_, iEvent);
  auto handle26 = connect(trackalgo, trackalgo_token_, iEvent);

  processEvent();
}

DEFINE_FWK_MODULE(SiStripGainFromCalibTree);
