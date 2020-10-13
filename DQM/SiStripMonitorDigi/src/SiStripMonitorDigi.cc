// -*- c++ -*-
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc
 */
// Original Author:  Dorian Kcira
//         Created:  Sat Feb  4 20:49:10 CET 2006
//         Updated:  May 2020, Suvankar Roy Chowdhury
#include <fstream>
#include "TNamed.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"

#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

#include "TMath.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <iostream>

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

/* mia: but is there not a smarter way ?!?!?! */
namespace {
  const double NORBITS_PER_LS = 262144.;  // per-second value would be 11223
}

//--------------------------------------------------------------------------------------------
SiStripMonitorDigi::SiStripMonitorDigi(const edm::ParameterSet& iConfig)
    :  //dqmStore_(edm::Service<DQMStore>().operator->()),
      conf_(iConfig),
      show_mechanical_structure_view(true),
      show_readout_view(false),
      show_control_view(false),
      select_all_detectors(true),
      reset_each_run(false),
      folder_organizer(),
      m_cacheID_(0) {
  firstEvent = -1;
  eventNb = 0;

  // Detector Partitions
  SubDetPhasePartMap["TIB"] = "TI";
  SubDetPhasePartMap["TID__MINUS"] = "TI";
  SubDetPhasePartMap["TID__PLUS"] = "TI";
  SubDetPhasePartMap["TOB"] = "TO";
  SubDetPhasePartMap["TEC__MINUS"] = "TM";
  SubDetPhasePartMap["TEC__PLUS"] = "TP";

  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");

  // get Digi Producer List
  digiProducerList = conf_.getParameter<std::vector<edm::InputTag> >("DigiProducersList");
  for (auto const& tag : digiProducerList) {
    digiProducerTokenList.push_back(consumes<edm::DetSetVector<SiStripDigi> >(tag));
  }

  //get on/off option for every cluster from cfi
  edm::ParameterSet ParametersNumberOfDigis = conf_.getParameter<edm::ParameterSet>("TH1NumberOfDigis");
  layerswitchnumdigison = ParametersNumberOfDigis.getParameter<bool>("layerswitchon");
  moduleswitchnumdigison = ParametersNumberOfDigis.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersNumberOfDigisPerStrip = conf_.getParameter<edm::ParameterSet>("TH1NumberOfDigisPerStrip");
  moduleswitchnumdigispstripon = ParametersNumberOfDigisPerStrip.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersADCsHottestStrip = conf_.getParameter<edm::ParameterSet>("TH1ADCsHottestStrip");
  layerswitchadchotteston = ParametersADCsHottestStrip.getParameter<bool>("layerswitchon");
  moduleswitchadchotteston = ParametersADCsHottestStrip.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersADCsCoolestStrip = conf_.getParameter<edm::ParameterSet>("TH1ADCsCoolestStrip");
  layerswitchadccooleston = ParametersADCsCoolestStrip.getParameter<bool>("layerswitchon");
  moduleswitchadccooleston = ParametersADCsCoolestStrip.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersDigiADCs = conf_.getParameter<edm::ParameterSet>("TH1DigiADCs");
  layerswitchdigiadcson = ParametersDigiADCs.getParameter<bool>("layerswitchon");
  moduleswitchdigiadcson = ParametersDigiADCs.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersStripOccupancy = conf_.getParameter<edm::ParameterSet>("TH1StripOccupancy");
  layerswitchstripoccupancyon = ParametersStripOccupancy.getParameter<bool>("layerswitchon");
  moduleswitchstripoccupancyon = ParametersStripOccupancy.getParameter<bool>("moduleswitchon");

  edm::ParameterSet ParametersDigiProf = conf_.getParameter<edm::ParameterSet>("TProfNumberOfDigi");
  layerswitchnumdigisprofon = ParametersDigiProf.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersDigiADC = conf_.getParameter<edm::ParameterSet>("TProfDigiADC");
  layerswitchdigiadcprofon = ParametersDigiProf.getParameter<bool>("layerswitchon");

  edm::ParameterSet ParametersTotDigiProf = conf_.getParameter<edm::ParameterSet>("TProfTotalNumberOfDigis");
  subdetswitchtotdigiprofon = ParametersTotDigiProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersTotDigiFailure = conf_.getParameter<edm::ParameterSet>("TotalNumberOfDigisFailure");
  subdetswitchtotdigifailureon = ParametersTotDigiFailure.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersDigiApvProf = conf_.getParameter<edm::ParameterSet>("TProfDigiApvCycle");
  subdetswitchapvcycleprofon = ParametersDigiApvProf.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersDigiApvTH2 = conf_.getParameter<edm::ParameterSet>("TH2DigiApvCycle");
  subdetswitchapvcycleth2on = ParametersDigiApvTH2.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersNApvShots = conf_.getParameter<edm::ParameterSet>("TH1NApvShots");
  subdetswitchnapvshotson = ParametersNApvShots.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersNStripApvShots = conf_.getParameter<edm::ParameterSet>("TH1NStripsApvShots");
  subdetswitchnstripsapvshotson = ParametersNStripApvShots.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersChargeMedianApvShots = conf_.getParameter<edm::ParameterSet>("TH1ChargeMedianApvShots");
  subdetswitchchargemedianapvshotson = ParametersChargeMedianApvShots.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersApvNumberApvShots = conf_.getParameter<edm::ParameterSet>("TH1ApvNumApvShots");
  subdetswitchapvshotsApvon = ParametersChargeMedianApvShots.getParameter<bool>("subdetswitchon");

  edm::ParameterSet ParametersNApvShotsProf = conf_.getParameter<edm::ParameterSet>("TProfNShotsVsTime");
  subdetswitchapvshotsonprof = ParametersNApvShotsProf.getParameter<bool>("subdetswitchon");

  //Global Histograms

  edm::ParameterSet ParametersGlobalNApvShots = conf_.getParameter<edm::ParameterSet>("TH1NApvShots");
  globalswitchnapvshotson = ParametersGlobalNApvShots.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersGlobalNApvShotsProf = conf_.getParameter<edm::ParameterSet>("TProfGlobalNShots");
  globalsummaryapvshotson = ParametersGlobalNApvShotsProf.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersGlobalNStripApvShots = conf_.getParameter<edm::ParameterSet>("TH1NStripsApvShots");
  globalswitchnstripsapvshotson = ParametersGlobalNStripApvShots.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersGlobalApvNumApvShots = conf_.getParameter<edm::ParameterSet>("TH1ApvNumApvShots");
  globalswitchapvshotsApvon = ParametersGlobalApvNumApvShots.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersGlobalChargeMedianApvShots =
      conf_.getParameter<edm::ParameterSet>("TH1ChargeMedianApvShots");
  globalswitchchargemedianapvshotson = ParametersGlobalChargeMedianApvShots.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersGlobalNApvShotsTimeProf = conf_.getParameter<edm::ParameterSet>("TProfNShotsVsTime");
  globalswitchapvshotsonprof = ParametersGlobalNApvShotsTimeProf.getParameter<bool>("globalswitchon");

  edm::ParameterSet ParametersGlobalNDigisFEDID = conf_.getParameter<edm::ParameterSet>("TProfNDigisFED");
  globalswitchNDigisFEDID = ParametersGlobalNDigisFEDID.getParameter<bool>("globalswitchon");

  //Digi and APV Shots Maps

  digitkhistomapon = conf_.getParameter<bool>("TkHistoMap_On");

  shotshistomapon = conf_.getParameter<bool>("TkHistoMapNApvShots_On");
  shotsstripshistomapon = conf_.getParameter<bool>("TkHistoMapNStripApvShots_On");
  shotschargehistomapon = conf_.getParameter<bool>("TkHistoMapMedianChargeApvShots_On");

  createTrendMEs = conf_.getParameter<bool>("CreateTrendMEs");
  Mod_On_ = conf_.getParameter<bool>("Mod_On");
  m_trendVs10LS = conf_.getParameter<bool>("TrendVs10LS");
  //  xLumiProf             = conf_.getParameter<int>("xLumiProf");
  // Event History Producer
  historyProducer_ = conf_.getParameter<edm::InputTag>("HistoryProducer");
  historyProducerToken_ = consumes<EventWithHistory>(conf_.getParameter<edm::InputTag>("HistoryProducer"));

  // Apv Phase Producer
  apvPhaseProducer_ = conf_.getParameter<edm::InputTag>("ApvPhaseProducer");
  apvPhaseProducerToken_ = consumes<APVCyclePhaseCollection>(conf_.getParameter<edm::InputTag>("ApvPhaseProducer"));

  gtEvmToken_ = consumes<L1GlobalTriggerEvmReadoutRecord>(edm::InputTag("gtEvmDigis"));

  if (show_mechanical_structure_view) {
    tTopoTokenRun_ = esConsumes<edm::Transition::BeginRun>();
    tkDetMapTokenRun_ = esConsumes<edm::Transition::BeginRun>();
    SiStripDetCablingTokenRun_ = esConsumes<edm::Transition::BeginRun>();
  }
  if (subdetswitchtotdigifailureon) {
    runInfoTokenRun_ = esConsumes<edm::Transition::BeginRun>();
  }
  tTopoToken_ = esConsumes();

  // Create DCS Status
  bool checkDCS = conf_.getParameter<bool>("UseDCSFiltering");
  if (checkDCS)
    dcsStatus_ = new SiStripDCSStatus(consumesCollector());
  else
    dcsStatus_ = nullptr;

  //initialize boolean for the data-presence check (needed for TotalNumberOfDigisFailure histogram)
  SBTransitionDone = false;
  SBDeclaredAt = 0;
  ignoreFirstNLumisections_ = TMath::Max(0, ParametersTotDigiFailure.getParameter<int32_t>("ignoreFirstNLumisections"));
  integrateNLumisections_ = TMath::Max(1, ParametersTotDigiFailure.getParameter<int32_t>("integrateNLumisections"));
  vecSubDetTotDigiProfLS.reserve(7);
}
//------------------------------------------------------------------------------------------

SiStripMonitorDigi::~SiStripMonitorDigi() {
  if (dcsStatus_)
    delete dcsStatus_;
}

void SiStripMonitorDigi::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& run, const edm::EventSetup& es) {
  if (show_mechanical_structure_view) {
    unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
    if (m_cacheID_ != cacheID) {
      m_cacheID_ = cacheID;
      edm::LogInfo("SiStripMonitorDigi") << "SiStripMonitorDigi::bookHistograms: "
                                         << " Creating MEs for new Cabling ";
      createMEs(ibooker, es);
    }
  } else if (reset_each_run) {
    edm::LogInfo("SiStripMonitorDigi") << "SiStripMonitorDigi::bookHistograms: "
                                       << " Resetting MEs ";
    for (std::map<uint32_t, ModMEs>::const_iterator idet = DigiMEs.begin(); idet != DigiMEs.end(); idet++) {
      ResetModuleMEs(idet->first);
    }
  }
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::dqmBeginRun(const edm::Run& run, const edm::EventSetup& es) {
  if (subdetswitchtotdigifailureon) {
    //get FED cabling to know if SiStrip is in DAQ or no
    unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
    if (m_cacheID_ != cacheID) {
      m_cacheID_ = cacheID;
    }

    //nFEDConnected = 0;
    nFedTIB = 0;
    nFedTIDm = 0;
    nFedTIDp = 0;
    nFedTECm = 0;
    nFedTECp = 0;
    nFedTOB = 0;

    //const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
    //const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;

    if (auto runInfoRec = es.tryToGet<RunInfoRcd>()) {
      edm::ESHandle<RunInfo> sumFED = runInfoRec->getHandle(runInfoTokenRun_);

      if (sumFED.isValid()) {
        std::vector<int> FedsInIds = sumFED->m_fed_in;
        for (unsigned int it = 0; it < FedsInIds.size(); ++it) {
          int fedID = FedsInIds[it];
          //	  if(fedID>=siStripFedIdMin &&  fedID<=siStripFedIdMax)  ++nFEDConnected;
          /* mia: but is there not a smarter way !?!?!? */
          if (fedID >= 50 && fedID <= 133)
            ++nFedTIB;
          if (fedID >= 134 && fedID <= 148)
            ++nFedTIDm;
          if (fedID >= 149 && fedID <= 163)
            ++nFedTIDp;
          if (fedID >= 164 && fedID <= 259)
            ++nFedTECm;
          if (fedID >= 260 && fedID <= 355)
            ++nFedTECp;
          if (fedID >= 356 && fedID <= 489)
            ++nFedTOB;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------------
std::shared_ptr<bool> SiStripMonitorDigi::globalBeginLuminosityBlock(const edm::LuminosityBlock& lb,
                                                                     const edm::EventSetup& es) const {
  bool isStableBeams = false;
  return std::make_shared<bool>(isStableBeams);
}

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::globalEndLuminosityBlock(const edm::LuminosityBlock& lb, const edm::EventSetup& es) {
  unsigned int currentLS = lb.id().luminosityBlock();
  const bool isStableBeams = luminosityBlockCache(lb.index());
  if (subdetswitchtotdigifailureon && isStableBeams && !SBTransitionDone) {
    SBDeclaredAt = (int)currentLS;
    SBTransitionDone = true;
  }

  if (subdetswitchtotdigifailureon && currentLS % integrateNLumisections_ == 0) {
    int nFeds[6] = {96, 96, 84, 15, 15, 134};  // tec- , tec+ , tib , tid- , tid+ , tob
    int nFedsConnected[6] = {nFedTECm, nFedTECp, nFedTIB, nFedTIDm, nFedTIDp, nFedTOB};

    if (digiFailureMEs.SubDetTotDigiProfLS) {
      for (int ibin = 1; ibin < 7; ibin++) {
        //the vector now has the sum from previous integrateNLumisections_
        //whereas digiFailureMEs.SubDetTotDigiProfLS now has sum for all LS upto now.
        float value = digiFailureMEs.SubDetTotDigiProfLS->getBinContent(ibin) - vecSubDetTotDigiProfLS[ibin];
        vecSubDetTotDigiProfLS[ibin] = digiFailureMEs.SubDetTotDigiProfLS->getBinContent(ibin);

        float fillvalue = 2;
        if (isStableBeams
            //	     && (int)lb.id().luminosityBlock() > ignoreFirstNLumisections_     //ignore first X lumisections for HV rampup
            && ((int)currentLS - SBDeclaredAt) > ignoreFirstNLumisections_ &&
            (float)nFedsConnected[ibin - 1] / nFeds[ibin - 1] > 0.5 && value < 50.) {
          fillvalue = 1.01;
        }
        //account for integrated LS: fill previous bins as well
        for (int fillbin = (int)currentLS - integrateNLumisections_ + 1; fillbin <= (int)currentLS; fillbin++)
          digiFailureMEs.SubDetDigiFailures2D->Fill(fillbin, ibin - 1, fillvalue);
      }
    }
  }
}
//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::createMEs(DQMStore::IBooker& ibooker, const edm::EventSetup& es) {
  if (show_mechanical_structure_view) {
    //Retrieve tracker topology from geometry
    const TrackerTopology* const tTopo = &es.getData(tTopoTokenRun_);
    const TkDetMap* tkDetMap = &es.getData(tkDetMapTokenRun_);

    // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
    SiStripDetCabling_ = &es.getData(SiStripDetCablingTokenRun_);

    // get list of active detectors from SiStripDetCabling
    std::vector<uint32_t> activeDets;
    activeDets.clear();  // just in case
    SiStripDetCabling_->addActiveDetectorsRawIds(activeDets);

    // remove any eventual zero elements - there should be none, but just in case
    for (std::vector<uint32_t>::iterator idets = activeDets.begin(); idets != activeDets.end(); idets++) {
      if (*idets == 0)
        activeDets.erase(idets);
    }

    // create SiStripFolderOrganizer
    SiStripFolderOrganizer folder_organizer;

    // Create TkHistoMap for Digi and APV shots properies

    if (digitkhistomapon)
      tkmapdigi = std::make_unique<TkHistoMap>(tkDetMap, ibooker, topFolderName_, "TkHMap_NumberOfDigi", 0.0, true);
    if (shotshistomapon)
      tkmapNApvshots = std::make_unique<TkHistoMap>(tkDetMap, ibooker, topFolderName_, "TkHMap_NApvShots", 0.0, true);
    if (shotsstripshistomapon)
      tkmapNstripApvshot =
          std::make_unique<TkHistoMap>(tkDetMap, ibooker, topFolderName_, "TkHMap_NStripApvShots", 0.0, true);
    if (shotschargehistomapon)
      tkmapMedianChargeApvshots =
          std::make_unique<TkHistoMap>(tkDetMap, ibooker, topFolderName_, "TkHMap_MedianChargeApvShots", 0.0, true);

    std::vector<uint32_t> tibDetIds;

    // loop over detectors and book MEs
    edm::LogInfo("SiStripTkDQM|SiStripMonitorDigi") << "nr. of activeDets:  " << activeDets.size();
    for (std::vector<uint32_t>::const_iterator detid_iterator = activeDets.begin(); detid_iterator != activeDets.end();
         detid_iterator++) {
      uint32_t detid = (*detid_iterator);

      ModMEs local_modmes;

      local_modmes.NumberOfDigis = nullptr;
      local_modmes.NumberOfDigisPerStrip = nullptr;
      local_modmes.ADCsHottestStrip = nullptr;
      local_modmes.ADCsCoolestStrip = nullptr;
      local_modmes.DigiADCs = nullptr;
      local_modmes.StripOccupancy = nullptr;

      if (Mod_On_) {
        // set appropriate folder using SiStripFolderOrganizer
        folder_organizer.setDetectorFolder(detid, tTopo);  // pass the detid to this method
        if (reset_each_run)
          ResetModuleMEs(detid);
        createModuleMEs(ibooker, local_modmes, detid);

        // append to DigiMEs
        DigiMEs.insert(std::make_pair(detid, local_modmes));
      }

      // Create Layer Level MEs if they are not created already
      std::pair<std::string, int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid, tTopo);
      SiStripHistoId hidmanager;
      std::string label = hidmanager.getSubdetid(detid, tTopo, false);

      // get detids for the layer
      std::map<std::string, LayerMEs>::iterator iLayerME = LayerMEsMap.find(label);

      if (iLayerME == LayerMEsMap.end()) {
        int32_t lnumber = det_layer_pair.second;
        std::vector<uint32_t> layerDetIds;
        if (det_layer_pair.first == "TIB") {
          SiStripSubStructure::getTIBDetectors(activeDets, layerDetIds, tTopo, lnumber, 0, 0, 0);
        } else if (det_layer_pair.first == "TOB") {
          SiStripSubStructure::getTOBDetectors(activeDets, layerDetIds, tTopo, lnumber, 0, 0);
        } else if (det_layer_pair.first == "TID" && lnumber > 0) {
          SiStripSubStructure::getTIDDetectors(activeDets, layerDetIds, tTopo, 2, abs(lnumber), 0, 0);
        } else if (det_layer_pair.first == "TID" && lnumber < 0) {
          SiStripSubStructure::getTIDDetectors(activeDets, layerDetIds, tTopo, 1, abs(lnumber), 0, 0);
        } else if (det_layer_pair.first == "TEC" && lnumber > 0) {
          SiStripSubStructure::getTECDetectors(activeDets, layerDetIds, tTopo, 2, abs(lnumber), 0, 0, 0, 0);
        } else if (det_layer_pair.first == "TEC" && lnumber < 0) {
          SiStripSubStructure::getTECDetectors(activeDets, layerDetIds, tTopo, 1, abs(lnumber), 0, 0, 0, 0);
        }

        LayerDetMap[label] = layerDetIds;

        // book Layer plots
        folder_organizer.setLayerFolder(detid, tTopo, det_layer_pair.second);
        createLayerMEs(ibooker, label, layerDetIds.size());
      }

      // book sub-detector plots
      auto sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
      if (SubDetMEsMap.find(sdet_pair.second) == SubDetMEsMap.end()) {
        ibooker.setCurrentFolder(sdet_pair.first);
        createSubDetMEs(ibooker, sdet_pair.second);
      }

    }  //end of loop over detectors

    //book some Summary histograms on APV shots in the MechanicalView

    ibooker.setCurrentFolder(topFolderName_ + "/MechanicalView/");

    if (globalsummaryapvshotson) {
      const char* HistoName = "Summary Mean Apv shots for SubDets";
      NApvShotsGlobalProf = ibooker.bookProfile(HistoName, HistoName, 6, 0.5, 6.5, 100, 0., 0., "");
      NApvShotsGlobalProf->setBinLabel(1, std::string("TEC-"));
      NApvShotsGlobalProf->setBinLabel(2, std::string("TEC+"));
      NApvShotsGlobalProf->setBinLabel(3, std::string("TIB"));
      NApvShotsGlobalProf->setBinLabel(4, std::string("TID-"));
      NApvShotsGlobalProf->setBinLabel(5, std::string("TID+"));
      NApvShotsGlobalProf->setBinLabel(6, std::string("TOB"));
      NApvShotsGlobalProf->setAxisTitle(" mean APV shots # / evt", 2);
    }

    //cumulative number of APV shots Vs Time
    if (globalswitchapvshotsonprof) {
      edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TProfNShotsVsTime");
      const char* HistoName = "NApv_Shots_vs_Time";
      ShotsVsTimeApvShotsGlobal = ibooker.bookProfile(HistoName,
                                                      HistoName,
                                                      Parameters.getParameter<int32_t>("Nbins"),
                                                      Parameters.getParameter<double>("xmin"),
                                                      Parameters.getParameter<double>("xmax"),
                                                      200,  //that parameter should not be there !?
                                                      Parameters.getParameter<double>("ymin"),
                                                      Parameters.getParameter<double>("ymax"),
                                                      "");
      ShotsVsTimeApvShotsGlobal->setAxisTitle("Time (s)", 1);
      ShotsVsTimeApvShotsGlobal->setAxisTitle("# Apv Shots", 2);
      if (ShotsVsTimeApvShotsGlobal->kind() == MonitorElement::Kind::TPROFILE)
        ShotsVsTimeApvShotsGlobal->getTH1()->SetCanExtend(TH1::kAllAxes);
    }

    //cumulative number of Strips in APV shots
    if (globalswitchnstripsapvshotson) {
      edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1NStripsApvShots");
      const char* HistoName = "Number_of_Strips_in_Apv_Shots";
      StripMultiplicityApvShotsGlobal = ibooker.book1D(HistoName,
                                                       HistoName,
                                                       Parameters.getParameter<int32_t>("Nbins"),
                                                       Parameters.getParameter<double>("xmin"),
                                                       Parameters.getParameter<double>("xmax"));
      StripMultiplicityApvShotsGlobal->setAxisTitle("# strips in Apv Shots", 1);
    }

    //cumulative number of APV shots
    if (globalswitchnapvshotson) {
      edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1NApvShots");
      const char* HistoName = "Number_of_Apv_Shots";
      NApvShotsGlobal = ibooker.book1D(HistoName,
                                       HistoName,
                                       Parameters.getParameter<int32_t>("Nbins"),
                                       Parameters.getParameter<double>("xmin"),
                                       Parameters.getParameter<double>("xmax"));
      NApvShotsGlobal->setAxisTitle("# Apv Shots", 1);
    }

    //cumulative Median Charge in APV shots
    if (globalswitchchargemedianapvshotson) {
      edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1ChargeMedianApvShots");
      //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
      const char* HistoName = "Apv_Shots_Charge_Median";
      MedianChargeApvShotsGlobal = ibooker.book1D(HistoName,
                                                  HistoName,
                                                  Parameters.getParameter<int32_t>("Nbins"),
                                                  Parameters.getParameter<double>("xmin"),
                                                  Parameters.getParameter<double>("xmax"));
      MedianChargeApvShotsGlobal->setAxisTitle("Apv Shots Charge Median (ADC)", 1);
    }

    //cmulative APV number with shots
    if (globalswitchapvshotsApvon) {
      edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1ApvNumApvShots");
      //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
      const char* HistoName = "Apv_Shots_Apv_Number";
      NApvApvShotsGlobal = ibooker.book1D(HistoName,
                                          HistoName,
                                          Parameters.getParameter<int32_t>("Nbins"),
                                          Parameters.getParameter<double>("xmin"),
                                          Parameters.getParameter<double>("xmax"));
      NApvApvShotsGlobal->setAxisTitle("Apv Number", 1);
    }

    if (globalswitchNDigisFEDID) {
      // Number of digis per FED
      edm::ParameterSet FEDDigi = conf_.getParameter<edm::ParameterSet>("TProfNDigisFED");
      const char* HistoName = "NumberOfDigisinFED_v_FEDID";
      //Using TProfile for average.
      NumberOfFEDDigis = ibooker.bookProfile(HistoName,
                                             HistoName,
                                             FEDDigi.getParameter<int32_t>("Nbinsx"),
                                             FEDDigi.getParameter<double>("xmin"),
                                             FEDDigi.getParameter<double>("xmax"),
                                             FEDDigi.getParameter<int32_t>("Nbinsy"),
                                             FEDDigi.getParameter<double>("ymin"),
                                             FEDDigi.getParameter<double>("ymax"),
                                             "");
      NumberOfFEDDigis->setAxisTitle("FED ID", 1);
      NumberOfFEDDigis->setAxisTitle("Mean # of Digis in FED", 2);
    }

    //
    // Book new histogram to monitor digi in last LS
    //

    digiFailureMEs.SubDetTotDigiProfLS = nullptr;
    digiFailureMEs.SubDetDigiFailures2D = nullptr;

    std::stringstream ss;

    folder_organizer.getLayerFolderName(ss, 0, tTopo);
    ibooker.setCurrentFolder(ss.str());

    if (subdetswitchtotdigifailureon) {
      const char* HistoName = "NumberOfDigisInLastLS";
      digiFailureMEs.SubDetTotDigiProfLS = ibooker.bookProfile(HistoName, HistoName, 6, 0.5, 6.5, 0., 0., "");
      digiFailureMEs.SubDetTotDigiProfLS->setBinLabel(1, std::string("TEC-"));
      digiFailureMEs.SubDetTotDigiProfLS->setBinLabel(2, std::string("TEC+"));
      digiFailureMEs.SubDetTotDigiProfLS->setBinLabel(3, std::string("TIB"));
      digiFailureMEs.SubDetTotDigiProfLS->setBinLabel(4, std::string("TID-"));
      digiFailureMEs.SubDetTotDigiProfLS->setBinLabel(5, std::string("TID+"));
      digiFailureMEs.SubDetTotDigiProfLS->setBinLabel(6, std::string("TOB"));
    }

    if (subdetswitchtotdigifailureon) {
      std::string HistoName = "DataPresentInLS";
      edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TotalNumberOfDigisFailure");

      digiFailureMEs.SubDetDigiFailures2D = ibooker.book2D(HistoName,
                                                           HistoName,
                                                           Parameters.getParameter<int32_t>("Nbins"),      //bins X
                                                           1,                                              //xmin
                                                           Parameters.getParameter<int32_t>("Nbins") + 1,  //xmax
                                                           6,                                              //bins Y
                                                           0,                                              //ymin
                                                           6);                                             //ymax

      digiFailureMEs.SubDetDigiFailures2D->setBinLabel(1, std::string("TEC-"), 2);
      digiFailureMEs.SubDetDigiFailures2D->setBinLabel(2, std::string("TEC+"), 2);
      digiFailureMEs.SubDetDigiFailures2D->setBinLabel(3, std::string("TIB"), 2);
      digiFailureMEs.SubDetDigiFailures2D->setBinLabel(4, std::string("TID-"), 2);
      digiFailureMEs.SubDetDigiFailures2D->setBinLabel(5, std::string("TID+"), 2);
      digiFailureMEs.SubDetDigiFailures2D->setBinLabel(6, std::string("TOB"), 2);
      digiFailureMEs.SubDetDigiFailures2D->setAxisTitle("Luminosity Section");
    }
  }  //end of if

}  //end of method

//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Filter out events if DCS Event if requested
  if (dcsStatus_ && !dcsStatus_->getStatus(iEvent, iSetup))
    return;

  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(tTopoToken_);

  TotalNShots = 0;

  runNb = iEvent.id().run();
  eventNb++;

  float iOrbitVar =
      m_trendVs10LS ? iEvent.orbitNumber() / (10 * NORBITS_PER_LS) : iEvent.orbitNumber() / NORBITS_PER_LS;

  digi_detset_handles.clear();

  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > >::const_iterator iToken =
      digiProducerTokenList.begin();
  for (std::vector<edm::InputTag>::const_iterator iter = digiProducerList.begin(), iEnd = digiProducerList.end();
       iter != iEnd;
       ++iter, ++iToken) {
    edm::Handle<edm::DetSetVector<SiStripDigi> > digi_handle;
    iEvent.getByToken(*iToken, digi_handle);
    if (digi_handle.isValid())
      digi_detset_handles.push_back(digi_handle.product());
  }

  // initialise # of clusters to zero
  for (std::map<std::string, SubDetMEs>::iterator iSubdet = SubDetMEsMap.begin(); iSubdet != SubDetMEsMap.end();
       iSubdet++) {
    iSubdet->second.totNDigis = 0;
    iSubdet->second.SubDetApvShots.clear();
  }

  std::map<int, int> FEDID_v_digisum;

  for (std::map<std::string, std::vector<uint32_t> >::const_iterator iterLayer = LayerDetMap.begin();
       iterLayer != LayerDetMap.end();
       iterLayer++) {
    std::string layer_label = iterLayer->first;

    std::vector<uint32_t> layer_dets = iterLayer->second;
    std::map<std::string, LayerMEs>::iterator iLayerME = LayerMEsMap.find(layer_label);

    //get Layer MEs
    LayerMEs local_layermes;

    if (iLayerME == LayerMEsMap.end())
      continue;
    else
      local_layermes = iLayerME->second;

    int largest_adc_layer = 0;
    int smallest_adc_layer = 99999;

    int ndigi_layer = 0;

    uint16_t iDet = 0;

    std::string subdet_label = "";

    // loop over all modules in the layer
    for (std::vector<uint32_t>::const_iterator iterDets = layer_dets.begin(); iterDets != layer_dets.end();
         iterDets++) {
      iDet++;

      // detid and type of ME
      uint32_t detid = (*iterDets);

      // Get SubDet label once
      if (subdet_label.empty())
        subdet_label = folder_organizer.getSubDetFolderAndTag(detid, tTopo).second;

      // DetId and corresponding set of MEs

      std::map<uint32_t, ModMEs>::iterator pos = DigiMEs.find(detid);
      ModMEs local_modmes = pos->second;

      // search  digis of detid
      int loc = getDigiSourceIndex(detid);

      int ndigi_det = 0;

      if (loc > -1) {
        ndigi_det = (*(digi_detset_handles[loc]))[detid].size();

        //////////////////////// FED N digis Monitoring //////////////////////

        //Get all FED connections associated with given detID.
        // All connections for a detid have same FED Id therefore one FEDID is associated with a given detID.
        std::vector<const FedChannelConnection*> fedConnections = SiStripDetCabling_->getConnections(detid);

        // Filling FED Id associated clusters map.

        int good_fcc_index = -999;
        for (unsigned int x = 0; x < fedConnections.size(); x++) {
          if (fedConnections[x] != nullptr) {
            good_fcc_index = x;
            break;
          }
        }
        if (fedConnections[good_fcc_index] != nullptr) {
          int temp_fedid = fedConnections[good_fcc_index]->fedId();
          if (FEDID_v_digisum.find(temp_fedid) != FEDID_v_digisum.end()) {
            if (ndigi_det < 1000 && ndigi_det > 0) {
              FEDID_v_digisum[temp_fedid] = FEDID_v_digisum.find(temp_fedid)->second + ndigi_det;
            }
          } else {
            if (ndigi_det < 1000 && ndigi_det > 0) {
              FEDID_v_digisum[temp_fedid] = ndigi_det;
            }
          }
        } else {
          edm::LogInfo("SiStripMonitorDigi")
              << "SiStripMonitorDigi::analyze WARNING! no good connections for detid = " << detid << std::endl;
        }
        ///////////////////////////////////////////////////////////////////////////

        APVShotFinder theShotFinder = APVShotFinder((*(digi_detset_handles[loc]))[detid]);
        const std::vector<APVShot>& shots = theShotFinder.getShots();
        AddApvShotsToSubDet(shots, SubDetMEsMap[subdet_label].SubDetApvShots);
        if (shotshistomapon)
          tkmapNApvshots->fill(detid, shots.size());
        if (shotsstripshistomapon)
          FillApvShotsMap(tkmapNstripApvshot.get(), shots, detid, 1);
        if (shotschargehistomapon)
          FillApvShotsMap(tkmapMedianChargeApvshots.get(), shots, detid, 2);
      }

      if (Mod_On_ && moduleswitchnumdigison && (local_modmes.NumberOfDigis != nullptr))
        (local_modmes.NumberOfDigis)->Fill(ndigi_det);

      if (layerswitchnumdigisprofon)
        local_layermes.LayerNumberOfDigisProfile->Fill(iDet * 1.0, ndigi_det);

      if (digitkhistomapon)
        tkmapdigi->fill(detid, ndigi_det);

      if (ndigi_det == 0)
        continue;  // no digis for this detid => jump to next step of loop

      const edm::DetSet<SiStripDigi>& digi_detset = (*(digi_detset_handles[loc]))[detid];

      ndigi_layer += ndigi_det;

      // ADCs
      int largest_adc = (digi_detset.data.begin())->adc();
      int smallest_adc = (digi_detset.data.begin())->adc();

      // Check if these parameters are really needed
      float det_occupancy = 0.0;

      for (edm::DetSet<SiStripDigi>::const_iterator digiIter = digi_detset.data.begin();
           digiIter != digi_detset.data.end();
           digiIter++) {
        int this_adc = digiIter->adc();

        if (this_adc > 0.0)
          det_occupancy++;

        if (this_adc > largest_adc)
          largest_adc = this_adc;
        if (this_adc < smallest_adc)
          smallest_adc = this_adc;

        if (Mod_On_ && moduleswitchnumdigispstripon && (local_modmes.NumberOfDigisPerStrip != nullptr) &&
            (this_adc > 0.0))
          (local_modmes.NumberOfDigisPerStrip)->Fill(digiIter->strip());

        if (Mod_On_ && moduleswitchdigiadcson && (local_modmes.DigiADCs != nullptr))
          (local_modmes.DigiADCs)->Fill(static_cast<float>(this_adc));

        //Fill #ADCs for this digi at layer level
        if (layerswitchdigiadcson) {
          fillME(local_layermes.LayerDigiADCs, this_adc);
          if (createTrendMEs)
            fillTrend(local_layermes.LayerDigiADCsTrend, this_adc, iOrbitVar);
        }

        if (layerswitchdigiadcprofon)
          local_layermes.LayerDigiADCProfile->Fill(iDet * 1.0, this_adc);

      }  //end of loop over digis in this det

      // Occupancy
      short nstrips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128;
      if (nstrips > 0 && det_occupancy > 0) {
        det_occupancy = det_occupancy / nstrips;
        if (Mod_On_ && moduleswitchstripoccupancyon && (local_modmes.StripOccupancy != nullptr))
          (local_modmes.StripOccupancy)->Fill(det_occupancy);
        if (layerswitchstripoccupancyon) {
          fillME(local_layermes.LayerStripOccupancy, det_occupancy);
          if (createTrendMEs)
            fillTrend(local_layermes.LayerStripOccupancyTrend, det_occupancy, iOrbitVar);
        }
      }

      if (largest_adc > largest_adc_layer)
        largest_adc_layer = largest_adc;
      if (smallest_adc < smallest_adc_layer)
        smallest_adc_layer = smallest_adc;

      // nr. of adcs for hottest strip
      if (Mod_On_ && moduleswitchadchotteston && (local_modmes.ADCsHottestStrip != nullptr))
        (local_modmes.ADCsHottestStrip)->Fill(static_cast<float>(largest_adc));

      // nr. of adcs for coolest strip
      if (Mod_On_ && moduleswitchadccooleston && (local_modmes.ADCsCoolestStrip != nullptr))
        (local_modmes.ADCsCoolestStrip)->Fill(static_cast<float>(smallest_adc));

    }  //end of loop over DetIds

    if (layerswitchnumdigison) {
      fillME(local_layermes.LayerNumberOfDigis, ndigi_layer);
      if (createTrendMEs)
        fillTrend(local_layermes.LayerNumberOfDigisTrend, ndigi_layer, iOrbitVar);
    }
    if (layerswitchadchotteston) {
      fillME(local_layermes.LayerADCsHottestStrip, largest_adc_layer);
      if (createTrendMEs)
        fillTrend(local_layermes.LayerADCsHottestStripTrend, largest_adc_layer, iOrbitVar);
    }
    if (layerswitchadccooleston) {
      fillME(local_layermes.LayerADCsCoolestStrip, smallest_adc_layer);
      if (createTrendMEs)
        fillTrend(local_layermes.LayerADCsCoolestStripTrend, smallest_adc_layer, iOrbitVar);
    }

    std::map<std::string, SubDetMEs>::iterator iSubdet = SubDetMEsMap.find(subdet_label);
    if (iSubdet != SubDetMEsMap.end()) {
      iSubdet->second.totNDigis += ndigi_layer;
      //std::cout << " totDigis" <<  iSubdet->second.totNDigis << " in "  << subdet_label << std::endl;
    }
  }

  bool& isStableBeams = *luminosityBlockCache(iEvent.getLuminosityBlock().index());
  if (subdetswitchtotdigifailureon) {
    //check Stable beams bit
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvm_handle;
    iEvent.getByToken(gtEvmToken_, gtEvm_handle);
    L1GlobalTriggerEvmReadoutRecord const* gtevm = gtEvm_handle.product();

    L1GtfeExtWord gtfeEvmExtWord;
    if (gtevm) {
      gtfeEvmExtWord = gtevm->gtfeWord();
    } else
      edm::LogInfo("DQMProvInfo") << " gtfeEvmWord inaccessible";

    /* mia: is there not a smarter way !?!?!?!? */
    if (gtfeEvmExtWord.beamMode() == 11)
      isStableBeams = true;
  }

  for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
    if (subdetswitchtotdigifailureon) {
      if (strcmp(it->first.c_str(), "TEC__MINUS") == 0) {
        digiFailureMEs.SubDetTotDigiProfLS->Fill(1, it->second.totNDigis);
      } else if (strcmp(it->first.c_str(), "TEC__PLUS") == 0) {
        digiFailureMEs.SubDetTotDigiProfLS->Fill(2, it->second.totNDigis);
      } else if (strcmp(it->first.c_str(), "TIB") == 0) {
        digiFailureMEs.SubDetTotDigiProfLS->Fill(3, it->second.totNDigis);
      } else if (strcmp(it->first.c_str(), "TID__MINUS") == 0) {
        digiFailureMEs.SubDetTotDigiProfLS->Fill(4, it->second.totNDigis);
      } else if (strcmp(it->first.c_str(), "TID__PLUS") == 0) {
        digiFailureMEs.SubDetTotDigiProfLS->Fill(5, it->second.totNDigis);
      } else if (strcmp(it->first.c_str(), "TOB") == 0) {
        digiFailureMEs.SubDetTotDigiProfLS->Fill(6, it->second.totNDigis);
      }
    }

    if (globalsummaryapvshotson) {
      if (strcmp(it->first.c_str(), "TEC__MINUS") == 0) {
        NApvShotsGlobalProf->Fill(1, it->second.SubDetApvShots.size());
      } else if (strcmp(it->first.c_str(), "TEC__PLUS") == 0) {
        NApvShotsGlobalProf->Fill(2, it->second.SubDetApvShots.size());
      } else if (strcmp(it->first.c_str(), "TIB") == 0) {
        NApvShotsGlobalProf->Fill(3, it->second.SubDetApvShots.size());
      } else if (strcmp(it->first.c_str(), "TID__MINUS") == 0) {
        NApvShotsGlobalProf->Fill(4, it->second.SubDetApvShots.size());
      } else if (strcmp(it->first.c_str(), "TID__PLUS") == 0) {
        NApvShotsGlobalProf->Fill(5, it->second.SubDetApvShots.size());
      } else if (strcmp(it->first.c_str(), "TOB") == 0) {
        NApvShotsGlobalProf->Fill(6, it->second.SubDetApvShots.size());
      }
    }

    SubDetMEs subdetmes = it->second;
    std::string subdet = it->first;

    // Fill APV shots histograms for SubDet

    uint ShotsSize = subdetmes.SubDetApvShots.size();
    TotalNShots += ShotsSize;  //Counter for total Shots in the SiStrip Tracker

    if (subdetswitchnapvshotson)
      subdetmes.SubDetNApvShotsTH1->Fill(ShotsSize);  // N shots
    if (subdetswitchapvshotsonprof)
      subdetmes.SubDetNApvShotsProf->Fill(iOrbitVar, ShotsSize);  //N shots vs time

    for (uint i = 0; i < ShotsSize;
         ++i) {  // Strip multiplicity, charge median and APV number distributions for APV shots

      if (subdetswitchapvshotsApvon)
        subdetmes.SubDetNApvShotsNApvTH1->Fill(
            (subdetmes.SubDetApvShots[i].apvNumber() + 1));  //APV are defined by 0 to 5 I want 1 to 6
      if (globalswitchapvshotsApvon)
        NApvApvShotsGlobal->Fill((subdetmes.SubDetApvShots[i].apvNumber() + 1));

      if (subdetswitchnstripsapvshotson)
        subdetmes.SubDetNStripsApvShotsTH1->Fill(subdetmes.SubDetApvShots[i].nStrips());
      if (globalswitchnstripsapvshotson)
        StripMultiplicityApvShotsGlobal->Fill(subdetmes.SubDetApvShots[i].nStrips());

      if (subdetswitchchargemedianapvshotson)
        subdetmes.SubDetChargeMedianApvShotsTH1->Fill(subdetmes.SubDetApvShots[i].median());
      if (globalswitchchargemedianapvshotson)
        MedianChargeApvShotsGlobal->Fill(subdetmes.SubDetApvShots[i].median());
    }

    if (subdetswitchtotdigiprofon)
      subdetmes.SubDetTotDigiProf->Fill(iOrbitVar, subdetmes.totNDigis);
  }

  if (globalswitchnapvshotson)
    NApvShotsGlobal->Fill(TotalNShots);
  if (globalswitchapvshotsonprof)
    ShotsVsTimeApvShotsGlobal->Fill(iOrbitVar, TotalNShots);

  if (globalswitchNDigisFEDID) {
    std::map<int, int>::iterator it;
    for (it = FEDID_v_digisum.begin(); it != FEDID_v_digisum.end(); it++) {
      NumberOfFEDDigis->Fill(it->first, it->second);
    }
  }

  // get EventHistory

  edm::Handle<EventWithHistory> event_history;
  iEvent.getByToken(historyProducerToken_, event_history);

  // get Phase of APV
  edm::Handle<APVCyclePhaseCollection> apv_phase_collection;
  iEvent.getByToken(apvPhaseProducerToken_, apv_phase_collection);

  if (event_history.isValid() && !event_history.failedToGet() && apv_phase_collection.isValid() &&
      !apv_phase_collection.failedToGet()) {
    long long tbx = event_history->absoluteBX();

    for (std::map<std::string, SubDetMEs>::iterator it = SubDetMEsMap.begin(); it != SubDetMEsMap.end(); it++) {
      SubDetMEs subdetmes;
      std::string subdet = it->first;
      subdetmes = it->second;

      int the_phase = APVCyclePhaseCollection::invalid;
      long long tbx_corr = tbx;

      if (SubDetPhasePartMap.find(subdet) != SubDetPhasePartMap.end())
        the_phase = apv_phase_collection->getPhase(SubDetPhasePartMap[subdet]);
      if (the_phase == APVCyclePhaseCollection::nopartition || the_phase == APVCyclePhaseCollection::multiphase ||
          the_phase == APVCyclePhaseCollection::invalid)
        the_phase = 30;
      tbx_corr -= the_phase;

      if (subdetswitchapvcycleprofon)
        subdetmes.SubDetDigiApvProf->Fill(tbx_corr % 70, subdetmes.totNDigis);
      if (subdetswitchapvcycleth2on)
        subdetmes.SubDetDigiApvTH2->Fill(tbx_corr % 70, subdetmes.totNDigis);
    }
  }
}  //end of method analyze
//--------------------------------------------------------------------------------------------
void SiStripMonitorDigi::ResetModuleMEs(uint32_t idet) {
  std::map<uint32_t, ModMEs>::iterator pos = DigiMEs.find(idet);
  ModMEs mod_me = pos->second;

  if (Mod_On_ && moduleswitchnumdigison)
    mod_me.NumberOfDigis->Reset();
  if (Mod_On_ && moduleswitchnumdigispstripon)
    mod_me.NumberOfDigisPerStrip->Reset();
  if (Mod_On_ && moduleswitchadchotteston)
    mod_me.ADCsHottestStrip->Reset();
  if (Mod_On_ && moduleswitchadccooleston)
    mod_me.ADCsCoolestStrip->Reset();
  if (Mod_On_ && moduleswitchdigiadcson)
    mod_me.DigiADCs->Reset();
  if (Mod_On_ && moduleswitchstripoccupancyon)
    mod_me.StripOccupancy->Reset();
}
//------------------------------------------------------------------------------------------
SiStripMonitorDigi::MonitorElement* SiStripMonitorDigi::bookMETrend(DQMStore::IBooker& ibooker,
                                                                    const char* ParameterSetLabel,
                                                                    const char* HistoName) {
  edm::ParameterSet ParametersTrend = conf_.getParameter<edm::ParameterSet>("Trending");
  MonitorElement* me =
      ibooker.bookProfile(HistoName,
                          HistoName,
                          ParametersTrend.getParameter<int32_t>("Nbins"),
                          // 					      0,
                          ParametersTrend.getParameter<double>("xmin"),
                          ParametersTrend.getParameter<double>("xmax"),
                          // 					      ParametersTrend.getParameter<int32_t>("Nbins"),
                          100,  //that parameter should not be there !?
                          ParametersTrend.getParameter<double>("ymin"),
                          ParametersTrend.getParameter<double>("ymax"),
                          "");
  if (!me)
    return me;

  me->setAxisTitle("Lumisection", 1);
  if (me->kind() == MonitorElement::Kind::TPROFILE)
    me->getTH1()->SetCanExtend(TH1::kAllAxes);
  return me;
}

//------------------------------------------------------------------------------------------
SiStripMonitorDigi::MonitorElement* SiStripMonitorDigi::bookME1D(DQMStore::IBooker& ibooker,
                                                                 const char* ParameterSetLabel,
                                                                 const char* HistoName) {
  edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book1D(HistoName,
                        HistoName,
                        Parameters.getParameter<int32_t>("Nbinx"),
                        Parameters.getParameter<double>("xmin"),
                        Parameters.getParameter<double>("xmax"));
}

//--------------------------------------------------------------------------------
void SiStripMonitorDigi::fillTrend(MonitorElement* me, float value, float timeinorbit) {
  if (!me)
    return;
  me->Fill(timeinorbit, value);
}

//
// -- Create Module Level MEs
//
void SiStripMonitorDigi::createModuleMEs(DQMStore::IBooker& ibooker, ModMEs& mod_single, uint32_t detid) {
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  std::string hid;

  //nr. of digis per module
  if (moduleswitchnumdigison) {
    hid = hidmanager.createHistoId("NumberOfDigis", "det", detid);
    mod_single.NumberOfDigis = ibooker.book1D(hid, hid, 21, -0.5, 20.5);
    mod_single.NumberOfDigis->setAxisTitle("number of digis in one detector module");
    mod_single.NumberOfDigis->setStatOverflows(kTRUE);  // over/underflows in Mean calculation
  }

  //nr. of digis per strip in module
  if (moduleswitchnumdigispstripon) {
    hid = hidmanager.createHistoId("NumberOfDigisPerStrip", "det", detid);
    short nstrips = SiStripDetCabling_->nApvPairs(detid) * 2 * 128;
    mod_single.NumberOfDigisPerStrip = ibooker.book1D(hid, hid, nstrips, -0.5, nstrips + 0.5);
    mod_single.NumberOfDigisPerStrip->setAxisTitle("number of (digis > 0) per strip");
    mod_single.NumberOfDigisPerStrip->setStatOverflows(kTRUE);  // over/underflows in Mean calculation
  }
  //#ADCs for hottest strip
  if (moduleswitchadchotteston) {
    hid = hidmanager.createHistoId("ADCsHottestStrip", "det", detid);
    mod_single.ADCsHottestStrip = bookME1D(ibooker, "TH1ADCsHottestStrip", hid.c_str());
    mod_single.ADCsHottestStrip->setAxisTitle("number of ADCs for hottest strip");
  }

  //#ADCs for coolest strip
  if (moduleswitchadccooleston) {
    hid = hidmanager.createHistoId("ADCsCoolestStrip", "det", detid);
    mod_single.ADCsCoolestStrip = bookME1D(ibooker, "TH1ADCsCoolestStrip", hid.c_str());
    mod_single.ADCsCoolestStrip->setAxisTitle("number of ADCs for coolest strip");
  }

  //#ADCs for each digi
  if (moduleswitchdigiadcson) {
    hid = hidmanager.createHistoId("DigiADCs", "det", detid);
    mod_single.DigiADCs = bookME1D(ibooker, "TH1DigiADCs", hid.c_str());
    mod_single.DigiADCs->setAxisTitle("number of ADCs for each digi");
  }

  //Strip occupancy
  if (moduleswitchstripoccupancyon) {
    hid = hidmanager.createHistoId("StripOccupancy", "det", detid);
    mod_single.StripOccupancy = bookME1D(ibooker, "TH1StripOccupancy", hid.c_str());
    mod_single.StripOccupancy->setAxisTitle("strip occupancy");
  }
}

//
// -- Create Module Level MEs
//

void SiStripMonitorDigi::createLayerMEs(DQMStore::IBooker& ibooker, std::string label, int ndets) {
  std::map<std::string, LayerMEs>::iterator iLayerME = LayerMEsMap.find(label);
  if (iLayerME == LayerMEsMap.end()) {
    SiStripHistoId hidmanager;
    LayerMEs layerMEs;
    layerMEs.LayerNumberOfDigis = nullptr;
    layerMEs.LayerNumberOfDigisTrend = nullptr;
    layerMEs.LayerADCsHottestStrip = nullptr;
    layerMEs.LayerADCsHottestStripTrend = nullptr;
    layerMEs.LayerADCsCoolestStrip = nullptr;
    layerMEs.LayerADCsCoolestStripTrend = nullptr;
    layerMEs.LayerDigiADCs = nullptr;
    layerMEs.LayerDigiADCsTrend = nullptr;
    layerMEs.LayerStripOccupancy = nullptr;
    layerMEs.LayerStripOccupancyTrend = nullptr;
    layerMEs.LayerNumberOfDigisProfile = nullptr;
    layerMEs.LayerDigiADCProfile = nullptr;

    //#Digis
    if (layerswitchnumdigison) {
      layerMEs.LayerNumberOfDigis =
          bookME1D(ibooker,
                   "TH1NumberOfDigis",
                   hidmanager.createHistoLayer("Summary_TotalNumberOfDigis", "layer", label, "").c_str());
      if (createTrendMEs)
        layerMEs.LayerNumberOfDigisTrend =
            bookMETrend(ibooker,
                        "TH1NumberOfDigis",
                        hidmanager.createHistoLayer("Trend_NumberOfDigis", "layer", label, "").c_str());
    }

    //#ADCs for hottest strip
    if (layerswitchadchotteston) {
      layerMEs.LayerADCsHottestStrip =
          bookME1D(ibooker,
                   "TH1ADCsHottestStrip",
                   hidmanager.createHistoLayer("Summary_ADCsHottestStrip", "layer", label, "").c_str());
      if (createTrendMEs)
        layerMEs.LayerADCsHottestStripTrend =
            bookMETrend(ibooker,
                        "TH1ADCsHottestStrip",
                        hidmanager.createHistoLayer("Trend_ADCsHottestStrip", "layer", label, "").c_str());
    }

    //#ADCs for coolest strip
    if (layerswitchadccooleston) {
      layerMEs.LayerADCsCoolestStrip =
          bookME1D(ibooker,
                   "TH1ADCsCoolestStrip",
                   hidmanager.createHistoLayer("Summary_ADCsCoolestStrip", "layer", label, "").c_str());
      if (createTrendMEs)
        layerMEs.LayerADCsCoolestStripTrend =
            bookMETrend(ibooker,
                        "TH1ADCsCoolestStrip",
                        hidmanager.createHistoLayer("Trend_ADCsCoolestStrip", "layer", label, "").c_str());
    }

    //#ADCs for each digi
    if (layerswitchdigiadcson) {
      layerMEs.LayerDigiADCs =
          bookME1D(ibooker, "TH1DigiADCs", hidmanager.createHistoLayer("Summary_DigiADCs", "layer", label, "").c_str());
      if (createTrendMEs)
        layerMEs.LayerDigiADCsTrend = bookMETrend(
            ibooker, "TH1DigiADCs", hidmanager.createHistoLayer("Trend_DigiADCs", "layer", label, "").c_str());
    }

    //Strip Occupancy
    if (layerswitchstripoccupancyon) {
      layerMEs.LayerStripOccupancy =
          bookME1D(ibooker,
                   "TH1StripOccupancy",
                   hidmanager.createHistoLayer("Summary_StripOccupancy", "layer", label, "").c_str());
      if (createTrendMEs)
        layerMEs.LayerStripOccupancyTrend =
            bookMETrend(ibooker,
                        "TH1StripOccupancy",
                        hidmanager.createHistoLayer("Trend_StripOccupancy", "layer", label, "").c_str());
    }
    // # of Digis
    if (layerswitchnumdigisprofon) {
      std::string hid = hidmanager.createHistoLayer("NumberOfDigiProfile", "layer", label, "");
      layerMEs.LayerNumberOfDigisProfile = ibooker.bookProfile(hid, hid, ndets, 0.5, ndets + 0.5, 21, -0.5, 200.5);
    }

    // # of Digis
    if (layerswitchdigiadcprofon) {
      std::string hid = hidmanager.createHistoLayer("DigiADCProfile", "layer", label, "");
      layerMEs.LayerDigiADCProfile = ibooker.bookProfile(hid, hid, ndets, 0.5, ndets + 0.5, 64, -0.5, 255.5);
    }

    LayerMEsMap[label] = layerMEs;
  }
}
//
// -- Create SubDetector MEs
//
void SiStripMonitorDigi::createSubDetMEs(DQMStore::IBooker& ibooker, std::string label) {
  SubDetMEs subdetMEs;
  subdetMEs.totNDigis = 0;
  subdetMEs.SubDetTotDigiProf = nullptr;
  subdetMEs.SubDetDigiApvProf = nullptr;
  subdetMEs.SubDetDigiApvTH2 = nullptr;

  subdetMEs.SubDetApvShots.clear();
  subdetMEs.SubDetNApvShotsTH1 = nullptr;
  subdetMEs.SubDetChargeMedianApvShotsTH1 = nullptr;
  subdetMEs.SubDetNStripsApvShotsTH1 = nullptr;
  subdetMEs.SubDetNApvShotsProf = nullptr;

  std::string HistoName;

  // Total Number of Digi - Profile
  if (subdetswitchtotdigiprofon) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TProfTotalNumberOfDigis");
    HistoName = "TotalNumberOfDigiProfile__" + label;
    subdetMEs.SubDetTotDigiProf = ibooker.bookProfile(HistoName,
                                                      HistoName,
                                                      Parameters.getParameter<int32_t>("Nbins"),
                                                      Parameters.getParameter<double>("xmin"),
                                                      Parameters.getParameter<double>("xmax"),
                                                      100,  //that parameter should not be there !?
                                                      Parameters.getParameter<double>("ymin"),
                                                      Parameters.getParameter<double>("ymax"),
                                                      "");
    subdetMEs.SubDetTotDigiProf->setAxisTitle("Lumisection", 1);

    if (subdetMEs.SubDetTotDigiProf->kind() == MonitorElement::Kind::TPROFILE)
      subdetMEs.SubDetTotDigiProf->getTH1()->SetCanExtend(TH1::kAllAxes);
  }

  // Number of Digi vs Bx - Profile
  if (subdetswitchapvcycleprofon) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TProfDigiApvCycle");
    HistoName = "Digi_vs_ApvCycle__" + label;
    subdetMEs.SubDetDigiApvProf = ibooker.bookProfile(HistoName,
                                                      HistoName,
                                                      Parameters.getParameter<int32_t>("Nbins"),
                                                      Parameters.getParameter<double>("xmin"),
                                                      Parameters.getParameter<double>("xmax"),
                                                      200,  //that parameter should not be there !?
                                                      Parameters.getParameter<double>("ymin"),
                                                      Parameters.getParameter<double>("ymax"),
                                                      "");
    subdetMEs.SubDetDigiApvProf->setAxisTitle("ApvCycle (Corrected Absolute Bx % 70)", 1);
  }

  // Number of Digi vs Bx - TH2
  if (subdetswitchapvcycleth2on) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH2DigiApvCycle");
    //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "Digi_vs_ApvCycle_2D__" + label;
    // Adjusting the scale for 2D histogram
    double h2ymax = 9999.0;
    double yfact = Parameters.getParameter<double>("yfactor");
    if (label.find("TIB") != std::string::npos)
      h2ymax = (6984. * 256.) * yfact;
    else if (label.find("TID") != std::string::npos)
      h2ymax = (2208. * 256.) * yfact;
    else if (label.find("TOB") != std::string::npos)
      h2ymax = (12906. * 256.) * yfact;
    else if (label.find("TEC") != std::string::npos)
      h2ymax = (7552. * 2. * 256.) * yfact;
    subdetMEs.SubDetDigiApvTH2 =
        ibooker.book2D(HistoName,
                       HistoName,
                       Parameters.getParameter<int32_t>("Nbins"),
                       Parameters.getParameter<double>("xmin"),
                       Parameters.getParameter<double>("xmax"),
                       Parameters.getParameter<int32_t>("Nbinsy"),  //it was 100 that parameter should not be there !?
                       Parameters.getParameter<double>("ymin"),
                       h2ymax);
    subdetMEs.SubDetDigiApvTH2->setAxisTitle("absolute Bx mod(70)", 1);
  }

  //Number of APV Shots
  if (subdetswitchnapvshotson) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1NApvShots");
    //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "Number_of_Apv_Shots_" + label;
    subdetMEs.SubDetNApvShotsTH1 = ibooker.book1D(HistoName,
                                                  HistoName,
                                                  Parameters.getParameter<int32_t>("Nbins"),
                                                  Parameters.getParameter<double>("xmin"),
                                                  Parameters.getParameter<double>("xmax"));
    subdetMEs.SubDetNApvShotsTH1->setAxisTitle("# Apv Shots", 1);
  }

  //Strip multiplicity of APV Shots
  if (subdetswitchnstripsapvshotson) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1NStripsApvShots");
    //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "Number_of_Strips_in_Apv_Shots_" + label;
    subdetMEs.SubDetNStripsApvShotsTH1 = ibooker.book1D(HistoName,
                                                        HistoName,
                                                        Parameters.getParameter<int32_t>("Nbins"),
                                                        Parameters.getParameter<double>("xmin"),
                                                        Parameters.getParameter<double>("xmax"));
    subdetMEs.SubDetNStripsApvShotsTH1->setAxisTitle("# strips in Apv Shots", 1);
  }

  //Charge median of APV Shots
  if (subdetswitchchargemedianapvshotson) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1ChargeMedianApvShots");
    //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "Apv_Shots_Charge_Median_" + label;
    subdetMEs.SubDetChargeMedianApvShotsTH1 = ibooker.book1D(HistoName,
                                                             HistoName,
                                                             Parameters.getParameter<int32_t>("Nbins"),
                                                             Parameters.getParameter<double>("xmin"),
                                                             Parameters.getParameter<double>("xmax"));
    subdetMEs.SubDetChargeMedianApvShotsTH1->setAxisTitle("Apv Shots Charge Median (ADC)", 1);
  }

  if (subdetswitchchargemedianapvshotson) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TH1ApvNumApvShots");
    //dqmStore_->setCurrentFolder("SiStrip/MechanicalView/"+label);
    HistoName = "Apv_Shots_Apv_Number_" + label;
    subdetMEs.SubDetNApvShotsNApvTH1 = ibooker.book1D(HistoName,
                                                      HistoName,
                                                      Parameters.getParameter<int32_t>("Nbins"),
                                                      Parameters.getParameter<double>("xmin"),
                                                      Parameters.getParameter<double>("xmax"));
    subdetMEs.SubDetNApvShotsNApvTH1->setAxisTitle("Apv Number", 1);
  }

  //APV Shots number Vs time
  if (subdetswitchapvshotsonprof) {
    edm::ParameterSet Parameters = conf_.getParameter<edm::ParameterSet>("TProfNShotsVsTime");
    HistoName = "NApv_Shots_vs_Time_" + label;
    subdetMEs.SubDetNApvShotsProf = ibooker.bookProfile(HistoName,
                                                        HistoName,
                                                        Parameters.getParameter<int32_t>("Nbins"),
                                                        Parameters.getParameter<double>("xmin"),
                                                        Parameters.getParameter<double>("xmax"),
                                                        200,  //that parameter should not be there !?
                                                        Parameters.getParameter<double>("ymin"),
                                                        Parameters.getParameter<double>("ymax"),
                                                        "");
    subdetMEs.SubDetNApvShotsProf->setAxisTitle("Time (s)", 1);
    subdetMEs.SubDetNApvShotsProf->setAxisTitle("# Apv Shots", 2);
    if (subdetMEs.SubDetNApvShotsProf->kind() == MonitorElement::Kind::TPROFILE)
      subdetMEs.SubDetNApvShotsProf->getTH1()->SetCanExtend(TH1::kAllAxes);
  }

  SubDetMEsMap[label] = subdetMEs;
}
//
// -- Get DetSet vector for a given Detector
//
int SiStripMonitorDigi::getDigiSourceIndex(uint32_t id) {
  int location = -1;
  for (unsigned int ival = 0; ival < digi_detset_handles.size(); ++ival) {
    edm::DetSetVector<SiStripDigi>::const_iterator isearch = digi_detset_handles[ival]->find(id);
    if (isearch != digi_detset_handles[ival]->end()) {
      location = ival;
      break;
    }
  }
  return location;
}

void SiStripMonitorDigi::AddApvShotsToSubDet(const std::vector<APVShot>& moduleShots,
                                             std::vector<APVShot>& subdetShots) {
  for (uint i = 0; i < moduleShots.size(); i++) {
    subdetShots.push_back(moduleShots[i]);
  }
}

void SiStripMonitorDigi::FillApvShotsMap(TkHistoMap* the_map, const std::vector<APVShot>& shots, uint32_t id, int mode) {
  for (uint i = 0; i < shots.size(); i++) {
    if (mode == 1)
      the_map->fill(id, shots[i].nStrips());  //mode == 1 fill with strip multiplicity
    if (mode == 2)
      the_map->fill(id, shots[i].median());  // mode == 2 fill with charge median
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorDigi);
