#ifndef SiStripMonitorDigi_SiStripMonitorDigi_h
#define SiStripMonitorDigi_SiStripMonitorDigi_h
// -*- C++ -*-
// Package:     SiStripMonitorDigi
// Class  :     SiStripMonitorDigi
/**\class SiStripMonitorDigi SiStripMonitorDigi.h DQM/SiStripMonitorDigi/interface/SiStripMonitorDigi.h
   Data Quality Monitoring source of the Silicon Strip Tracker. Produces histograms related to digis.
*/
// Original Author:  dkcira
//         Created:  Sat Feb  4 20:49:51 CET 2006
#include <memory>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQM/SiStripCommon/interface/APVShotFinder.h"
#include "DQM/SiStripCommon/interface/APVShot.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

class DQMStore;
class SiStripDCSStatus;
class SiStripDetCabling;

class EventWithHistory;
class L1GlobalTriggerEvmReadoutRecord;
class APVCyclePhaseCollection;

class SiStripMonitorDigi : public DQMEDAnalyzer {
 public:
  explicit SiStripMonitorDigi(const edm::ParameterSet&);
  ~SiStripMonitorDigi() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) override;

  std::string topFolderName_;

  struct ModMEs{

    MonitorElement* NumberOfDigis;
    MonitorElement* NumberOfDigisPerStrip;
    MonitorElement* ADCsHottestStrip;
    MonitorElement* ADCsCoolestStrip;
    MonitorElement* DigiADCs;
    MonitorElement* StripOccupancy;
  };

  struct LayerMEs{

    MonitorElement* LayerNumberOfDigis;
    MonitorElement* LayerNumberOfDigisTrend;
    MonitorElement* LayerADCsHottestStrip;
    MonitorElement* LayerADCsHottestStripTrend;
    MonitorElement* LayerADCsCoolestStrip;
    MonitorElement* LayerADCsCoolestStripTrend;
    MonitorElement* LayerDigiADCs;
    MonitorElement* LayerDigiADCsTrend;
    MonitorElement* LayerStripOccupancy;
    MonitorElement* LayerStripOccupancyTrend;
    MonitorElement* LayerNumberOfDigisProfile;
    MonitorElement* LayerDigiADCProfile;

  };

  struct SubDetMEs{

    int totNDigis;
    MonitorElement* SubDetTotDigiProf;
    MonitorElement* SubDetDigiApvProf;
    MonitorElement* SubDetDigiApvTH2;

    //int totApvShots;
    std::vector<APVShot> SubDetApvShots;
    MonitorElement* SubDetNApvShotsTH1;
    MonitorElement* SubDetChargeMedianApvShotsTH1;
    MonitorElement* SubDetNStripsApvShotsTH1;
    MonitorElement* SubDetNApvShotsProf;
    MonitorElement* SubDetNApvShotsNApvTH1;
  };

  struct DigiFailureMEs{
    MonitorElement* SubDetTotDigiProfLS;
    MonitorElement* SubDetDigiFailures2D;
  };

  MonitorElement* NumberOfFEDDigis = nullptr;

 private:
  void createMEs(DQMStore::IBooker & ibooker , const edm::EventSetup& es );
  void ResetModuleMEs(uint32_t idet);
  void bookLayer( DQMStore::IBooker & ibooker );
  MonitorElement* bookMETrend( DQMStore::IBooker & ibooker , const char* ParameterSetLabel, const char* HistoName);
  MonitorElement* bookME1D( DQMStore::IBooker & ibooker , const char* ParameterSetLabel, const char* HistoName);
  void bookTrendMEs( DQMStore::IBooker & ibooker , const TString& name,int32_t layer,uint32_t id,std::string flag);
  void fillDigiADCsMEs(int value, std::string name);
  void fillTrend(MonitorElement* me ,float value, float timeinorbit);
  inline void fillME(MonitorElement* ME,float value1){if (ME!=nullptr)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=nullptr)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=nullptr)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=nullptr)ME->Fill(value1,value2,value3,value4);}
  bool AllDigis( const edm::EventSetup& es);

  void createModuleMEs( DQMStore::IBooker & ibooker , ModMEs& mod_single, uint32_t detid);
  void createLayerMEs( DQMStore::IBooker & ibooker , std::string label, int ndet);
  void createSubDetMEs( DQMStore::IBooker & ibooker , std::string label);
  void createSubDetTH2( DQMStore::IBooker & ibooker , std::string label);
  int getDigiSourceIndex(uint32_t id);
  void AddApvShotsToSubDet(const std::vector<APVShot> &, std::vector<APVShot> &);
  void FillApvShotsMap(TkHistoMap*, const std::vector<APVShot> &, uint32_t id ,int);

 private:

  edm::ParameterSet conf_;
  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > > digiProducerTokenList;
  std::vector<edm::InputTag> digiProducerList;
  std::map<uint32_t, ModMEs> DigiMEs; // uint32_t me_type: 1=#digis/module; 2=adcs of hottest strip/module; 3= adcs of coolest strips/module.
  bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, calculate_strip_occupancy, reset_each_run;

  std::map<std::string, std::vector< uint32_t > > LayerDetMap;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::string> SubDetPhasePartMap;
  DigiFailureMEs digiFailureMEs;

  TString name;
  SiStripFolderOrganizer folder_organizer;
  std::map<std::pair<std::string,int32_t>,bool> DetectedLayers;
  std::vector<const edm::DetSetVector<SiStripDigi> *> digi_detset_handles;

  unsigned long long m_cacheID_;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  std::vector<uint32_t> ModulesToBeExcluded_;

  //Global MEs to monitor APV Shots properties
  MonitorElement *NApvShotsGlobal, *NApvShotsGlobalProf, *MedianChargeApvShotsGlobal, *NApvApvShotsGlobal, *StripMultiplicityApvShotsGlobal, *ShotsVsTimeApvShotsGlobal;

  TkHistoMap* tkmapdigi, *tkmapNApvshots, *tkmapNstripApvshot, *tkmapMedianChargeApvshots;

  int runNb, eventNb;
  int firstEvent;
  int TotalNShots;

  bool globalsummaryapvshotson;

  bool layerswitchnumdigison;
  bool layerswitchnumdigisapvon;
  bool layerswitchadchotteston;
  bool layerswitchadccooleston;
  bool layerswitchdigiadcson;
  bool layerswitchstripoccupancyon;
  bool layerswitchnumdigisprofon;
  bool layerswitchdigiadcprofon;

  bool moduleswitchnumdigison;
  bool moduleswitchnumdigispstripon;
  bool moduleswitchadchotteston;
  bool moduleswitchadccooleston;
  bool moduleswitchdigiadcson;
  bool moduleswitchstripoccupancyon;

  bool subdetswitchtotdigiprofon;
  bool subdetswitchapvcycleprofon;
  bool subdetswitchapvcycleth2on;

  bool subdetswitchtotdigifailureon;

  bool subdetswitchnapvshotson;
  bool subdetswitchnstripsapvshotson;
  bool subdetswitchapvshotsonprof;
  bool subdetswitchchargemedianapvshotson;
  bool subdetswitchapvshotsApvon;

  bool globalswitchnapvshotson;
  bool globalswitchnstripsapvshotson;
  bool globalswitchapvshotsonprof;
  bool globalswitchchargemedianapvshotson;
  bool globalswitchapvshotsApvon;
  bool globalswitchNDigisFEDID;

  //  int xLumiProf;
  bool isStableBeams;
  int ignoreFirstNLumisections_;
  int integrateNLumisections_;
  int SBDeclaredAt;
  bool SBTransitionDone;
  int nFedTIB;
  int nFedTIDp;
  int nFedTIDm;
  int nFedTECp;
  int nFedTECm;
  int nFedTOB;
  //  int nFEDConnected;

  bool Mod_On_;

  bool digitkhistomapon;
  bool shotshistomapon;
  bool shotsstripshistomapon;
  bool shotschargehistomapon;

  bool createTrendMEs;

  bool m_trendVsLS;

  edm::InputTag historyProducer_;
  edm::InputTag apvPhaseProducer_;

  edm::EDGetTokenT<EventWithHistory> historyProducerToken_;
  edm::EDGetTokenT<APVCyclePhaseCollection> apvPhaseProducerToken_;
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> gtEvmToken_;

  SiStripDCSStatus* dcsStatus_;
};
#endif
