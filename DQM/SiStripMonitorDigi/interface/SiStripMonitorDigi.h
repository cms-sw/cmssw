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
// $Id: SiStripMonitorDigi.h,v 1.14 2008/12/04 21:03:06 dutta Exp $
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;

class SiStripMonitorDigi : public edm::EDAnalyzer {
 public:
  explicit SiStripMonitorDigi(const edm::ParameterSet&);
  ~SiStripMonitorDigi();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(edm::EventSetup const&) ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  struct ModMEs{
	
    MonitorElement* NumberOfDigis;
    MonitorElement* ADCsHottestStrip;
    MonitorElement* ADCsCoolestStrip;
    MonitorElement* DigiADCs;
    MonitorElement* StripOccupancy;
    uint16_t nStrip;	
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
    MonitorElement* SubDetTotDigiProf;
    MonitorElement* SubDetDigiApvProf;
  };

 private:
  void createMEs(const edm::EventSetup& es);
  void ResetModuleMEs(uint32_t idet);
  void bookLayer(); 
  MonitorElement* bookMETrend(const char* ParameterSetLabel, const char* HistoName);
  MonitorElement* bookME1D(const char* ParameterSetLabel, const char* HistoName);
  void bookTrendMEs(TString name,int32_t layer,uint32_t id,std::string flag);
  void fillDigiADCsMEs(int value, std::string name);
  void fillTrend(MonitorElement* me ,float value, float timeinorbit);
  inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}
  bool AllDigis( const edm::EventSetup& es);

  void createModuleMEs(ModMEs& mod_single, uint32_t detid);
  void createLayerMEs(std::string label, int ndet);
  void createSubDetMEs(std::string label);
  void createSubDetTH2(std::string label);
  void getLayerLabel(uint32_t detid, std::string& label, int& subdetid, int& subsubdetid);
  int getDigiSource(uint32_t id, edm::DetSet<SiStripDigi>& digi_detset);
      
 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  std::map<uint32_t, ModMEs> DigiMEs; // uint32_t me_type: 1=#digis/module; 2=adcs of hottest strip/module; 3= adcs of coolest strips/module.
  bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, calculate_strip_occupancy, reset_each_run;
  unsigned long long m_cacheID_;

  std::map<std::string, std::vector< uint32_t > > LayerDetMap;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;
       
  edm::ParameterSet Parameters;
  
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  TString name;
  SiStripFolderOrganizer folder_organizer;
  std::map<std::pair<std::string,int32_t>,bool> DetectedLayers;
  /*        edm::Handle< edmNew::DetSetVector<SiStripDigi> > digi_detsetvektor; */
  edm::Handle< edm::DetSetVector<SiStripDigi> > digi_detsetvektor[4];
  std::vector<uint32_t> ModulesToBeExcluded_;

  int runNb, eventNb;
  int firstEvent;

  bool layerswitchnumdigison;
  bool layerswitchadchotteston;
  bool layerswitchadccooleston;
  bool layerswitchdigiadcson;
  bool layerswitchstripoccupancyon;
  bool layerswitchnumdigisprofon;
  bool layerswitchdigiadcprofon;
  bool layerswitchdigitkhistomapon;

  bool moduleswitchnumdigison;
  bool moduleswitchadchotteston;
  bool moduleswitchadccooleston;
  bool moduleswitchdigiadcson;
  bool moduleswitchstripoccupancyon;
  bool subdetswitchtotdigiprofon;
  bool subdetswitchapvcycleprofon;

  bool Mod_On_;

  bool createTrendMEs;
};
#endif

