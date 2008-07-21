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
// $Id: SiStripMonitorDigi.h,v 1.7 2008/03/01 00:37:46 dutta Exp $
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
	ModMEs():    
	  LayerNumberOfDigis(0),
	     LayerNumberOfDigisTrend(0),
	     LayerADCsHottestStrip(0),
	     LayerADCsHottestStripTrend(0),
	     LayerADCsCoolestStrip(0),
	     LayerADCsCoolestStripTrend(0),
	     LayerDigiADCs(0),
	     LayerDigiADCsTrend(0),
	     LayerStripOccupancy(0),
	     LayerStripOccupancyTrend(0){};
	     
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
	     
	     MonitorElement* NumberOfDigis;
	     MonitorElement* ADCsHottestStrip;
	     MonitorElement* ADCsCoolestStrip;
	     MonitorElement* DigiADCs;
	     MonitorElement* StripOccupancy;
	
      };
  private:
      void FillStripOccupancy(MonitorElement*,  std::vector<uint16_t> &);
      void createMEs(const edm::EventSetup& es);
      void ResetModuleMEs(uint32_t idet);
      void bookLayer(); 
      MonitorElement* bookMETrend(const char* ParameterSetLabel, const char* HistoName);
      MonitorElement* bookME1D(const char* ParameterSetLabel, const char* HistoName);
      void bookTrendMEs(TString name,int32_t layer,uint32_t id,std::string flag);
/*       void fillTrendMEs(edm::DetSetVector<SiStripDigi>::const_iterator iterdigi, std::string name); */
      void fillTrendMEs(int subdetid, int subsubdetid, std::string name);
      void fillDigiADCsMEs(int value, std::string name);
      void fillTrend(MonitorElement* me ,float value);
      inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
      inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
      inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
      inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}
      bool AllDigis( const edm::EventSetup& es);
      
   private:
       DQMStore* dqmStore_;
       edm::ParameterSet conf_;
       std::map<uint32_t, ModMEs> DigiMEs; // uint32_t me_type: 1=#digis/module; 2=adcs of hottest strip/module; 3= adcs of coolest strips/module.
       bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, calculate_strip_occupancy, reset_each_run;
       unsigned long long m_cacheID_;

       std::map<TString, ModMEs> ModMEsMap;
       std::map<TString, MonitorElement*> MEMap;
       
       edm::ParameterSet Parameters;
  
       edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
       TString name;
       SiStripFolderOrganizer folder_organizer;
       std::map<std::pair<std::string,int32_t>,bool> DetectedLayers;
/*        edm::Handle< edmNew::DetSetVector<SiStripDigi> > digi_detsetvektor; */
       edm::Handle< edm::DetSetVector<SiStripDigi> > digi_detsetvektor;
       std::vector<uint32_t> ModulesToBeExcluded_;

       int runNb, eventNb;
       int firstEvent;
       int **NDigi;
       int **ADCHottest;
       int **ADCCoolest;

       bool layerswitchnumdigison;
       bool layerswitchadchotteston;
       bool layerswitchadccooleston;
       bool layerswitchdigiadcson;

       bool moduleswitchnumdigison;
       bool moduleswitchadchotteston;
       bool moduleswitchadccooleston;
       bool moduleswitchdigiadcson;

       bool tibon;
       bool tidon;
       bool tobon;
       bool tecon;

};
#endif

