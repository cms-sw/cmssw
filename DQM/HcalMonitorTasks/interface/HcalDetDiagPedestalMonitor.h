#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGPEDESTALMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGPEDESTALMONITOR_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <math.h>
using namespace edm;
using namespace std;
// this is to retrieve HCAL digi's
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// to retrive trigger information (local runs only)
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
// to retrive GMT information, for cosmic runs muon triggers can be used as pedestal (global runs only)
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
// to retrive trigger desition words, to select pedestal (from hcal point of view) triggers (global runs only)
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "CondTools/Hcal/interface/HcalLogicalMapGenerator.h"
#include "CondTools/Hcal/interface/HcalLogicalMap.h"

/** \class HcalDetDiagPedestalMonitor
  *  
  * $Date: 2009/01/21 15:01:37 $
  * $Revision: 1.20 $
  * \author D. Vishnevskiy
  */

class HcalDetDiagPedestalData{
public: 
   HcalDetDiagPedestalData(){ 
             reset();
	     IsRefetence=false;
	     status=0;   
	  }
   void   reset(){
             for(int i=0;i<10;i++) adc[i]=0; 
	     overflow=0;
          }	  
   void   add_statistics(unsigned int val){
             if(val<10) adc[val]++; else overflow++;    
	  }
   void   set_reference(float val,float rms){
             ref_ped=val; ref_rms=rms;
	     IsRefetence=true;
          }	  
   void   change_status(int val){
             status|=val;
          }	  
   int    get_status(){
             return status;
          }	  
   bool   get_reference(double *val,double *rms){
             *val=ref_ped; *rms=ref_rms;
	     return IsRefetence;
          }	  
   bool   get_average(double *ave,double *rms){
             double Sum=0,nSum=0; 
             for(int i=0;i<10;i++){
                Sum+=i*adc[i];
	        nSum+=adc[i];
             } 
             if(nSum>0) *ave=Sum/nSum; else return false;
             Sum=0;
             for(int i=0;i<10;i++) Sum+=adc[i]*(i-*ave)*(i-*ave);
             *rms=sqrt(Sum/nSum);
             return true; 
          }
   int    get_statistics(){
             int nSum=0;  
             for(int i=0;i<10;i++) nSum+=adc[i];
	     return nSum;
	  } 
   int    get_overflow(){
             return overflow;
          }   
private:   
   int   adc[10];
   int   overflow;
   bool  IsRefetence;
   float ref_ped;
   float ref_rms;
   int   status;
};

class HcalDetDiagPedestalMonitor:public HcalBaseMonitor {
public:
  HcalDetDiagPedestalMonitor(); 
  ~HcalDetDiagPedestalMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond);
  void done();
  void reset();
  void clearME(); 
  void fillHistos();
  // in principle functions below shold use DB interface (will be modefied when DB will be ready...)  
  void SaveReference();
  void LoadReference();
  void CheckStatus();
  int  GetStatistics(){ return ievt_; }
private:
 // HcalLogicalMap            *lmap;
  const HcalElectronicsMap  *emap;
  
  int         ievt_;
  int         run_number;
  int         dataset_seq_number;
  bool        IsReference;
  
  double      HBMeanTreshold;
  double      HBRmsTreshold;
  double      HEMeanTreshold;
  double      HERmsTreshold;
  double      HOMeanTreshold;
  double      HORmsTreshold;
  double      HFMeanTreshold;
  double      HFRmsTreshold;
  bool        UseDB;
   
  std::string ReferenceData;
  std::string OutputFilePath;

  MonitorElement *meEVT_;
  MonitorElement *PedestalsAve4HB;
  MonitorElement *PedestalsAve4HE;
  MonitorElement *PedestalsAve4HO;
  MonitorElement *PedestalsAve4HF;
  
  MonitorElement *PedestalsRefAve4HB;
  MonitorElement *PedestalsRefAve4HE;
  MonitorElement *PedestalsRefAve4HO;
  MonitorElement *PedestalsRefAve4HF;
  
  MonitorElement *PedestalsAve4HBref;
  MonitorElement *PedestalsAve4HEref;
  MonitorElement *PedestalsAve4HOref;
  MonitorElement *PedestalsAve4HFref;
  MonitorElement *PedestalsRmsHB;
  MonitorElement *PedestalsRmsHE;
  MonitorElement *PedestalsRmsHO;
  MonitorElement *PedestalsRmsHF;
  
  MonitorElement *PedestalsRmsRefHB;
  MonitorElement *PedestalsRmsRefHE;
  MonitorElement *PedestalsRmsRefHO;
  MonitorElement *PedestalsRmsRefHF;
  
  MonitorElement *PedestalsRmsHBref;
  MonitorElement *PedestalsRmsHEref;
  MonitorElement *PedestalsRmsHOref;
  MonitorElement *PedestalsRmsHFref;
  
  HcalDetDiagPedestalData hb_data[85][72][4][4];
  HcalDetDiagPedestalData he_data[85][72][4][4];
  HcalDetDiagPedestalData ho_data[85][72][4][4];
  HcalDetDiagPedestalData hf_data[85][72][4][4];
  
  std::vector<MonitorElement*> ChannelStatusMissingChannels;
  std::vector<MonitorElement*> ChannelStatusUnstableChannels;
  std::vector<MonitorElement*> ChannelStatusBadPedestalMean;
  std::vector<MonitorElement*> ChannelStatusBadPedestalRMS;
  void fill_channel_status(char *subdet,int eta,int phi,int depth,int type,double status);
};

#endif
