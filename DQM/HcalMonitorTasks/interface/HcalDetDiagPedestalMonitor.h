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

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

/** \class HcalDetDiagPedestalMonitor
  *  
  * $Date: 2009/07/01 06:09:04 $
  * $Revision: 1.1.2.6 $
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
             for(int i=0;i<128;i++) adc[i]=0; 
	     overflow=0;
          }	  
   void   add_statistics(unsigned int val){
             if(val<25) adc[val&0x7F]++; else overflow++;    
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
	     int from,to,max=adc[0],maxi=0;
	     for(int i=1;i<25;i++) if(adc[i]>max){ max=adc[i]; maxi=i;} 
	     from=0; to=maxi+6;
             for(int i=from;i<=to;i++){
                Sum+=i*adc[i];
	        nSum+=adc[i];
             } 
             if(nSum>0) *ave=Sum/nSum; else return false;
             Sum=0;
             for(int i=from;i<=to;i++) Sum+=adc[i]*(i-*ave)*(i-*ave);
             *rms=sqrt(Sum/nSum);
             return true; 
          }
   int    get_statistics(){
             int nSum=0;  
             for(int i=0;i<25;i++) nSum+=adc[i];
	     return nSum;
	  } 
   int    get_overflow(){
             return overflow;
          }   
private:   
   int   adc[128];
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
  edm::InputTag inputLabelDigi_;
  
  const HcalElectronicsMap  *emap;
  
  int         ievt_;
  int         run_number;
  int         dataset_seq_number;
  bool        IsReference;
  Timestamp   time_min,time_max;
  
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
  std::string ReferenceRun;
  std::string OutputFilePath;

  MonitorElement *meEVT_;
  MonitorElement *RefRun_;
  MonitorElement *PedestalsAve4HB;
  MonitorElement *PedestalsAve4HE;
  MonitorElement *PedestalsAve4HO;
  MonitorElement *PedestalsAve4HF;
  MonitorElement *PedestalsAve4Simp;
  MonitorElement *PedestalsAve4ZDC;
  
  MonitorElement *PedestalsRefAve4HB;
  MonitorElement *PedestalsRefAve4HE;
  MonitorElement *PedestalsRefAve4HO;
  MonitorElement *PedestalsRefAve4HF;
  MonitorElement *PedestalsRefAve4Simp;
  MonitorElement *PedestalsRefAve4ZDC;
  
  MonitorElement *PedestalsAve4HBref;
  MonitorElement *PedestalsAve4HEref;
  MonitorElement *PedestalsAve4HOref;
  MonitorElement *PedestalsAve4HFref;
  MonitorElement *PedestalsRmsHB;
  MonitorElement *PedestalsRmsHE;
  MonitorElement *PedestalsRmsHO;
  MonitorElement *PedestalsRmsHF;
  MonitorElement *PedestalsRmsSimp;
  MonitorElement *PedestalsRmsZDC;
  
  MonitorElement *PedestalsRmsRefHB;
  MonitorElement *PedestalsRmsRefHE;
  MonitorElement *PedestalsRmsRefHO;
  MonitorElement *PedestalsRmsRefHF;
  MonitorElement *PedestalsRmsRefSimp;
  MonitorElement *PedestalsRmsRefZDC;
  
  MonitorElement *PedestalsRmsHBref;
  MonitorElement *PedestalsRmsHEref;
  MonitorElement *PedestalsRmsHOref;
  MonitorElement *PedestalsRmsHFref;
  
  MonitorElement *Pedestals2DRmsHBHEHF;
  MonitorElement *Pedestals2DRmsHO;
  MonitorElement *Pedestals2DHBHEHF;
  MonitorElement *Pedestals2DHO;
  MonitorElement *Pedestals2DErrorHBHEHF;
  MonitorElement *Pedestals2DErrorHO;
  
  HcalDetDiagPedestalData hb_data[85][72][4][4];
  HcalDetDiagPedestalData he_data[85][72][4][4];
  HcalDetDiagPedestalData ho_data[85][72][4][4];
  HcalDetDiagPedestalData hf_data[85][72][4][4];
  HcalDetDiagPedestalData zdc_data[5][7][7][4];
  
  std::vector<MonitorElement*> ChannelStatusMissingChannels;
  std::vector<MonitorElement*> ChannelStatusUnstableChannels;
  std::vector<MonitorElement*> ChannelStatusBadPedestalMean;
  std::vector<MonitorElement*> ChannelStatusBadPedestalRMS;
  void fill_channel_status(char *subdet,int eta,int phi,int depth,int type,double status);
};

#endif
