#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


/** \class HcalTrigPrimMonitor
  *  
  * $Date: 2008/03/01 00:39:58 $
  * $Revision: 1.7 $
  * \author W. Fisher - FNAL
  */
class HcalTrigPrimMonitor: public HcalBaseMonitor {
public:
  HcalTrigPrimMonitor(); 
  ~HcalTrigPrimMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
		    const HBHEDigiCollection& hbhedigi,
		    const HODigiCollection& hodigi,
		    const HFDigiCollection& hfdigi,		    
		    const HcalTrigPrimDigiCollection& tpDigis);
  void clearME();
  void reset();

private:  ///Monitoring elements

  int ievt_;
  int etaBins_, phiBins_;

  double occThresh_;
  double etaMax_, etaMin_, phiMax_, phiMin_;

  MonitorElement* meEVT_;
  MonitorElement* tpCount_;
  MonitorElement* tpCountThr_;
  MonitorElement* tpSize_;

  MonitorElement* tpSpectrum_[10];
  MonitorElement* tpSpectrumAll_;
  MonitorElement* tpETSumAll_;
  MonitorElement* tpSOI_ET_;

  MonitorElement* TPTiming_; 
  MonitorElement* TPTimingTop_; 
  MonitorElement* TPTimingBot_; 
  MonitorElement* TPOcc_;
  MonitorElement* TP_ADC_;
  MonitorElement* TPvsDigi_;

  MonitorElement* OCC_MAP_SLB;
  MonitorElement* OCC_ETA;
  MonitorElement* OCC_PHI;
  MonitorElement* OCC_MAP_GEO;
  MonitorElement* OCC_MAP_THR;
  MonitorElement* OCC_ELEC_VME;
  MonitorElement* OCC_ELEC_DCC;
  MonitorElement* EN_ETA;
  MonitorElement* EN_PHI;
  MonitorElement* EN_MAP_GEO;
  MonitorElement* EN_ELEC_VME;
  MonitorElement* EN_ELEC_DCC;
// not so nice , but very useful for correlation plots...
  void   ClearEvent(){
             memset(adc_data,  0,(sizeof(float)*100*72*5*10));
             memset(tp_data,   0,(sizeof(float)*100*72*5*10));
             memset(Is_adc_Data,0,(sizeof(char)*100*72*5));	  
             memset(Is_tp_Data, 0,(sizeof(char)*100*72*5));	  
	  }
  float *get_adc(int eta,int phi,int depth=1){
             return &adc_data[eta+50][phi][depth][0];}
  void   set_adc(int eta,int phi,int depth,float *val){ 
             if(eta<-42 || eta>42 || eta==0) return;
	     if(phi<1 || phi>72)             return;
	     if(depth<1 || depth>4)          return;
             for(int i=0;i<10;i++) adc_data[eta+50][phi][depth][i]=val[i];
             Is_adc_Data[eta+50][phi][depth]=1;
          }
  float *get_tp(int eta,int phi,int depth=1){
             return &tp_data[eta+50][phi][depth][0];}
  void   set_tp(int eta,int phi,int depth,float *val){ 
             if(eta<-42 || eta>42 || eta==0) return;
	     if(phi<1 || phi>72)             return;
	     if(depth<1 || depth>4)          return;
             for(int i=0;i<10;i++) tp_data[eta+50][phi][depth][i]=val[i];
             Is_tp_Data[eta+50][phi][depth]=1;
          }	
  char   IsSet_adc(int eta,int phi,int depth){ 
             return Is_adc_Data[eta+50][phi][depth];
	  }
  char   IsSet_tp(int eta,int phi,int depth){ 
             return Is_tp_Data[eta+50][phi][depth];
	  }
	    
  float adc_data   [100][73][5][10];
  float tp_data    [100][73][5][10];
  char  Is_adc_Data[100][73][5];
  char  Is_tp_Data [100][73][5];
};

#endif
