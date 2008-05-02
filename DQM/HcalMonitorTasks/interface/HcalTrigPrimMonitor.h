#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


/** \class HcalTrigPrimMonitor
  *  
  * $Date: 2008/04/03 12:54:54 $
  * $Revision: 1.8 $
  * \author W. Fisher - FNAL
  */
 static const float TrigMonAdc2fc[128]={-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                                     8.5,  9.5, 10.5, 11.5, 12.5, 13.5, 15., 17.,
                                     19., 21., 23., 25., 27., 29.5,
                                     32.5, 35.5, 38.5, 42., 46., 50., 54.5, 59.5,
                                     64.5, 59.5, 64.5, 69.5, 74.5,
                                     79.5, 84.5, 89.5, 94.5, 99.5, 104.5, 109.5,
                                     114.5, 119.5, 124.5, 129.5, 137.,
                                     147., 157., 167., 177., 187., 197., 209.5, 224.5,
                                     239.5, 254.5, 272., 292.,
                                     312., 334.5, 359.5, 384.5, 359.5, 384.5, 409.5,
                                     434.5, 459.5, 484.5, 509.5,
                                     534.5, 559.5, 584.5, 609.5, 634.5, 659.5, 684.5, 709.5,
                                     747., 797., 847.,
                                     897.,  947., 997., 1047., 1109.5, 1184.5, 1259.5,
                                     1334.5, 1422., 1522., 1622.,
                                     1734.5, 1859.5, 1984.5, 1859.5, 1984.5, 2109.5,
                                     2234.5, 2359.5, 2484.5,
                                     2609.5, 2734.5, 2859.5, 2984.5, 3109.5,
                                     3234.5, 3359.5, 3484.5, 3609.5, 3797.,
                                     4047., 4297., 4547., 4797., 5047., 5297.,
                                     5609.5, 5984.5, 6359.5, 6734.5,
                                     7172., 7672., 8172., 8734.5, 9359.5, 9984.5};


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
  double TPThresh_;
  int TPdigi_;
  int ADCdigi_;
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
  void   set_tp(int eta,int phi,int depth=1,float *val){ 
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
