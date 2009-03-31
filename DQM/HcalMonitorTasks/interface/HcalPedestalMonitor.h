#ifndef DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H
#define DQM_HCALMONITORTASKS_HCALPEDESTALMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <cmath>

/** \class HcalPedestalMonitor
  *  
  * $Date: 2009/03/28 13:58:18 $
  * $Revision: 1.20.2.2 $
  * \author W. Fisher - FNAL
  */


class HcalPedestalMonitor: public HcalBaseMonitor {
public:
  HcalPedestalMonitor(); 
  ~HcalPedestalMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    // const ZDCDigiCollection& zdc,
		    const HcalDbService& cond);
  void done();
  void reset();
  void fillDBValues(const HcalDbService& cond);
  void fillPedestalHistos(void); // fills histograms once every (checkNevents_) events
  void clearME(); // overrides base class function
  void zeroCounters();

private:

  // Configurable parameters
  //bool doPerChannel_; // enable histograms for each channel (not yet (or ever?) operational)
  bool doFCpeds_; // pedestal units in fC (if false, assume ADC)
  // specify time slices over which to calculate pedestals
  bool startingTimeSlice_;
  bool endingTimeSlice_;
  
  // Specify maximum allowed difference between ADC pedestal and nominal value
  double nominalPedMeanInADC_;
  double nominalPedWidthInADC_;
  double maxPedMeanDiffADC_;
  double maxPedWidthDiffADC_; // specify maximum width of pedestal (in ADC)
  unsigned int minEntriesPerPed_; // minimum # of events needed to calculate pedestals
  // Haven't yet figured out how to implement these reasonably.
  // I'd like for them to default to whatever the global minErrorFlag_ has been set to,
  // but user should be able to also set them directly.  Hmm... 
  double pedmon_minErrorFlag_;
  int pedmon_checkNevents_;

  // Pedestal ADC/fC conversion stuffx
  const HcalQIEShape* shape_;
  const HcalQIECoder* channelCoder_;
  HcalCalibrations calibs_;

  MonitorElement* meEVT_;
  int ievt_;
  
  // Store means, RMS of pedestals by depth
  std::vector<MonitorElement*> MeanMapByDepth;
  std::vector<MonitorElement*> RMSMapByDepth;

  // Original pedestal read info from database
  MonitorElement* ADC_PedestalFromDB;
  std::vector<MonitorElement*> ADC_PedestalFromDBByDepth;
  MonitorElement* ADC_WidthFromDB;
  std::vector<MonitorElement*> ADC_WidthFromDBByDepth;
  MonitorElement* fC_PedestalFromDB;
  std::vector<MonitorElement*> fC_PedestalFromDBByDepth;
  MonitorElement* fC_WidthFromDB;
  std::vector<MonitorElement*> fC_WidthFromDBByDepth;
  std::vector<MonitorElement*> ADC_1D_PedestalFromDBByDepth;
  std::vector<MonitorElement*> ADC_1D_WidthFromDBByDepth;
  std::vector<MonitorElement*> fC_1D_PedestalFromDBByDepth;
  std::vector<MonitorElement*> fC_1D_WidthFromDBByDepth;
  // Reference Pedestal info by capid
  std::vector<std::vector<MonitorElement*> > ADC_PedestalFromDBByDepth_bycapid;
  std::vector<std::vector<MonitorElement*> > ADC_WidthFromDBByDepth_bycapid;
  std::vector<std::vector<MonitorElement*> > fC_PedestalFromDBByDepth_bycapid;
  std::vector<std::vector<MonitorElement*> > fC_WidthFromDBByDepth_bycapid;
  std::vector<std::vector<MonitorElement*> > ADC_PedestalFromDBByDepth_1D_bycapid;
  std::vector<std::vector<MonitorElement*> > ADC_WidthFromDBByDepth_1D_bycapid;
  std::vector<std::vector<MonitorElement*> > fC_PedestalFromDBByDepth_1D_bycapid;
  std::vector<std::vector<MonitorElement*> > fC_WidthFromDBByDepth_1D_bycapid;

  
  std::vector<MonitorElement*> PedestalOcc;

  // "raw" pedestal plots in ADC
  std::vector<MonitorElement*> ADCPedestalMean;
  std::vector<MonitorElement*> ADCPedestalRMS;
  std::vector<MonitorElement*> ADCPedestalMean_1D;
  std::vector<MonitorElement*> ADCPedestalRMS_1D;

  std::vector <std::vector<MonitorElement*> >  ADCPedestalMean_bycapid;
  std::vector <std::vector<MonitorElement*> >  ADCPedestalRMS_bycapid;
  std::vector <std::vector<MonitorElement*> >  ADCPedestalMean_1D_bycapid;
  std::vector <std::vector<MonitorElement*> >  ADCPedestalRMS_1D_bycapid;


  // subtracted ADC pedestal plots
  std::vector<MonitorElement*> subADCPedestalMean;
  std::vector<MonitorElement*> subADCPedestalRMS;
  std::vector<MonitorElement*> subADCPedestalMean_1D;
  std::vector<MonitorElement*> subADCPedestalRMS_1D;

  std::vector <std::vector<MonitorElement*> >  subADCPedestalMean_bycapid;
  std::vector <std::vector<MonitorElement*> >  subADCPedestalRMS_bycapid;
  std::vector <std::vector<MonitorElement*> >  subADCPedestalMean_1D_bycapid;
  std::vector <std::vector<MonitorElement*> >  subADCPedestalRMS_1D_bycapid;


  //  pedestal plots in femtocoulombs
  std::vector<MonitorElement*> fCPedestalMean;
  std::vector<MonitorElement*> fCPedestalRMS;
  std::vector<MonitorElement*> fCPedestalMean_1D;
  std::vector<MonitorElement*> fCPedestalRMS_1D;

  std::vector <std::vector<MonitorElement*> > fCPedestalMean_bycapid;
  std::vector <std::vector<MonitorElement*> > fCPedestalRMS_bycapid;
  std::vector <std::vector<MonitorElement*> > fCPedestalMean_1D_bycapid;
  std::vector <std::vector<MonitorElement*> > fCPedestalRMS_1D_bycapid;

  // subtracted pedestal plots in femtocoulombs
  std::vector<MonitorElement*> subfCPedestalMean;
  std::vector<MonitorElement*> subfCPedestalRMS;
  std::vector<MonitorElement*> subfCPedestalMean_1D;
  std::vector<MonitorElement*> subfCPedestalRMS_1D;

  std::vector <std::vector<MonitorElement*> > subfCPedestalMean_bycapid;
  std::vector <std::vector<MonitorElement*> > subfCPedestalRMS_bycapid;
  std::vector <std::vector<MonitorElement*> > subfCPedestalMean_1D_bycapid;
  std::vector <std::vector<MonitorElement*> > subfCPedestalRMS_1D_bycapid;

  // Problem Histograms 
  MonitorElement* ProblemPedestals;
  std::vector<MonitorElement*> ProblemPedestalsByDepth;


  //Quick pedestal arrays -- these store the values that are used to compute pedestals
  unsigned int pedcounts[ETABINS][PHIBINS][6];
  unsigned int pedcounts_bycapid[ETABINS][PHIBINS][6][4];
  float ADC_pedsum[ETABINS][PHIBINS][6];
  float ADC_pedsum2[ETABINS][PHIBINS][6];
  float fC_pedsum[ETABINS][PHIBINS][6];
  float fC_pedsum2[ETABINS][PHIBINS][6];

  float ADC_pedsum_bycapid[ETABINS][PHIBINS][6][4];
  float ADC_pedsum2_bycapid[ETABINS][PHIBINS][6][4];
  float fC_pedsum_bycapid[ETABINS][PHIBINS][6][4];
  float fC_pedsum2_bycapid[ETABINS][PHIBINS][6][4];



};

#endif
