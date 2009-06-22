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
  * $Date: 2009/05/01 14:06:09 $
  * $Revision: 1.25 $
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
		    const ZDCDigiCollection& zdc,
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
  EtaPhiHists MeanMapByDepth;
  EtaPhiHists RMSMapByDepth;

  // Original pedestal read info from database
  MonitorElement* ADC_PedestalFromDB;
  EtaPhiHists ADC_PedestalFromDBByDepth;
  MonitorElement* ADC_WidthFromDB;
  EtaPhiHists ADC_WidthFromDBByDepth;
  MonitorElement* fC_PedestalFromDB;
  EtaPhiHists fC_PedestalFromDBByDepth;
  MonitorElement* fC_WidthFromDB;
  EtaPhiHists fC_WidthFromDBByDepth;
  std::vector<MonitorElement*> ADC_1D_PedestalFromDBByDepth;
  std::vector<MonitorElement*> ADC_1D_WidthFromDBByDepth;
  std::vector<MonitorElement*> fC_1D_PedestalFromDBByDepth;
  std::vector<MonitorElement*> fC_1D_WidthFromDBByDepth;

  
  EtaPhiHists PedestalOcc;

  // "raw" pedestal plots in ADC
  EtaPhiHists ADCPedestalMean;
  EtaPhiHists ADCPedestalRMS;
  std::vector<MonitorElement*> ADCPedestalMean_1D;
  std::vector<MonitorElement*> ADCPedestalRMS_1D;

  // subtracted ADC pedestal plots
  EtaPhiHists subADCPedestalMean;
  EtaPhiHists subADCPedestalRMS;
  std::vector<MonitorElement*> subADCPedestalMean_1D;
  std::vector<MonitorElement*> subADCPedestalRMS_1D;

  //  pedestal plots in femtocoulombs
  EtaPhiHists fCPedestalMean;
  EtaPhiHists fCPedestalRMS;
  std::vector<MonitorElement*> fCPedestalMean_1D;
  std::vector<MonitorElement*> fCPedestalRMS_1D;

  // subtracted pedestal plots in femtocoulombs
  EtaPhiHists subfCPedestalMean;
  EtaPhiHists subfCPedestalRMS;
  std::vector<MonitorElement*> subfCPedestalMean_1D;
  std::vector<MonitorElement*> subfCPedestalRMS_1D;

  // Problem Histograms 
  MonitorElement* ProblemPedestals;
  EtaPhiHists ProblemPedestalsByDepth;


  //Quick pedestal arrays -- these store the values that are used to compute pedestals
  unsigned int pedcounts[85][72][4];
  float ADC_pedsum[85][72][4];
  float ADC_pedsum2[85][72][4];
  float fC_pedsum[85][72][4];
  float fC_pedsum2[85][72][4];

  // ZDC pedestals
  std::vector<MonitorElement*> zdc_pedestals;
  float zdc_ADC_peds[2][2][5][26]; // zside, section, channel, 
  float zdc_ADC_count[2][2][5];

};

#endif
