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
  * $Date: 2009/01/21 15:01:37 $
  * $Revision: 1.20 $
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

  // "raw" pedestal plots in ADC
  std::vector<MonitorElement*> rawADCPedestalMean;
  std::vector<MonitorElement*> rawADCPedestalRMS;
  std::vector<MonitorElement*> rawADCPedestalMean_1D;
  std::vector<MonitorElement*> rawADCPedestalRMS_1D;

  // subtracted ADC pedestal plots
  std::vector<MonitorElement*> subADCPedestalMean;
  std::vector<MonitorElement*> subADCPedestalRMS;
  std::vector<MonitorElement*> subADCPedestalMean_1D;
  std::vector<MonitorElement*> subADCPedestalRMS_1D;

  // raw pedestal plots in femtocoulombs
  std::vector<MonitorElement*> rawFCPedestalMean;
  std::vector<MonitorElement*> rawFCPedestalRMS;
  std::vector<MonitorElement*> rawFCPedestalMean_1D;
  std::vector<MonitorElement*> rawFCPedestalRMS_1D;

  // subtracted pedestal plots in femtocoulombs
  std::vector<MonitorElement*> subFCPedestalMean;
  std::vector<MonitorElement*> subFCPedestalRMS;
  std::vector<MonitorElement*> subFCPedestalMean_1D;
  std::vector<MonitorElement*> subFCPedestalRMS_1D;

  // Problem Histograms 
  MonitorElement* ProblemPedestals;
  std::vector<MonitorElement*> ProblemPedestalsByDepth;


  //Quick pedestal arrays -- these store the values that are used to compute pedestals
  unsigned int pedcounts[ETABINS][PHIBINS][6];
  float rawpedsum[ETABINS][PHIBINS][6];
  float rawpedsum2[ETABINS][PHIBINS][6];
  float subpedsum[ETABINS][PHIBINS][6];
  float subpedsum2[ETABINS][PHIBINS][6];
  float fC_rawpedsum[ETABINS][PHIBINS][6];
  float fC_rawpedsum2[ETABINS][PHIBINS][6];
  float fC_subpedsum[ETABINS][PHIBINS][6];
  float fC_subpedsum2[ETABINS][PHIBINS][6];



};

#endif
