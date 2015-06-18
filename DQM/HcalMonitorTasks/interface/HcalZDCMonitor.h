#ifndef DQM_HCALMONITORTASKS_HCALZDCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALZDCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDigi/interface/HcalUnpackerReport.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include <cmath>

/** \class HcalZDCMonitor
 *
 * \author S.Sen
 */

class HcalZDCMonitor: public HcalBaseMonitor
{
public:
	HcalZDCMonitor();
	~HcalZDCMonitor();
	void setup(const edm::ParameterSet& ps, DQMStore::IBooker& ib);
	void processEvent(const ZDCDigiCollection& digi,
			const ZDCRecHitCollection& rechit,
			const HcalUnpackerReport& report);
	void reset();
	void endLuminosityBlock(void);
private:
	double getTime(const std::vector<double>& fData, unsigned int ts_min, unsigned int ts_max, double& fSum);
	//int getTSMax(std::vector<double> fData);
	//bool isGood(std::vector<double> fData, double fCut, double fPercentage);

	bool checkZDC_;
	int NumBadZDC;
	MonitorElement* ProblemsVsLB_ZDC;

	int NLumiBlocks_;
	float ChannelRatio[18];//errorevents/total events in 1 LS
	int EventCounter;//events in lumi

	int OfflineDeadThreshold_;
	int OnlineDeadThreshold_;
	int OnlineColdThreshold_;

	int TotalChannelErrors[18];//total events with an error per channel in a LS
	int DeadChannelCounter[18];
	int ColdChannelCounter[18];
	bool DeadChannelError[18];
	bool HotChannelError[18];
	bool DigiErrorCAPID[18];
	bool DigiErrorDVER[18];
	bool ChannelHasDigiError[18];

	std::vector<double> ChannelWeighting_; //Quality index(QI) see description below
	std::vector<double> MaxErrorRates_; //the fractional error rate before a channel is called bad for a LS
	int OfflineColdThreshold_;

//	const HcalQIEShape* shape_;
//	const HcalQIECoder* channelCoder_;
//	HcalCalibrations calibs_;
	int ievt_;
	//int zdc_checkNevents_;
	MonitorElement* meEVT_;
	MonitorElement* h_2D_saturation;
	MonitorElement* h_2D_charge;
	MonitorElement* h_2D_TSMean;
	MonitorElement* h_2D_RecHitEnergy;
	MonitorElement* h_2D_RecHitTime;
	MonitorElement* h_ZDCP_EM_Pulse[5];
	MonitorElement* h_ZDCM_EM_Pulse[5];
	MonitorElement* h_ZDCP_EM_Charge[5];
	MonitorElement* h_ZDCM_EM_Charge[5];
	MonitorElement* h_ZDCP_EM_TSMean[5];
	MonitorElement* h_ZDCM_EM_TSMean[5];
	MonitorElement* h_ZDCP_HAD_Pulse[4];
	MonitorElement* h_ZDCM_HAD_Pulse[4];
	MonitorElement* h_ZDCP_HAD_Charge[4];
	MonitorElement* h_ZDCM_HAD_Charge[4];
	MonitorElement* h_ZDCP_HAD_TSMean[4];
	MonitorElement* h_ZDCM_HAD_TSMean[4];
	MonitorElement* h_ZDCP_EM_RecHitEnergy[5];
	MonitorElement* h_ZDCM_EM_RecHitEnergy[5];
	MonitorElement* h_ZDCP_EM_RecHitTiming[5];
	MonitorElement* h_ZDCM_EM_RecHitTiming[5];
	MonitorElement* h_ZDCP_HAD_RecHitEnergy[4];
	MonitorElement* h_ZDCM_HAD_RecHitEnergy[4];
	MonitorElement* h_ZDCP_HAD_RecHitTiming[4];
	MonitorElement* h_ZDCM_HAD_RecHitTiming[4];


	MonitorElement* ZDC_Digi_Errors;
	MonitorElement* ZDC_DigiErrorsVsLS;
	MonitorElement* ZDC_DigiErrors_DVER;
	MonitorElement* ZDC_DigiErrors_CAPID;

	MonitorElement* ZDC_Hot_Channel_Errors;
	MonitorElement* ZDC_HotChannelErrorsVsLS;

	MonitorElement* ZDC_Cold_Channel_Errors;
	MonitorElement* ZDC_ColdChannelErrorsVsLS;

	MonitorElement* ZDC_Dead_Channel_Errors;
	MonitorElement* ZDC_DeadChannelErrorsVsLS;

	MonitorElement* ZDC_TotalChannelErrors;

	// Quality index(QI) per LS...QI is a number between 0 and 1. 
	// Each ZDC channel is assigned a custom weighting factor, 
	// found in /python/HcalZDCMonitor_cfi.py. 
	// This weighting factor is representative of the "value" of that channel to data taking. 
	// For example, HAD1 is more valuable than HAD4, 
	// this is because HAD1 is closer to the IP and will absorb more of the energy of the neutron, 
	// so if it is broken, we care more about it being broken than if HAD4 is broken. 
	// If all channels are working properly in a given ZDC (ZDC+/-),
	// the QI for that ZDC will be 1. 
	// If a channel is deemed malfunctional for a given LS, 
	// then the assigned weightfactor will be subtracted from 1.
	MonitorElement* PZDC_QualityIndexVsLB_;
	MonitorElement* NZDC_QualityIndexVsLB_;
	MonitorElement* EventsVsLS;
};

#endif
