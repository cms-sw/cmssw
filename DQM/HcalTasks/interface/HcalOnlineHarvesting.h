#ifndef DQM_HcalTasks_HcalOnlineHarvesting_h
#define DQM_HcalTasks_HcalOnlineHarvesting_h

/**
 *	file:		HcalOnlineHarvesting.h
 *	Author:		VK
 *	Date:		..
 *	Description: 
 *		This is DQMEDAnalyzer which is a edm::one module. edm::one enforces
 *		that only 1 run is being processed.
 *		https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkOneModuleInterface - for details.
 *
 *		HcalOnlineHarvesting is responsible for Status Evaluation and Summary
 *		Generation. In this step RAW, DIGI, RECO + TP Data Tiers 
 *		are evaluated and Summary is generated. 
 *		___________
 *		Online:
 *		There is always a Current Summary - Summary for the Current LS 
 *		being Evaluated. It might and might not include the information 
 *		from previous LSs, depending on the Quantity.
 *		___________
 *		Offline:
 *		For Offline only Run Summary is being generated. As it is meaningless
 *		to have current LS information being delivered. Only Total Summary
 *		makes sense
 */

#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"
#include "DQM/HcalCommon/interface/DQHarvester.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/DQClient.h"

#include "DQM/HcalTasks/interface/RawRunSummary.h"
#include "DQM/HcalTasks/interface/DigiRunSummary.h"
#include "DQM/HcalTasks/interface/RecoRunSummary.h"
#include "DQM/HcalTasks/interface/TPRunSummary.h"
#include "DQM/HcalTasks/interface/PedestalRunSummary.h"

using namespace hcaldqm;

class HcalOnlineHarvesting : public DQHarvester
{
	public:
		HcalOnlineHarvesting(edm::ParameterSet const&);
		virtual ~HcalOnlineHarvesting(){}
		virtual void beginRun(edm::Run const&, edm::EventSetup const&);

	protected:
		virtual void _dqmEndLuminosityBlock(DQMStore::IBooker&,
			DQMStore::IGetter&, edm::LuminosityBlock const&,
			edm::EventSetup const&);
		virtual void _dqmEndJob(DQMStore::IBooker&,
			DQMStore::IGetter&);

		enum Summary
		{
			fRaw=0,
			fDigi=1,
			fReco=2,
			fTP=3,
			fPedestal=4,
			nSummary=5
		};

		//	flags to harvest...
		std::vector<bool> _vmarks;
		std::vector<DQClient*> _vsumgen;
		std::vector<std::string> _vnames;

		//	summaries
		std::vector<ContainerSingle2D> _vcSummaryvsLS;

		MonitorElement *_me;

		//	reportSummaryMap
		MonitorElement *_reportSummaryMap;
};

#endif
