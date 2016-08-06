#ifndef HcalOfflineHarvesting_h
#define HcalOfflineHarvesting_h

/**
 *	file:		HcalOffineHarvesting.h
 *	Author:		VK
 *	Date:		..
 *	Description: 
 *		This is DQMEDAnalyzer which is a edm::one module. edm::one enforces
 *		that only 1 run is being processed.
 *		https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkOneModuleInterface - for details.
 *
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
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/DQClient.h"

#include "DQM/HcalTasks/interface/RawRunSummary.h"
#include "DQM/HcalTasks/interface/DigiRunSummary.h"
#include "DQM/HcalTasks/interface/RecoRunSummary.h"
#include "DQM/HcalTasks/interface/TPRunSummary.h"

using namespace hcaldqm;

class HcalOfflineHarvesting : public DQHarvester
{
	public:
		HcalOfflineHarvesting(edm::ParameterSet const&);
		virtual ~HcalOfflineHarvesting(){}

		virtual void beginRun(edm::Run const&,
			edm::EventSetup const&);

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
			nSummary=4
		};

		//	vector of Summary Generators and marks of being present
		//	by default all false
		std::vector<DQClient*> _vsumgen;
		std::vector<bool> _vmarks;
		std::vector<std::string> _vnames;
		
		//	reportSummaryMap
		MonitorElement *_reportSummaryMap;
		MonitorElement *me;
};

#endif
