#ifndef HCALDQCLIENT_H
#define HCALDQCLIENT_H

/*
 *	file:				HcalDQClient.h
 *	Author:				VK
 *	Start Date:			05/06/2015
 */

#include "DQM/HcalCommon/interface/HcalDQMonitor.h"
#include "DQM/HcalCommon/interface/HcalMECollection.h"
#include "DQM/HcalCommon/interface/HcalCommonHeaders.h"

namespace hcaldqm
{
	/*
	 *	HcalDQClient class - Base class for the DQClients
	 */
	class HcalDQClient : public DQMEDHarvester, public HcalDQMonitor
	{
		public:
			HcalDQClient(edm::ParameterSet const&);
			virtual ~HcalDQClient();

			//	Generic doWork functions for all DQClients
			//	per LS or per Run
			virtual void doWork(DQMStore::IGetter&,
					edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
	//		virtual void doWork(DQMStore::IBooker&, DQMStore::IGetter&) = 0;

			virtual void beginJob();
			//	EndJob is mandatory and EndLumiBlock is optional
			virtual void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&);
			virtual void dqmEndLuminosityBlock(DQMStore::IBooker&, 
					DQMStore::IGetter&, 
					edm::LuminosityBlock const&, edm::EventSetup const&);

		protected:
			//	Apply Reset/Update if neccessary
			//	periodfalg: 0 for per Event Reset and 1 for per LS
			virtual void reset(int const periodflag);

			//	Functions specific to Clients

		protected:
			HcalMECollection			_bmes;
			HcalMECollection			_rmes;
	};
}

#endif 
