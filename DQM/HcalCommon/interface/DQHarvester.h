#ifndef DQHarvester_h
#define DQHarvester_h

/*
 *	file:		DQHarvester.h
 *	Author:		VK
 */

#include "DQM/HcalCommon/interface/DQModule.h"
#include "DQM/HcalCommon/interface/DQClient.h"

#include <vector>
#include <string>

namespace hcaldqm
{
	class DQHarvester : public DQMEDHarvester, public DQModule
	{
		public:
			DQHarvester(edm::ParameterSet const&);
			virtual ~DQHarvester() {}

			virtual void dqmEndLuminosityBlock(DQMStore::IGetter&, 
				edm::LuminosityBlock const&, edm::EventSetup const&);
			virtual void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
			std::vector<DQClient*>			_clients;
	};
}

#endif









