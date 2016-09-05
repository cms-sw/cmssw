#ifndef DQHarvester_h
#define DQHarvester_h

/*
 *	file:		DQHarvester.h
 *	Author:		VK
 */

#include "DQM/HcalCommon/interface/DQModule.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"

#include <vector>
#include <string>

namespace hcaldqm
{
	class DQHarvester : public DQMEDHarvester, public DQModule
	{
		public:
			DQHarvester(edm::ParameterSet const&);
			virtual ~DQHarvester() {}

			virtual void beginRun(edm::Run const&, edm::EventSetup const&);
			virtual void dqmEndLuminosityBlock(
				DQMStore::IBooker&, DQMStore::IGetter&, 
				edm::LuminosityBlock const&, edm::EventSetup const&);
			virtual void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&);

		protected:
			//	empa
			HcalElectronicsMap const* _emap;

			//	some counters
			int _totalLS;

			//	all FEDs
			std::vector<int>		_vFEDs;
			std::vector<uint32_t>	_vhashFEDs;
			//	container of quality masks from conddb
			ContainerXXX<uint32_t> _xQuality;
			//	vector of Electronics raw Ids of HCAL FEDs
			//	that were registered at cDAQ for the Run
			std::vector<uint32_t> _vcdaqEids;

			virtual void _dqmEndLuminosityBlock(
				DQMStore::IBooker&, DQMStore::IGetter&, 
				edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
			virtual void _dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) = 0;
	};
}

#endif









