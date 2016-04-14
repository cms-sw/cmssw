#include "DQM/HcalCommon/interface/DQHarvester.h"

namespace hcaldqm
{
	DQHarvester::DQHarvester(edm::ParameterSet const& ps) :
		DQModule(ps)
	{}

	/* virtual */ void DQHarvester::beginRun(edm::Run const&,
		edm::EventSetup const& es)
	{
		edm::ESHandle<HcalDbService> dbs;
		es.get<HcalDbRecord>().get(dbs);
		_emap = dbs->getHcalMapping();

		_vFEDs = utilities::getFEDList(_emap);
		for (std::vector<int>::const_iterator it=_vFEDs.begin();
			it!=_vFEDs.end(); ++it)
			if (*it>FED_VME_MAX)
				_vhashFEDs.push_back(HcalElectronicsId(
					utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
					FIBERCH_MIN, false).rawId());
			else
				_vhashFEDs.push_back(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, SPIGOT_MIN, (*it)-FED_VME_MIN).rawId());
	}

	/* virtual */ void DQHarvester::dqmEndLuminosityBlock(
		DQMStore::IBooker& ib, DQMStore::IGetter& ig,
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
	{
		_currentLS = lb.luminosityBlock();
		_dqmEndLuminosityBlock(ib, ig, lb, es);
	}
	/* virtual */ void DQHarvester::dqmEndJob(DQMStore::IBooker& ib, 
		DQMStore::IGetter& ig)
	{
		_dqmEndJob(ib, ig);
	}
}
