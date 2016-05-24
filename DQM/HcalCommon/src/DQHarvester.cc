#include "DQM/HcalCommon/interface/DQHarvester.h"

namespace hcaldqm
{
	DQHarvester::DQHarvester(edm::ParameterSet const& ps) :
		DQModule(ps)
	{}

	/* virtual */ void DQHarvester::beginRun(edm::Run const& r,
		edm::EventSetup const& es)
	{
		if (_ptype==fLocal)
			if (r.runAuxiliary().run()==1)
				return;

		//	TEMPORARY FIX
		_vhashFEDs.clear(); _vcdaqEids.clear();

		//	- get the Hcal Electronics Map
		//	- collect all the FED numbers and FED's rawIds
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

		//	get the FEDs registered at cDAQ
		edm::eventsetup::EventSetupRecordKey recordKey(
			edm::eventsetup::EventSetupRecordKey::TypeTag::findType(
			"RunInfoRcd"));
		if (es.find(recordKey))
		{
			edm::ESHandle<RunInfo> ri;
			es.get<RunInfoRcd>().get(ri);
			std::vector<int> vfeds= ri->m_fed_in;
			for (std::vector<int>::const_iterator it=vfeds.begin();
				it!=vfeds.end(); ++it)
			{
				if (*it>=constants::FED_VME_MIN && *it<=FED_VME_MAX)
					_vcdaqEids.push_back(HcalElectronicsId(
						constants::FIBERCH_MIN,
						constants::FIBER_VME_MIN, SPIGOT_MIN,
						(*it)-FED_VME_MIN).rawId());
				else if (*it>=constants::FED_uTCA_MIN && 
					*it<=FEDNumbering::MAXHCALuTCAFEDID)
					_vcdaqEids.push_back(HcalElectronicsId(
						utilities::fed2crate(*it), SLOT_uTCA_MIN,
						FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
			}
		}

		//	get the Hcal Channels Quality for channels that are not 0
		_xQuality.initialize(hashfunctions::fDChannel);
		edm::ESHandle<HcalChannelQuality> hcq;
		es.get<HcalChannelQualityRcd>().get("withTopo", hcq);
		const HcalChannelQuality *cq = hcq.product();
		std::vector<DetId> detids = cq->getAllChannels();
		for (std::vector<DetId>::const_iterator it=detids.begin();
			it!=detids.end(); ++it)
		{
			 if (HcalGenericDetId(*it).genericSubdet()==
				HcalGenericDetId::HcalGenUnknown)
				 continue;

			 if (HcalGenericDetId(*it).isHcalDetId())
			 {
				 HcalDetId did(*it);
				  uint32_t mask = (cq->getValues(did))->getValue();
				   if (mask!=0)
					   _xQuality.push(did, mask);
			 }
		}
	}

	/* virtual */ void DQHarvester::dqmEndLuminosityBlock(
		DQMStore::IBooker& ib, DQMStore::IGetter& ig,
		edm::LuminosityBlock const& lb, edm::EventSetup const& es)
	{
		//	should be the same - just in case!
		_currentLS = lb.luminosityBlock();
		_totalLS++;
		_dqmEndLuminosityBlock(ib, ig, lb, es);
	}
	/* virtual */ void DQHarvester::dqmEndJob(DQMStore::IBooker& ib, 
		DQMStore::IGetter& ig)
	{
		_dqmEndJob(ib, ig);
	}
}
