#include "DQM/HcalCommon/interface/DQClient.h"

namespace hcaldqm
{
	DQClient::DQClient(std::string const& name, std::string const& taskname,
		edm::ParameterSet const& ps) :
		DQModule(ps),_taskname(taskname), _maxProcessedLS(0)
	{
		//	- SET THE TASK NAME YOU REFER TO
		//	- SET THE CLIENT'S NAME AS WELL - RUN SUMMARY PLOTS
		//	WILL BE GENERATED UNDER THAT FOLDER
		_name = name;
	}

	/* virtual */ void DQClient::beginRun(edm::Run const& r,
		edm::EventSetup const& es)
	{
		//	TEMPORARY
		_vhashFEDs.clear(); _vcdaqEids.clear();

		//	get various FED lists
		edm::ESHandle<HcalDbService> dbs;
		es.get<HcalDbRecord>().get(dbs);
		_emap = dbs->getHcalMapping();
		_vFEDs = utilities::getFEDList(_emap);
		for (std::vector<int>::const_iterator it=_vFEDs.begin();
			it!=_vFEDs.end(); ++it)
		{
			//
			//	FIXME
			//	until there exists a map of FED2Crate and Crate2FED,
			//	all the unknown Crates will be mapped to 0...
			//
			if (*it==0)
			{
				_vhashFEDs.push_back(HcalElectronicsId(
					0, SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
					FIBERCH_MIN, false).rawId());
				continue;
			}

			if (*it>FED_VME_MAX)
				_vhashFEDs.push_back(HcalElectronicsId(
					utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
					FIBERCH_MIN, false).rawId());
			else
				_vhashFEDs.push_back(HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, SPIGOT_MIN, (*it)-FED_VME_MIN).rawId());
		}

		//	get FEDs registered @cDAQ
		edm::eventsetup::EventSetupRecordKey recordKey(
			edm::eventsetup::EventSetupRecordKey::TypeTag::findType(
				"RunInfoRcd"));
		if (es.find(recordKey))
		{
			edm::ESHandle<RunInfo> ri;
			es.get<RunInfoRcd>().get(ri);
			std::vector<int> vfeds=ri->m_fed_in;
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

		//	get the Channel Quality masks
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
				uint32_t mask=(cq->getValues(did))->getValue();
				if (mask!=0)
					_xQuality.push(did, mask);
			}
		}
	}

	/* virtual */ void DQClient::endLuminosityBlock(DQMStore::IBooker&,
		DQMStore::IGetter&, edm::LuminosityBlock const& lb,
		edm::EventSetup const&)
	{
		_currentLS=lb.luminosityBlock();
		_totalLS++;
		if (_maxProcessedLS<_currentLS)
			_maxProcessedLS=_currentLS;
	}

	/* virtual */ std::vector<flag::Flag> DQClient::endJob(DQMStore::IBooker&,
		DQMStore::IGetter&)
	{
		return std::vector<flag::Flag>();
	}
}
