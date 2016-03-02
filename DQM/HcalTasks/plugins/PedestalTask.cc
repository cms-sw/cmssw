
#include "DQM/HcalTasks/interface/PedestalTask.h"

using namespace hcaldqm;
PedestalTask::PedestalTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	//	tags
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));
	_tagTrigger = ps.getUntrackedParameter<edm::InputTag>("tagTrigger",
		edm::InputTag("tbunpacker"));
	_tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<HFDigiCollection>(_tagHF);
	_tokTrigger = consumes<HcalTBTriggerData>(_tagTrigger);
}

/* virtual */ void PedestalTask::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	if (_ptype==fLocal)
		if (r.runAuxiliary().run()==1)
			return;
	DQTask::bookHistograms(ib, r, es);
	
	edm::ESHandle<HcalDbService> dbs;
	es.get<HcalDbRecord>().get(dbs);
	_emap = dbs->getHcalMapping();
	std::vector<uint32_t> vhashVME;
	std::vector<uint32_t> vhashuTCA;
	std::vector<uint32_t> vhashC36;
	vhashVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN,
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	vhashC36.push_back(HcalElectronicsId(36, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
		vhashVME);
	_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
		vhashuTCA);
	_filter_C36.initialize(filter::fFilter, hashfunctions::fCrate,
		vhashC36);

	//	Containers XXX
	_xPedSum.initialize(hashfunctions::fDChannel);
	_xPedSum2.initialize(hashfunctions::fDChannel);
	_xPedRefMean.initialize(hashfunctions::fDChannel);
	_xPedEntries.initialize(hashfunctions::fDChannel);
	_xPedRefRMS.initialize(hashfunctions::fDChannel);

	//	Containers
	_cMean_Subdet.initialize(_name, "Mean",hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fADC_15),
		new quantity::ValueQuantity(quantity::fN, true));
	_cRMS_Subdet.initialize(_name, "RMS", hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fADC_5),
		new quantity::ValueQuantity(quantity::fN, true));
	_cMean_depth.initialize(_name, "Mean", hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cRMS_depth.initialize(_name, "RMS", hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cMean_FEDVME.initialize(_name, "Mean", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cMean_FEDuTCA.initialize(_name, "Mean", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cRMS_FEDVME.initialize(_name, "RMS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMS_FEDuTCA.initialize(_name, "RMS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));

	_cMeanDBRef_Subdet.initialize(_name, "MeanDBRef", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fAroundZero),
		new quantity::ValueQuantity(quantity::fN, true));
	_cRMSDBRef_Subdet.initialize(_name, "RMSDBRef", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fAroundZero),
		new quantity::ValueQuantity(quantity::fN, true));
	_cMeanDBRef_depth.initialize(_name, "MeanDBRef", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fAroundZero));
	_cRMSDBRef_depth.initialize(_name, "RMSDBRef", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fAroundZero));
	_cMeanDBRef_FEDVME.initialize(_name, "MeanDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cMeanDBRef_FEDuTCA.initialize(_name, "MeanDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSDBRef_FEDVME.initialize(_name, "RMSDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSDBRef_FEDuTCA.initialize(_name, "RMSDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));

	_cMissing_depth.initialize(_name, "Missing", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBad_depth.initialize(_name, "MeanBad", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBad_depth.initialize(_name, "RMSBad", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing_FEDVME.initialize(_name, "Missing", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing_FEDuTCA.initialize(_name, "Missing", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBad_FEDVME.initialize(_name, "MeanBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBad_FEDuTCA.initialize(_name, "MeanBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBad_FEDVME.initialize(_name, "RMSBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBad_FEDuTCA.initialize(_name, "RMSBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));

	//	book plots
	_cMean_Subdet.book(ib, _emap, _subsystem);
	_cRMS_Subdet.book(ib, _emap, _subsystem);
	_cMean_depth.book(ib, _emap, _subsystem);
	_cRMS_depth.book(ib, _emap, _subsystem);
	_cMeanDBRef_Subdet.book(ib, _emap, _subsystem);
	_cRMSDBRef_Subdet.book(ib, _emap, _subsystem);
	_cMeanDBRef_depth.book(ib, _emap, _subsystem);
	_cRMSDBRef_depth.book(ib, _emap, _subsystem);
	_cMissing_depth.book(ib, _emap, _subsystem);
	_cMeanBad_depth.book(ib, _emap, _subsystem);
	_cRMSBad_depth.book(ib, _emap, _subsystem);

	_cMean_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMeanDBRef_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanDBRef_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSDBRef_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSDBRef_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMissing_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMissing_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSBad_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSBad_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMeanBad_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanBad_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	
	//	book compact containers
	_xPedSum.book(_emap);
	_xPedSum2.book(_emap);
	_xPedEntries.book(_emap);
	_xPedRefMean.book(_emap);
	_xPedRefRMS.book(_emap);

	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);

	//	load conditions pedestals
	std::vector<HcalGenericDetId> dids = _emap->allPrecisionId();
	for (std::vector<HcalGenericDetId>::const_iterator it=dids.begin();
		it!=dids.end(); ++it)
	{
		//	skip if calib or whatever
		if (!it->isHcalDetId())
			continue;
		//	skip Crate 36
		if (_filter_C36.filter(HcalElectronicsId(_ehashmap.lookup(*it))))
			continue;
		HcalDetId did = HcalDetId(it->rawId());

		HcalPedestal const* peds = dbs->getPedestal(did);
		float const *means = peds->getValues();
		float const *rmss = peds->getWidths();
		double msum=0; double rsum=0;
		for (uint32_t i=0; i<4; i++)
		{msum+=means[i]; rsum+=rmss[i];}
		msum/=4; rsum/=4;
		_xPedRefMean.set(did, msum);
		_xPedRefRMS.set(did, rsum);
	}
}

/* virtual */ void PedestalTask::_resetMonitors(UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);
}

/* virtual */ void PedestalTask::endRun(edm::Run const& r, 
	edm::EventSetup const&)
{	
	if (_ptype==fLocal)
		if (r.runAuxiliary().run()==1)
			return;
	this->_dump();
}

/* virtual */ void PedestalTask::_dump()
{
	//	reset what's needed
	_cMean_Subdet.reset();
	_cRMS_Subdet.reset();
	_cMean_depth.reset();
	_cRMS_depth.reset();
	_cMean_FEDVME.reset();
	_cMean_FEDuTCA.reset();
	_cRMS_FEDVME.reset();
	_cRMS_FEDuTCA.reset();
	
	_cMeanDBRef_Subdet.reset();
	_cMeanDBRef_depth.reset();
	_cRMSDBRef_Subdet.reset();
	_cRMSDBRef_depth.reset();
	_cMissing_depth.reset();
	_cMeanBad_depth.reset();
	_cRMSBad_depth.reset();

	_cMeanDBRef_FEDVME.reset();
	_cMeanDBRef_FEDuTCA.reset();
	_cRMSDBRef_FEDVME.reset();
	_cRMSDBRef_FEDuTCA.reset();
	_cMissing_FEDVME.reset();
	_cMissing_FEDuTCA.reset();
	_cMeanBad_FEDVME.reset();
	_cMeanBad_FEDuTCA.reset();
	_cRMSBad_FEDVME.reset();
	_cRMSBad_FEDuTCA.reset();

	std::vector<HcalGenericDetId> dids = _emap->allPrecisionId();
	for (std::vector<HcalGenericDetId>::const_iterator it=dids.begin();
		it!=dids.end(); ++it)
	{
		if (!it->isHcalDetId())
			continue;
		HcalElectronicsId eid(_ehashmap.lookup(*it));
		if (_filter_C36.filter(eid))
			continue;
		HcalDetId did = HcalDetId(it->rawId());
		double sum = _xPedSum.get(did); double refm = _xPedRefMean.get(did);
		double sum2 = _xPedSum2.get(did); double refr = _xPedRefRMS.get(did);
		double n = _xPedEntries.get(did);

		//	if channel is missing
		if (n==0)
		{
			_cMissing_depth.fill(did);
			if (eid.isVMEid())
				_cMissing_FEDVME.fill(eid);
			else
				_cMissing_FEDuTCA.fill(eid);
			continue;
		}

		//	if not missing
		sum/=n; double rms = sqrt(sum2/n-sum*sum);
		double diffm = sum-refm;
		double diffr = rms - refr;
		_cMean_Subdet.fill(did, sum);
		_cMean_depth.fill(did, sum);
		_cRMS_Subdet.fill(did, rms);
		_cRMS_depth.fill(did, rms);
		_cMeanDBRef_Subdet.fill(did, diffm);
		_cMeanDBRef_depth.fill(did, diffm);
		_cRMSDBRef_Subdet.fill(did, diffr);
		_cRMSDBRef_depth.fill(did, diffr);
		if (eid.isVMEid())
		{
			_cMean_FEDVME.fill(eid, sum);
			_cRMS_FEDVME.fill(eid, rms);
			_cMeanDBRef_FEDVME.fill(eid, diffm);
			_cRMSDBRef_FEDVME.fill(eid, diffr);
		}
		else
		{
			_cMean_FEDuTCA.fill(eid, sum);
			_cRMS_FEDuTCA.fill(eid, rms);
			_cMeanDBRef_FEDuTCA.fill(eid, diffm);
			_cRMSDBRef_FEDuTCA.fill(eid, diffr);
		}

		//	if bad quality...
		if (fabs(diffm)>0.2)
		{
			_cMeanBad_depth.fill(did);
			if (eid.isVMEid())
				_cMeanBad_FEDVME.fill(eid);
			else
				_cMeanBad_FEDuTCA.fill(eid);
		}
		if (fabs(diffr)>0.2)
		{
			_cRMSBad_depth.fill(did);
			if (eid.isVMEid())
				_cRMSBad_FEDVME.fill(eid);
			else 
				_cRMSBad_FEDuTCA.fill(eid);
		}
	}
}

/* virtual */ void PedestalTask::_process(edm::Event const& e,
	edm::EventSetup const& es)
{
	edm::Handle<HBHEDigiCollection>		chbhe;
	edm::Handle<HODigiCollection>		cho;
	edm::Handle<HFDigiCollection>		chf;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHEDigiCollection isn't available"
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiCollection isn't available"
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection HFDigiCollection isn't available"
			+ _tagHF.label() + " " + _tagHF.instance());

	for (HBHEDigiCollection::const_iterator it=chbhe->begin();
		it!=chbhe->end(); ++it)
	{
		const HBHEDataFrame digi = (const HBHEDataFrame)(*it);
		HcalDetId did = digi.id();
		int digiSizeToUse = floor(digi.size()/constants::CAPS_NUM)*
			constants::CAPS_NUM-1;
		for (int i=0; i<digiSizeToUse; i++)
		{
			_xPedSum.get(did)+=it->sample(i).adc();
			_xPedSum2.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntries.get(did)++;
		}
	}
	for (HODigiCollection::const_iterator it=cho->begin();
		it!=cho->end(); ++it)
	{
		const HODataFrame digi = (const HODataFrame)(*it);
		HcalDetId did = digi.id();
		int digiSizeToUse = floor(digi.size()/constants::CAPS_NUM)*
			constants::CAPS_NUM-1;
		for (int i=0; i<digiSizeToUse; i++)
		{
			_xPedSum.get(did)+=it->sample(i).adc();
			_xPedSum2.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntries.get(did)++;
		}
	}
	for (HFDigiCollection::const_iterator it=chf->begin();
		it!=chf->end(); ++it)
	{
		const HFDataFrame digi = (const HFDataFrame)(*it);
		HcalDetId did = digi.id();
		int digiSizeToUse = floor(digi.size()/constants::CAPS_NUM)*
			constants::CAPS_NUM-1;
		for (int i=0; i<digiSizeToUse; i++)
		{
			_xPedSum.get(did)+=it->sample(i).adc();
			_xPedSum2.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntries.get(did)++;
		}
	}

	if (_ptype==fOnline && _evsTotal>0 && 
		_evsTotal%constants::CALIBEVENTS_MIN==0)
		this->_dump();
}

/* virtual */ bool PedestalTask::_isApplicable(edm::Event const& e)
{
	if (_ptype==fOnline)
	{
		//	online-global
		return this->_getCalibType(e)==hc_Pedestal;
	}
	else 
	{
		//	local
		edm::Handle<HcalTBTriggerData>	ctrigger;
		if (!e.getByToken(_tokTrigger, ctrigger))
			_logger.dqmthrow("Collection HcalTBTriggerData isn't available"
				+ _tagTrigger.label() + " " + _tagTrigger.instance());
		return ctrigger->wasSpillIgnorantPedestalTrigger();
	}

	return false;
}

DEFINE_FWK_MODULE(PedestalTask);


