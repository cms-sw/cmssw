
#include "DQM/HcalTasks/interface/LaserTask.h"

using namespace hcaldqm;
LaserTask::LaserTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_nevents = ps.getUntrackedParameter<int>("nevents", 2000);

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

	//	constants
	_lowHBHE = ps.getUntrackedParameter<double>("lowHBHE",
		20);
	_lowHO = ps.getUntrackedParameter<double>("lowHO",
		20);
	_lowHF = ps.getUntrackedParameter<double>("lowHF",
		20);
}
	
/* virtual */ void LaserTask::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	if (_ptype==fLocal)
		if (r.runAuxiliary().run()==1)
			return;

	DQTask::bookHistograms(ib, r, es);
	
	edm::ESHandle<HcalDbService> dbService;
	es.get<HcalDbRecord>().get(dbService);
	_emap = dbService->getHcalMapping();

	std::vector<uint32_t> vhashVME;
	std::vector<uint32_t> vhashuTCA;
	std::vector<uint32_t> vhashC36;
	vhashVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN,
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
		vhashVME);
	_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
		vhashuTCA);

	//	INITIALIZE
	_cSignalMean_Subdet.initialize(_name, "SignalMean",
		hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::ffC_3000),
		new quantity::ValueQuantity(quantity::fN, true));
	_cSignalRMS_Subdet.initialize(_name, "SignalRMS",
		hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::ffC_1000),
		new quantity::ValueQuantity(quantity::fN, true));
	_cTimingMean_Subdet.initialize(_name, "TimingMean",
		hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fTiming_TS200),
		new quantity::ValueQuantity(quantity::fN, true));
	_cTimingRMS_Subdet.initialize(_name, "TimingRMS",
		hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fTiming_TS200), 
		new quantity::ValueQuantity(quantity::fN, true));

	_cSignalMean_FEDVME.initialize(_name, "SignalMean",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cSignalMean_FEDuTCA.initialize(_name, "SignalMean",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cSignalRMS_FEDVME.initialize(_name, "SignalRMS",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cSignalRMS_FEDuTCA.initialize(_name, "SignalRMS",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cTimingMean_FEDVME.initialize(_name, "TimingMean",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingMean_FEDuTCA.initialize(_name, "TimingMean",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingRMS_FEDVME.initialize(_name, "TimingRMS",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingRMS_FEDuTCA.initialize(_name, "TimingRMS",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));

	_cShapeCut_FEDSlot.initialize(_name, "Shape", 
		hashfunctions::fFEDSlot,
		new quantity::ValueQuantity(quantity::fTiming_TS),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cTimingvsEvent_SubdetPM.initialize(_name, "TimingvsEvent",
		hashfunctions::fSubdetPM,
		new quantity::EventNumber(_nevents),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cSignalvsEvent_SubdetPM.initialize(_name, "SignalvsEvent",
		hashfunctions::fSubdetPM,
		new quantity::EventNumber(_nevents),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cTimingvsLS_SubdetPM.initialize(_name, "TimingvsLS",
		hashfunctions::fSubdetPM,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cSignalvsLS_SubdetPM.initialize(_name, "SignalvsLS",
		hashfunctions::fSubdetPM,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::ffC_3000));

	_cSignalMean_depth.initialize(_name, "SignalMean",
		hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::ffC_3000));
	_cSignalRMS_depth.initialize(_name, "SignalRMS",
		hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::ffC_1000));
	_cTimingMean_depth.initialize(_name, "TimingMean",
		hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingRMS_depth.initialize(_name, "TimingRMS",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fTiming_TS200));

	_cMissing_depth.initialize(_name, "Missing",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing_FEDVME.initialize(_name, "Missing",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing_FEDuTCA.initialize(_name, "Missing",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));

	
	//	initialize compact containers
	_xSignalSum.initialize(hashfunctions::fDChannel);
	_xSignalSum2.initialize(hashfunctions::fDChannel);
	_xTimingSum.initialize(hashfunctions::fDChannel);
	_xTimingSum2.initialize(hashfunctions::fDChannel);
	_xEntries.initialize(hashfunctions::fDChannel);

	//	BOOK
	_cSignalMean_Subdet.book(ib, _emap, _subsystem);
	_cSignalRMS_Subdet.book(ib, _emap, _subsystem);
	_cTimingMean_Subdet.book(ib, _emap, _subsystem);
	_cTimingRMS_Subdet.book(ib, _emap, _subsystem);

	_cSignalMean_depth.book(ib, _emap, _subsystem);
	_cSignalRMS_depth.book(ib, _emap, _subsystem);
	_cTimingMean_depth.book(ib, _emap, _subsystem);
	_cTimingRMS_depth.book(ib, _emap, _subsystem);

	if (_ptype==fLocal)
	{
		_cTimingvsEvent_SubdetPM.book(ib, _emap, _subsystem);
		_cSignalvsEvent_SubdetPM.book(ib, _emap, _subsystem);
	}
	else
	{	
		_cTimingvsLS_SubdetPM.book(ib, _emap, _subsystem);
		_cSignalvsLS_SubdetPM.book(ib, _emap, _subsystem);
	}

	_cSignalMean_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cSignalMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cSignalRMS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cSignalRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingMean_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingRMS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);

	_cShapeCut_FEDSlot.book(ib, _emap, _subsystem);
	_cMissing_depth.book(ib, _emap,_subsystem);
	_cMissing_FEDVME.book(ib, _emap, _filter_uTCA,_subsystem);
	_cMissing_FEDuTCA.book(ib, _emap, _filter_VME,_subsystem);

	_xSignalSum.book(_emap);
	_xSignalSum2.book(_emap);
	_xEntries.book(_emap);
	_xTimingSum.book(_emap);
	_xTimingSum2.book(_emap);

	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
}

/* virtual */ void LaserTask::_resetMonitors(UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);
}

/* virtual */ void LaserTask::_dump()
{
	_cSignalMean_Subdet.reset();
	_cSignalRMS_Subdet.reset();
	_cTimingMean_Subdet.reset();
	_cTimingRMS_Subdet.reset();
	_cSignalMean_depth.reset();
	_cSignalRMS_depth.reset();
	_cTimingMean_depth.reset();
	_cTimingRMS_depth.reset();

	_cSignalMean_FEDVME.reset();
	_cSignalMean_FEDuTCA.reset();
	_cSignalRMS_FEDVME.reset();
	_cSignalRMS_FEDuTCA.reset();
	_cTimingMean_FEDVME.reset();
	_cTimingMean_FEDuTCA.reset();
	_cTimingRMS_FEDVME.reset();
	_cTimingRMS_FEDuTCA.reset();

	std::vector<HcalGenericDetId> dids = _emap->allPrecisionId();
	for (std::vector<HcalGenericDetId>::const_iterator it=dids.begin();
		it!=dids.end(); ++it)
	{
		if (!it->isHcalDetId())
			continue;
		HcalDetId did = HcalDetId(it->rawId());
		HcalElectronicsId eid(_ehashmap.lookup(*it));
		int n = _xEntries.get(did);
		double msig = _xSignalSum.get(did)/n; 
		double mtim = _xTimingSum.get(did)/n;
		double rsig = sqrt(_xSignalSum2.get(did)/n-msig*msig);
		double rtim = sqrt(_xTimingSum2.get(did)/n-mtim*mtim);

		//	channels missing or low signal
		if (n==0)
		{
			_cMissing_depth.fill(did);
			if (eid.isVMEid())
				_cMissing_FEDVME.fill(eid);
			else
				_cMissing_FEDuTCA.fill(eid);
			continue;
		}
		_cSignalMean_Subdet.fill(did, msig);
		_cSignalMean_depth.fill(did, msig);
		_cSignalRMS_Subdet.fill(did, rsig);
		_cSignalRMS_depth.fill(did, rsig);
		_cTimingMean_Subdet.fill(did, mtim);
		_cTimingMean_depth.fill(did, mtim);
		_cTimingRMS_Subdet.fill(did, rtim);
		_cTimingRMS_depth.fill(did, rtim);
		if (eid.isVMEid())
		{
			_cSignalMean_FEDVME.fill(eid, msig);
			_cSignalRMS_FEDVME.fill(eid, rsig);
			_cTimingMean_FEDVME.fill(eid, mtim);
			_cTimingRMS_FEDVME.fill(eid, rtim);
		}
		else
		{
			_cSignalMean_FEDuTCA.fill(eid, msig);
			_cSignalRMS_FEDuTCA.fill(eid, rsig);
			_cTimingMean_FEDuTCA.fill(eid, mtim);
			_cTimingRMS_FEDuTCA.fill(eid, rtim);
		}
	}
}

/* virtual */ void LaserTask::_process(edm::Event const& e,
	edm::EventSetup const& es)
{
	edm::Handle<HBHEDigiCollection>		chbhe;
	edm::Handle<HODigiCollection>		cho;
	edm::Handle<HFDigiCollection>		chf;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHEDigiCollection isn't available "
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiCollection isn't available "
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection HFDigiCollection isn't available "
			+ _tagHF.label() + " " + _tagHF.instance());

//	int currentEvent = e.eventAuxiliary().id().event();

	for (HBHEDigiCollection::const_iterator it=chbhe->begin();
		it!=chbhe->end(); ++it)
	{
		const HBHEDataFrame digi = (const HBHEDataFrame)(*it);
		double sumQ = utilities::sumQ<HBHEDataFrame>(digi, 2.5, 0, 
			digi.size()-1);
		if (sumQ<_lowHBHE)
			continue;
		HcalDetId did = digi.id();
		HcalElectronicsId eid = digi.elecId();

		double aveTS = utilities::aveTS<HBHEDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		for (int i=0; i<digi.size(); i++)
			_cShapeCut_FEDSlot.fill(eid, i, 
				digi.sample(i).nominal_fC()-2.5);

		//	select based on local global
		if (_ptype==fLocal)
		{
			int currentEvent = e.eventAuxiliary().id().event();
			_cTimingvsEvent_SubdetPM.fill(did, currentEvent, aveTS);
			_cSignalvsEvent_SubdetPM.fill(did, currentEvent, sumQ);
		}
		else
		{
			_cTimingvsLS_SubdetPM.fill(did, _currentLS, aveTS);
			_cSignalvsLS_SubdetPM.fill(did, _currentLS, sumQ);
		}
	}
	for (HODigiCollection::const_iterator it=cho->begin();
		it!=cho->end(); ++it)
	{
		const HODataFrame digi = (const HODataFrame)(*it);
		double sumQ = utilities::sumQ<HODataFrame>(digi, 8.5, 0, 
			digi.size()-1);
		if (sumQ<_lowHO)
			continue;
		HcalDetId did = digi.id();
		HcalElectronicsId eid = digi.elecId();

		double aveTS = utilities::aveTS<HODataFrame>(digi, 8.5, 0,
			digi.size()-1);
		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		for (int i=0; i<digi.size(); i++)
			_cShapeCut_FEDSlot.fill(eid, i, 
				digi.sample(i).nominal_fC()-8.5);

		//	select based on local global
		if (_ptype==fLocal)
		{
			int currentEvent = e.eventAuxiliary().id().event();
			_cTimingvsEvent_SubdetPM.fill(did, currentEvent, aveTS);
			_cSignalvsEvent_SubdetPM.fill(did, currentEvent, sumQ);
		}
		else
		{
			_cTimingvsLS_SubdetPM.fill(did, _currentLS, aveTS);
			_cSignalvsLS_SubdetPM.fill(did, _currentLS, sumQ);
		}
	}
	for (HFDigiCollection::const_iterator it=chf->begin();
		it!=chf->end(); ++it)
	{
		const HFDataFrame digi = (const HFDataFrame)(*it);
		double sumQ = utilities::sumQ<HFDataFrame>(digi, 2.5, 0, 
			digi.size()-1);
		if (sumQ<_lowHF)
			continue;
		HcalDetId did = digi.id();
		HcalElectronicsId eid = digi.elecId();

		double aveTS = utilities::aveTS<HFDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		for (int i=0; i<digi.size(); i++)
			_cShapeCut_FEDSlot.fill(eid, i, 
				digi.sample(i).nominal_fC()-2.5);

		//	select based on local global
		if (_ptype==fLocal)
		{
			int currentEvent = e.eventAuxiliary().id().event();
			_cTimingvsEvent_SubdetPM.fill(did, currentEvent, aveTS);
			_cSignalvsEvent_SubdetPM.fill(did, currentEvent, sumQ);
		}
		else
		{
			_cTimingvsLS_SubdetPM.fill(did, _currentLS, aveTS);
			_cSignalvsLS_SubdetPM.fill(did, _currentLS, sumQ);
		}
	}

	if (_ptype==fOnline && _evsTotal>0 &&
		_evsTotal%constants::CALIBEVENTS_MIN==0)
		this->_dump();
}

/* virtual */ bool LaserTask::_isApplicable(edm::Event const& e)
{
	if (_ptype!=fOnline)
	{
		//	local
		edm::Handle<HcalTBTriggerData> ctrigger;
		if (!e.getByToken(_tokTrigger, ctrigger))
			_logger.dqmthrow("Collection HcalTBTriggerData isn't available "
				+ _tagTrigger.label() + " " + _tagTrigger.instance());
		return ctrigger->wasLaserTrigger();
	}

	return false;
}

DEFINE_FWK_MODULE(LaserTask);


