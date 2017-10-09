
#include "DQM/HcalTasks/interface/LEDTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;
LEDTask::LEDTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	//	tags
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHEP17 = ps.getUntrackedParameter<edm::InputTag>("tagHEP17",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));
	_tagTrigger = ps.getUntrackedParameter<edm::InputTag>("tagTrigger",
		edm::InputTag("tbunpacker"));
	_tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
	_tokHEP17 = consumes<QIE11DigiCollection>(_tagHEP17);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<QIE10DigiCollection>(_tagHF);
	_tokTrigger = consumes<HcalTBTriggerData>(_tagTrigger);

	//	constants
	_lowHBHE = ps.getUntrackedParameter<double>("lowHBHE",
		20);
	_lowHEP17 = ps.getUntrackedParameter<double>("lowHEP17",
		20);
	_lowHO = ps.getUntrackedParameter<double>("lowHO",
		20);
	_lowHF = ps.getUntrackedParameter<double>("lowHF",
		20);
}
	
/* virtual */ void LEDTask::bookHistograms(DQMStore::IBooker &ib,
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
	_filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics,
		vhashVME);
	_filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics,
		vhashuTCA);

	//	INITIALIZE
	_cSignalMean_Subdet.initialize(_name, "SignalMean",
		hcaldqm::hashfunctions::fSubdet, 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_3000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cSignalRMS_Subdet.initialize(_name, "SignalRMS",
		hcaldqm::hashfunctions::fSubdet, 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_1000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cTimingMean_Subdet.initialize(_name, "TimingMean",
		hcaldqm::hashfunctions::fSubdet, 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cTimingRMS_Subdet.initialize(_name, "TimingRMS",
		hcaldqm::hashfunctions::fSubdet, 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200), 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);

	if (_ptype != fOffline) { // hidefed2crate
		_cSignalMean_FEDVME.initialize(_name, "SignalMean",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_400000, true),0);
		_cSignalMean_FEDuTCA.initialize(_name, "SignalMean",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_400000, true),0);
		_cSignalRMS_FEDVME.initialize(_name, "SignalRMS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_10000, true),0);
		_cSignalRMS_FEDuTCA.initialize(_name, "SignalRMS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_10000, true),0);
		_cTimingMean_FEDVME.initialize(_name, "TimingMean",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingMean_FEDuTCA.initialize(_name, "TimingMean",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingRMS_FEDVME.initialize(_name, "TimingRMS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingRMS_FEDuTCA.initialize(_name, "TimingRMS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);

		_cShapeCut_FEDSlot.initialize(_name, "Shape", 
			hcaldqm::hashfunctions::fFEDSlot,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_400000),0);
	}

	_cSignalMean_depth.initialize(_name, "SignalMean",
		hcaldqm::hashfunctions::fdepth, 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta), 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_400000, true),0);
	_cSignalRMS_depth.initialize(_name, "SignalRMS",
		hcaldqm::hashfunctions::fdepth, 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta), 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_10000, true),0);
	_cTimingMean_depth.initialize(_name, "TimingMean",
		hcaldqm::hashfunctions::fdepth, 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta), 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
	_cTimingRMS_depth.initialize(_name, "TimingRMS",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta), 
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);

	_cMissing_depth.initialize(_name, "Missing",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	if (_ptype != fOffline) { // hidefed2crate
		_cMissing_FEDVME.initialize(_name, "Missing",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cMissing_FEDuTCA.initialize(_name, "Missing",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	}
	
	//	initialize compact containers
	_xSignalSum.initialize(hcaldqm::hashfunctions::fDChannel);
	_xSignalSum2.initialize(hcaldqm::hashfunctions::fDChannel);
	_xTimingSum.initialize(hcaldqm::hashfunctions::fDChannel);
	_xTimingSum2.initialize(hcaldqm::hashfunctions::fDChannel);
	_xEntries.initialize(hcaldqm::hashfunctions::fDChannel);

	//	BOOK
	_cSignalMean_Subdet.book(ib, _emap, _subsystem);
	_cSignalRMS_Subdet.book(ib, _emap, _subsystem);
	_cTimingMean_Subdet.book(ib, _emap, _subsystem);
	_cTimingRMS_Subdet.book(ib, _emap, _subsystem);

	_cSignalMean_depth.book(ib, _emap, _subsystem);
	_cSignalRMS_depth.book(ib, _emap, _subsystem);
	_cTimingMean_depth.book(ib, _emap, _subsystem);
	_cTimingRMS_depth.book(ib, _emap, _subsystem);

	_cMissing_depth.book(ib, _emap, _subsystem);
	if (_ptype != fOffline) { // hidefed2crate
		_cSignalMean_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cSignalMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cSignalRMS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cSignalRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cTimingMean_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cTimingMean_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cTimingRMS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cTimingRMS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cShapeCut_FEDSlot.book(ib, _emap, _subsystem);
		_cMissing_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cMissing_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	}

	_xSignalSum.book(_emap);
	_xSignalSum2.book(_emap);
	_xEntries.book(_emap);
	_xTimingSum.book(_emap);
	_xTimingSum2.book(_emap);

	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
}

/* virtual */ void LEDTask::_resetMonitors(hcaldqm::UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);
}

/* virtual */ void LEDTask::_dump()
{
	_cSignalMean_Subdet.reset();
	_cSignalRMS_Subdet.reset();
	_cTimingMean_Subdet.reset();
	_cTimingRMS_Subdet.reset();
	_cSignalMean_depth.reset();
	_cSignalRMS_depth.reset();
	_cTimingMean_depth.reset();
	_cTimingRMS_depth.reset();

	if (_ptype != fOffline) { // hidefed2crate
		_cSignalMean_FEDVME.reset();
		_cSignalMean_FEDuTCA.reset();
		_cSignalRMS_FEDVME.reset();
		_cSignalRMS_FEDuTCA.reset();
		_cTimingMean_FEDVME.reset();
		_cTimingMean_FEDuTCA.reset();
		_cTimingRMS_FEDVME.reset();
		_cTimingRMS_FEDuTCA.reset();
	}

	std::vector<HcalGenericDetId> dids = _emap->allPrecisionId();
	for (std::vector<HcalGenericDetId>::const_iterator it=dids.begin();
		it!=dids.end(); ++it)
	{
		if (!it->isHcalDetId())
			continue;
		HcalDetId did = HcalDetId(it->rawId());
		HcalElectronicsId eid(_ehashmap.lookup(did));
		int n = _xEntries.get(did);
		double msig = _xSignalSum.get(did)/n; 
		double mtim = _xTimingSum.get(did)/n;
		double rsig = sqrt(_xSignalSum2.get(did)/n-msig*msig);
		double rtim = sqrt(_xTimingSum2.get(did)/n-mtim*mtim);

		//	channels missing or low signal
		if (n==0)
		{
			_cMissing_depth.fill(did);
			if (_ptype != fOffline) { // hidefed2crate
				if (eid.isVMEid())
					_cMissing_FEDVME.fill(eid);
				else
					_cMissing_FEDuTCA.fill(eid);
			}
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
		if (_ptype != fOffline) { // hidefed2crate
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
}

/* virtual */ void LEDTask::_process(edm::Event const& e,
	edm::EventSetup const& es)
{
	edm::Handle<HBHEDigiCollection>		chbhe;
	edm::Handle<HODigiCollection>		cho;
	edm::Handle<QIE10DigiCollection>		chf;
	edm::Handle<QIE11DigiCollection>		chep17;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHEDigiCollection isn't available "
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiCollection isn't available "
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection QIE10DigiCollection isn't available "
			+ _tagHF.label() + " " + _tagHF.instance());
	if (!e.getByToken(_tokHEP17, chep17))
		_logger.dqmthrow("Collection QIE11DigiCollection isn't available "
			+ _tagHEP17.label() + " " + _tagHEP17.instance());

//	int currentEvent = e.eventAuxiliary().id().event();

	for (HBHEDigiCollection::const_iterator it=chbhe->begin();
		it!=chbhe->end(); ++it)
	{
		const HBHEDataFrame digi = (const HBHEDataFrame)(*it);
		HcalDetId did = digi.id();
		HcalElectronicsId eid = digi.elecId();

		// Get total charge and apply charge cut
		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<HBHEDataFrame>(_dbService, did, digi);
		//double sumQ = hcaldqm::utilities::sumQ<HBHEDataFrame>(digi, 2.5, 0, digi.size()-1);
		double sumQ = hcaldqm::utilities::sumQDB<HBHEDataFrame>(_dbService, digi_fC, did, digi, 0, digi.size()-1);
		if (sumQ<_lowHBHE)
			continue;

		//double aveTS = hcaldqm::utilities::aveTS<HBHEDataFrame>(digi, 2.5, 0,digi.size()-1);
		double aveTS = hcaldqm::utilities::aveTSDB<HBHEDataFrame>(_dbService, digi_fC, did, digi, 0, digi.size()-1);

		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		if (_ptype != fOffline) { // hidefed2crate
			for (int i=0; i<digi.size(); i++) {
				//_cShapeCut_FEDSlot.fill(eid, i, digi.sample(i).nominal_fC()-2.5);
				_cShapeCut_FEDSlot.fill(eid, i, hcaldqm::utilities::adc2fCDBMinusPedestal<HBHEDataFrame>(_dbService, digi_fC, did, digi, i));
			}
		}
	}

	for (QIE11DigiCollection::const_iterator it=chep17->begin(); it!=chep17->end();
		++it)
	{
		const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*it);
		HcalDetId const& did = digi.detid();
		if (did.subdet() != HcalEndcap) {
			continue;
		}
		uint32_t rawid = _ehashmap.lookup(did);
		if (!rawid) {
		  std::string unknown_id_string="Detid "+std::to_string(int(did))+", ieta "+std::to_string(did.ieta());
		  unknown_id_string+=", iphi "+std::to_string(did.iphi())+", depth "+std::to_string(did.depth());
		  unknown_id_string+=", is not in emap. Skipping.";
		  _logger.warn(unknown_id_string.c_str());
		  continue;
		}
		HcalElectronicsId const& eid(rawid);

		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE11DataFrame>(_dbService, did, digi);
		//double sumQ = hcaldqm::utilities::sumQ_v10<QIE11DataFrame>(digi, 2.5, 0, digi.samples()-1);
		double sumQ = hcaldqm::utilities::sumQDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples()-1);
		if (sumQ<_lowHEP17)
			continue;

		//double aveTS = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(digi, 2.5, 0,digi.samples()-1);
		double aveTS = hcaldqm::utilities::aveTSDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.size()-1);

		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		if (_ptype != fOffline) { // hidefed2crate
			for (int i=0; i<digi.samples(); i++) {
				//_cShapeCut_FEDSlot.fill(eid, i, digi.sample(i).nominal_fC()-2.5);
				_cShapeCut_FEDSlot.fill(eid, i, hcaldqm::utilities::adc2fCDBMinusPedestal<QIE11DataFrame>(_dbService, digi_fC, did, digi, i));
			}
		}
	}
	for (HODigiCollection::const_iterator it=cho->begin();
		it!=cho->end(); ++it)
	{
		const HODataFrame digi = (const HODataFrame)(*it);
		HcalDetId did = digi.id();
		HcalElectronicsId eid = digi.elecId();
		//double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(digi, 8.5, 0, digi.size()-1);
		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<HODataFrame>(_dbService, did, digi);
		double sumQ = hcaldqm::utilities::sumQDB<HODataFrame>(_dbService, digi_fC, did, digi, 0, digi.size()-1);
		if (sumQ<_lowHO)
			continue;

		//double aveTS = hcaldqm::utilities::aveTS<HODataFrame>(digi, 8.5, 0, digi.size()-1);
		double aveTS = hcaldqm::utilities::aveTSDB<HODataFrame>(_dbService, digi_fC, did, digi, 0, digi.size()-1);

		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		if (_ptype != fOffline) { // hidefed2crate
			for (int i=0; i<digi.size(); i++) {
				//_cShapeCut_FEDSlot.fill(eid, i, digi.sample(i).nominal_fC()-8.5);
				_cShapeCut_FEDSlot.fill(eid, i, hcaldqm::utilities::adc2fCDBMinusPedestal<HODataFrame>(_dbService, digi_fC, did, digi, i));
			}
		}
	}
	for (QIE10DigiCollection::const_iterator it=chf->begin();
		it!=chf->end(); ++it)
	{
		const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);
		HcalDetId did = digi.detid();
		if (did.subdet() != HcalForward) {
			continue;
		}
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
		//double sumQ = hcaldqm::utilities::sumQ_v10<QIE10DataFrame>(digi, 2.5, 0, digi.samples()-1);
		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(_dbService, did, digi);
		double sumQ = hcaldqm::utilities::sumQDB<QIE10DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples()-1);
		if (sumQ<_lowHF)
			continue;

		//double aveTS = hcaldqm::utilities::aveTS_v10<QIE10DataFrame>(digi, 2.5, 0, digi.samples()-1);
		double aveTS = hcaldqm::utilities::aveTSDB<QIE10DataFrame>(_dbService, digi_fC, did, digi, 0, digi.size()-1);

		_xSignalSum.get(did)+=sumQ;
		_xSignalSum2.get(did)+=sumQ*sumQ;
		_xTimingSum.get(did)+=aveTS;
		_xTimingSum2.get(did)+=aveTS*aveTS;
		_xEntries.get(did)++;

		if (_ptype != fOffline) { // hidefed2crate
			for (int i = 0; i < digi.samples(); ++i) {
				// Note: this used to be digi.sample(i).nominal_fC() - 2.5, but this branch doesn't exist in QIE10DataFrame.
				// Instead, use lookup table.
				//_cShapeCut_FEDSlot.fill(eid, i, constants::adc2fC[digi[i].adc()]);
				_cShapeCut_FEDSlot.fill(eid, i, hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, digi, i));
			}
		}
	}

	if (_ptype==fOnline && _evsTotal>0 &&
		_evsTotal%constants::CALIBEVENTS_MIN==0)
		this->_dump();
}

/* virtual */ bool LEDTask::_isApplicable(edm::Event const& e)
{
	if (_ptype!=fOnline)
	{
		//	local
		edm::Handle<HcalTBTriggerData> ctrigger;
		if (!e.getByToken(_tokTrigger, ctrigger))
			_logger.dqmthrow("Collection HcalTBTriggerData isn't available "
				+ _tagTrigger.label() + " " + _tagTrigger.instance());
		return ctrigger->wasLEDTrigger();
	}

	return false;
}

DEFINE_FWK_MODULE(LEDTask);


