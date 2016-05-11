#include "DQM/HcalTasks/interface/RawTask.h"

RawTask::RawTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_tagFEDs = ps.getUntrackedParameter<edm::InputTag>("tagFEDs",
		edm::InputTag("rawDataCollector"));
	_tagReport = ps.getUntrackedParameter<edm::InputTag>("tagReport",
		edm::InputTag("hcalDigis"));

	_tokFEDs = consumes<FEDRawDataCollection>(_tagFEDs);
	_tokReport = consumes<HcalUnpackerReport>(_tagReport);

	_vflags.resize(nRawFlag);
	_vflags[fEvnMsm]=flag::Flag("EvnMsm");
	_vflags[fBcnMsm]=flag::Flag("BcnMsm");
	_vflags[fBadQ]=flag::Flag("BadQ");
}

/* virtual */ void RawTask::bookHistograms(DQMStore::IBooker& ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	DQTask::bookHistograms(ib,r,es);

	//	GET WHAT YOU NEED
	edm::ESHandle<HcalDbService> dbs;
	es.get<HcalDbRecord>().get(dbs);
	_emap = dbs->getHcalMapping();
	std::vector<int> vFEDs = utilities::getFEDList(_emap);
	std::vector<int> vFEDsVME = utilities::getFEDVMEList(_emap);
	std::vector<int> vFEDsuTCA = utilities::getFEDuTCAList(_emap);
	std::vector<uint32_t> vVME;
	std::vector<uint32_t> vuTCA;
	std::vector<uint32_t> vhashFEDsVME;
	std::vector<uint32_t> vhashFEDsuTCA;
	vVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN, 
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
		vVME);
	_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
		vuTCA);

	for (std::vector<int>::const_iterator it=vFEDsVME.begin();
		it!=vFEDsVME.end(); ++it)
	{
		vhashFEDsVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN,
			constants::FIBER_VME_MIN, SPIGOT_MIN,
			(*it)-constants::FED_VME_MIN).rawId());
		_vhashFEDs.push_back(HcalElectronicsId(constants::FIBERCH_MIN,
			constants::FIBER_VME_MIN, SPIGOT_MIN,
			(*it)-constants::FED_VME_MIN).rawId());
	}
	for (std::vector<int>::const_iterator it=vFEDsuTCA.begin();
		it!=vFEDsuTCA.end(); ++it)
	{
		vhashFEDsuTCA.push_back(HcalElectronicsId(
		utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
		FIBERCH_MIN, false).rawId());
		_vhashFEDs.push_back(HcalElectronicsId(
		utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
		FIBERCH_MIN, false).rawId());
	}
	_filter_FEDsVME.initialize(filter::fPreserver, 
		hashfunctions::fFED, vhashFEDsVME);
	_filter_FEDsuTCA.initialize(filter::fPreserver,
		hashfunctions::fFED, vhashFEDsuTCA);

	//	INITIALIZE FIRST
	_cEvnMsm_ElectronicsVME.initialize(_name, "EvnMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cBcnMsm_ElectronicsVME.initialize(_name, "BcnMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cOrnMsm_ElectronicsVME.initialize(_name, "OrnMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cEvnMsm_ElectronicsuTCA.initialize(_name, "EvnMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cBcnMsm_ElectronicsuTCA.initialize(_name, "BcnMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cOrnMsm_ElectronicsuTCA.initialize(_name, "OrnMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));

	//	Bad Quality
	_cBadQuality_FEDVME.initialize(_name, "BadQuality",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cBadQuality_FEDuTCA.initialize(_name, "BadQuality",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cBadQualityvsLS.initialize(_name, "BadQualityvsLS",
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fN_m0to10000));
	_cBadQualityvsBX.initialize(_name, "BadQualityvsBX",
		new quantity::ValueQuantity(quantity::fBX),
		new quantity::ValueQuantity(quantity::fN_m0to10000));
	_cBadQuality_depth.initialize(_name, "BadQuality",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));

	//	INITIALIZE HISTOGRAMS to be used in Online Only
	if (_ptype==fOnline)
	{
		_xEvnMsmLS.initialize(hashfunctions::fFED);
		_xBcnMsmLS.initialize(hashfunctions::fFED);
		_xBadQLS.initialize(hashfunctions::fFED);
		_cSummaryvsLS_FED.initialize(_name, "SummaryvsLS",
			hashfunctions::fFED,
			new quantity::LumiSection(_maxLS),
			new quantity::FlagQuantity(_vflags),
			new quantity::ValueQuantity(quantity::fState));
		_cSummaryvsLS.initialize(_name, "SummaryvsLS",
			new quantity::LumiSection(_maxLS),
			new quantity::FEDQuantity(vFEDs),
			new quantity::ValueQuantity(quantity::fState));
	}

	//	BOOK HISTOGRAMS
	_cEvnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cBcnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOrnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cEvnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cBcnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOrnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);


	_cBadQuality_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cBadQuality_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cBadQuality_depth.book(ib, _emap, _subsystem);
	_cBadQualityvsLS.book(ib, _subsystem);
	_cBadQualityvsBX.book(ib, _subsystem);

	// BOOK HISTOGRAMS to be used in ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_xEvnMsmLS.book(_emap);
		_xBcnMsmLS.book(_emap);
		_xBadQLS.book(_emap);
		_cSummaryvsLS_FED.book(ib, _emap, _subsystem);
		_cSummaryvsLS.book(ib, _subsystem);
	}

	//	FOR OFFLINE PROCESSING MARK THESE HISTOGRAMS AS LUMI BASED
	if (_ptype==fOffline)
	{
		_cEvnMsm_ElectronicsVME.setLumiFlag();
		_cBcnMsm_ElectronicsVME.setLumiFlag();
		_cEvnMsm_ElectronicsuTCA.setLumiFlag();
		_cBcnMsm_ElectronicsuTCA.setLumiFlag();
		_cBadQuality_depth.setLumiFlag();
	}

	//	initialize hash map
	_ehashmap.initialize(_emap, hcaldqm::electronicsmap::fD2EHashMap);
}

/* virtual */ void RawTask::_resetMonitors(UpdateFreq uf)
{
	//	base reset
	DQTask::_resetMonitors(uf);
}

/* virtual */ void RawTask::_process(edm::Event const& e,
	edm::EventSetup const&)
{
	edm::Handle<FEDRawDataCollection> craw;
	edm::Handle<HcalUnpackerReport> creport;
	if (!e.getByToken(_tokFEDs, craw))
		_logger.dqmthrow("Collection FEDRawDataCollection isn't available"+
			_tagFEDs.label()+" " +_tagFEDs.instance());
	if (!e.getByToken(_tokReport, creport))
		_logger.dqmthrow("Collection HcalUnpackerReport isn't available"+
			_tagReport.label()+" " +_tagReport.instance());

	//	extract some info
	int bx = e.bunchCrossing();

	
	int nn = 0;
	//	loop thru and fill the detIds with bad quality
	//	NOTE: Calibration Channels are skipped!
	//	TODO: Include for Online Calibration Channels marked as bad
	//	a comment below is left on purpose!
	//_cBadQualityvsBX.fill(bx, creport->badQualityDigis());
	for (std::vector<DetId>::const_iterator it=creport->bad_quality_begin();
		it!=creport->bad_quality_end(); ++it)
	{
		if (!HcalGenericDetId(*it).isHcalDetId())
			continue;

		nn++;
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(*it));
		_cBadQuality_depth.fill(HcalDetId(*it));
		//	ONLINE ONLY!
		if (_ptype==fOnline)
			_xBadQLS.get(eid)++;
		if (eid.isVMEid())
		{
			if (_filter_FEDsVME.filter(eid))
				continue;
			_cBadQuality_FEDVME.fill(eid);
		}
		else
		{
			if (_filter_FEDsuTCA.filter(eid))
				continue;
			_cBadQuality_FEDuTCA.fill(eid);
		}
	}
	_cBadQualityvsLS.fill(_currentLS,nn);
	_cBadQualityvsBX.fill(bx, nn);

	for (int fed=FEDNumbering::MINHCALFEDID; 
		fed<=FEDNumbering::MAXHCALuTCAFEDID; fed++)
	{
		//	skip nonHCAL FEDs
		if ((fed>FEDNumbering::MAXHCALFEDID &&
			fed<FEDNumbering::MINHCALuTCAFEDID) || 
			fed>FEDNumbering::MAXHCALuTCAFEDID)
			continue;
		FEDRawData const& raw = craw->FEDData(fed);
		if (raw.size()<constants::RAW_EMPTY)
			continue;

		if (fed<=FEDNumbering::MAXHCALFEDID)	// VME
		{
			HcalDCCHeader const* hdcc = (HcalDCCHeader const*)(raw.data());
			if (!hdcc)
				continue;

			uint32_t bcn = hdcc->getBunchId();
			uint32_t orn = hdcc->getOrbitNumber();
			uint32_t evn = hdcc->getDCCEventNumber();
			int dccId = hdcc->getSourceId()-constants::FED_VME_MIN;

			//	iterate over spigots
			HcalHTRData htr;
			for (int is=0; is<HcalDCCHeader::SPIGOT_COUNT; is++)
			{
				int r = hdcc->getSpigotData(is, htr, raw.size());
				if (r!=0)
					continue;
				HcalElectronicsId eid = HcalElectronicsId(
					constants::FIBERCH_MIN, constants::FIBER_VME_MIN,
					is, dccId);
				if (_filter_FEDsVME.filter(eid))
					continue;

				uint32_t htr_evn = htr.getL1ANumber();
				uint32_t htr_orn = htr.getOrbitNumber();
				uint32_t htr_bcn = htr.getBunchNumber();
				bool qevn = (htr_evn!=evn);
				bool qbcn = (htr_bcn!=bcn);
				bool qorn = (htr_orn!=orn);
				if (qevn)
				{
					_cEvnMsm_ElectronicsVME.fill(eid);

					if (_ptype==fOnline && is<=constants::SPIGOT_MAX)
						_xEvnMsmLS.get(eid)++;
				}
				if (qorn)
					_cOrnMsm_ElectronicsVME.fill(eid);
				if (qbcn)
				{
					_cBcnMsm_ElectronicsVME.fill(eid);

					if (_ptype==fOnline && is<=constants::SPIGOT_MAX)
						_xBcnMsmLS.get(eid)++;
				}
			}
		}
		else	// uTCA
		{
			hcal::AMC13Header const* hamc13 = (hcal::AMC13Header const*)
				raw.data();
			if (!hamc13)
				continue;

			uint32_t bcn = hamc13->bunchId();
			uint32_t orn = hamc13->orbitNumber();
			uint32_t evn = hamc13->l1aNumber();
			int namc = hamc13->NAMC();

			for (int iamc=0; iamc<namc; iamc++)
			{
				int slot = hamc13->AMCSlot(iamc);
				int crate = hamc13->AMCId(iamc)&0xFF;
				HcalElectronicsId eid(crate, slot, FIBER_uTCA_MIN1,
					FIBERCH_MIN, false);
				if (_filter_FEDsuTCA.filter(eid))
					continue;
				HcalUHTRData uhtr(hamc13->AMCPayload(iamc),
					hamc13->AMCSize(iamc));

				uint32_t uhtr_evn = uhtr.l1ANumber();
				uint32_t uhtr_bcn = uhtr.bunchNumber();
				uint32_t uhtr_orn = uhtr.orbitNumber();
				bool qevn = (uhtr_evn!=evn);
				bool qbcn = (uhtr_bcn!=bcn);
				bool qorn = (uhtr_orn!=orn);
				if (qevn)
				{
					_cEvnMsm_ElectronicsuTCA.fill(eid);

					if (_ptype==fOnline)
						_xEvnMsmLS.get(eid)++;
				}
				if (qorn)
					_cOrnMsm_ElectronicsuTCA.fill(eid);
				if (qbcn)
				{
					_cBcnMsm_ElectronicsuTCA.fill(eid);

					if (_ptype==fOnline)
						_xBcnMsmLS.get(eid)++;
				}
			}
		}
	}
}

/* virtual */ void RawTask::beginLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	DQTask::beginLuminosityBlock(lb, es);

//	_cBadQualityvsLS.extendAxisRange(_currentLS);

	//	ONLINE ONLY!
	if (_ptype!=fOnline)
		return;
//	_cSummaryvsLS_FED.extendAxisRange(_currentLS);
//	_cSummaryvsLS.extendAxisRange(_currentLS);
	
}

/* virtual */ void RawTask::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	if (_ptype!=fOnline)
		return;

	//	
	//	GENERATE STATUS ONLY FOR ONLINE!
	//
	for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
		it!=_vhashFEDs.end(); ++it)
	{
		flag::Flag fSum("RAW");
		HcalElectronicsId eid = HcalElectronicsId(*it);

		std::vector<uint32_t>::const_iterator cit=std::find(
			_vcdaqEids.begin(), _vcdaqEids.end(), *it);
		if (cit==_vcdaqEids.end())
		{
			// not @cDAQ
			for (uint32_t iflag=0; iflag<_vflags.size(); iflag++)
				_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
					int(flag::fNCDAQ));
			_cSummaryvsLS.setBinContent(eid, _currentLS, int(flag::fNCDAQ));
			continue;
		}

		//	FED is @cDAQ
		if (utilities::isFEDHBHE(eid) || utilities::isFEDHF(eid) ||
			utilities::isFEDHO(eid))
		{
			if (_xEvnMsmLS.get(eid)>0)
				_vflags[fEvnMsm]._state = flag::fBAD;
			else
				_vflags[fEvnMsm]._state = flag::fGOOD;
			if (_xBcnMsmLS.get(eid)>0)
				_vflags[fBcnMsm]._state = flag::fBAD;
			else
				_vflags[fBcnMsm]._state = flag::fGOOD;
			if (_xBadQLS.get(eid)>0)
				_vflags[fBadQ]._state = flag::fBAD;
			else
				_vflags[fBadQ]._state = flag::fGOOD;
		}

		int iflag=0;
		//	iterate over all flags:
		//	- sum them all up in summary flag for this FED
		//	- reset each flag right after using it
		for (std::vector<flag::Flag>::iterator ft=_vflags.begin();
			ft!=_vflags.end(); ++ft)
		{
			_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
				ft->_state);
			fSum+=(*ft);
			iflag++;

			//	this is the MUST! We don't keep flags per FED, reset
			//	each one of them after using
			ft->reset();
		}
		_cSummaryvsLS.setBinContent(eid, _currentLS, fSum._state);
	}

	//	reset...
	_xEvnMsmLS.reset(); _xBcnMsmLS.reset(); _xBadQLS.reset();

	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(RawTask);

