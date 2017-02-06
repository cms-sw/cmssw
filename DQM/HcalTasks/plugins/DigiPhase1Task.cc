#include "DQM/HcalTasks/interface/DigiPhase1Task.h"

using namespace hcaldqm; 
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

DigiPhase1Task::DigiPhase1Task(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));

	_tokHBHE = consumes<QIE11DigiCollection>(_tagHBHE);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<QIE10DigiCollection>(_tagHF);

	_cutSumQ_HBHE = ps.getUntrackedParameter<double>("cutSumQ_HBHE", 20);
	_cutSumQ_HO = ps.getUntrackedParameter<double>("cutSumQ_HO", 20);
	_cutSumQ_HF = ps.getUntrackedParameter<double>("cutSumQ_HF", 20);
	_thresh_unihf = ps.getUntrackedParameter<double>("thresh_unihf", 0.2);
}

/* virtual */ void DigiPhase1Task::bookHistograms(DQMStore::IBooker& ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	DQTask::bookHistograms(ib,r,es);

	//	GET WHAT YOU NEED
	edm::ESHandle<HcalDbService> dbs;
	es.get<HcalDbRecord>().get(dbs);
	_emap = dbs->getHcalMapping();
	std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
	std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
	std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);
	std::vector<uint32_t> vVME;
	std::vector<uint32_t> vuTCA;
	vVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN, 
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics,
		vVME);
	_filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics,
		vuTCA);

	//	push the rawIds of each fed into the vector...
	for (std::vector<int>::const_iterator it=vFEDsVME.begin();
		it!=vFEDsVME.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(
			constants::FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN,
			(*it)-FED_VME_MIN).rawId());
	for (std::vector<int>::const_iterator it=vFEDsuTCA.begin();
		it!=vFEDsuTCA.end(); ++it)
    {
        std::pair<uint16_t, uint16_t> cspair = hcaldqm::utilities::fed2crate(*it);
		_vhashFEDs.push_back(HcalElectronicsId(
			cspair.first, cspair.second, FIBER_uTCA_MIN1,
			FIBERCH_MIN, false).rawId());
    }

	//	INITIALIZE FIRST
	_cADC_SubdetPM.initialize(_name, "ADC", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cfC_SubdetPM.initialize(_name, "fC", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cSumQ_SubdetPM.initialize(_name, "SumQ", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cSumQ_depth.initialize(_name, "SumQ", hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
	_cSumQvsLS_SubdetPM.initialize(_name, "SumQvsLS",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
	_cShapeCut_FED.initialize(_name, "Shape",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
	_cTimingCut_SubdetPM.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cTimingCut_FEDVME.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
	_cTimingCut_FEDuTCA.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
	_cTimingCut_ElectronicsVME.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fElectronics,
		new hcaldqm::quantity::FEDQuantity(vFEDsVME),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
	_cTimingCut_ElectronicsuTCA.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fElectronics,
		new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
	_cTimingCutvsLS_FED.initialize(_name, "TimingvsLS",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
	_cTimingCut_depth.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));

	//	Occupancy w/o a cut
	_cOccupancy_FEDVME.initialize(_name, "Occupancy",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancy_FEDuTCA.initialize(_name, "Occupancy",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancy_ElectronicsVME.initialize(_name, "Occupancy",
		hcaldqm::hashfunctions::fElectronics,
		new hcaldqm::quantity::FEDQuantity(vFEDsVME),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancy_ElectronicsuTCA.initialize(_name, "Occupancy",
		hcaldqm::hashfunctions::fElectronics,
		new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancyvsLS_Subdet.initialize(_name, "OccupancyvsLS",
		hcaldqm::hashfunctions::fSubdet,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to3000));
	_cOccupancy_depth.initialize(_name, "Occupancy",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));

	//	Occupancy w/ a cut
	_cOccupancyCut_FEDVME.initialize(_name, "OccupancyCut",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancyCut_FEDuTCA.initialize(_name, "OccupancyCut",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancyCut_ElectronicsVME.initialize(_name, "OccupancyCut",
		hcaldqm::hashfunctions::fElectronics,
		new hcaldqm::quantity::FEDQuantity(vFEDsVME),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancyCut_ElectronicsuTCA.initialize(_name, "OccupancyCut",
		hcaldqm::hashfunctions::fElectronics,
		new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
		new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
	_cOccupancyCutvsLS_Subdet.initialize(_name, "OccupancyCutvsLS",
		hcaldqm::hashfunctions::fSubdet,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to3000));
	_cOccupancyCut_depth.initialize(_name, "OccupancyCut",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));

	_cDigiSize_FED.initialize(_name, "DigiSize",
		hcaldqm::hashfunctions::fFED,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDigiSize),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));

	//	INITIALIZE HISTOGRAMS that are only for Online
	if (_ptype==fOnline)
	{
		_cSumQvsBX_SubdetPM.initialize(_name, "SumQvsBX",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
		_cDigiSizevsLS_FED.initialize(_name, "DigiSizevsLS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDigiSize));
		_cTimingCutvsiphi_SubdetPM.initialize(_name, "TimingCutvsiphi",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
		_cTimingCutvsieta_Subdet.initialize(_name, "TimingCutvsieta",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200));
		_cOccupancyvsiphi_SubdetPM.initialize(_name, "Occupancyvsiphi",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
		_cOccupancyvsieta_Subdet.initialize(_name, "Occupancyvsieta",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
		_cOccupancyCutvsiphi_SubdetPM.initialize(_name, "OccupancyCutvsiphi",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
		_cOccupancyCutvsieta_Subdet.initialize(_name, "OccupancyCutvsieta",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
		_cOccupancyCutvsLS_Subdet.initialize(_name, "OccupancyCutvsLS",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to3000));
		_cOccupancyCutvsBX_Subdet.initialize(_name, "OccupancyCutvsBX",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to3000));
//		_cOccupancyCutvsSlotvsLS_HFPM.initialize(_name, 
//			"OccupancyCutvsSlotvsLS", hcaldqm::hashfunctions::fSubdetPM,
//			new hcaldqm::quantity::LumiSection(_maxLS),
//			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
//			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));
		_cOccupancyCutvsiphivsLS_SubdetPM.initialize(_name, 
			"OccupancyCutvsiphivsLS", hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN));

		_xUni.initialize(hcaldqm::hashfunctions::fFED);
		_xDigiSize.initialize(hcaldqm::hashfunctions::fFED);
		_xNChs.initialize(hcaldqm::hashfunctions::fFED);
		_xNChsNominal.initialize(hcaldqm::hashfunctions::fFED);
	}

	//	BOOK HISTOGRAMS
	char cutstr[200];
	sprintf(cutstr, "_SumQHBHE%dHO%dHF%d", int(_cutSumQ_HBHE),
		int(_cutSumQ_HO), int(_cutSumQ_HF));
	char cutstr2[200];
	sprintf(cutstr2, "_SumQHF%d", int(_cutSumQ_HF));

	_cADC_SubdetPM.book(ib, _emap, _subsystem);

	_cfC_SubdetPM.book(ib, _emap, _subsystem);
	_cSumQ_SubdetPM.book(ib, _emap, _subsystem);
	_cSumQ_depth.book(ib, _emap, _subsystem);
	_cSumQvsLS_SubdetPM.book(ib, _emap, _subsystem);

	_cShapeCut_FED.book(ib, _emap, _subsystem);

	_cTimingCut_SubdetPM.book(ib, _emap, _subsystem);
	_cTimingCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingCutvsLS_FED.book(ib, _emap, _subsystem);
	_cTimingCut_depth.book(ib, _emap, _subsystem);

	_cOccupancy_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancy_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancy_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancy_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyvsLS_Subdet.book(ib, _emap, _subsystem);
	_cOccupancy_depth.book(ib, _emap, _subsystem);
	_cOccupancyCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCut_depth.book(ib, _emap, _subsystem);

	_cDigiSize_FED.book(ib, _emap, _subsystem);

	//	BOOK HISTOGRAMS that are only for Online
	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
	if (_ptype==fOnline)
	{
		_cSumQvsBX_SubdetPM.book(ib, _emap, _subsystem);
		_cDigiSizevsLS_FED.book(ib, _emap, _subsystem);
		_cTimingCutvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cTimingCutvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsLS_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsBX_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyCutvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsiphivsLS_SubdetPM.book(ib, _emap, _subsystem);

		_xNChs.book(_emap);
		_xNChsNominal.book(_emap);
		_xUni.book(_emap);
		_xDigiSize.book(_emap);

		// just PER HF FED RECORD THE #CHANNELS
		// ONLY WAY TO DO THAT AUTOMATICALLY AND W/O HARDCODING 1728
		// or ANY OTHER VALUES LIKE 2592, 2192
		std::vector<HcalGenericDetId> gids = _emap->allPrecisionId();
		for (std::vector<HcalGenericDetId>::const_iterator it=gids.begin();
			it!=gids.end(); ++it)
		{
			if (!it->isHcalDetId())
				continue;
			HcalDetId did(it->rawId());
			if (_xQuality.exists(did)) 
			{
				HcalChannelStatus cs(it->rawId(), _xQuality.get(
					HcalDetId(*it)));
				if (
					cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
					cs.isBitSet(HcalChannelStatus::HcalCellDead))
					continue;
			}
			HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
			_xNChsNominal.get(eid)++;	// he will know the nominal #channels per FED
		}
	}

	//	MARK THESE HISTOGRAMS AS LUMI BASED FOR OFFLINE PROCESSING
	if (_ptype==fOffline)
	{
		_cDigiSize_FED.setLumiFlag();
		_cOccupancy_depth.setLumiFlag();
	}

	ib.setCurrentFolder(_subsystem+"/RunInfo");
	meNumEvents1LS = ib.book1D("NumberOfEvents", "NumberOfEvents",
		1, 0, 1);
	meNumEvents1LS->setLumiFlag();
}

/* virtual */ void DigiPhase1Task::_resetMonitors(hcaldqm::UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);

	switch(uf)
	{
		case hcaldqm::f50LS:
			//	^^^ONLINE ONLY!
			if (_ptype==fOnline)
				_cOccupancyvsiphi_SubdetPM.reset();
			//	^^^
			break;
		default:
			break;
	}
}

/* virtual */ void DigiPhase1Task::_process(edm::Event const& e,
	edm::EventSetup const&)
{
	edm::Handle<QIE11DigiCollection>     chbhe;
	edm::Handle<HODigiCollection>       cho;
	edm::Handle<QIE10DigiCollection>       chf;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHE QIE11DigiCollection isn't available"
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiPhase1Collection isn't available"
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection HF QIE10Collection isn't available"
			+ _tagHF.label() + " " + _tagHF.instance());

	//	extract some info per event
	int bx = e.bunchCrossing();
	meNumEvents1LS->Fill(0.5); // just increment

	//	HB collection
	int numChs = 0;
	int numChsCut = 0;
	int numChsHE = 0;
	int numChsCutHE = 0;

	/*
	 *	Suggested Reading for CMSSW & Boost interaction of iterators for 
	 *	DataFrameContainer, etc....
	 *
	 *	http://www.boost.org/doc/libs/1_53_0/libs/iterator/doc/transform_iterator.html
	 *	http://www.boost.org/doc/libs/1_50_0/libs/iterator/doc/counting_iterator.html
	 *	helps to understand how to unwrap the boost iterator classes...
	 */
	for (QIE11DigiCollection::const_iterator it=chbhe->begin(); it!=chbhe->end();
		++it)
	{
		QIE11DataFrame const& frame = *it;
		double sumQ = hcaldqm::utilities::sumQ_v10<QIE11DataFrame>(frame, 
			2.5, 0, frame.samples()-1);
		HcalDetId const& did = frame.detid();
		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}
		HcalElectronicsId const& eid = _ehashmap.lookup(did);

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype==fOnline)
		{
			_cDigiSizevsLS_FED.fill(eid, _currentLS, frame.samples());
			frame.samples()!=constants::DIGISIZE[did.subdet()-1]?
				_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		_cDigiSize_FED.fill(eid, frame.samples());
		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
			/*
			if (!it->validate(0, it->size()))
			{
				_cCapIdRots_depth.fill(did);
				_cCapIdRots_FEDuTCA.fill(eid, 1);
			}*/
		}

		for (int i=0; i<frame.samples(); i++)
		{
			_cADC_SubdetPM.fill(did, frame[i].adc());
			_cfC_SubdetPM.fill(did, 
				constants::adc2fC[frame[i].adc()]);
			if (sumQ>_cutSumQ_HBHE)
				_cShapeCut_FED.fill(eid, i, 
					constants::adc2fC[frame[i].adc()]);
		}

		if (sumQ>_cutSumQ_HBHE)
		{
			double timing = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(frame, 2.5, 0,
				frame.samples()-1);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cOccupancyCut_depth.fill(did);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			_cSumQ_depth.fill(did, sumQ);
			_cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);
			if (_ptype==fOnline)
			{
				_cSumQvsBX_SubdetPM.fill(did, bx, sumQ);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
			}
			if (eid.isVMEid())
			{
				_cTimingCut_FEDVME.fill(eid, timing);
				_cTimingCut_ElectronicsVME.fill(eid, timing);
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
			}
			else 
			{
				_cTimingCut_FEDuTCA.fill(eid, timing);
				_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
			}
			did.subdet()==HcalBarrel?numChsCut++:numChsCutHE++;
		}
		did.subdet()==HcalBarrel?numChs++:numChsHE++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), _currentLS, 
		numChs);
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), _currentLS,
		numChsHE);
	//	ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), 
			_currentLS, numChsCut);
		_cOccupancyCutvsBX_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), bx,
			numChsCut);
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), 
			_currentLS, numChsCutHE);
		_cOccupancyCutvsBX_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), bx,
			numChsCutHE);
	}
	//	^^^ONLINE ONLY!
	numChs=0;
	numChsCut = 0;

	//	HO collection
	for (HODigiCollection::const_iterator it=cho->begin(); it!=cho->end();
		++it)
	{
		double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size()-1);
		HcalDetId const& did = it->id();
		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}
		HcalElectronicsId const& eid = it->elecId();

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype==fOnline)
		{
			_cDigiSizevsLS_FED.fill(eid, _currentLS, it->size());
			it->size()!=constants::DIGISIZE[did.subdet()-1]?
				_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		_cDigiSize_FED.fill(eid, it->size());
		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
			/*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);
				*/
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
			/*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);*/
		}

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (sumQ>_cutSumQ_HO)
				_cShapeCut_FED.fill(eid, i, it->sample(i).nominal_fC());
		}

		if (sumQ>_cutSumQ_HO)
		{
			double timing = hcaldqm::utilities::aveTS<HODataFrame>(*it, 8.5, 0,
				it->size()-1);
			_cSumQ_depth.fill(did, sumQ);
			_cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			if (_ptype==fOnline)
			{
				_cSumQvsBX_SubdetPM.fill(did, bx, sumQ);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
			}
			if (eid.isVMEid())
			{
				_cTimingCut_FEDVME.fill(eid, timing);
				_cTimingCut_ElectronicsVME.fill(eid, timing);
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
			}
			else 
			{
				_cTimingCut_FEDuTCA.fill(eid, timing);
				_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
			}
			numChsCut++;
		}
		numChs++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 1), _currentLS,
		numChs);

	if (_ptype==fOnline)
	{
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 1), 
			_currentLS, numChsCut);
		_cOccupancyCutvsBX_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 1), bx,
			numChsCut);
	}
	numChs=0; numChsCut=0;

	//	HF collection
	for (QIE10DigiCollection::const_iterator it=chf->begin(); it!=chf->end();
		++it)
	{
		QIE10DataFrame frame = *it;
		double sumQ = hcaldqm::utilities::sumQ_v10<QIE10DataFrame>(frame, 
			2.5, 0, frame.samples()-1);
		HcalDetId const& did = frame.detid();
		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}
		HcalElectronicsId const& eid = _ehashmap.lookup(did);

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype==fOnline)
		{
			_xNChs.get(eid)++;
			_cDigiSizevsLS_FED.fill(eid, _currentLS, frame.samples());
			frame.samples()!=constants::DIGISIZE[did.subdet()-1]?
				_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		_cDigiSize_FED.fill(eid, frame.samples());
		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
			/*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);*/
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
			/*
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);*/
		}

		for (int i=0; i<frame.samples(); i++)
		{
			_cADC_SubdetPM.fill(did, frame[i].adc());
			_cfC_SubdetPM.fill(did, 
				constants::adc2fC[frame[i].adc()]);
			if (sumQ>_cutSumQ_HF)
				_cShapeCut_FED.fill(eid, i, 
					constants::adc2fC[frame[i].adc()]);
		}

		if (sumQ>_cutSumQ_HF)
		{
			double timing = hcaldqm::utilities::aveTS_v10<QIE10DataFrame>(frame, 2.5, 0,
				frame.samples()-1);
			_cSumQ_depth.fill(did, sumQ);
			_cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);
			if (_ptype==fOnline)
			{
				_cSumQvsBX_SubdetPM.fill(did, bx, sumQ);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
//				_cOccupancyCutvsSlotvsLS_HFPM.fill(did, _currentLS);
			}
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			_cOccupancyCut_depth.fill(did);
			if (eid.isVMEid())
			{
				_cTimingCut_FEDVME.fill(eid, timing);
				_cTimingCut_ElectronicsVME.fill(eid, timing);
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
			}
			else 
			{
				_cTimingCut_FEDuTCA.fill(eid, timing);
				_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
			}
			numChsCut++;
		}
		numChs++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), _currentLS, 
		numChs);

	if (_ptype==fOnline)
	{
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), 
			_currentLS, numChsCut);
		_cOccupancyCutvsBX_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), bx,
			numChsCut);
	}
}

/* virtual */ void DigiPhase1Task::beginLuminosityBlock(
	edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	DQTask::beginLuminosityBlock(lb, es);

}

/* virtual */ void DigiPhase1Task::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiPhase1Task);

