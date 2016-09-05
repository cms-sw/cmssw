#include "DQM/HcalTasks/interface/DigiTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

DigiTask::DigiTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));

	_tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<HFDigiCollection>(_tagHF);

	_cutSumQ_HBHE = ps.getUntrackedParameter<double>("cutSumQ_HBHE", 20);
	_cutSumQ_HO = ps.getUntrackedParameter<double>("cutSumQ_HO", 20);
	_cutSumQ_HF = ps.getUntrackedParameter<double>("cutSumQ_HF", 20);
	_thresh_unihf = ps.getUntrackedParameter<double>("thresh_unihf", 0.2);

	_vflags.resize(nDigiFlag);
	_vflags[fUni]=hcaldqm::flag::Flag("UniSlotHF");
	_vflags[fDigiSize]=hcaldqm::flag::Flag("DigiSize");
	_vflags[fNChsHF]=hcaldqm::flag::Flag("NChsHF");
	_vflags[fUnknownIds]=hcaldqm::flag::Flag("UnknownIds");
}

/* virtual */ void DigiTask::bookHistograms(DQMStore::IBooker& ib,
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
	std::vector<uint32_t> vFEDHF;
	vVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN, 
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics,
		vVME);
	_filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics,
		vuTCA);
	vFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	vFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	vFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());

	//	initialize filters
	_filter_FEDHF.initialize(filter::fPreserver, hcaldqm::hashfunctions::fFED,
		vFEDHF);

	//	push the rawIds of each fed into the vector...
	for (std::vector<int>::const_iterator it=vFEDsVME.begin();
		it!=vFEDsVME.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(
			constants::FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN,
			(*it)-FED_VME_MIN).rawId());
	for (std::vector<int>::const_iterator it=vFEDsuTCA.begin();
		it!=vFEDsuTCA.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(
			hcaldqm::utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
			FIBERCH_MIN, false).rawId());

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
		std::vector<uint32_t> vhashHF; 
		vhashHF.push_back(HcalDetId(HcalForward, 31,1,1).rawId());
		_filter_HF.initialize(filter::fPreserver, hcaldqm::hashfunctions::fSubdet,
			vhashHF);

		//	Charge sharing
		_cQ2Q12CutvsLS_FEDHF.initialize(_name, "Q2Q12vsLS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio_0to2));
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
		_cSummaryvsLS_FED.initialize(_name, "SummaryvsLS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::FlagQuantity(_vflags),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState));
		_cSummaryvsLS.initialize(_name, "SummaryvsLS",
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::FEDQuantity(vFEDs),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState));

		_xUniHF.initialize(hcaldqm::hashfunctions::fFEDSlot);
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
	_dhashmap.initialize(_emap, electronicsmap::fE2DHashMap);
	if (_ptype==fOnline)
	{
		_cQ2Q12CutvsLS_FEDHF.book(ib, _emap, _filter_FEDHF, _subsystem);
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
//		_cOccupancyCutvsSlotvsLS_HFPM.book(ib, _emap, _filter_HF, _subsystem);
		_cOccupancyCutvsiphivsLS_SubdetPM.book(ib, _emap, _subsystem);
		_cSummaryvsLS_FED.book(ib, _emap, _subsystem);
		_cSummaryvsLS.book(ib, _subsystem);

		_xUniHF.book(_emap, _filter_FEDHF);
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

	//	book Number of Events vs LS histogram
	ib.setCurrentFolder(_subsystem+"/RunInfo");
	meNumEvents1LS = ib.book1D("NumberOfEvents", "NumberOfEvents",
		1, 0, 1);
	meNumEvents1LS->setLumiFlag();

	//	book the flag for unknown ids and the online guy as well
	ib.setCurrentFolder(_subsystem+"/"+_name);
	meUnknownIds1LS = ib.book1D("UnknownIds", "UnknownIds",
		1, 0, 1);
	_unknownIdsPresent = false;
	meUnknownIds1LS->setLumiFlag();
}

/* virtual */ void DigiTask::_resetMonitors(hcaldqm::UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);

	switch(uf)
	{
		case hcaldqm::f1LS:
			_unknownIdsPresent = false;
			break;
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

/* virtual */ void DigiTask::_process(edm::Event const& e,
	edm::EventSetup const&)
{
	edm::Handle<HBHEDigiCollection>     chbhe;
	edm::Handle<HODigiCollection>       cho;
	edm::Handle<HFDigiCollection>       chf;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHEDigiCollection isn't available"
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiCollection isn't available"
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection HFDigiCollection isn't available"
			+ _tagHF.label() + " " + _tagHF.instance());

	//	extract some info per event
	int bx = e.bunchCrossing();
	meNumEvents1LS->Fill(0.5); // just increment

	//	To fill histograms outside of the loop, you need to determine if there were
	//	any valid det ids first
	uint32_t rawidValid = 0;
	uint32_t rawidHBValid = 0;
	uint32_t rawidHEValid = 0;

	//	HB collection
	int numChs = 0;
	int numChsCut = 0;
	int numChsHE = 0;
	int numChsCutHE = 0;
	for (HBHEDigiCollection::const_iterator it=chbhe->begin(); it!=chbhe->end();
		++it)
	{
		double sumQ = hcaldqm::utilities::sumQ<HBHEDataFrame>(*it, 2.5, 0, it->size()-1);

		//	Explicit check on the DetIds present in the Collection
		HcalDetId const& did = it->id();
		uint32_t rawid = _ehashmap.lookup(did);
		if (rawid==0) 
		{meUnknownIds1LS->Fill(1); _unknownIdsPresent=true;continue;}
		HcalElectronicsId const& eid(rawid);
		if (did.subdet()==HcalBarrel)
			rawidHBValid = did.rawId();
		else if (did.subdet()==HcalEndcap) 
			rawidHEValid = did.rawId();

		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}

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

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (sumQ>_cutSumQ_HBHE)
				_cShapeCut_FED.fill(eid, i, it->sample(i).nominal_fC());
		}

		if (sumQ>_cutSumQ_HBHE)
		{
			double timing = hcaldqm::utilities::aveTS<HBHEDataFrame>(*it, 2.5, 0,
				it->size()-1);
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

	if (rawidHBValid!=0 && rawidHEValid!=0)
	{
		_cOccupancyvsLS_Subdet.fill(HcalDetId(rawidHBValid), _currentLS, 
			numChs);
		_cOccupancyvsLS_Subdet.fill(HcalDetId(rawidHEValid), _currentLS,
			numChsHE);
		//	ONLINE ONLY!
		if (_ptype==fOnline)
		{
			_cOccupancyCutvsLS_Subdet.fill(HcalDetId(rawidHBValid), 
				_currentLS, numChsCut);
			_cOccupancyCutvsBX_Subdet.fill(HcalDetId(rawidHBValid), bx,
				numChsCut);
			_cOccupancyCutvsLS_Subdet.fill(HcalDetId(rawidHEValid), 
				_currentLS, numChsCutHE);
			_cOccupancyCutvsBX_Subdet.fill(HcalDetId(rawidHEValid), bx,
				numChsCutHE);
		}
		//	^^^ONLINE ONLY!
	}
	numChs=0;
	numChsCut = 0;

	//	reset
	rawidValid = 0;

	//	HO collection
	for (HODigiCollection::const_iterator it=cho->begin(); it!=cho->end();
		++it)
	{
		double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size()-1);

		//	Explicit check on the DetIds present in the Collection
		HcalDetId const& did = it->id();
		uint32_t rawid = _ehashmap.lookup(did);
		if (rawid==0) 
		{meUnknownIds1LS->Fill(1); _unknownIdsPresent=true;continue;}
		HcalElectronicsId const& eid(rawid);
		if (did.subdet()==HcalOuter)
			rawidValid = did.rawId();

		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}

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

	if (rawidValid!=0)
	{
		_cOccupancyvsLS_Subdet.fill(HcalDetId(rawidValid), _currentLS,
			numChs);
	
		if (_ptype==fOnline)
		{
			_cOccupancyCutvsLS_Subdet.fill(HcalDetId(rawidValid), 
				_currentLS, numChsCut);
			_cOccupancyCutvsBX_Subdet.fill(HcalDetId(rawidValid), bx,
				numChsCut);
		}
	}
	numChs=0; numChsCut=0;

	//	reset
	rawidValid = 0;

	//	HF collection
	for (HFDigiCollection::const_iterator it=chf->begin(); it!=chf->end();
		++it)
	{
		double sumQ = hcaldqm::utilities::sumQ<HFDataFrame>(*it, 2.5, 0, it->size()-1);

		//	Explicit check on the DetIds present in the Collection
		HcalDetId const& did = it->id();
		uint32_t rawid = _ehashmap.lookup(did);
		if (rawid==0) 
		{meUnknownIds1LS->Fill(1); _unknownIdsPresent=true;continue;}
		HcalElectronicsId const& eid(rawid);
		if (did.subdet()==HcalForward)
			rawidValid = did.rawId();

		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype==fOnline)
		{
			_xNChs.get(eid)++;
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

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (sumQ>_cutSumQ_HF)
				_cShapeCut_FED.fill(eid, i, it->sample(i).nominal_fC());
		}

		if (sumQ>_cutSumQ_HF)
		{
			double timing = hcaldqm::utilities::aveTS<HFDataFrame>(*it, 2.5, 0,
				it->size()-1);
			double q1 = it->sample(1).nominal_fC()-2.5;
			double q2 = it->sample(2).nominal_fC()-2.5;
			double q2q12 = q2/(q1+q2);
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
				_xUniHF.get(eid)++;
			}
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			_cOccupancyCut_depth.fill(did);
			if (!eid.isVMEid())
				if (_ptype==fOnline)
					_cQ2Q12CutvsLS_FEDHF.fill(eid, _currentLS, q2q12);
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

	if (rawidValid!=0)
	{
		_cOccupancyvsLS_Subdet.fill(HcalDetId(rawidValid), _currentLS, 
			numChs);
	
		if (_ptype==fOnline)
		{
			_cOccupancyCutvsLS_Subdet.fill(HcalDetId(rawidValid), 
				_currentLS, numChsCut);
			_cOccupancyCutvsBX_Subdet.fill(HcalDetId(rawidValid), bx,
				numChsCut);
		}
	}
}

/* virtual */ void DigiTask::beginLuminosityBlock(
	edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	DQTask::beginLuminosityBlock(lb, es);
}

/* virtual */ void DigiTask::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	if (_ptype!=fOnline)
		return;

	for (uintCompactMap::const_iterator it=_xUniHF.begin();
		it!=_xUniHF.end(); ++it)
	{
		uint32_t hash1 = it->first;
		HcalElectronicsId eid1(hash1);
		double x1 = it->second;

		for (uintCompactMap::const_iterator jt=_xUniHF.begin();
			jt!=_xUniHF.end(); ++jt)
		{
			if (jt==it)
				continue;
			double x2 = jt->second;
			if (x2==0)
				continue;
			if (x1/x2<_thresh_unihf)
				_xUni.get(eid1)++;
		}
	}

	for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
		it!=_vhashFEDs.end(); ++it)
	{
		hcaldqm::flag::Flag fSum("DIGI");
		HcalElectronicsId eid = HcalElectronicsId(*it);

		std::vector<uint32_t>::const_iterator cit=std::find(
			_vcdaqEids.begin(), _vcdaqEids.end(), *it);
		if (cit==_vcdaqEids.end())
		{
			//	not @cDAQ
			for (uint32_t iflag=0; iflag<_vflags.size(); iflag++)
				_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
					int(hcaldqm::flag::fNCDAQ));
			_cSummaryvsLS.setBinContent(eid, _currentLS, int(hcaldqm::flag::fNCDAQ));
			continue;
		}

		//	FED is @cDAQ		
		if (hcaldqm::utilities::isFEDHBHE(eid) || hcaldqm::utilities::isFEDHF(eid) ||
			hcaldqm::utilities::isFEDHO(eid))
		{
			if (_xDigiSize.get(eid)>0)
				_vflags[fDigiSize]._state = hcaldqm::flag::fBAD;
			else
				_vflags[fDigiSize]._state = hcaldqm::flag::fGOOD;
			if (hcaldqm::utilities::isFEDHF(eid))
			{
				double fr = double(_xNChs.get(eid))/double(
					_xNChsNominal.get(eid)*_evsPerLS);
				if (_runkeyVal==0 || _runkeyVal==4)
				{
					//	only for pp or hi
					if (_xUni.get(eid)>0)
						_vflags[fUni]._state = hcaldqm::flag::fPROBLEMATIC;
					else
						_vflags[fUni]._state = hcaldqm::flag::fGOOD;
				}
				if (fr<0.95)
					_vflags[fNChsHF]._state = hcaldqm::flag::fBAD;
				else if (fr<1.0)
					_vflags[fNChsHF]._state = hcaldqm::flag::fPROBLEMATIC;
				else
					_vflags[fNChsHF]._state = hcaldqm::flag::fGOOD;
			}
		}
		if (_unknownIdsPresent) 
			_vflags[fUnknownIds]._state = hcaldqm::flag::fBAD;
		else
			_vflags[fUnknownIds]._state = hcaldqm::flag::fGOOD;

		int iflag=0;
		for (std::vector<hcaldqm::flag::Flag>::iterator ft=_vflags.begin();
			ft!=_vflags.end(); ++ft)
		{
			_cSummaryvsLS_FED.setBinContent(eid, _currentLS, iflag,
				int(ft->_state));
			fSum+=(*ft);
			iflag++;

			//	reset!
			ft->reset();
		}
		_cSummaryvsLS.setBinContent(eid, _currentLS, fSum._state);
	}

	_xDigiSize.reset(); _xUniHF.reset(); _xUni.reset();
	_xNChs.reset();

	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiTask);

