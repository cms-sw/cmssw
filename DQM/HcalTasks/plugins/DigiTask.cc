#include "DQM/HcalTasks/interface/DigiTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

DigiTask::DigiTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHE = ps.getUntrackedParameter<edm::InputTag>("tagHE",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));

	_tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
	_tokHE = consumes<QIE11DigiCollection>(_tagHE);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<QIE10DigiCollection>(_tagHF);

	_cutSumQ_HBHE = ps.getUntrackedParameter<double>("cutSumQ_HBHE", 20);
	_cutSumQ_HE = ps.getUntrackedParameter<double>("cutSumQ_HE", 20);
	_cutSumQ_HO = ps.getUntrackedParameter<double>("cutSumQ_HO", 20);
	_cutSumQ_HF = ps.getUntrackedParameter<double>("cutSumQ_HF", 20);
	_thresh_unihf = ps.getUntrackedParameter<double>("thresh_unihf", 0.2);
	_thresh_led = ps.getUntrackedParameter<double>("thresh_led", 20);

	_vflags.resize(nDigiFlag);
	_vflags[fUni]=hcaldqm::flag::Flag("UniSlotHF");
	_vflags[fDigiSize]=hcaldqm::flag::Flag("DigiSize");
	_vflags[fNChsHF]=hcaldqm::flag::Flag("NChsHF");
	_vflags[fUnknownIds]=hcaldqm::flag::Flag("UnknownIds");
	_vflags[fLED]=hcaldqm::flag::Flag("LEDMisfire");
	_vflags[fCapId]=hcaldqm::flag::Flag("BadCapId");

	_qie10InConditions = ps.getUntrackedParameter<bool>("qie10InConditions", true);

	// Get reference digi sizes. Convert from unsigned to signed int, because <digi>::size()/samples() return ints for some reason.
	std::vector<uint32_t> vrefDigiSize = ps.getUntrackedParameter<std::vector<uint32_t>>("refDigiSize");
	_refDigiSize[HcalBarrel]  = (int)vrefDigiSize[0];
	_refDigiSize[HcalEndcap]  = (int)vrefDigiSize[1];
	_refDigiSize[HcalOuter]   = (int)vrefDigiSize[2];
	_refDigiSize[HcalForward] = (int)vrefDigiSize[3];

	// (capid - BX) % 4 to 1
	_capidmbx[HcalBarrel] = 1;
	_capidmbx[HcalEndcap] = 1;
	_capidmbx[HcalOuter] = 1;
	_capidmbx[HcalForward] = 1;

	// LED calibration channels
	std::vector<edm::ParameterSet> vLedCalibChannels = ps.getParameter<std::vector<edm::ParameterSet>>("ledCalibrationChannels");
	for (int i = 0; i <= 3; ++i) {
		HcalSubdetector this_subdet = HcalEmpty;
		switch (i) {
			case 0:
				this_subdet = HcalBarrel;
				break;
			case 1:
				this_subdet = HcalEndcap;
				break;
			case 2:
				this_subdet = HcalOuter;
				break;
			case 3:
				this_subdet = HcalForward;
				break;
			default:
				this_subdet = HcalEmpty;
				break;
		}
		std::vector<int32_t> subdet_calib_ietas = vLedCalibChannels[i].getUntrackedParameter<std::vector<int32_t>>("ieta");
		std::vector<int32_t> subdet_calib_iphis = vLedCalibChannels[i].getUntrackedParameter<std::vector<int32_t>>("iphi");
		std::vector<int32_t> subdet_calib_depths = vLedCalibChannels[i].getUntrackedParameter<std::vector<int32_t>>("depth");
		for (unsigned int ichannel = 0; ichannel < subdet_calib_ietas.size(); ++ichannel) {
			_ledCalibrationChannels[this_subdet].push_back(HcalDetId(HcalOther, subdet_calib_ietas[ichannel], subdet_calib_iphis[ichannel], subdet_calib_depths[ichannel]));
		}
	}
}

/* virtual */ void DigiTask::bookHistograms(DQMStore::IBooker& ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	DQTask::bookHistograms(ib,r,es);

	//	GET WHAT YOU NEED
	edm::ESHandle<HcalDbService> dbs;
	es.get<HcalDbRecord>().get(dbs);
	_emap = dbs->getHcalMapping();
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

	// Filters for QIE8 vs QIE10/11
	std::vector<uint32_t> vhashQIE1011; 
	vhashQIE1011.push_back(hcaldqm::hashfunctions::hash_did[hcaldqm::hashfunctions::fSubdet](HcalDetId(HcalEndcap, 20,1,1)));
	vhashQIE1011.push_back(hcaldqm::hashfunctions::hash_did[hcaldqm::hashfunctions::fSubdet](HcalDetId(HcalForward, 29,1,1)));
	_filter_QIE1011.initialize(filter::fPreserver, hcaldqm::hashfunctions::fSubdet,
		vhashQIE1011);
	_filter_QIE8.initialize(filter::fFilter, hcaldqm::hashfunctions::fSubdet,
		vhashQIE1011);

	//	INITIALIZE FIRST
	_cADC_SubdetPM.initialize(_name, "ADC", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cfC_SubdetPM.initialize(_name, "fC", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cSumQ_SubdetPM.initialize(_name, "SumQ", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cSumQ_depth.initialize(_name, "SumQ", hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_generic_400000, true),0);
	_cSumQvsLS_SubdetPM.initialize(_name, "SumQvsLS",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),0);

	_cADC_SubdetPM_QIE1011.initialize(_name, "ADC", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cfC_SubdetPM_QIE1011.initialize(_name, "fC", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_400000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cSumQ_SubdetPM_QIE1011.initialize(_name, "SumQ", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_400000),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cSumQvsLS_SubdetPM_QIE1011.initialize(_name, "SumQvsLS",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_400000),0);

	_cTimingCut_SubdetPM.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	_cTimingCut_depth.initialize(_name, "TimingCut",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
	_cTimingCutvsLS_SubdetPM.initialize(_name, "TimingvsLS",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);

	//	Occupancy w/o a cut
	_cOccupancyvsLS_Subdet.initialize(_name, "OccupancyvsLS",
		hcaldqm::hashfunctions::fSubdet,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to8000),0);
	_cOccupancy_depth.initialize(_name, "Occupancy",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

	//	Occupancy w/ a cut
	_cOccupancyCutvsLS_Subdet.initialize(_name, "OccupancyCutvsLS",
		hcaldqm::hashfunctions::fSubdet,
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to8000),0);
	_cOccupancyCut_depth.initialize(_name, "OccupancyCut",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

	// Digi size
	_cDigiSize_Crate.initialize(_name, "DigiSize",
		hcaldqm::hashfunctions::fCrate,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDigiSize),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	_cADCvsTS_SubdetPM.initialize(_name, "ADCvsTS",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	_cADCvsTS_SubdetPM_QIE1011.initialize(_name, "ADCvsTS",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

	_cLETDCTimevsADC_SubdetPM.initialize(_name, "LETDCTimevsADC",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250_coarse),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cLETDCvsADC_SubdetPM.initialize(_name, "LETDCvsADC",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cLETDCvsTS_SubdetPM.initialize(_name, "LETDCvsTS", 
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64), 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));		
	_cLETDCTime_SubdetPM.initialize(_name, "LETDCTime",
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cLETDCTime_depth.initialize(_name, "LETDCTime",
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

	_cBadTDCValues_SubdetPM.initialize(_name, "BadTDCValues", 
		hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBadTDC),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cBadTDCvsBX_SubdetPM.initialize(_name, "BadTDCvsBX", 
		hcaldqm::hashfunctions::fSubdetPM, 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cBadTDCvsLS_SubdetPM.initialize(_name, "BadTDCvsLS", 
		hcaldqm::hashfunctions::fSubdetPM, 
		new hcaldqm::quantity::LumiSection(_maxLS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cBadTDCCount_depth.initialize(_name, "BadTDCCount", 
		hcaldqm::hashfunctions::fdepth,
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
		new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

	if (_ptype == fOnline || _ptype == fLocal) {
		_cOccupancy_Crate.initialize(_name,
			 "Occupancy", hashfunctions::fCrate,
			 new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
			 new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
			 new quantity::ValueQuantity(quantity::fN),0);
		_cOccupancy_CrateSlot.initialize(_name,
			 "Occupancy", hashfunctions::fCrateSlot,
			 new quantity::ElectronicsQuantity(quantity::fFiberuTCA),
			 new quantity::ElectronicsQuantity(quantity::fFiberCh),
			 new quantity::ValueQuantity(quantity::fN),0);		
	}

	//	INITIALIZE HISTOGRAMS that are only for Online
	if (_ptype==fOnline)
	{
		//	Charge sharing
		_cQ2Q12CutvsLS_FEDHF.initialize(_name, "Q2Q12vsLS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fRatio_0to2),0);
		_cSumQvsBX_SubdetPM.initialize(_name, "SumQvsBX",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),0);
		_cSumQvsBX_SubdetPM_QIE1011.initialize(_name, "SumQvsBX",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_10000),0);
		_cDigiSizevsLS_FED.initialize(_name, "DigiSizevsLS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDigiSize),0);
		_cTimingCutvsiphi_SubdetPM.initialize(_name, "TimingCutvsiphi",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingCutvsieta_Subdet.initialize(_name, "TimingCutvsieta",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cOccupancyvsiphi_SubdetPM.initialize(_name, "Occupancyvsiphi",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyvsieta_Subdet.initialize(_name, "Occupancyvsieta",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCutvsiphi_SubdetPM.initialize(_name, "OccupancyCutvsiphi",
			hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCutvsieta_Subdet.initialize(_name, "OccupancyCutvsieta",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCutvsLS_Subdet.initialize(_name, "OccupancyCutvsLS",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to8000),0);
		_cOccupancyCutvsBX_Subdet.initialize(_name, "OccupancyCutvsBX",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_to8000),0);
//		_cOccupancyCutvsSlotvsLS_HFPM.initialize(_name, 
//			"OccupancyCutvsSlotvsLS", hcaldqm::hashfunctions::fSubdetPM,
//			new hcaldqm::quantity::LumiSection(_maxLS),
//			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
//			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCutvsiphivsLS_SubdetPM.initialize(_name, 
			"OccupancyCutvsiphivsLS", hcaldqm::hashfunctions::fSubdetPM,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	}
	_cCapidMinusBXmod4_SubdetPM.initialize(_name, 
		"CapID", hcaldqm::hashfunctions::fSubdetPM,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fCapidMinusBXmod4),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));

	for (int i = 0; i < 4; ++i) {
		_cCapidMinusBXmod4_CrateSlotuTCA[i].initialize(_name, "CapID", 
			new quantity::ElectronicsQuantity(quantity::fCrateuTCA),
			new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
		_cCapidMinusBXmod4_CrateSlotVME[i].initialize(_name, "CapID", 
			new quantity::ElectronicsQuantity(quantity::fCrateVME),
			new quantity::ElectronicsQuantity(quantity::fSlotVME),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	}

	if (_ptype != fOffline) { // hidefed2crate
		std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
		std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
		std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);

		if (_ptype == fOnline) {
			_cCapid_BadvsFEDvsLS.initialize(_name, "CapID", 
				new hcaldqm::quantity::LumiSectionCoarse(_maxLS, 10),
				new hcaldqm::quantity::FEDQuantity(vFEDs),		
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);

			_cCapid_BadvsFEDvsLSmod60.initialize(_name, "CapID", 
				new hcaldqm::quantity::LumiSection(60),
				new hcaldqm::quantity::FEDQuantity(vFEDs),		
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
		}
	
		std::vector<uint32_t> vFEDHF;
		vFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN+6,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN+6,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN+6,
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
		{
			std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(*it);
			_vhashFEDs.push_back(HcalElectronicsId(
				cspair.first, cspair.second, FIBER_uTCA_MIN1,
				FIBERCH_MIN, false).rawId());
		}
	
		_cShapeCut_FED.initialize(_name, "ShapeCut",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000),0);
	
		_cTimingCut_FEDVME.initialize(_name, "TimingCut",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingCut_FEDuTCA.initialize(_name, "TimingCut",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingCut_ElectronicsVME.initialize(_name, "TimingCut",
			hcaldqm::hashfunctions::fElectronics,
			new hcaldqm::quantity::FEDQuantity(vFEDsVME),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingCut_ElectronicsuTCA.initialize(_name, "TimingCut",
			hcaldqm::hashfunctions::fElectronics,
			new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
		_cTimingCutvsLS_FED.initialize(_name, "TimingvsLS",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS200),0);
	
		_cOccupancy_FEDVME.initialize(_name, "Occupancy",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancy_FEDuTCA.initialize(_name, "Occupancy",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancy_ElectronicsVME.initialize(_name, "Occupancy",
			hcaldqm::hashfunctions::fElectronics,
			new hcaldqm::quantity::FEDQuantity(vFEDsVME),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancy_ElectronicsuTCA.initialize(_name, "Occupancy",
			hcaldqm::hashfunctions::fElectronics,
			new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
	
		_cOccupancyCut_FEDVME.initialize(_name, "OccupancyCut",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCut_FEDuTCA.initialize(_name, "OccupancyCut",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCut_ElectronicsVME.initialize(_name, "OccupancyCut",
			hcaldqm::hashfunctions::fElectronics,
			new hcaldqm::quantity::FEDQuantity(vFEDsVME),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		_cOccupancyCut_ElectronicsuTCA.initialize(_name, "OccupancyCut",
			hcaldqm::hashfunctions::fElectronics,
			new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
			new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

		_cDigiSize_FED.initialize(_name, "DigiSize",
			hcaldqm::hashfunctions::fFED,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDigiSize),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

		if (_ptype == fOnline) {
			_cSummaryvsLS_FED.initialize(_name, "SummaryvsLS",
				hcaldqm::hashfunctions::fFED,
				new hcaldqm::quantity::LumiSection(_maxLS),
				new hcaldqm::quantity::FlagQuantity(_vflags),
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState),0);
			_cSummaryvsLS.initialize(_name, "SummaryvsLS",
				new hcaldqm::quantity::LumiSection(_maxLS),
				new hcaldqm::quantity::FEDQuantity(vFEDs),
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState),0);

			_xUniHF.initialize(hcaldqm::hashfunctions::fFEDSlot);
			_xUni.initialize(hcaldqm::hashfunctions::fFED);
			_xDigiSize.initialize(hcaldqm::hashfunctions::fFED);
			_xNChs.initialize(hcaldqm::hashfunctions::fFED);
			_xNChsNominal.initialize(hcaldqm::hashfunctions::fFED);
			_xBadCapid.initialize(hcaldqm::hashfunctions::fFED);
		}
	}
	if (_ptype != fLocal) {
		_LED_ADCvsBX_Subdet.initialize(_name, "LED_ADCvsBX", 
			hcaldqm::hashfunctions::fSubdet, 
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX_36),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_256_4),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);

		_LED_CUCountvsLS_Subdet.initialize(_name, "LED_CUCountvsLS",
			hcaldqm::hashfunctions::fSubdet,
			new hcaldqm::quantity::LumiSection(_maxLS),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		if (_ptype == fOnline) {
			_LED_CUCountvsLSmod60_Subdet.initialize(_name, "LED_CUCountvsLSmod60",
				hcaldqm::hashfunctions::fSubdet,
				new hcaldqm::quantity::LumiSection(60),
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),0);
		}
	}

	//	BOOK HISTOGRAMS
	char cutstr[200];
	sprintf(cutstr, "_SumQHBHE%dHO%dHF%d", int(_cutSumQ_HBHE),
		int(_cutSumQ_HO), int(_cutSumQ_HF));
	char cutstr2[200];
	sprintf(cutstr2, "_SumQHF%d", int(_cutSumQ_HF));

	_cADC_SubdetPM.book(ib, _emap, _filter_QIE8, _subsystem);
	_cADC_SubdetPM_QIE1011.book(ib, _emap, _filter_QIE1011, _subsystem);
	_cfC_SubdetPM.book(ib, _emap, _filter_QIE8, _subsystem);
	_cfC_SubdetPM_QIE1011.book(ib, _emap, _filter_QIE1011, _subsystem);
	_cSumQ_SubdetPM.book(ib, _emap, _filter_QIE8, _subsystem);
	_cSumQ_SubdetPM_QIE1011.book(ib, _emap, _filter_QIE1011, _subsystem);
	_cSumQ_depth.book(ib, _emap, _subsystem);
	_cSumQvsLS_SubdetPM.book(ib, _emap, _filter_QIE8, _subsystem);
	_cSumQvsLS_SubdetPM_QIE1011.book(ib, _emap, _filter_QIE1011, _subsystem);
	_cDigiSize_Crate.book(ib, _emap, _subsystem);
	_cADCvsTS_SubdetPM.book(ib, _emap, _filter_QIE8, _subsystem);
	_cADCvsTS_SubdetPM_QIE1011.book(ib, _emap, _filter_QIE1011, _subsystem);

	if (_ptype != fOffline) { // hidefed2crate
		_cShapeCut_FED.book(ib, _emap, _subsystem);
		_cTimingCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cTimingCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cTimingCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cTimingCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cTimingCutvsLS_FED.book(ib, _emap, _subsystem);
		_cOccupancy_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cOccupancy_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cOccupancy_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cOccupancy_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cOccupancyCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cOccupancyCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cOccupancyCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
		_cOccupancyCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
		_cDigiSize_FED.book(ib, _emap, _subsystem);
	}

	_cTimingCut_SubdetPM.book(ib, _emap, _subsystem);
	_cTimingCut_depth.book(ib, _emap, _subsystem);
	_cTimingCutvsLS_SubdetPM.book(ib, _emap, _subsystem);

	_cOccupancyvsLS_Subdet.book(ib, _emap, _subsystem);
	_cOccupancy_depth.book(ib, _emap, _subsystem);
	_cOccupancyCut_depth.book(ib, _emap, _subsystem);

	_cLETDCTimevsADC_SubdetPM.book(ib, _emap, _subsystem);
	_cLETDCvsADC_SubdetPM.book(ib, _emap, _subsystem);
	_cLETDCvsTS_SubdetPM.book(ib, _emap, _subsystem);
	_cLETDCTime_SubdetPM.book(ib, _emap, _subsystem);
	_cLETDCTime_depth.book(ib, _emap, _subsystem);

	_cBadTDCValues_SubdetPM.book(ib, _emap, _subsystem);
	_cBadTDCvsBX_SubdetPM.book(ib, _emap, _subsystem);
	_cBadTDCvsLS_SubdetPM.book(ib, _emap, _subsystem);
	_cBadTDCCount_depth.book(ib, _emap, _subsystem);

	_cCapidMinusBXmod4_SubdetPM.book(ib, _emap, _subsystem);
	if (_ptype == fOnline) {
		_cCapid_BadvsFEDvsLS.book(ib, _subsystem, "BadvsLS");
		_cCapid_BadvsFEDvsLSmod60.book(ib, _subsystem, "BadvsLSmod60");
	}
	for (int i = 0; i < 4; ++i) {
		constexpr unsigned int kSize=16;
		char aux[kSize];
		snprintf(aux, kSize, "%d_uTCA", i);
		_cCapidMinusBXmod4_CrateSlotuTCA[i].book(ib, _subsystem, aux);

		snprintf(aux, kSize, "%d_VME", i);
		_cCapidMinusBXmod4_CrateSlotVME[i].book(ib, _subsystem, aux);
	}

	if (_ptype != fLocal) {
		_LED_ADCvsBX_Subdet.book(ib, _emap, _subsystem);
		_LED_CUCountvsLS_Subdet.book(ib, _emap, _subsystem);
		if (_ptype == fOnline) {
			_LED_CUCountvsLSmod60_Subdet.book(ib, _emap, _subsystem);
		}
	}

	//	BOOK HISTOGRAMS that are only for Online
	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
	_dhashmap.initialize(_emap, electronicsmap::fE2DHashMap);

	if (_ptype == fOnline || _ptype == fLocal) {
		_cOccupancy_Crate.book(ib, _emap, _subsystem);
		_cOccupancy_CrateSlot.book(ib, _emap, _subsystem);
	}

	if (_ptype==fOnline)
	{
		_cQ2Q12CutvsLS_FEDHF.book(ib, _emap, _filter_FEDHF, _subsystem);
		_cSumQvsBX_SubdetPM.book(ib, _emap, _filter_QIE8, _subsystem);
		_cSumQvsBX_SubdetPM_QIE1011.book(ib, _emap, _filter_QIE1011, _subsystem);
		_cDigiSizevsLS_FED.book(ib, _emap, _subsystem);
		_cTimingCutvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cTimingCutvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsLS_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsBX_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyCutvsieta_Subdet.book(ib, _emap, _subsystem);
//		_cOccupancyCutvsSlotvsLS_HFPM.book(ib, _emap, _filter_QIE1011, _subsystem);
		_cOccupancyCutvsiphivsLS_SubdetPM.book(ib, _emap, _subsystem);
		_cSummaryvsLS_FED.book(ib, _emap, _subsystem);
		_cSummaryvsLS.book(ib, _subsystem);

		_xUniHF.book(_emap, _filter_FEDHF);
		_xNChs.book(_emap);
		_xNChsNominal.book(_emap);
		_xUni.book(_emap);
		_xDigiSize.book(_emap);
		_xBadCapid.book(_emap);

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
		_cDigiSize_Crate.setLumiFlag();
		//_cDigiSize_FED.setLumiFlag();
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
	edm::Handle<QIE11DigiCollection>     che_qie11;
	edm::Handle<HODigiCollection>       cho;
	edm::Handle<QIE10DigiCollection>       chf;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHEDigiCollection isn't available"
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHE, che_qie11))
		_logger.dqmthrow("Collection QIE11DigiCollection isn't available"
			+ _tagHE.label() + " " + _tagHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiCollection isn't available"
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection QIE10DigiCollection isn't available"
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
		//	Explicit check on the DetIds present in the Collection
		HcalDetId const& did = it->id();
		if (did.subdet() != HcalBarrel) {
			continue;
		}
		uint32_t rawid = _ehashmap.lookup(did);
		if (rawid == 0) {
			meUnknownIds1LS->Fill(1); 
			_unknownIdsPresent=true;
			continue;
		} else {
			if (did.subdet()==HcalBarrel) {
				rawidHBValid = did.rawId();
			} else if (did.subdet()==HcalEndcap) {
				rawidHEValid = did.rawId();
			}
		}
		HcalElectronicsId const& eid(rawid);

		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}

		if (_ptype == fOnline) {
			short this_capidmbx = (it->sample(it->presamples()).capid() - bx) % 4;
			if (this_capidmbx < 0) {
				this_capidmbx += 4;
			}
			_cCapidMinusBXmod4_SubdetPM.fill(did, this_capidmbx);
			bool good_capidmbx = (_capidmbx[did.subdet()] == this_capidmbx);
			if (!good_capidmbx) {
				_xBadCapid.get(eid)++;
				_cCapid_BadvsFEDvsLS.fill(eid, _currentLS);
				_cCapid_BadvsFEDvsLSmod60.fill(eid, _currentLS % 60);
			}
			if (eid.isVMEid()) {
				_cCapidMinusBXmod4_CrateSlotVME[this_capidmbx].fill(eid);

			} else {
				_cCapidMinusBXmod4_CrateSlotuTCA[this_capidmbx].fill(eid);
			}
		}

		//double sumQ = hcaldqm::utilities::sumQ<HBHEDataFrame>(*it, 2.5, 0, it->size()-1);
		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<HBHEDataFrame>(_dbService, did, *it);
		double sumQ = hcaldqm::utilities::sumQDB<HBHEDataFrame>(_dbService, digi_fC, did, *it, 0, it->size()-1);

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype == fOnline || _ptype == fLocal) {
			_cOccupancy_Crate.fill(eid);
			_cOccupancy_CrateSlot.fill(eid);
		}
		if (_ptype==fOnline)
		{
			_cDigiSizevsLS_FED.fill(eid, _currentLS, it->size());
			it->size()!=_refDigiSize[did.subdet()]?
				_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		_cDigiSize_Crate.fill(eid, it->size());
		if (_ptype != fOffline) { // hidefed2crate
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
		}

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (_ptype != fOffline) { // hidefed2crate
				_cADCvsTS_SubdetPM.fill(did, i, it->sample(i).adc());
				if (sumQ>_cutSumQ_HBHE) {
					_cShapeCut_FED.fill(eid, i, it->sample(i).nominal_fC());
				}
			}
		}

		if (sumQ>_cutSumQ_HBHE)
		{
			//double timing = hcaldqm::utilities::aveTS<HBHEDataFrame>(*it, 2.5, 0, it->size()-1);
			double timing = hcaldqm::utilities::aveTSDB<HBHEDataFrame>(_dbService, digi_fC, did, *it, 0, it->size()-1);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cOccupancyCut_depth.fill(did);
			_cTimingCutvsLS_SubdetPM.fill(did, _currentLS, timing);
			if (_ptype != fOffline) { // hidefed2crate
				_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			}
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
			if (_ptype != fOffline) { // hidefed2crate
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
			}
			did.subdet()==HcalBarrel?numChsCut++:numChsCutHE++;
		}
		did.subdet()==HcalBarrel?numChs++:numChsHE++;
	}

	// HE QIE11 collection
	for (QIE11DigiCollection::const_iterator it=che_qie11->begin(); it!=che_qie11->end();
		++it)
	{
		const QIE11DataFrame digi = static_cast<const QIE11DataFrame>(*it);

		//	Explicit check on the DetIds present in the Collection
		HcalDetId const& did = digi.detid();
		if (did.subdet() != HcalEndcap) {
			// LED monitoring from calibration channels
			if (_ptype != fLocal) {
				if (did.subdet() == HcalOther) {
					HcalOtherDetId hodid(digi.detid());
					if (hodid.subdet() == HcalCalibration) {
						// New method: use configurable list of channels
						if (std::find(_ledCalibrationChannels[HcalEndcap].begin(), _ledCalibrationChannels[HcalEndcap].end(), did) != _ledCalibrationChannels[HcalEndcap].end()) {
							bool channelLEDSignalPresent = false;
							for (int i=0; i<digi.samples(); i++) {
								_LED_ADCvsBX_Subdet.fill(HcalDetId(HcalEndcap, 16, 1, 1), bx, digi[i].adc());

								if (digi[i].adc() > _thresh_led) {
									channelLEDSignalPresent = true;
								}
							}
							if (channelLEDSignalPresent) {
								_LED_CUCountvsLS_Subdet.fill(HcalDetId(HcalEndcap, 16, 1, 1), _currentLS);
								if (_ptype == fOnline) {
									_LED_CUCountvsLSmod60_Subdet.fill(HcalDetId(HcalEndcap, 16, 1, 1), _currentLS % 60);
								}
							}
						}
					}
				}
			}
			continue;
		}

		uint32_t rawid = _ehashmap.lookup(did);
		if (rawid == 0) {
			meUnknownIds1LS->Fill(1);
			_unknownIdsPresent=true;
			continue;
		} else {
			if (did.subdet()==HcalBarrel) { // Note: since this is HE, we obviously expect did.subdet() always to be HcalEndcap, but QIE11DigiCollection will have HB for Run 3.
				rawidHBValid = did.rawId();
			} else if (did.subdet()==HcalEndcap) {
				rawidHEValid = did.rawId();
			}
		}
		HcalElectronicsId const& eid(rawid);

		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}

		// (capid - BX) % 4
		if (_ptype == fOnline) {
			short soi = -1;
			for (int i=0; i<digi.samples(); i++) {
				if (digi[i].soi()) {
					soi = i;
					break;
				}
			}
			short this_capidmbx = (digi[soi].capid() - bx) % 4;
			if (this_capidmbx < 0) {
				this_capidmbx += 4;
			}
			_cCapidMinusBXmod4_SubdetPM.fill(did, this_capidmbx);
			bool good_capidmbx = (_capidmbx[did.subdet()] == this_capidmbx);
			if (!good_capidmbx) {
				_xBadCapid.get(eid)++;
				_cCapid_BadvsFEDvsLS.fill(eid, _currentLS);
				_cCapid_BadvsFEDvsLSmod60.fill(eid, _currentLS % 60);
			}
			if (eid.isVMEid()) {
				_cCapidMinusBXmod4_CrateSlotVME[this_capidmbx].fill(eid);

			} else {
				_cCapidMinusBXmod4_CrateSlotuTCA[this_capidmbx].fill(eid);
			}
		}

		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE11DataFrame>(_dbService, did, digi);
		double sumQ = hcaldqm::utilities::sumQDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples()-1);

		_cSumQ_SubdetPM_QIE1011.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype == fOnline || _ptype == fLocal) {
			_cOccupancy_Crate.fill(eid);
			_cOccupancy_CrateSlot.fill(eid);
		}
		if (_ptype==fOnline)
		{
			_cDigiSizevsLS_FED.fill(eid, _currentLS, digi.samples());
			digi.samples()!=_refDigiSize[did.subdet()]?
				_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		_cDigiSize_Crate.fill(eid, digi.samples());
		if (_ptype != fOffline) { // hidefed2crate
			_cDigiSize_FED.fill(eid, digi.samples());
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
				if (!digi.validate(0, digi.size()))
				{
					_cCapIdRots_depth.fill(did);
					_cCapIdRots_FEDuTCA.fill(eid, 1);
				}*/
			}
		}
		for (int i=0; i<digi.samples(); i++) {
			double q = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE11DataFrame>(_dbService, digi_fC, did, digi, i);
			_cADC_SubdetPM_QIE1011.fill(did, digi[i].adc());
			_cfC_SubdetPM_QIE1011.fill(did, q);
			_cLETDCvsADC_SubdetPM.fill(did, digi[i].adc(), digi[i].tdc());
			_cLETDCvsTS_SubdetPM.fill(did, (int)i, digi[i].tdc());
			if (digi[i].tdc() <50) {
				double time = i*25. + (digi[i].tdc() / 2.);
				_cLETDCTime_SubdetPM.fill(did, time);
				_cLETDCTime_depth.fill(did, time);
				_cLETDCTimevsADC_SubdetPM.fill(did, digi[i].adc(), time);
			}
			// Bad TDC values: 50-61 should never happen in QIE10 or QIE11, but we saw some in 2017 data.
			if ((50 <= digi[i].tdc()) && (digi[i].tdc() <= 61)) {
				_cBadTDCValues_SubdetPM.fill(did, digi[i].tdc());
				_cBadTDCvsBX_SubdetPM.fill(did, bx);
				_cBadTDCvsLS_SubdetPM.fill(did, _currentLS);
				_cBadTDCCount_depth.fill(did);
			}
			if (_ptype != fOffline) { // hidefed2crate
				_cADCvsTS_SubdetPM_QIE1011.fill(did, i, digi[i].adc());
				if (sumQ>_cutSumQ_HE) {
					_cShapeCut_FED.fill(eid, i, q);
				}
			}			
		}

		if (sumQ>_cutSumQ_HE)
		{
			//double timing = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(digi, 2.5, 0,digi.samples()-1);
			double timing = hcaldqm::utilities::aveTSDB<QIE11DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples()-1);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cOccupancyCut_depth.fill(did);
			_cTimingCutvsLS_SubdetPM.fill(did, _currentLS, timing);
			if (_ptype != fOffline) { // hidefed2crate
				_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			}
			_cSumQ_depth.fill(did, sumQ);
			_cSumQvsLS_SubdetPM_QIE1011.fill(did, _currentLS, sumQ);
			if (_ptype==fOnline)
			{
				_cSumQvsBX_SubdetPM_QIE1011.fill(did, bx, sumQ);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
			}
			if (_ptype != fOffline) { // hidefed2crate
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
		//	Explicit check on the DetIds present in the Collection
		HcalDetId const& did = it->id();
		if (did.subdet() != HcalOuter) {
			continue;
		}
		uint32_t rawid = _ehashmap.lookup(did);
		if (rawid == 0) {
			meUnknownIds1LS->Fill(1);
			_unknownIdsPresent = true;
			continue;
		} else {
			rawidValid = did.rawId();
		}
		HcalElectronicsId const& eid(rawid);

		//	filter out channels that are masked out
		if (_xQuality.exists(did)) 
		{
			HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
			if (
				cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
				cs.isBitSet(HcalChannelStatus::HcalCellDead))
				continue;
		}

		if (_ptype == fOnline) {
			short this_capidmbx = (it->sample(it->presamples()).capid() - bx) % 4;
			if (this_capidmbx < 0) {
				this_capidmbx += 4;
			}
			_cCapidMinusBXmod4_SubdetPM.fill(did, this_capidmbx);
			bool good_capidmbx = (_capidmbx[did.subdet()] == this_capidmbx);
			if (!good_capidmbx) {
				_xBadCapid.get(eid)++;
				_cCapid_BadvsFEDvsLS.fill(eid, _currentLS);
				_cCapid_BadvsFEDvsLSmod60.fill(eid, _currentLS % 60);
			}
			if (eid.isVMEid()) {
				_cCapidMinusBXmod4_CrateSlotVME[this_capidmbx].fill(eid);

			} else {
				_cCapidMinusBXmod4_CrateSlotuTCA[this_capidmbx].fill(eid);
			}
		}

		//double sumQ = hcaldqm::utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size()-1);
		CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<HODataFrame>(_dbService, did, *it);
		double sumQ = hcaldqm::utilities::sumQDB<HODataFrame>(_dbService, digi_fC, did, *it, 0, it->size()-1);

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (_ptype==fOnline)
		{
			_cDigiSizevsLS_FED.fill(eid, _currentLS, it->size());
			it->size()!=_refDigiSize[did.subdet()]?
				_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		_cDigiSize_Crate.fill(eid, it->size());
		if (_ptype != fOffline) { // hidefed2crate
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
		}

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (_ptype != fOffline) { // hidefed2crate
				_cADCvsTS_SubdetPM.fill(did, i, it->sample(i).adc());
				if (sumQ>_cutSumQ_HO)
					_cShapeCut_FED.fill(eid, i, it->sample(i).nominal_fC());
			}
		}

		if (sumQ>_cutSumQ_HO)
		{
			//double timing = hcaldqm::utilities::aveTS<HODataFrame>(*it, 8.5, 0,it->size()-1);
			double timing = hcaldqm::utilities::aveTSDB<HODataFrame>(_dbService, digi_fC, did, *it, 0, it->size()-1);
			_cSumQ_depth.fill(did, sumQ);
			_cSumQvsLS_SubdetPM.fill(did, _currentLS, sumQ);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cTimingCutvsLS_SubdetPM.fill(did, _currentLS, timing);
			if (_ptype != fOffline) { // hidefed2crate
				_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			}
			if (_ptype==fOnline)
			{
				_cSumQvsBX_SubdetPM.fill(did, bx, sumQ);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
			}
			if (_ptype != fOffline) { // hidefed2crate
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
	if (_qie10InConditions) {
		for (QIE10DigiCollection::const_iterator it=chf->begin(); it!=chf->end(); ++it) {
			const QIE10DataFrame digi = static_cast<const QIE10DataFrame>(*it);

			//	Explicit check on the DetIds present in the Collection
			HcalDetId const& did = digi.detid();
			if (did.subdet() != HcalForward) {
				// LED monitoring from calibration channels
				if (_ptype != fLocal) {
					if (did.subdet() == HcalOther) {
						HcalOtherDetId hodid(digi.detid());
						if (hodid.subdet() == HcalCalibration) {
							// New method: use configurable list of channels
							if (std::find(_ledCalibrationChannels[HcalForward].begin(), _ledCalibrationChannels[HcalForward].end(), did) != _ledCalibrationChannels[HcalForward].end()) {
								bool channelLEDSignalPresent = false;
								for (int i=0; i<digi.samples(); i++) {
									_LED_ADCvsBX_Subdet.fill(HcalDetId(HcalForward, 16, 1, 1), bx, digi[i].adc());

									if (digi[i].adc() > _thresh_led) {
										channelLEDSignalPresent = true;
									}
								}
								if (channelLEDSignalPresent) {
									_LED_CUCountvsLS_Subdet.fill(HcalDetId(HcalForward, 16, 1, 1), _currentLS);
									if (_ptype == fOnline) { 
										_LED_CUCountvsLSmod60_Subdet.fill(HcalDetId(HcalForward, 16, 1, 1), _currentLS % 60);
									}
								}
							}
						}
					}
				}
				continue;
			}

			uint32_t rawid = _ehashmap.lookup(did);
			if (rawid == 0) {
				meUnknownIds1LS->Fill(1); 
				_unknownIdsPresent=true;
				continue;
			} else {
				rawidValid = did.rawId();
			}
			HcalElectronicsId const& eid(rawid);

			//	filter out channels that are masked out
			if (_xQuality.exists(did)) 
			{
				HcalChannelStatus cs(did.rawId(), _xQuality.get(did));
				if (
					cs.isBitSet(HcalChannelStatus::HcalCellMask) ||
					cs.isBitSet(HcalChannelStatus::HcalCellDead))
					continue;
			}

			// (capid - BX) % 4
			if (_ptype == fOnline) {
				short soi = -1;
				for (int i=0; i<digi.samples(); i++) {
					if (digi[i].soi()) {
						soi = i;
						break;
					}
				}
				short this_capidmbx = (digi[soi].capid() - bx) % 4;
				if (this_capidmbx < 0) {
					this_capidmbx += 4;
				}
				_cCapidMinusBXmod4_SubdetPM.fill(did, this_capidmbx);
				bool good_capidmbx = (_capidmbx[did.subdet()] == this_capidmbx);
				if (!good_capidmbx) {
					_xBadCapid.get(eid)++;
					_cCapid_BadvsFEDvsLS.fill(eid, _currentLS);
					_cCapid_BadvsFEDvsLSmod60.fill(eid, _currentLS % 60);
				}
				if (eid.isVMEid()) {
					_cCapidMinusBXmod4_CrateSlotVME[this_capidmbx].fill(eid);

				} else {
					_cCapidMinusBXmod4_CrateSlotuTCA[this_capidmbx].fill(eid);
				}
			}

			CaloSamples digi_fC = hcaldqm::utilities::loadADC2fCDB<QIE10DataFrame>(_dbService, did, digi);
			double sumQ = hcaldqm::utilities::sumQDB<QIE10DataFrame>(_dbService, digi_fC, did, digi, 0, digi.samples()-1);
			//double sumQ = hcaldqm::utilities::sumQ_v10<QIE10DataFrame>(digi, 2.5, 0, digi.samples()-1);

			//if (!_filter_QIE1011.filter(did)) {
			_cSumQ_SubdetPM_QIE1011.fill(did, sumQ);
			//}
			_cOccupancy_depth.fill(did);
			if (_ptype==fOnline)
			{
				_xNChs.get(eid)++;
				_cDigiSizevsLS_FED.fill(eid, _currentLS, digi.samples());
				digi.samples()!=_refDigiSize[did.subdet()]?
					_xDigiSize.get(eid)++:_xDigiSize.get(eid)+=0;
				_cOccupancyvsiphi_SubdetPM.fill(did);
				_cOccupancyvsieta_Subdet.fill(did);
			}
			_cDigiSize_Crate.fill(eid, digi.samples());
			if (_ptype != fOffline) { // hidefed2crate
				_cDigiSize_FED.fill(eid, digi.samples());
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
			}

			for (int i=0; i<digi.samples(); i++)
			{
				double q = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, digi, i);
				//if (!_filter_QIE1011.filter(did)) {
				_cADC_SubdetPM_QIE1011.fill(did, digi[i].adc());
				_cfC_SubdetPM_QIE1011.fill(did, q);
				_cLETDCvsADC_SubdetPM.fill(did, digi[i].adc(), digi[i].le_tdc());
				_cLETDCvsTS_SubdetPM.fill(did, (int)i, digi[i].le_tdc());
				if (digi[i].le_tdc() <50) {
					double time = i*25. + (digi[i].le_tdc() / 2.);
					_cLETDCTime_SubdetPM.fill(did, time);
					_cLETDCTime_depth.fill(did, time);
					_cLETDCTimevsADC_SubdetPM.fill(did, digi[i].adc(), time);
				}

				// Bad TDC values: 50-61 should never happen in QIE10 or QIE11, but we are seeing some in 2017 data.
				if ((50 <= digi[i].le_tdc()) && (digi[i].le_tdc() <= 61)) {
					_cBadTDCValues_SubdetPM.fill(did, digi[i].le_tdc());
					_cBadTDCvsBX_SubdetPM.fill(did, bx);
					_cBadTDCvsLS_SubdetPM.fill(did, _currentLS);
					_cBadTDCCount_depth.fill(did);
				}
				if (_ptype != fOffline) { // hidefed2crate
					_cADCvsTS_SubdetPM_QIE1011.fill(did, (int)i, digi[i].adc());
					if (sumQ>_cutSumQ_HF)
						_cShapeCut_FED.fill(eid, (int)i, q);
				}
				//}
			}

			if (sumQ>_cutSumQ_HF)
			{
				double timing = hcaldqm::utilities::aveTS_v10<QIE10DataFrame>(digi, 2.5, 0,
					digi.samples()-1);
				double q1 = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, digi, 1);
				double q2 = hcaldqm::utilities::adc2fCDBMinusPedestal<QIE10DataFrame>(_dbService, digi_fC, did, digi, 2);
				double q2q12 = q2/(q1+q2);
				_cSumQ_depth.fill(did, sumQ);
				//if (!_filter_QIE1011.filter(did)) {
				_cSumQvsLS_SubdetPM_QIE1011.fill(did, _currentLS, sumQ);
				//}
				_cTimingCut_SubdetPM.fill(did, timing);
				_cTimingCut_depth.fill(did, timing);
				_cTimingCutvsLS_SubdetPM.fill(did, _currentLS, timing);
				if (_ptype==fOnline)
				{
					//if (!_filter_QIE1011.filter(did)) {
					_cSumQvsBX_SubdetPM_QIE1011.fill(did, bx, sumQ);
					//}
					_cTimingCutvsiphi_SubdetPM.fill(did, timing);
					_cTimingCutvsieta_Subdet.fill(did, timing);
					_cOccupancyCutvsiphi_SubdetPM.fill(did);
					_cOccupancyCutvsieta_Subdet.fill(did);
					_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
	//				_cOccupancyCutvsSlotvsLS_HFPM.fill(did, _currentLS);
					_xUniHF.get(eid)++;
				}
				if (_ptype != fOffline) { // hidefed2crate
					_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
				}
				_cOccupancyCut_depth.fill(did);
				if (!eid.isVMEid())
					if (_ptype==fOnline)
						_cQ2Q12CutvsLS_FEDHF.fill(eid, _currentLS, q2q12);
				if (_ptype != fOffline) { // hidefed2crate
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
				}
				numChsCut++;
			}
			numChs++;
		}
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
	if (_ptype == fOnline) {
		// Reset the bin for _cCapid_BadvsFEDvsLSmod60
		for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
				it!=_vhashFEDs.end(); ++it) {
			HcalElectronicsId eid = HcalElectronicsId(*it);
			_cCapid_BadvsFEDvsLSmod60.setBinContent(eid, _currentLS % 50, 0);
		}	
	}
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

	if (_ptype != fOffline) { // hidefed2crate
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

				if (_xBadCapid.get(eid) > 0) {
					_vflags[fCapId]._state = hcaldqm::flag::fBAD;
				} else {
					_vflags[fCapId]._state = hcaldqm::flag::fGOOD;
				}

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

			// LED misfires
			if (_ptype != fLocal) {
				if (hcaldqm::utilities::isFEDHBHE(eid)) {
					HcalDetId did_hb(hcaldqm::hashfunctions::hash_Subdet(HcalDetId(HcalBarrel, 1, 1, 1)));
					HcalDetId did_he(hcaldqm::hashfunctions::hash_Subdet(HcalDetId(HcalEndcap, 16, 1, 1)));

					if (_LED_CUCountvsLS_Subdet.getBinContent(did_hb, _currentLS) > 0 || _LED_CUCountvsLS_Subdet.getBinContent(did_he, _currentLS) > 0) {
						_vflags[fLED]._state = hcaldqm::flag::fBAD;
					} else {
						_vflags[fLED]._state = hcaldqm::flag::fGOOD;
					}
				} else if (hcaldqm::utilities::isFEDHF(eid)) {
					HcalDetId did_hf(hcaldqm::hashfunctions::hash_Subdet(HcalDetId(HcalForward, 29, 1, 1)));
					if (_LED_CUCountvsLS_Subdet.getBinContent(did_hf, _currentLS) > 0) {
						_vflags[fLED]._state = hcaldqm::flag::fBAD;
					} else {
						_vflags[fLED]._state = hcaldqm::flag::fGOOD;
					}
				} else if (hcaldqm::utilities::isFEDHO(eid)) {
					HcalDetId did_ho(hcaldqm::hashfunctions::hash_Subdet(HcalDetId(HcalOuter, 1, 1, 1)));
					if (_LED_CUCountvsLS_Subdet.getBinContent(did_ho, _currentLS) > 0) {
						_vflags[fLED]._state = hcaldqm::flag::fBAD;
					} else {
						_vflags[fLED]._state = hcaldqm::flag::fGOOD;
					}
				}
			}

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
	}

	_xDigiSize.reset(); _xUniHF.reset(); _xUni.reset(); 
	_xNChs.reset();
	_xBadCapid.reset();

	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiTask);

