#include "DQM/HcalTasks/interface/TPTask.h"

TPTask::TPTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_tagData = ps.getUntrackedParameter<edm::InputTag>("tagData",
		edm::InputTag("hcalDigis"));
	_tagEmul = ps.getUntrackedParameter<edm::InputTag>("tagEmul",
		edm::InputTag("emulDigis"));

	_tokData = consumes<HcalTrigPrimDigiCollection>(_tagData);
	_tokEmul = consumes<HcalTrigPrimDigiCollection>(_tagEmul);

	_skip1x1 = ps.getUntrackedParameter<bool>("skip1x1", true);
	_cutEt = ps.getUntrackedParameter<int>("cutEt", 3);
	_thresh_EtMsmRate = ps.getUntrackedParameter<double>("thresh_EtMsmRate",
		0.1);
	_thresh_FGMsmRate = ps.getUntrackedParameter<double>("thresh_FGMsmRate",
		0.1);
	_thresh_DataMsn = ps.getUntrackedParameter<double>("thresh_DataMsn",
		0.1);
	_thresh_EmulMsn = ps.getUntrackedParameter<double>("thresh_EmulMsn");

	_vflags.resize(nTPFlag);
	_vflags[fEtMsm]=flag::Flag("EtMsm");
	_vflags[fFGMsm]=flag::Flag("FGMsm");
	_vflags[fDataMsn]=flag::Flag("DataMsn");
	_vflags[fEmulMsn]=flag::Flag("EmulMsn");
}

/* virtual */ void TPTask::bookHistograms(DQMStore::IBooker& ib,
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
	std::vector<uint32_t> depth0;
	vVME.push_back(HcalElectronicsId(FIBERCH_MIN, 
		FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
		vVME);
	_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
		vuTCA);
	depth0.push_back(HcalTrigTowerDetId(1, 1, 0).rawId());
	_filter_depth0.initialize(filter::fPreserver, hashfunctions::fTTdepth,
		depth0);

	//	push the rawIds of each fed into the vector
	//	this vector is used at endlumi for online state generation
	for (std::vector<int>::const_iterator it=vFEDsVME.begin();
		it!=vFEDsVME.end(); ++it)
	{
		_vhashFEDs.push_back(HcalElectronicsId(FIBERCH_MIN, FIBER_VME_MIN,
			SPIGOT_MIN, (*it)-FED_VME_MIN).rawId());
	}
	for (std::vector<int>::const_iterator it=vFEDsuTCA.begin();
		it!=vFEDsuTCA.end(); ++it)
	{
		_vhashFEDs.push_back(HcalElectronicsId(utilities::fed2crate(*it), 
			SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	}

	//	INITIALIZE FIRST
	//	Et/FG
	_cEtData_TTSubdet.initialize(_name, "EtData", hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fEt_128),
		new quantity::ValueQuantity(quantity::fN, true));
	_cEtEmul_TTSubdet.initialize(_name, "EtEmul", hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fEt_128),
		new quantity::ValueQuantity(quantity::fN, true));
	_cEtCorr_TTSubdet.initialize(_name, "EtCorr", hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fEtCorr_256),
		new quantity::ValueQuantity(quantity::fEtCorr_256),
		new quantity::ValueQuantity(quantity::fN, true));
	_cFGCorr_TTSubdet.initialize(_name, "FGCorr", hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fFG),
		new quantity::ValueQuantity(quantity::fFG),
		new quantity::ValueQuantity(quantity::fN, true));

	_cEtData_ElectronicsVME.initialize(_name, "EtData", 
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtData_ElectronicsuTCA.initialize(_name, "EtData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtEmul_ElectronicsVME.initialize(_name, "EtEmul", 
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtEmul_ElectronicsuTCA.initialize(_name, "EtEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtData_depthlike.initialize(_name, "EtData",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtEmul_depthlike.initialize(_name, "EtEmul",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtCutData_depthlike.initialize(_name, "EtCutData",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fEt_256));
	_cEtCutEmul_depthlike.initialize(_name, "EtCutEmul",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fEt_256));

	//	Occupancies
	_cOccupancyData_ElectronicsVME.initialize(_name, "OccupancyData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyEmul_ElectronicsVME.initialize(_name, "OccupancyEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyData_ElectronicsuTCA.initialize(_name, "OccupancyData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyEmul_ElectronicsuTCA.initialize(_name, "OccupancyEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN, true));

	_cOccupancyCutData_ElectronicsVME.initialize(_name, "OccupancyCutData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyCutEmul_ElectronicsVME.initialize(_name, "OccupancyCutEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyCutData_ElectronicsuTCA.initialize(_name, "OccupancyCutData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyCutEmul_ElectronicsuTCA.initialize(_name, "OccupancyCutEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN, true));

	_cOccupancyData_depthlike.initialize(_name, "OccupancyData",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyEmul_depthlike.initialize(_name, "OccupancyEmul",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyCutData_depthlike.initialize(_name, "OccupancyCutData",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN, true));
	_cOccupancyCutEmul_depthlike.initialize(_name, "OccupancyCutEmul",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN, true));

	//	Mismatches
	_cEtMsm_ElectronicsVME.initialize(_name, "EtMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cFGMsm_ElectronicsVME.initialize(_name, "FGMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cEtMsm_ElectronicsuTCA.initialize(_name, "EtMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cFGMsm_ElectronicsuTCA.initialize(_name, "FGMsm",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cEtMsm_depthlike.initialize(_name, "EtMsm",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cFGMsm_depthlike.initialize(_name, "FGMsm",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN));

	//	Missing Data w.r.t. Emulator
	_cMsnData_ElectronicsVME.initialize(_name, "MsnData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cMsnData_ElectronicsuTCA.initialize(_name, "MsnData",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cMsnEmul_ElectronicsVME.initialize(_name, "MsnEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cMsnEmul_ElectronicsuTCA.initialize(_name, "MsnEmul",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cMsnData_depthlike.initialize(_name, "MsnData",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMsnEmul_depthlike.initialize(_name, "MsnEmul",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cEtCorrRatio_ElectronicsVME.initialize(_name, "EtCorrRatio",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fRatio_0to2));
	_cEtCorrRatio_ElectronicsuTCA.initialize(_name, "EtCorrRatio",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fRatio_0to2));
	_cEtCorrRatio_depthlike.initialize(_name, "EtCorrRatio",
		new quantity::TrigTowerQuantity(quantity::fTTieta),
		new quantity::TrigTowerQuantity(quantity::fTTiphi),
		new quantity::ValueQuantity(quantity::fRatio_0to2));

	_cOccupancyDatavsBX_TTSubdet.initialize(_name, "OccupancyDatavsBX",
		hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fBX),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyEmulvsBX_TTSubdet.initialize(_name, "OccupancyEmulvsBX",
		hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fBX),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCutDatavsBX_TTSubdet.initialize(_name, "OccupancyCutDatavsBX",
		hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fBX),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCutEmulvsBX_TTSubdet.initialize(_name, "OccupancyCutEmulvsBX",
		hashfunctions::fTTSubdet,
		new quantity::ValueQuantity(quantity::fBX),
		new quantity::ValueQuantity(quantity::fN));

	//	INITIALIZE HISTOGRAMS to be used in Online only!
	if (_ptype==fOnline)
	{
		_cEtCorr2x3_TTSubdet.initialize(_name, "EtCorr2x3", 
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fEtCorr_256),
			new quantity::ValueQuantity(quantity::fEtCorr_256),
			new quantity::ValueQuantity(quantity::fN, true));
		_cOccupancyData2x3_depthlike.initialize(_name, "OccupancyData2x3",
			new quantity::TrigTowerQuantity(quantity::fTTieta2x3),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fN, true));
		_cOccupancyEmul2x3_depthlike.initialize(_name, "OccupancyEmul2x3",
			new quantity::TrigTowerQuantity(quantity::fTTieta2x3),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fN, true));
		_cEtCutDatavsLS_TTSubdet.initialize(_name, "EtCutDatavsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fEtCorr_256));
		_cEtCutEmulvsLS_TTSubdet.initialize(_name, "EtCutEmulvsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fEtCorr_256));
		_cEtCutDatavsBX_TTSubdet.initialize(_name, "EtCutDatavsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fEtCorr_256));
		_cEtCutEmulvsBX_TTSubdet.initialize(_name, "EtCutEmulvsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fEtCorr_256));
		_cEtCorrRatiovsLS_TTSubdet.initialize(_name, "EtCorrRatiovsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fRatio_0to2));
		_cEtCorrRatiovsBX_TTSubdet.initialize(_name, "EtCorrRatiovsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fRatio_0to2));
		_cEtMsmRatiovsLS_TTSubdet.initialize(_name, "EtMsmRatiovsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fRatio));
		_cEtMsmRatiovsBX_TTSubdet.initialize(_name, "EtMsmRatiovsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fRatio));
		_cEtMsmvsLS_TTSubdet.initialize(_name, "EtMsmvsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cEtMsmvsBX_TTSubdet.initialize(_name, "EtMsmvsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnDatavsLS_TTSubdet.initialize(_name, "MsnDatavsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnCutDatavsLS_TTSubdet.initialize(_name, "MsnCutDatavsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnDatavsBX_TTSubdet.initialize(_name, "MsnDatavsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnCutDatavsBX_TTSubdet.initialize(_name, "MsnCutDatavsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnEmulvsLS_TTSubdet.initialize(_name, "MsnEmulvsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnCutEmulvsLS_TTSubdet.initialize(_name, "MsnCutEmulvsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnEmulvsBX_TTSubdet.initialize(_name, "MsnEmulvsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cMsnCutEmulvsBX_TTSubdet.initialize(_name, "MsnCutEmulvsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyDatavsLS_TTSubdet.initialize(_name, "OccupancyDatavsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutDatavsLS_TTSubdet.initialize(_name, 
			"OccupancyCutDatavsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyDatavsBX_TTSubdet.initialize(_name, "OccupancyDatavsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutDatavsBX_TTSubdet.initialize(_name, 
			"OccupancyCutDatavsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyEmulvsLS_TTSubdet.initialize(_name, "OccupancyEmulvsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutEmulvsLS_TTSubdet.initialize(_name, 
			"OccupancyCutEmulvsLS",
			hashfunctions::fTTSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyEmulvsBX_TTSubdet.initialize(_name, "OccupancyEmulvsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutEmulvsBX_TTSubdet.initialize(_name, 
			"OccupancyCutEmulvsBX",
			hashfunctions::fTTSubdet,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cSummaryvsLS_FED.initialize(_name, "SummaryvsLS",
			hashfunctions::fFED,
			new quantity::LumiSection(_maxLS),
			new quantity::FlagQuantity(_vflags),
			new quantity::ValueQuantity(quantity::fState));
		_cSummaryvsLS.initialize(_name, "SummaryvsLS",
			new quantity::LumiSection(_maxLS),
			new quantity::FEDQuantity(vFEDs),
			new quantity::ValueQuantity(quantity::fState));

		_xEtMsm.initialize(hashfunctions::fFED);
		_xFGMsm.initialize(hashfunctions::fFED);
		_xNumCorr.initialize(hashfunctions::fFED);
		_xDataMsn.initialize(hashfunctions::fFED);
		_xDataTotal.initialize(hashfunctions::fFED);
		_xEmulMsn.initialize(hashfunctions::fFED);
		_xEmulTotal.initialize(hashfunctions::fFED);
	}

	//	BOOK HISTOGRAMS
	_cEtData_TTSubdet.book(ib, _emap, _subsystem);
	_cEtEmul_TTSubdet.book(ib, _emap, _subsystem);
	_cEtCorr_TTSubdet.book(ib, _emap, _subsystem);
	_cFGCorr_TTSubdet.book(ib, _emap, _subsystem);
	_cEtData_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cEtData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cEtEmul_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cEtEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cEtData_depthlike.book(ib, _subsystem);
	_cEtEmul_depthlike.book(ib, _subsystem);
	_cEtCutData_depthlike.book(ib, _subsystem);
	_cEtCutEmul_depthlike.book(ib, _subsystem);
	_cOccupancyData_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyEmul_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCutData_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCutEmul_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCutData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCutEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyData_depthlike.book(ib, _subsystem);
	_cOccupancyEmul_depthlike.book(ib, _subsystem);
	_cOccupancyCutData_depthlike.book(ib, _subsystem);
	_cOccupancyCutEmul_depthlike.book(ib, _subsystem);
	_cEtMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cEtMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cFGMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cFGMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMsnData_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMsnData_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMsnEmul_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMsnEmul_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cEtCorrRatio_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cEtCorrRatio_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cEtCorrRatio_depthlike.book(ib, _subsystem);
	_cEtMsm_depthlike.book(ib, _subsystem);
	_cFGMsm_depthlike.book(ib, _subsystem);
	_cMsnData_depthlike.book(ib, _subsystem);
	_cMsnEmul_depthlike.book(ib, _subsystem);

	//	whatever has to go online only goes here
	if (_ptype==fOnline)
	{
		_cEtCorr2x3_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyData2x3_depthlike.book(ib, _subsystem);
		_cOccupancyEmul2x3_depthlike.book(ib, _subsystem);
		_cEtCutDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cEtCutEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cEtCutDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cEtCutEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cEtCorrRatiovsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cEtCorrRatiovsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cEtMsmvsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cEtMsmvsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cEtMsmRatiovsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cEtMsmRatiovsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnCutDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnCutDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnCutEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cMsnCutEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyCutDatavsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyCutEmulvsBX_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyCutDatavsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cOccupancyCutEmulvsLS_TTSubdet.book(ib, _emap, _subsystem);
		_cSummaryvsLS_FED.book(ib, _emap, _subsystem);
		_cSummaryvsLS.book(ib, _subsystem);

		_xEtMsm.book(_emap);
		_xFGMsm.book(_emap);
		_xNumCorr.book(_emap);
		_xDataMsn.book(_emap);
		_xDataTotal.book(_emap);
		_xEmulMsn.book(_emap);
		_xEmulTotal.book(_emap);
	}
	
	//	initialize the hash map
	_ehashmap.initialize(_emap, hcaldqm::electronicsmap::fT2EHashMap);
}

/* virtual */ void TPTask::_resetMonitors(UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);
}

/* virtual */ void TPTask::_process(edm::Event const& e,
	edm::EventSetup const&)
{
	edm::Handle<HcalTrigPrimDigiCollection> cdata;
	edm::Handle<HcalTrigPrimDigiCollection> cemul;
	if (!e.getByToken(_tokData, cdata))
		_logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available"
			+ _tagData.label() + " " + _tagData.instance());
	if (!e.getByToken(_tokEmul, cemul))
		_logger.dqmthrow("Collection HcalTrigPrimDigiCollection isn't available"
			+ _tagEmul.label() + " " + _tagEmul.instance());

	//	extract some info per event
	int bx = e.bunchCrossing();

	//	some summaries... per event
	int numHBHE(0), numHF(0), numCutHBHE(0), numCutHF(0);
	int numCorrHBHE(0), numCorrHF(0);
	int numMsmHBHE(0), numMsmHF(0);
	int numMsnHBHE(0), numMsnHF(0), numMsnCutHBHE(0), numMsnCutHF(0);

	/*
	 * STEP1: 
	 * Loop over the data digis and 
	 * - do ... for all the data digis
	 * - find the emulator digi
	 * --- compare soi Et
	 * --- compare soi FG
	 * --- Do not fill anything for emulator Et!!!
	 */
	for (HcalTrigPrimDigiCollection::const_iterator it=cdata->begin();
		it!=cdata->end(); ++it)
	{
		HcalTrigTowerDetId tid = it->id();

		//
		//	HF 2x3 TPs Treat theam separately and only for ONLINE!
		//
		if (tid.version()==0 && tid.ietaAbs()>=29)
		{
			//	do this only for online processing
			if (_ptype==fOnline)
			{
				_cOccupancyData2x3_depthlike.fill(tid);
				HcalTrigPrimDigiCollection::const_iterator jt=cemul->find(tid);
				if (jt!=cemul->end())
					_cEtCorr2x3_TTSubdet.fill(tid, it->SOI_compressedEt(),
						jt->SOI_compressedEt());
			}

			//	skip to the next tp digi
			continue;
		}

		//	FROM THIS POINT, HBHE + 1x1 HF TPs
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(tid));
		int soiEt_d = it->SOI_compressedEt();
		int soiFG_d = it->SOI_fineGrain()?1:0;
		tid.ietaAbs()>=29?numHF++:numHBHE++;

		//	 fill w/o a cut
		_cEtData_TTSubdet.fill(tid, soiEt_d);
		_cEtData_depthlike.fill(tid, soiEt_d);
		_cOccupancyData_depthlike.fill(tid);
		if (eid.isVMEid())
		{
			_cOccupancyData_ElectronicsVME.fill(eid);
			_cEtData_ElectronicsVME.fill(eid, soiEt_d);
		}
		else
		{
			_cOccupancyData_ElectronicsuTCA.fill(eid);
			_cEtData_ElectronicsuTCA.fill(eid, soiEt_d);
		}
		
		//	FILL w/a CUT
		if (soiEt_d>_cutEt)
		{
			tid.ietaAbs()>=29?numCutHF++:numCutHBHE++;
			_cOccupancyCutData_depthlike.fill(tid);
			_cEtCutData_depthlike.fill(tid, soiEt_d);

			//	ONLINE ONLY!
			if (_ptype==fOnline)
			{
				_cEtCutDatavsLS_TTSubdet.fill(tid, _currentLS, soiEt_d);
				_cEtCutDatavsBX_TTSubdet.fill(tid, bx, soiEt_d);
				_xDataTotal.get(eid)++;
			}
			//	^^^ONLINE ONLY!

			if (eid.isVMEid())
				_cOccupancyCutData_ElectronicsVME.fill(eid);
			else
				_cOccupancyCutData_ElectronicsuTCA.fill(eid);
		}

		//	FIND the EMULATOR DIGI
		HcalTrigPrimDigiCollection::const_iterator jt=cemul->find(tid);
		if (jt!=cemul->end())
		{
			//	if PRESENT!
			int soiEt_e = jt->SOI_compressedEt();
			int soiFG_e = jt->SOI_fineGrain()?1:0;
			//	if both are zeroes => set 1
			double rEt = soiEt_d==0 && soiEt_e==0?1:
				double(std::min(soiEt_d, soiEt_e))/
				double(std::max(soiEt_e, soiEt_d));

			//	ONLINE ONLY!
			if (_ptype==fOnline)
			{		
				_xNumCorr.get(eid)++;
				tid.ietaAbs()>=29?numCorrHF++:numCorrHBHE++;
				_cEtCorrRatiovsLS_TTSubdet.fill(tid, _currentLS, rEt);
				_cEtCorrRatiovsBX_TTSubdet.fill(tid, bx, rEt);
			}
			//	^^^ONLINE ONLY!

			_cEtCorrRatio_depthlike.fill(tid, rEt);
			_cEtCorr_TTSubdet.fill(tid, soiEt_d, soiEt_e);
			_cFGCorr_TTSubdet.fill(tid, soiFG_d, soiFG_e);
			//	FILL w/o a CUT
			if (eid.isVMEid())
			{
				_cEtCorrRatio_ElectronicsVME.fill(eid, rEt);
			}
			else
			{
				_cEtCorrRatio_ElectronicsuTCA.fill(eid, rEt);
			}

			//	if SOI Et are not equal
			//	fill mismatched
			if (soiEt_d!=soiEt_e)
			{
				tid.ietaAbs()>=29?numMsmHF++:numMsmHBHE++;
				_cEtMsm_depthlike.fill(tid);
				if (eid.isVMEid())
					_cEtMsm_ElectronicsVME.fill(eid);
				else
					_cEtMsm_ElectronicsuTCA.fill(eid);
				if (_ptype==fOnline)
					_xEtMsm.get(eid)++;
			}
			//	 if SOI FG are not equal
			//	 fill mismatched
			if (soiFG_d!=soiFG_e)
			{
				_cFGMsm_depthlike.fill(tid);
				if (eid.isVMEid())
					_cFGMsm_ElectronicsVME.fill(eid);
				else
					_cFGMsm_ElectronicsuTCA.fill(eid);
				if (_ptype==fOnline)
					_xFGMsm.get(eid)++;
			}
		}
		else
		{
			//	IF MISSING
			_cEtCorr_TTSubdet.fill(tid, soiEt_d, -2);
			_cMsnEmul_depthlike.fill(tid);
			tid.ietaAbs()>=29?numMsnHF++:numMsnHBHE++;
			if (eid.isVMEid())
				_cMsnEmul_ElectronicsVME.fill(eid);
			else
				_cMsnEmul_ElectronicsuTCA.fill(eid);

			if (soiEt_d>_cutEt)
			{
				tid.ietaAbs()>=29?numMsnCutHF++:numMsnCutHBHE++;
				if (_ptype==fOnline)
					_xEmulMsn.get(eid)++;
			}
		}
	}
	
	//	ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_cOccupancyDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1), bx,
			numHBHE);
		_cOccupancyDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1), bx,
			numHF);
		_cOccupancyCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1), bx,
			numCutHBHE);
		_cOccupancyCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1), bx,
			numCutHF);
		_cOccupancyDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1), 
			_currentLS, numHBHE);
		_cOccupancyDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1), 
			_currentLS,numHF);
		_cOccupancyCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			_currentLS, numCutHBHE);
		_cOccupancyCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1), 
			_currentLS, numCutHF);

		_cEtMsmvsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1), _currentLS,
			numMsmHBHE);
		_cEtMsmvsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1), _currentLS, 
			numMsmHF);
		_cEtMsmvsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1), bx,
			numMsmHBHE);
		_cEtMsmvsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1), bx, 
			numMsmHF);
		
		_cEtMsmRatiovsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1), _currentLS,
			double(numMsmHBHE)/double(numCorrHBHE));
		_cEtMsmRatiovsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1), _currentLS, 
			double(numMsmHF)/double(numCorrHF));
		_cEtMsmRatiovsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1), bx,
			double(numMsmHBHE)/double(numCorrHBHE));
		_cEtMsmRatiovsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1), bx, 
			double(numMsmHF)/double(numCorrHF));

		_cMsnEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			_currentLS, numMsnHBHE);
		_cMsnEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			_currentLS, numMsnHF);
		_cMsnCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			_currentLS, numMsnCutHBHE);
		_cMsnCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			_currentLS, numMsnCutHF);

		_cMsnEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			bx, numMsnHBHE);
		_cMsnEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			bx, numMsnHF);
		_cMsnCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			bx, numMsnCutHBHE);
		_cMsnCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			bx, numMsnCutHF);
	}

	numHBHE=0; numHF=0; numCutHBHE=0; numCutHF=0;
	numMsnHBHE=0; numMsnHF=0; numCutHBHE=0; numCutHF=0;

	/*
	 *	STEP2:
	 *	Loop over the emulator digis and 
	 *	- do ... for all the emulator digis
	 *	- find data digi and 
	 *	--- if found skip
	 *	--- if not found - fill the missing Data plot
	 */
	for (HcalTrigPrimDigiCollection::const_iterator it=cemul->begin();
		it!=cemul->end(); ++it)
	{
		HcalTrigTowerDetId tid = it->id();

		//	HF 2x3 TPs. Only do it for Online!!!
		if (tid.version()==0 && tid.ietaAbs()>=29)
		{
			//	only do this for online processing
			if (_ptype==fOnline)
				_cOccupancyEmul2x3_depthlike.fill(tid);
			continue;
		}
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(tid));
		int soiEt = it->SOI_compressedEt();

		//	FILL/INCREMENT w/o a CUT
		tid.ietaAbs()>=29?numHF++:numHBHE++;
		_cEtEmul_TTSubdet.fill(tid, soiEt);
		_cEtEmul_depthlike.fill(tid, soiEt);
		_cOccupancyEmul_depthlike.fill(tid);
		if (eid.isVMEid())
		{
			_cOccupancyEmul_ElectronicsVME.fill(eid);
			_cEtEmul_ElectronicsVME.fill(eid, soiEt);
		}
		else
		{
			_cOccupancyEmul_ElectronicsuTCA.fill(eid);
			_cEtEmul_ElectronicsuTCA.fill(eid, soiEt);
		}

		//	FILL w/ a CUT
		if (soiEt>_cutEt)
		{
			tid.ietaAbs()>=29?numCutHF++:numCutHBHE++;
			_cOccupancyCutEmul_depthlike.fill(tid);
			_cEtCutEmul_depthlike.fill(tid, soiEt);
			if (eid.isVMEid())
				_cOccupancyCutEmul_ElectronicsVME.fill(eid);
			else 
				_cOccupancyCutEmul_ElectronicsuTCA.fill(eid);

			//	ONLINE ONLY!
			if (_ptype==fOnline)
			{
				_cEtCutEmulvsLS_TTSubdet.fill(tid, _currentLS, soiEt);
				_cEtCutEmulvsBX_TTSubdet.fill(tid, bx, soiEt);
				_xEmulTotal.get(eid)++;
			}
			//	^^^ONLINE ONLY!
		}

		//	FIND a data digi
		HcalTrigPrimDigiCollection::const_iterator jt=cdata->find(tid);
		if (jt==cdata->end())
		{
			tid.ietaAbs()>=29?numMsnHF++:numMsnHBHE++;
			_cEtCorr_TTSubdet.fill(tid, -2, soiEt);
			if (eid.isVMEid())
				_cMsnData_ElectronicsVME.fill(eid);
			else
				_cMsnData_ElectronicsuTCA.fill(eid);
			if (soiEt>_cutEt)
			{
				tid.ietaAbs()>=29?numMsnCutHF++:numMsnCutHBHE++;
				if (_ptype==fOnline)
					_xDataMsn.get(eid)++;
			}
		}
	}

	//	ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_cOccupancyEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1), bx,
			numHBHE);
		_cOccupancyEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1), bx,
			numHF);
		_cOccupancyCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1), bx,
			numCutHBHE);
		_cOccupancyCutEmulvsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1), bx,
			numCutHF);

		_cOccupancyEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1), 
			_currentLS, numHBHE);
		_cOccupancyEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1), 
			_currentLS,numHF);
		_cOccupancyCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			_currentLS, numCutHBHE);
		_cOccupancyCutEmulvsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1), 
			_currentLS, numCutHF);

		_cMsnDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			_currentLS, numMsnHBHE);
		_cMsnDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			_currentLS, numMsnHF);
		_cMsnCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			_currentLS, numMsnCutHBHE);
		_cMsnCutDatavsLS_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			_currentLS, numMsnCutHF);

		_cMsnDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			bx, numMsnHBHE);
		_cMsnDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			bx, numMsnHF);
		_cMsnCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(1,1),
			bx, numMsnCutHBHE);
		_cMsnCutDatavsBX_TTSubdet.fill(HcalTrigTowerDetId(29,1),
			bx, numMsnCutHF);
	}
	//	^^^ONLINE ONLY!
}

/* virtual */ void TPTask::beginLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	DQTask::beginLuminosityBlock(lb, es);

	/*
	//	ONLINE ONLY!
	if (_ptype!=fOnline)
		return;
	_cEtCutDatavsLS_TTSubdet.extendAxisRange(_currentLS);
	_cEtCutEmulvsLS_TTSubdet.extendAxisRange(_currentLS);
	_cEtCorrRatiovsLS_TTSubdet.extendAxisRange(_currentLS);
	_cEtMsmvsLS_TTSubdet.extendAxisRange(_currentLS);
	_cEtMsmRatiovsLS_TTSubdet.extendAxisRange(_currentLS);
	_cMsnDatavsLS_TTSubdet.extendAxisRange(_currentLS);
	_cMsnCutDatavsLS_TTSubdet.extendAxisRange(_currentLS);
	_cMsnEmulvsLS_TTSubdet.extendAxisRange(_currentLS);
	_cMsnCutEmulvsLS_TTSubdet.extendAxisRange(_currentLS);
	_cOccupancyDatavsLS_TTSubdet.extendAxisRange(_currentLS);
	_cOccupancyEmulvsLS_TTSubdet.extendAxisRange(_currentLS);
	_cOccupancyCutDatavsLS_TTSubdet.extendAxisRange(_currentLS);
	_cOccupancyCutEmulvsLS_TTSubdet.extendAxisRange(_currentLS);
//	_cSummaryvsLS_FED.extendAxisRange(_currentLS);
//	_cSummaryvsLS.extendAxisRange(_currentLS);
//	*/
}

/* virtual */ void TPTask::endLuminosityBlock(edm::LuminosityBlock const& lb,
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
		flag::Flag fSum("TP");
		HcalElectronicsId eid = HcalElectronicsId(*it);

		std::vector<uint32_t>::const_iterator cit=std::find(
			_vcdaqEids.begin(), _vcdaqEids.end(), *it);
		if (cit==_vcdaqEids.end())
		{
			//	not @cDAQ
			for (uint32_t iflag=0; iflag<_vflags.size(); iflag++)
				_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
					int(flag::fNCDAQ));
			_cSummaryvsLS.setBinContent(eid, _currentLS, int(flag::fNCDAQ));
			continue;
		}

		if (utilities::isFEDHBHE(eid) || utilities::isFEDHF(eid))
		{
			//	FED is @cDAQ
			double etmsm = _xNumCorr.get(eid)>0?
				double(_xEtMsm.get(eid))/double(_xNumCorr.get(eid)):0;
			double fgmsm = _xNumCorr.get(eid)>0?
				double(_xFGMsm.get(eid))/double(_xNumCorr.get(eid)):0;
			/*	
			 * UNUSED VARS
			 * double dmsm = _xDataTotal.get(eid)>0?
				double(_xDataMsn.get(eid))/double(_xDataTotal.get(eid)):0;
			double emsm = _xEmulTotal.get(eid)>0?
				double(_xEmulMsn.get(eid))/double(_xEmulTotal.get(eid)):0;
				*/
			if (etmsm>=_thresh_EtMsmRate)
				_vflags[fEtMsm]._state = flag::fBAD;
			else
				_vflags[fEtMsm]._state = flag::fGOOD;
			if (fgmsm>=_thresh_FGMsmRate)
				_vflags[fFGMsm]._state = flag::fBAD;
			else
				_vflags[fFGMsm]._state = flag::fGOOD;
			/*
			 *	DISABLE THESE FLAGS FOR ONLINE FOR NOW!
			if (dmsm>=_thresh_DataMsn)
				_vflags[fDataMsn]._state = flag::fBAD;
			else
				_vflags[fDataMsn]._state = flag::fGOOD;
			if (emsm>=_thresh_EmulMsn)
				_vflags[fEmulMsn]._state = flag::fBAD;
			else
				_vflags[fEmulMsn]._state = flag::fGOOD;
				*/
		}

		int iflag=0;
		for (std::vector<flag::Flag>::iterator ft=_vflags.begin();
			ft!=_vflags.end(); ++ft)
		{
			_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
				ft->_state);
			fSum+=(*ft);
			iflag++;

			//	this is the MUST!
			//	reset after using this flag
			ft->reset();
		}
		_cSummaryvsLS.setBinContent(eid, _currentLS, int(fSum._state));
	}

	//	reset...
	_xEtMsm.reset(); _xFGMsm.reset(); _xNumCorr.reset();
	_xDataMsn.reset(); _xDataTotal.reset(); _xEmulMsn.reset(); 
	_xEmulTotal.reset();
	
	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(TPTask);

