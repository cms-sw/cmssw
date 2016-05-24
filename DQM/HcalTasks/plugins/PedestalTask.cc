
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

	_vflags.resize(nPedestalFlag);
	_vflags[fMsn]=flag::Flag("Msn");
	_vflags[fBadM]=flag::Flag("BadM");
	_vflags[fBadR]=flag::Flag("BadR");

	_thresh_mean = ps.getUntrackedParameter<double>("thresh_mean",
		0.25);
	_thresh_rms = ps.getUntrackedParameter<double>("thresh_mean",
		0.25);
	_thresh_badm = ps.getUntrackedParameter<double>("thresh_badm", 0.1);
	_thresh_badr = ps.getUntrackedParameter<double>("thresh_badr", 0.1);
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
	std::vector<int> vFEDs = utilities::getFEDList(_emap);
	std::vector<int> vFEDsVME = utilities::getFEDVMEList(_emap);
	std::vector<int> vFEDsuTCA = utilities::getFEDuTCAList(_emap);
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

	for (std::vector<int>::const_iterator it=vFEDsVME.begin();
		it!=vFEDsVME.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(constants::FIBERCH_MIN,
			FIBER_VME_MIN, SPIGOT_MIN, (*it)-FED_VME_MIN).rawId());
	for (std::vector<int>::const_iterator it=vFEDsuTCA.begin();
		it!=vFEDsuTCA.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(
			utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
			FIBERCH_MIN, false).rawId());

	//	Containers XXX
	_xPedSum1LS.initialize(hashfunctions::fDChannel);
	_xPedSum21LS.initialize(hashfunctions::fDChannel);
	_xPedEntries1LS.initialize(hashfunctions::fDChannel);
	_xPedSumTotal.initialize(hashfunctions::fDChannel);
	_xPedSum2Total.initialize(hashfunctions::fDChannel);
	_xPedEntriesTotal.initialize(hashfunctions::fDChannel);

	_xPedRefMean.initialize(hashfunctions::fDChannel);
	_xPedRefRMS.initialize(hashfunctions::fDChannel);

	_xNChs.initialize(hashfunctions::fFED);
	_xNMsn1LS.initialize(hashfunctions::fFED);
	_xNBadMean1LS.initialize(hashfunctions::fFED);
	_xNBadRMS1LS.initialize(hashfunctions::fFED);

	//	Containers
	_cMean1LS_Subdet.initialize(_name, "Mean",hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fADC_15),
		new quantity::ValueQuantity(quantity::fN, true));
	_cRMS1LS_Subdet.initialize(_name, "RMS", hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fADC_5),
		new quantity::ValueQuantity(quantity::fN, true));
	_cMean1LS_depth.initialize(_name, "Mean", hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cRMS1LS_depth.initialize(_name, "RMS", hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cMean1LS_FEDVME.initialize(_name, "Mean", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cMean1LS_FEDuTCA.initialize(_name, "Mean", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cRMS1LS_FEDVME.initialize(_name, "RMS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMS1LS_FEDuTCA.initialize(_name, "RMS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	
	_cMeanTotal_Subdet.initialize(_name, "Mean",hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fADC_15),
		new quantity::ValueQuantity(quantity::fN, true));
	_cRMSTotal_Subdet.initialize(_name, "RMS", hashfunctions::fSubdet, 
		new quantity::ValueQuantity(quantity::fADC_5),
		new quantity::ValueQuantity(quantity::fN, true));
	_cMeanTotal_depth.initialize(_name, "Mean", hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cRMSTotal_depth.initialize(_name, "RMS", hashfunctions::fdepth, 
		new quantity::DetectorQuantity(quantity::fieta), 
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cMeanTotal_FEDVME.initialize(_name, "Mean", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cMeanTotal_FEDuTCA.initialize(_name, "Mean", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_15));
	_cRMSTotal_FEDVME.initialize(_name, "RMS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSTotal_FEDuTCA.initialize(_name, "RMS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));

	_cMeanDBRef1LS_Subdet.initialize(_name, "MeanDBRef1LS", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fAroundZero),
		new quantity::ValueQuantity(quantity::fN, true));
	_cRMSDBRef1LS_Subdet.initialize(_name, "RMSDBRef1LS", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fAroundZero),
		new quantity::ValueQuantity(quantity::fN, true));
	_cMeanDBRef1LS_depth.initialize(_name, "MeanDBRef1LS", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fAroundZero));
	_cRMSDBRef1LS_depth.initialize(_name, "RMSDBRef1LS", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fAroundZero));
	_cMeanDBRef1LS_FEDVME.initialize(_name, "MeanDBRef1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cMeanDBRef1LS_FEDuTCA.initialize(_name, "MeanDBRef1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSDBRef1LS_FEDVME.initialize(_name, "RMSDBRef1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSDBRef1LS_FEDuTCA.initialize(_name, "RMSDBRef1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	
	_cMeanDBRefTotal_Subdet.initialize(_name, "MeanDBRef", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fAroundZero),
		new quantity::ValueQuantity(quantity::fN, true));
	_cRMSDBRefTotal_Subdet.initialize(_name, "RMSDBRef", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fAroundZero),
		new quantity::ValueQuantity(quantity::fN, true));
	_cMeanDBRefTotal_depth.initialize(_name, "MeanDBRef", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fAroundZero));
	_cRMSDBRefTotal_depth.initialize(_name, "RMSDBRef", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fAroundZero));
	_cMeanDBRefTotal_FEDVME.initialize(_name, "MeanDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cMeanDBRefTotal_FEDuTCA.initialize(_name, "MeanDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSDBRefTotal_FEDVME.initialize(_name, "RMSDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));
	_cRMSDBRefTotal_FEDuTCA.initialize(_name, "RMSDBRef", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fADC_5));

	_cMissingvsLS_Subdet.initialize(_name, "MissingvsLS", 
		hashfunctions::fSubdet,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyvsLS_Subdet.initialize(_name, "OccupancyvsLS", 
		hashfunctions::fSubdet,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fN));
	_cNBadMeanvsLS_Subdet.initialize(_name, "NBadMeanvsLS", 
		hashfunctions::fSubdet,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fN));
	_cNBadRMSvsLS_Subdet.initialize(_name, "NBadRMSvsLS", 
		hashfunctions::fSubdet,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fN));

	_cMissing1LS_depth.initialize(_name, "Missing1LS", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBad1LS_depth.initialize(_name, "MeanBad1LS", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBad1LS_depth.initialize(_name, "RMSBad1LS", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing1LS_FEDVME.initialize(_name, "Missing1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing1LS_FEDuTCA.initialize(_name, "Missing1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBad1LS_FEDVME.initialize(_name, "MeanBad1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBad1LS_FEDuTCA.initialize(_name, "MeanBad1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBad1LS_FEDVME.initialize(_name, "RMSBad1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBad1LS_FEDuTCA.initialize(_name, "RMSBad1LS", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	
	_cMissingTotal_depth.initialize(_name, "Missing", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBadTotal_depth.initialize(_name, "MeanBad", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBadTotal_depth.initialize(_name, "RMSBad", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cMissingTotal_FEDVME.initialize(_name, "Missing", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMissingTotal_FEDuTCA.initialize(_name, "Missing", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBadTotal_FEDVME.initialize(_name, "MeanBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMeanBadTotal_FEDuTCA.initialize(_name, "MeanBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBadTotal_FEDVME.initialize(_name, "RMSBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cRMSBadTotal_FEDuTCA.initialize(_name, "RMSBad", hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
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

	//	book plots
	_cMean1LS_Subdet.book(ib, _emap, _subsystem);
	_cRMS1LS_Subdet.book(ib, _emap, _subsystem);
	_cMean1LS_depth.book(ib, _emap, _subsystem);
	_cRMS1LS_depth.book(ib, _emap, _subsystem);
	_cMeanDBRef1LS_Subdet.book(ib, _emap, _subsystem);
	_cRMSDBRef1LS_Subdet.book(ib, _emap, _subsystem);
	_cMeanDBRef1LS_depth.book(ib, _emap, _subsystem);
	_cRMSDBRef1LS_depth.book(ib, _emap, _subsystem);
	_cMissing1LS_depth.book(ib, _emap, _subsystem);
	_cMeanBad1LS_depth.book(ib, _emap, _subsystem);
	_cRMSBad1LS_depth.book(ib, _emap, _subsystem);
	
	_cMeanTotal_Subdet.book(ib, _emap, _subsystem);
	_cRMSTotal_Subdet.book(ib, _emap, _subsystem);
	_cMeanTotal_depth.book(ib, _emap, _subsystem);
	_cRMSTotal_depth.book(ib, _emap, _subsystem);
	_cMeanDBRefTotal_Subdet.book(ib, _emap, _subsystem);
	_cRMSDBRefTotal_Subdet.book(ib, _emap, _subsystem);
	_cMeanDBRefTotal_depth.book(ib, _emap, _subsystem);
	_cRMSDBRefTotal_depth.book(ib, _emap, _subsystem);
	_cMissingTotal_depth.book(ib, _emap, _subsystem);
	_cMeanBadTotal_depth.book(ib, _emap, _subsystem);
	_cRMSBadTotal_depth.book(ib, _emap, _subsystem);

	_cMean1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMean1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMS1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMS1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMeanDBRef1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanDBRef1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSDBRef1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSDBRef1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMissing1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMissing1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSBad1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSBad1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMeanBad1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanBad1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);

	_cMeanTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMeanDBRefTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanDBRefTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSDBRefTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSDBRefTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMissingTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMissingTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cRMSBadTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cRMSBadTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMeanBadTotal_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMeanBadTotal_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);

	_cMissingvsLS_Subdet.book(ib, _emap, _subsystem);
	_cOccupancyvsLS_Subdet.book(ib, _emap, _subsystem);
	_cNBadMeanvsLS_Subdet.book(ib, _emap, _subsystem);
	_cNBadRMSvsLS_Subdet.book(ib, _emap, _subsystem);
	_cSummaryvsLS_FED.book(ib, _emap, _subsystem);
	_cSummaryvsLS.book(ib, _subsystem);

	//	book compact containers
	_xPedSum1LS.book(_emap);
	_xPedSum21LS.book(_emap);
	_xPedEntries1LS.book(_emap);
	_xPedSumTotal.book(_emap);
	_xPedSum2Total.book(_emap);
	_xPedEntriesTotal.book(_emap);

	_xPedRefMean.book(_emap);
	_xPedRefRMS.book(_emap);

	_xNChs.book(_emap);
	_xNMsn1LS.book(_emap);
	_xNBadMean1LS.book(_emap);
	_xNBadRMS1LS.book(_emap);

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

	switch(uf)
	{
		case hcaldqm::f50LS:
			//	reset the containers for Sums, SumSqr, #Entries
			_xPedSum1LS.reset();
			_xPedSum21LS.reset();
			_xPedEntries1LS.reset();
			break;
		default:
			break;
	}
}

/* virtual */ void PedestalTask::endRun(edm::Run const& r, 
	edm::EventSetup const&)
{}

/* virtual */ void PedestalTask::_dump()
{
	//	reset what's needed
	
	//	Mean/RMS actual values
	_cMean1LS_Subdet.reset();
	_cRMS1LS_Subdet.reset();
	_cMean1LS_depth.reset();
	_cRMS1LS_depth.reset();
	_cMean1LS_FEDVME.reset();
	_cMean1LS_FEDuTCA.reset();
	_cRMS1LS_FEDVME.reset();
	_cRMS1LS_FEDuTCA.reset();
	
	_cMeanTotal_Subdet.reset();
	_cRMSTotal_Subdet.reset();
	_cMeanTotal_depth.reset();
	_cRMSTotal_depth.reset();
	_cMeanTotal_FEDVME.reset();
	_cMeanTotal_FEDuTCA.reset();
	_cRMSTotal_FEDVME.reset();
	_cRMSTotal_FEDuTCA.reset();
	

	//	DB Conditions Comparison
	_cMeanDBRef1LS_Subdet.reset();
	_cMeanDBRef1LS_depth.reset();
	_cRMSDBRef1LS_Subdet.reset();
	_cRMSDBRef1LS_depth.reset();
	
	_cMeanDBRefTotal_Subdet.reset();
	_cMeanDBRefTotal_depth.reset();
	_cRMSDBRefTotal_Subdet.reset();
	_cRMSDBRefTotal_depth.reset();

	_cMeanDBRef1LS_FEDVME.reset();
	_cMeanDBRef1LS_FEDuTCA.reset();
	_cRMSDBRef1LS_FEDVME.reset();
	_cRMSDBRef1LS_FEDuTCA.reset();
	
	_cMeanDBRefTotal_FEDVME.reset();
	_cMeanDBRefTotal_FEDuTCA.reset();
	_cRMSDBRefTotal_FEDVME.reset();
	_cRMSDBRefTotal_FEDuTCA.reset();

	//	missing channels
	_cMissing1LS_depth.reset();
	_cMeanBad1LS_depth.reset();
	_cRMSBad1LS_depth.reset();
	
//	_cMissingTotal_depth.reset();
//	_cMeanBadTotal_depth.reset();
//	_cRMSBadTotal_depth.reset();

	//	Missing or Bad
	_cMissing1LS_FEDVME.reset();
	_cMissing1LS_FEDuTCA.reset();
	_cMeanBad1LS_FEDVME.reset();
	_cMeanBad1LS_FEDuTCA.reset();
	_cRMSBad1LS_FEDVME.reset();
	_cRMSBad1LS_FEDuTCA.reset();

	_cMissingTotal_FEDVME.reset();
	_cMissingTotal_FEDuTCA.reset();
	_cMeanBadTotal_FEDVME.reset();
	_cMeanBadTotal_FEDuTCA.reset();
	_cRMSBadTotal_FEDVME.reset();
	_cRMSBadTotal_FEDuTCA.reset();

	//	reset some XXX containers
	_xNChs.reset();
	_xNMsn1LS.reset();
	_xNBadMean1LS.reset(); _xNBadRMS1LS.reset();

	// - ITERATE OVER ALL TEH CHANNELS
	// - FIND THE ONES THAT ARE MISSING
	// - FIND THE ONES WITH BAD PEDESTAL MEANs
	// - FIND THE ONES WITH BAD PEDESTAL RMSs
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
		double sum1LS = _xPedSum1LS.get(did); 
		double refm = _xPedRefMean.get(did);
		double sum21LS = _xPedSum21LS.get(did); 
		double refr = _xPedRefRMS.get(did);
		double n1LS = _xPedEntries1LS.get(did);
		
		double sumTotal = _xPedSumTotal.get(did);
		double sum2Total = _xPedSum2Total.get(did);
		double nTotal = _xPedEntriesTotal.get(did);

		_xNChs.get(eid)++;
		// IF A CHANNEL IS MISSING FOR THIS LS
		if (n1LS==0)
		{
			_cMissing1LS_depth.fill(did);
			_cMissingvsLS_Subdet.fill(did, _currentLS);
			eid.isVMEid()?_cMissing1LS_FEDVME.fill(eid):
				_cMissing1LS_FEDuTCA.fill(eid);
			_xNMsn1LS.get(eid)++;
			//	ALSO CHECK
			//	IF A CHANNEL HAS BEEN MISSING FOR ALL LSs SO FAR
			if (nTotal==0)
			{
				_cMissingTotal_depth.fill(did);
				eid.isVMEid()?_cMissingTotal_FEDVME.fill(eid):
				_cMissingTotal_FEDuTCA.fill(eid);
			}
			continue;
		}

		//	if not missing, fill the occupancy...
		_cOccupancyvsLS_Subdet.fill(did, _currentLS);

		//	compute the means and diffs for this LS
		sum1LS/=n1LS; double rms1LS = sqrt(sum21LS/n1LS-sum1LS*sum1LS);
		double diffm1LS = sum1LS-refm;
		double diffr1LS = rms1LS - refr;
		
		//	compute the means and diffs for the whole Run
		sumTotal/=nTotal; 
		double rmsTotal = sqrt(sum2Total/nTotal-sumTotal*sumTotal);
		double diffmTotal = sumTotal-refm;
		double diffrTotal = rmsTotal - refr;

		//	FILL ACTUAL MEANs AND RMSs FOR THIS LS
		_cMean1LS_Subdet.fill(did, sum1LS);
		_cMean1LS_depth.fill(did, sum1LS);
		_cRMS1LS_Subdet.fill(did, rms1LS);
		_cRMS1LS_depth.fill(did, rms1LS);

		//	FILL THE DIFFERENCES FOR THIS LS
		_cMeanDBRef1LS_Subdet.fill(did, diffm1LS);
		_cMeanDBRef1LS_depth.fill(did, diffm1LS);
		_cRMSDBRef1LS_Subdet.fill(did, diffr1LS);
		_cRMSDBRef1LS_depth.fill(did, diffr1LS);
		
			//	FILL ACTUAL MEANs AND RMSs FOR THE WHOLE RUN
		_cMeanTotal_Subdet.fill(did, sumTotal);
		_cMeanTotal_depth.fill(did, sumTotal);
		_cRMSTotal_Subdet.fill(did, rmsTotal);
		_cRMSTotal_depth.fill(did, rmsTotal);

		//	FILL THE DIFFERENCES FOR THE WHOLE RUN
		_cMeanDBRefTotal_Subdet.fill(did, diffmTotal);
		_cMeanDBRefTotal_depth.fill(did, diffmTotal);
		_cRMSDBRefTotal_Subdet.fill(did, diffrTotal);
		_cRMSDBRefTotal_depth.fill(did, diffrTotal);

		//	FOR THIS LS
		if (eid.isVMEid())
		{
			_cMean1LS_FEDVME.fill(eid, sum1LS);
			_cRMS1LS_FEDVME.fill(eid, rms1LS);
			_cMeanDBRef1LS_FEDVME.fill(eid, diffm1LS);
			_cRMSDBRef1LS_FEDVME.fill(eid, diffr1LS);
		}
		else
		{
			_cMean1LS_FEDuTCA.fill(eid, sum1LS);
			_cRMS1LS_FEDuTCA.fill(eid, rms1LS);
			_cMeanDBRef1LS_FEDuTCA.fill(eid, diffm1LS);
			_cRMSDBRef1LS_FEDuTCA.fill(eid, diffr1LS);
		}
		
		//	FOR THE WHOLE RUN
		if (eid.isVMEid())
		{
			_cMeanTotal_FEDVME.fill(eid, sumTotal);
			_cRMSTotal_FEDVME.fill(eid, rmsTotal);
			_cMeanDBRefTotal_FEDVME.fill(eid, diffmTotal);
			_cRMSDBRefTotal_FEDVME.fill(eid, diffrTotal);
		}
		else
		{
			_cMeanTotal_FEDuTCA.fill(eid, sumTotal);
			_cRMSTotal_FEDuTCA.fill(eid, rmsTotal);
			_cMeanDBRefTotal_FEDuTCA.fill(eid, diffmTotal);
			_cRMSDBRefTotal_FEDuTCA.fill(eid, diffrTotal);
		}

		//	FOR THE CURRENT LS COMPARE MEANS AND RMSS
		if (fabs(diffm1LS)>_thresh_mean)
		{
			_cMeanBad1LS_depth.fill(did);
			_cNBadMeanvsLS_Subdet.fill(did, _currentLS);
			if (eid.isVMEid())
				_cMeanBad1LS_FEDVME.fill(eid);
			else
				_cMeanBad1LS_FEDuTCA.fill(eid);
			_xNBadMean1LS.get(eid)++;
		}
		if (fabs(diffr1LS)>_thresh_rms)
		{
			_cRMSBad1LS_depth.fill(did);
			_cNBadRMSvsLS_Subdet.fill(did, _currentLS);
			if (eid.isVMEid())
				_cRMSBad1LS_FEDVME.fill(eid);
			else 
				_cRMSBad1LS_FEDuTCA.fill(eid);
			_xNBadRMS1LS.get(eid)++;
		}

		//	FOR THIS RUN 
		if (fabs(diffmTotal)>_thresh_mean)
		{
			_cMeanBadTotal_depth.fill(did);
			if (eid.isVMEid())
				_cMeanBadTotal_FEDVME.fill(eid);
			else
				_cMeanBadTotal_FEDuTCA.fill(eid);
		}
		if (fabs(diffrTotal)>_thresh_rms)
		{
			_cRMSBadTotal_depth.fill(did);
			if (eid.isVMEid())
				_cRMSBadTotal_FEDVME.fill(eid);
			else 
				_cRMSBadTotal_FEDuTCA.fill(eid);
		}

	}

	//	SET THE FLAGS FOR THIS LS
	for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
		it!=_vhashFEDs.end(); ++it)
	{
		flag::Flag fSum("PED");
		HcalElectronicsId eid = HcalElectronicsId(*it);

		std::vector<uint32_t>::const_iterator jt=
			std::find(_vcdaqEids.begin(), _vcdaqEids.end(), (*it));
		if (jt==_vcdaqEids.end())
		{
			//	not @cDAQ
			for (uint32_t iflag=0; iflag<_vflags.size(); iflag++)
				_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
					int(flag::fNCDAQ));
			_cSummaryvsLS.setBinContent(eid, _currentLS, int(flag::fNCDAQ));
			continue;
		}

		//	@cDAQ
		if (utilities::isFEDHBHE(eid) || utilities::isFEDHO(eid) ||
			utilities::isFEDHF(eid))
		{
			double frbadm = _xNBadMean1LS.get(eid)/_xNChs.get(eid);
			double frbadr = _xNBadRMS1LS.get(eid)/_xNChs.get(eid);

			if (_xNMsn1LS.get(eid)>0)
				_vflags[fMsn]._state = flag::fBAD;
			else
				_vflags[fMsn]._state = flag::fGOOD;
			if (frbadm>=_thresh_badm)
				_vflags[fBadM]._state = flag::fBAD;
			else
				_vflags[fBadM]._state = flag::fGOOD;
			if (frbadr>=_thresh_badr)
				_vflags[fBadR]._state = flag::fBAD;
			else
				_vflags[fBadR]._state = flag::fGOOD;
		}

		int iflag=0;
		for (std::vector<flag::Flag>::iterator ft=_vflags.begin();
			ft!=_vflags.end(); ++ft)
		{
			_cSummaryvsLS_FED.setBinContent(eid, _currentLS, iflag,
				int(ft->_state));
			fSum+=(*ft);
			iflag++;
			ft->reset();
		}
		_cSummaryvsLS.setBinContent(eid, _currentLS, fSum._state);
	}

	//	reset the pedestal containers instead of writting reset... func
	_xPedSum1LS.reset(); _xPedSum21LS.reset(); _xPedEntries1LS.reset();
	
}

/* virtual */ void PedestalTask::endLuminosityBlock(edm::LuminosityBlock const&,
	edm::EventSetup const&)
{
	if (_ptype==fLocal)
		return;
	this->_dump();
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
			_xPedSum1LS.get(did)+=it->sample(i).adc();
			_xPedSum21LS.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntries1LS.get(did)++;
			
			_xPedSumTotal.get(did)+=it->sample(i).adc();
			_xPedSum2Total.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntriesTotal.get(did)++;
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
			_xPedSum1LS.get(did)+=it->sample(i).adc();
			_xPedSum21LS.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntries1LS.get(did)++;
			
			_xPedSumTotal.get(did)+=it->sample(i).adc();
			_xPedSum2Total.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntriesTotal.get(did)++;
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
			_xPedSum1LS.get(did)+=it->sample(i).adc();
			_xPedSum21LS.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntries1LS.get(did)++;
			
			_xPedSumTotal.get(did)+=it->sample(i).adc();
			_xPedSum2Total.get(did)+=it->sample(i).adc()*it->sample(i).adc();
			_xPedEntriesTotal.get(did)++;
		}
	}
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


