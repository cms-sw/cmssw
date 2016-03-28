#include "DQM/HcalTasks/interface/DigiTask.h"

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
}

/* virtual */ void DigiTask::bookHistograms(DQMStore::IBooker& ib,
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
	std::vector<uint32_t> vFEDHF;
	vVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN, 
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
		vVME);
	_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
		vuTCA);
	vFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	vFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	vFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());

	//	initialize filters
	_filter_FEDHF.initialize(filter::fPreserver, hashfunctions::fFED,
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
			utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
			FIBERCH_MIN, false).rawId());

	//	INITIALIZE FIRST
	_cADC_SubdetPM.initialize(_name, "ADC", hashfunctions::fSubdetPM,
		new quantity::ValueQuantity(quantity::fADC_128),
		new quantity::ValueQuantity(quantity::fN, true));
	_cfC_SubdetPM.initialize(_name, "fC", hashfunctions::fSubdetPM,
		new quantity::ValueQuantity(quantity::ffC_10000),
		new quantity::ValueQuantity(quantity::fN, true));
	_cSumQ_SubdetPM.initialize(_name, "SumQ", hashfunctions::fSubdetPM,
		new quantity::ValueQuantity(quantity::ffC_10000),
		new quantity::ValueQuantity(quantity::fN, true));
	_cSumQ_depth.initialize(_name, "SumQ", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::ffC_10000));
	_cSumQvsLS_FED.initialize(_name, "SumQvsLS",
		hashfunctions::fFED,
		new quantity::LumiSection(_numLSstart),
		new quantity::ValueQuantity(quantity::ffC_10000));
	_cShapeCut_FEDSlot.initialize(_name, "Shape",
		hashfunctions::fFEDSlot,
		new quantity::ValueQuantity(quantity::fTiming_TS),
		new quantity::ValueQuantity(quantity::ffC_10000));
	_cTimingCut_SubdetPM.initialize(_name, "Timing",
		hashfunctions::fSubdetPM,
		new quantity::ValueQuantity(quantity::fTiming_TS200),
		new quantity::ValueQuantity(quantity::fN));
	_cTimingCut_FEDVME.initialize(_name, "Timing",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingCut_FEDuTCA.initialize(_name, "Timing",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingCut_ElectronicsVME.initialize(_name, "Timing",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingCut_ElectronicsuTCA.initialize(_name, "Timing",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cTimingCutvsLS_FED.initialize(_name, "TimingvsLS",
		hashfunctions::fFED,
		new quantity::LumiSection(),
		new quantity::ValueQuantity(quantity::fTiming_TS200));

	//	Charge sharing
	_cQ2Q12CutvsLS_FEDHF.initialize(_name, "Q2Q12vsLS",
		hashfunctions::fFED,
		new quantity::LumiSection(_numLSstart),
		new quantity::ValueQuantity(quantity::fRatio_0to2));

	//	Occupancy w/o a cut
	_cOccupancy_FEDVME.initialize(_name, "Occupancy",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancy_FEDuTCA.initialize(_name, "Occupancy",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancy_ElectronicsVME.initialize(_name, "Occupancy",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancy_ElectronicsuTCA.initialize(_name, "Occupancy",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyvsLS_Subdet.initialize(_name, "OccupancyvsLS",
		hashfunctions::fSubdet,
		new quantity::LumiSection(_numLSstart),
		new quantity::ValueQuantity(quantity::fN_to3000));
	_cOccupancy_depth.initialize(_name, "Occupancy",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyNR_FEDVME.initialize(_name, "OccupancyNR",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyNR_FEDuTCA.initialize(_name, "OccupancyNR",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));

	//	Occupancy w/ a cut
	_cOccupancyCut_FEDVME.initialize(_name, "OccupancyCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyCut_FEDuTCA.initialize(_name, "OccupancyCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyCut_ElectronicsVME.initialize(_name, "OccupancyCut",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyCut_ElectronicsuTCA.initialize(_name, "OccupancyCut",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyCutvsLS_Subdet.initialize(_name, "OccupancyCutvsLS",
		hashfunctions::fSubdet,
		new quantity::LumiSection(_numLSstart),
		new quantity::ValueQuantity(quantity::fN_to3000));
	_cOccupancyCut_depth.initialize(_name, "OccupancyCut",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCutNR_FEDVME.initialize(_name, "OccupancyNRCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancyCutNR_FEDuTCA.initialize(_name, "OccupancyNRCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_TS200));

	_cCapIdRots_FEDVME.initialize(_name, "CapId",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cCapIdRots_FEDuTCA.initialize(_name, "CapId",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing1LS_FEDVME.initialize(_name, "Missing1LS",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cMissing1LS_FEDuTCA.initialize(_name, "Missing1LS",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cDigiSize_FEDVME.initialize(_name, "DigiSize",
		hashfunctions::fFED,
		new quantity::ValueQuantity(quantity::fDigiSize),
		new quantity::ValueQuantity(quantity::fN));
	_cDigiSize_FEDuTCA.initialize(_name, "DigiSize",
		hashfunctions::fFED,
		new quantity::ValueQuantity(quantity::fDigiSize),
		new quantity::ValueQuantity(quantity::fN));

	std::vector<std::string> fnames;
	fnames.push_back("UniSlot");
	fnames.push_back("Msn1LS");
	fnames.push_back("CapIdRot");
	fnames.push_back("DigiSize");
	_cSummary.initialize(_name, "Summary",
		new quantity::FEDQuantity(vFEDs),
		new quantity::FlagQuantity(fnames),
		new quantity::QualityQuantity());

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
	_cSumQvsLS_FED.book(ib, _emap, _subsystem);

	_cShapeCut_FEDSlot.book(ib, _emap, _subsystem);

	_cTimingCut_SubdetPM.book(ib, _emap, _subsystem);
	_cTimingCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingCutvsLS_FED.book(ib, _emap, _subsystem);

	_cQ2Q12CutvsLS_FEDHF.book(ib, _emap, _filter_FEDHF, _subsystem);

	_cOccupancy_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancy_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancy_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancy_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyvsLS_Subdet.book(ib, _emap, _subsystem);
	_cOccupancy_depth.book(ib, _emap, _subsystem);
	_cOccupancyNR_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyNR_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCutvsLS_Subdet.book(ib, _emap, _subsystem);
	_cOccupancyCut_depth.book(ib, _emap, _subsystem);
	_cOccupancyCutNR_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCutNR_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);


	_cCapIdRots_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cCapIdRots_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cMissing1LS_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cMissing1LS_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cDigiSize_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cDigiSize_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cSummary.book(ib, _subsystem);

	_ehashmapVME.initialize(_emap, electronicsmap::fE2DHashMap,
		_filter_uTCA);
	_ehashmapuTCA.initialize(_emap, electronicsmap::fE2DHashMap,
		_filter_VME);
}

/* virtual */ void DigiTask::_resetMonitors(UpdateFreq uf)
{
	switch(uf)
	{
		case hcaldqm::fLS:
			_cDigiSize_FEDVME.reset();
			_cDigiSize_FEDuTCA.reset();
			_cCapIdRots_FEDVME.reset();
			_cCapIdRots_FEDuTCA.reset();
			_cOccupancy_FEDVME.reset();
			_cOccupancy_FEDuTCA.reset();
			break;
		case hcaldqm::f10LS:
			_cMissing1LS_FEDVME.reset();
			_cMissing1LS_FEDuTCA.reset();
			_cOccupancy_ElectronicsVME.reset();
			_cOccupancy_ElectronicsuTCA.reset();
			_cOccupancyCut_ElectronicsVME.reset();
			_cOccupancyCut_ElectronicsuTCA.reset();
			break;
		default:
			break;
	}

	DQTask::_resetMonitors(uf);
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

	//	HB collection
	int numChs = 0;
	int numChsCut = 0;
	int numChsHE = 0;
	int numChsCutHE = 0;
	for (HBHEDigiCollection::const_iterator it=chbhe->begin(); it!=chbhe->end();
		++it)
	{
		double sumQ = utilities::sumQ<HBHEDataFrame>(*it, 2.5, 0, it->size()-1);
		HcalDetId const& did = it->id();
		HcalElectronicsId const& eid = it->elecId();

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cSumQ_depth.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		_cSumQvsLS_FED.fill(eid, _currentLS, sumQ);
		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
			_cOccupancyNR_FEDVME.fill(eid);
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);
			_cDigiSize_FEDVME.fill(eid, it->size());
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
			_cOccupancyNR_FEDuTCA.fill(eid);
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);
			_cDigiSize_FEDuTCA.fill(eid, it->size());
		}

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (sumQ>_cutSumQ_HBHE)
				_cShapeCut_FEDSlot.fill(eid, i, it->sample(i).nominal_fC());
		}

		if (sumQ>_cutSumQ_HBHE)
		{
			double timing = utilities::aveTS<HBHEDataFrame>(*it, 2.5, 0,
				it->size()-1);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cOccupancyCut_depth.fill(did);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			if (eid.isVMEid())
			{
				_cTimingCut_FEDVME.fill(eid, timing);
				_cTimingCut_ElectronicsVME.fill(eid, timing);
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
				_cOccupancyCutNR_FEDVME.fill(eid);
			}
			else 
			{
				_cTimingCut_FEDuTCA.fill(eid, timing);
				_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
				_cOccupancyCutNR_FEDuTCA.fill(eid);
			}
			did.subdet()==HcalBarrel?numChsCut++:numChsCutHE++;
		}
		did.subdet()==HcalBarrel?numChs++:numChsHE++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), _currentLS, 
		numChs);
	_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), _currentLS,
		numChsCut);
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), _currentLS,
		numChsHE);
	_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), _currentLS,
		numChsCutHE);
	numChs=0;
	numChsCut = 0;

	//	HO collection
	for (HODigiCollection::const_iterator it=cho->begin(); it!=cho->end();
		++it)
	{
		double sumQ = utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size()-1);
		HcalDetId const& did = it->id();
		HcalElectronicsId const& eid = it->elecId();

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cSumQ_depth.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		_cSumQvsLS_FED.fill(eid, _currentLS, sumQ);
		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
			_cOccupancyNR_FEDVME.fill(eid);
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);
			_cDigiSize_FEDVME.fill(eid, it->size());
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
			_cOccupancyNR_FEDuTCA.fill(eid);
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);
			_cDigiSize_FEDuTCA.fill(eid, it->size());
		}

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (sumQ>_cutSumQ_HO)
				_cShapeCut_FEDSlot.fill(eid, i, it->sample(i).nominal_fC());
		}

		if (sumQ>_cutSumQ_HO)
		{
			double timing = utilities::aveTS<HODataFrame>(*it, 8.5, 0,
				it->size()-1);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			if (eid.isVMEid())
			{
				_cTimingCut_FEDVME.fill(eid, timing);
				_cTimingCut_ElectronicsVME.fill(eid, timing);
				_cOccupancyCutNR_FEDVME.fill(eid);
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
			}
			else 
			{
				_cTimingCut_FEDuTCA.fill(eid, timing);
				_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				_cOccupancyCutNR_FEDuTCA.fill(eid);
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
			}
			numChsCut++;
		}
		numChs++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 1), _currentLS,
		numChs);
	_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 1), _currentLS,
		numChsCut);
	numChs=0; numChsCut=0;

	//	HF collection
	for (HFDigiCollection::const_iterator it=chf->begin(); it!=chf->end();
		++it)
	{
		double sumQ = utilities::sumQ<HFDataFrame>(*it, 2.5, 0, it->size()-1);
		HcalDetId const& did = it->id();
		HcalElectronicsId const& eid = it->elecId();

		_cSumQ_SubdetPM.fill(did, sumQ);
		_cOccupancy_depth.fill(did);
		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
			_cOccupancyNR_FEDVME.fill(eid);
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDVME.fill(eid, 1);
			_cDigiSize_FEDVME.fill(eid, it->size());
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
			_cOccupancyNR_FEDuTCA.fill(eid);
			if (!it->validate(0, it->size()))
				_cCapIdRots_FEDuTCA.fill(eid, 1);
			_cDigiSize_FEDuTCA.fill(eid, it->size());
		}

		for (int i=0; i<it->size(); i++)
		{
			_cADC_SubdetPM.fill(did, it->sample(i).adc());
			_cfC_SubdetPM.fill(did, it->sample(i).nominal_fC());
			if (sumQ>_cutSumQ_HF)
				_cShapeCut_FEDSlot.fill(eid, i, it->sample(i).nominal_fC());
		}

		if (sumQ>_cutSumQ_HF)
		{
			double timing = utilities::aveTS<HFDataFrame>(*it, 2.5, 0,
				it->size()-1);
			double q1 = it->sample(1).nominal_fC()-2.5;
			double q2 = it->sample(2).nominal_fC()-2.5;
			double q2q12 = q2/(q1+q2);
			_cSumQ_depth.fill(did, sumQ);
			_cSumQvsLS_FED.fill(eid, _currentLS, sumQ);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			_cOccupancyCut_depth.fill(did);
			if (!eid.isVMEid())
				_cQ2Q12CutvsLS_FEDHF.fill(eid, _currentLS, q2q12);
			if (eid.isVMEid())
			{
				_cTimingCut_FEDVME.fill(eid, timing);
				_cTimingCut_ElectronicsVME.fill(eid, timing);
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
				_cOccupancyCutNR_FEDVME.fill(eid);
			}
			else 
			{
				_cTimingCut_FEDuTCA.fill(eid, timing);
				_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
				_cOccupancyCutNR_FEDuTCA.fill(eid);
			}
			numChsCut++;
		}
		numChs++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), _currentLS, 
		numChs);
	_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), _currentLS,
		numChsCut);
}

/* virtual */ void DigiTask::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	// iterate over each fed and set its status
	for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
		it!=_vhashFEDs.end(); ++it)
	{
		HcalElectronicsId eid = HcalElectronicsId(*it);
		//	set flag as unapplicable for this FED first
		//	TEMPORARY - skip crate36!
		for (int flag=fUniSlot; flag<nDigiFlag; flag++)
			_cSummary.setBinContent(eid, flag, fNA);
		if (eid.crateId()==36)
			continue;

		//	#channels per FED
		int numChs = eid.isVMEid() ? SPIGOT_NUM*FIBER_VME_NUM*FIBERCH_NUM:
			SLOT_uTCA_NUM*FIBER_uTCA_NUM*FIBERCH_NUM;
		//	threshold for #channels 0 for HF and 5% for the rest
		double numChsThr = eid.crateId()==22 || eid.crateId()==29 ||
			eid.crateId()==32 ? 0 : 0.05;

		//	init some flags/constants/whatever before checks
		int ncapid = 0;
		int nmissing = 0;
		bool uniSlot = false;
		bool digisize = false;

		//	electronics type specific
		if (eid.isVMEid())
		{
			double mdsize = _cDigiSize_FEDVME.getMean(eid);
			double rdsize = _cDigiSize_FEDVME.getRMS(eid);
			//	check if rms>0 or mean!=expected(10)
			//	HARDCODED EXPECTED here! Only for VME
			digisize = rdsize>0 || mdsize!=10;

			//	VME
			for (int is=SPIGOT_MIN; is<=SPIGOT_MAX; is++)
			{
				eid = HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, is, eid.dccid());
				HcalElectronicsId ejd = HcalElectronicsId(FIBERCH_MIN,
					FIBER_VME_MIN, is==SPIGOT_MAX?SPIGOT_MIN:is+1,eid.dccid());

				//	note, Occupancy w/o a cut is used - cause no ZS 
				int niscut = _cOccupancy_ElectronicsVME.getBinContent(eid);
				int njscut = _cOccupancy_ElectronicsVME.getBinContent(ejd);
				for (int ifib=FIBER_VME_MIN;ifib<=FIBER_VME_MAX;ifib++)
					for (int ifc=FIBERCH_MIN; ifc<=FIBERCH_MAX; ifc++)
					{
						eid=HcalElectronicsId(ifc, ifib, eid.spigot(),
							eid.dccid());

						//	get the capid rotated channels checked
						ncapid+=_cCapIdRots_FEDVME.getBinContent(eid);

						//	get the occupancy checked - what misses out...
						if (_cOccupancy_FEDVME.getBinContent(eid)<1)
						{
							_cMissing1LS_FEDVME.fill(eid);
							nmissing++;
						}
					}

				//	set/check the ratio for unislot
				double ratio = niscut==0 && njscut==0? 1:
					double(std::min(niscut, njscut))/double(std::max(niscut, 
					njscut));
				//	when at least x5 difference
				if (ratio<0.2)
					uniSlot = true;
			}
		}
		else
		{
			double mdsize = _cDigiSize_FEDuTCA.getMean(eid);
			double rdsize = _cDigiSize_FEDuTCA.getRMS(eid);
			HcalDetId did = HcalDetId(_ehashmapuTCA.lookup(eid));
			//	check if rms>0 or mean!=expected
			digisize = rdsize>0 || mdsize!=constants::TS_NUM[did.subdet()-1];
			
			//	uTCA
			for (int is=constants::SLOT_uTCA_MIN;
				is<=SLOT_uTCA_MAX; is++)
			{
				eid = HcalElectronicsId(eid.crateId(), is,
					FIBER_uTCA_MIN1, FIBERCH_MIN, false);
				HcalElectronicsId ejd = HcalElectronicsId(eid.crateId(), 
					is==SLOT_uTCA_MAX?SLOT_uTCA_MIN:is+1, FIBER_uTCA_MIN1,
					FIBERCH_MIN, false);

				//	HARDCODED HF SELECTION
				//	TODO: map back to HcalDetId
				int niscut = eid.crateId()==22 || eid.crateId()==29 || 
					eid.crateId()==32?
					_cOccupancyCut_ElectronicsuTCA.getBinContent(eid):
					_cOccupancy_ElectronicsuTCA.getBinContent(eid);
				int njscut = ejd.crateId()==22 || ejd.crateId()==29 || 
					ejd.crateId()==32?
					_cOccupancyCut_ElectronicsuTCA.getBinContent(ejd):
					_cOccupancy_ElectronicsuTCA.getBinContent(ejd);
				for (int ifib=FIBER_uTCA_MIN1;
					ifib<=FIBER_uTCA_MAX2; ifib++)
				{
					//	skip fibers in between for now
					if (ifib>FIBER_uTCA_MAX1 && ifib<FIBER_uTCA_MIN2)
						continue;
					for (int ifc = FIBERCH_MIN; ifc<=FIBERCH_MAX; ifc++)
					{
						eid = HcalElectronicsId(eid.crateId(), is,
							ifib, ifc, false);

						//	capid
						ncapid+=_cCapIdRots_FEDuTCA.getBinContent(eid);
						//	occupancy - missing chs
						if (_cOccupancy_FEDuTCA.getBinContent(eid)<1)
						{
							_cMissing1LS_FEDuTCA.fill(eid);
							nmissing++;
						}
					}
				}
				//	set/check the ratio for unislot
				double ratio = niscut==0 && njscut==0?1:
					double(std::min(niscut, njscut))/double(std::max(niscut, 
					njscut));
				//	when x5 difference
				if (ratio<0.2)
					uniSlot = true;
			}
		}

		if (ncapid>0)
			_cSummary.setBinContent(eid, fCapIdRot, fLow);
		else
			_cSummary.setBinContent(eid, fCapIdRot, fGood);
		if (uniSlot)
			_cSummary.setBinContent(eid, fUniSlot, fLow);
		else
			_cSummary.setBinContent(eid, fUniSlot, fGood);
		if (double(nmissing)/double(numChs)>numChsThr)	
			//	frac missing > threshold
			_cSummary.setBinContent(eid, fMsn1LS, fLow);
		else
			_cSummary.setBinContent(eid, fMsn1LS, fGood);
		digisize?
			_cSummary.setBinContent(eid, fDigiSize, fLow):
			_cSummary.setBinContent(eid, fDigiSize, fGood);
	}

	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiTask);

