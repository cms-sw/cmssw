#include "DQM/HcalTasks/interface/RecHitTask.h"
#include <math.h>

RecHitTask::RecHitTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hbhereco"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("horeco"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hfreco"));

	_tokHBHE = consumes<HBHERecHitCollection>(_tagHBHE);
	_tokHO = consumes<HORecHitCollection>(_tagHO);
	_tokHF = consumes<HFRecHitCollection>(_tagHF);

	_cutE_HBHE = ps.getUntrackedParameter<double>("cutE_HBHE", 5);
	_cutE_HO = ps.getUntrackedParameter<double>("cutE_HO", 5);
	_cutE_HF = ps.getUntrackedParameter<double>("cutE_HF", 5);
	_thresh_unihf = ps.getUntrackedParameter<double>("thresh_unihf", 0.2);

	//	order must be the same as in RecoFlag enum
	_vflags.resize(nRecoFlag);
	_vflags[fUni]=flag::Flag("UniSlotHF");
	_vflags[fTCDS]=flag::Flag("TCDS");
}

/* virtual */ void RecHitTask::bookHistograms(DQMStore::IBooker& ib,
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
	vVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN, 
		constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
	vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
		vVME);
	_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
		vuTCA);

	//	push the rawIds of each fed into the vector
	for (std::vector<int>::const_iterator it=vFEDsVME.begin();
		it!=vFEDsVME.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(
			FIBERCH_MIN, FIBER_VME_MIN, SPIGOT_MIN, (*it)-FED_VME_MIN).rawId());
	for (std::vector<int>::const_iterator it=vFEDsuTCA.begin(); 
		it!=vFEDsuTCA.end(); ++it)
		_vhashFEDs.push_back(HcalElectronicsId(
			utilities::fed2crate(*it), SLOT_uTCA_MIN, FIBER_uTCA_MIN1,
			FIBERCH_MIN, false).rawId());

	//	INITIALIZE FIRST
	//	Energy
	_cEnergy_Subdet.initialize(_name, "Energy", hashfunctions::fSubdet,
		new quantity::ValueQuantity(quantity::fEnergy),
		new quantity::ValueQuantity(quantity::fN, true));
	_cEnergy_depth.initialize(_name, "Energy", hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fEnergy, true));

	//	Timing
	_cTimingCut_SubdetPM.initialize(_name, "TimingCut", 
		hashfunctions::fSubdetPM,
		new quantity::ValueQuantity(quantity::fTiming_ns),
		new quantity::ValueQuantity(quantity::fN, true));
	_cTimingvsEnergy_SubdetPM.initialize(_name, "TimingvsEnergy",
		hashfunctions::fSubdetPM,
		new quantity::ValueQuantity(quantity::fEnergy, true),
		new quantity::ValueQuantity(quantity::fTiming_ns),
		new quantity::ValueQuantity(quantity::fN, true));
	_cTimingCut_FEDVME.initialize(_name, "TimingCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_ns));
	_cTimingCut_FEDuTCA.initialize(_name, "TimingCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fTiming_ns));
	_cTimingCutvsLS_FED.initialize(_name, "TimingCutvsLS",
		hashfunctions::fFED,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fTiming_ns));
	_cTimingCut_ElectronicsVME.initialize(_name, "TimingCut",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fTiming_ns));
	_cTimingCut_ElectronicsuTCA.initialize(_name, "TimingCut",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fTiming_ns));
	_cTimingCut_HBHEPartition.initialize(_name, "TimingCut",
		hashfunctions::fHBHEPartition,
		new quantity::ValueQuantity(quantity::fTiming_ns),
		new quantity::ValueQuantity(quantity::fN));
	_cTimingCut_depth.initialize(_name, "TimingCut",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fTiming_ns));

	//	Occupancy
	_cOccupancy_depth.initialize(_name, "Occupancy",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancy_FEDVME.initialize(_name, "Occupancy",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancy_FEDuTCA.initialize(_name, "Occupancy",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancy_ElectronicsVME.initialize(_name, "Occupancy",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancy_ElectronicsuTCA.initialize(_name, "Occupancy",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyvsLS_Subdet.initialize(_name, "OccupancyvsLS",
		hashfunctions::fSubdet,
		new quantity::LumiSection(_maxLS),
		new quantity::ValueQuantity(quantity::fN_to3000));
	_cOccupancyCut_FEDVME.initialize(_name, "OccupancyCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCut_FEDuTCA.initialize(_name, "OccupancyCut",
		hashfunctions::fFED,
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ElectronicsQuantity(quantity::fFiberuTCAFiberCh),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCut_ElectronicsVME.initialize(_name, "OccupancyCut",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsVME),
		new quantity::ElectronicsQuantity(quantity::fSpigot),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCut_ElectronicsuTCA.initialize(_name, "OccupancyCut",
		hashfunctions::fElectronics,
		new quantity::FEDQuantity(vFEDsuTCA),
		new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCut_depth.initialize(_name, "OccupancyCut",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));

	//	INITIALIZE HISTOGRAMS to be used only in Online
	if (_ptype==fOnline)
	{
		_cEnergyvsieta_Subdet.initialize(_name, "Energyvsieta",
			hashfunctions::fSubdet,
			new quantity::DetectorQuantity(quantity::fieta),
			new quantity::ValueQuantity(quantity::fEnergy_1TeV));
		_cEnergyvsiphi_SubdetPM.initialize(_name, "Energyvsiphi",
			hashfunctions::fSubdetPM,
			new quantity::DetectorQuantity(quantity::fiphi),
			new quantity::ValueQuantity(quantity::fEnergy_1TeV));
		_cEnergyvsLS_SubdetPM.initialize(_name, "EnergyvsLS",
			hashfunctions::fSubdetPM,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fEnergy_1TeV));
		_cEnergyvsBX_SubdetPM.initialize(_name, "EnergyvsBX",
			hashfunctions::fSubdetPM,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fEnergy_1TeV));
		_cTimingCutvsieta_Subdet.initialize(_name, "TimingCutvsieta",
			hashfunctions::fSubdet,
			new quantity::DetectorQuantity(quantity::fieta),
			new quantity::ValueQuantity(quantity::fTiming_ns));
		_cTimingCutvsiphi_SubdetPM.initialize(_name, "TimingCutvsiphi",
			hashfunctions::fSubdetPM,
			new quantity::DetectorQuantity(quantity::fiphi),
			new quantity::ValueQuantity(quantity::fTiming_ns));
		_cTimingCutvsBX_SubdetPM.initialize(_name, "TimingCutvsBX",
			hashfunctions::fSubdetPM,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fTiming_ns));
		_cOccupancyvsiphi_SubdetPM.initialize(_name, "Occupancyvsiphi",
			hashfunctions::fSubdetPM,
			new quantity::DetectorQuantity(quantity::fiphi),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyvsieta_Subdet.initialize(_name, "Occupancyvsieta",
			hashfunctions::fSubdet,
			new quantity::DetectorQuantity(quantity::fieta),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutvsiphi_SubdetPM.initialize(_name, "OccupancyCutvsiphi",
			hashfunctions::fSubdetPM,
			new quantity::DetectorQuantity(quantity::fiphi),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutvsieta_Subdet.initialize(_name, "OccupancyCutvsieta",
			hashfunctions::fSubdet,
			new quantity::DetectorQuantity(quantity::fieta),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutvsBX_SubdetPM.initialize(_name, "OccupancyCutvsBX",
			hashfunctions::fSubdetPM,
			new quantity::ValueQuantity(quantity::fBX),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutvsiphivsLS_SubdetPM.initialize(_name,
			"OccupancyCutvsiphivsLS", hashfunctions::fSubdetPM,
			new quantity::LumiSection(_maxLS),
			new quantity::DetectorQuantity(quantity::fiphi),
			new quantity::ValueQuantity(quantity::fN));
		_cOccupancyCutvsLS_Subdet.initialize(_name, "OccupancyCutvsLS",
			hashfunctions::fSubdet,
			new quantity::LumiSection(_maxLS),
			new quantity::ValueQuantity(quantity::fN_to3000));

		_cSummaryvsLS_FED.initialize(_name, "SummaryvsLS",
			hashfunctions::fFED,
			new quantity::LumiSection(_maxLS),
			new quantity::FlagQuantity(_vflags),
			new quantity::ValueQuantity(quantity::fState));
		_cSummaryvsLS.initialize(_name, "SummaryvsLS",
			new quantity::LumiSection(_maxLS),
			new quantity::FEDQuantity(vFEDs),
			new quantity::ValueQuantity(quantity::fState));

		_xUniHF.initialize(hashfunctions::fFEDSlot);
		_xUni.initialize(hashfunctions::fFED);
	}

	//	BOOK HISTOGRAMS
	char cutstr[200];
	sprintf(cutstr, "_EHBHE%dHO%dHF%d", int(_cutE_HBHE),
		int(_cutE_HO), int(_cutE_HF));
	char cutstr2[200];
	sprintf(cutstr2, "_EHF%d", int(_cutE_HF));

	//	Energy
	_cEnergy_Subdet.book(ib, _emap, _subsystem);
	_cEnergy_depth.book(ib, _emap, _subsystem);

	//	Timing
	_cTimingCut_SubdetPM.book(ib, _emap, _subsystem);
	_cTimingvsEnergy_SubdetPM.book(ib, _emap, _subsystem);
	_cTimingCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cTimingCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cTimingCut_HBHEPartition.book(ib, _emap, _subsystem);
	_cTimingCut_depth.book(ib, _emap, _subsystem);
	_cTimingCutvsLS_FED.book(ib, _emap, _subsystem);

	//	Occupancy
	_cOccupancy_depth.book(ib, _emap, _subsystem);
	_cOccupancy_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancy_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancy_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancy_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyvsLS_Subdet.book(ib, _emap, _subsystem);
	_cOccupancyCut_depth.book(ib, _emap, _subsystem);
	_cOccupancyCut_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCut_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
	_cOccupancyCut_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
	_cOccupancyCut_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);

	//	BOOK HISTOGRAMS to be used only in Online
	if (_ptype==fOnline)
	{
		_cEnergyvsLS_SubdetPM.book(ib, _emap, _subsystem);
		_cEnergyvsieta_Subdet.book(ib, _emap, _subsystem);
		_cEnergyvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cEnergyvsBX_SubdetPM.book(ib, _emap, _subsystem);
		_cTimingCutvsieta_Subdet.book(ib, _emap, _subsystem);
		_cTimingCutvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cTimingCutvsBX_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsiphi_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyCutvsieta_Subdet.book(ib, _emap, _subsystem);
		_cOccupancyCutvsBX_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyCutvsiphivsLS_SubdetPM.book(ib, _emap, _subsystem);
		_cOccupancyCutvsLS_Subdet.book(ib, _emap, _subsystem);
		_cSummaryvsLS_FED.book(ib, _emap, _subsystem);
		_cSummaryvsLS.book(ib, _subsystem);

		std::vector<uint32_t> vhashFEDHF;
		vhashFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vhashFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		vhashFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		HashFilter filter_FEDHF;
		filter_FEDHF.initialize(filter::fPreserver, hashfunctions::fFED,
			vhashFEDHF);

		_gids = _emap->allPrecisionId();
		_xUniHF.book(_emap, filter_FEDHF);
		_xUni.book(_emap);
	}

	//	initialize hash map
	_ehashmap.initialize(_emap, hcaldqm::electronicsmap::fD2EHashMap);
}

/* virtual */ void RecHitTask::_resetMonitors(UpdateFreq uf)
{
	/*
	switch(uf)
	{
		case hcaldqm::f1LS:
			_cOccupancy_FEDVME.reset();
			_cOccupancy_FEDuTCA.reset();
			break;
		case hcaldqm::f10LS:
			_cOccupancyCut_ElectronicsVME.reset();
			_cOccupancyCut_ElectronicsuTCA.reset();
			_cOccupancy_ElectronicsVME.reset();
			_cOccupancy_ElectronicsuTCA.reset();
			_cTimingCut_ElectronicsVME.reset();
			_cTimingCut_ElectronicsuTCA.reset();
			break;
		default:
			break;
	}
	*/

	DQTask::_resetMonitors(uf);
}

/* virtual */ void RecHitTask::_process(edm::Event const& e,
	edm::EventSetup const&)
{
	edm::Handle<HBHERecHitCollection> chbhe;
	edm::Handle<HORecHitCollection> cho;
	edm::Handle<HFRecHitCollection> chf;

	if (!(e.getByToken(_tokHBHE, chbhe)))
		_logger.dqmthrow("Collection HBHERecHitCollection not available "
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!(e.getByToken(_tokHO, cho)))
		_logger.dqmthrow("Collection HORecHitCollection not available "
			+ _tagHO.label() + " " + _tagHO.instance());
	if (!(e.getByToken(_tokHF, chf)))
		_logger.dqmthrow("Collection HFRecHitCollection not available "
			+ _tagHF.label() + " " + _tagHF.instance());

	//	extract some info per event
	int bx = e.bunchCrossing();

	double ehbm = 0; double ehbp = 0;
	double ehem = 0; double ehep = 0;
	int nChsHB = 0; int nChsHE = 0;
	int nChsHBCut = 0; int nChsHECut = 0;
	for (HBHERecHitCollection::const_iterator it=chbhe->begin();
		it!=chbhe->end(); ++it)
	{
		double energy = it->energy();
		double timing = it->time();
		HcalDetId did = it->id();
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
		_cEnergy_Subdet.fill(did, energy);
		_cTimingvsEnergy_SubdetPM.fill(did, energy, timing);
		_cOccupancy_depth.fill(did);
		did.subdet()==HcalBarrel?did.ieta()>0?ehbp+=energy:ehbm+=energy:
			did.ieta()>0?ehep+=energy:ehem+=energy;

		//	ONLINE ONLY!
		if (_ptype==fOnline)
		{
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		//	^^^ONLINE ONLY!

		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
		}

		if (energy>_cutE_HBHE)
		{
			//	ONLINE ONLY!
			if (_ptype==fOnline)
			{
				_cEnergyvsLS_SubdetPM.fill(did, _currentLS, energy);
				_cEnergyvsBX_SubdetPM.fill(did, bx, energy);
				_cEnergyvsieta_Subdet.fill(did, energy);
				_cEnergyvsiphi_SubdetPM.fill(did, energy);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsBX_SubdetPM.fill(did, bx, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
			}
			//	^^^ONLINE ONLY!
			_cEnergy_depth.fill(did, energy);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCut_HBHEPartition.fill(did, timing);
			if (timing>=-12.5 && timing<=12.5)
			{
				//	time constrains here are explicit!
				_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
				_cTimingCut_depth.fill(did, timing);
			}
			_cOccupancyCut_depth.fill(did);
			if (eid.isVMEid())
			{
				if (timing>=-12.5 && timing<=12.5)
				{
					//	time constraints are explicit!
					_cTimingCut_FEDVME.fill(eid, timing);
					_cTimingCut_ElectronicsVME.fill(eid, timing);
				}
				_cOccupancyCut_FEDVME.fill(eid);
				_cOccupancyCut_ElectronicsVME.fill(eid);
			}
			else
			{
				if (timing>=-12.5 && timing<=12.5)
				{
					//	time constraints are explicit!
					_cTimingCut_FEDuTCA.fill(eid, timing);
					_cTimingCut_ElectronicsuTCA.fill(eid, timing);
				}
				_cOccupancyCut_FEDuTCA.fill(eid);
				_cOccupancyCut_ElectronicsuTCA.fill(eid);
			}
			did.subdet()==HcalBarrel?nChsHBCut++:nChsHECut++;
		}
		did.subdet()==HcalBarrel?nChsHB++:nChsHE++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), _currentLS, 
		nChsHB);
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), _currentLS,
		nChsHE);

	//	ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_cOccupancyCutvsBX_SubdetPM.fill(HcalDetId(HcalBarrel, 1,1,1),
			bx, nChsHBCut);
		_cOccupancyCutvsBX_SubdetPM.fill(HcalDetId(HcalEndcap, 1,1,1),
			bx, nChsHECut);
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalBarrel, 1, 1, 1), 
			_currentLS, nChsHBCut);
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalEndcap, 1, 1, 1), 
			_currentLS, nChsHECut);
	}
	//	^^^ONLINE ONLY!

	int nChsHO = 0; int nChsHOCut = 0;
	double ehop = 0; double ehom = 0;
	for (HORecHitCollection::const_iterator it=cho->begin();
		it!=cho->end(); ++it)
	{
		double energy = it->energy();
		double timing = it->time();
		HcalDetId did = it->id();
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
		_cEnergy_Subdet.fill(did, energy);
		_cTimingvsEnergy_SubdetPM.fill(did, energy, timing);
		_cOccupancy_depth.fill(did);
		did.ieta()>0?ehop+=energy:ehom+=energy;

		//	IMPORTANT: ONLINE ONLY!
		if (_ptype==fOnline)
		{
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		//	ONLINE ONLY!

		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
		}

		if (energy>_cutE_HO)
		{
			//	ONLINE ONLY!
			if (_ptype==fOnline)
			{
				_cEnergyvsLS_SubdetPM.fill(did, _currentLS, energy);
				_cEnergyvsBX_SubdetPM.fill(did, bx, energy);
				_cEnergyvsieta_Subdet.fill(did, energy);
				_cEnergyvsiphi_SubdetPM.fill(did, energy);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsBX_SubdetPM.fill(did, bx, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
			}
			//	^^^ONLINE ONLY!
			
			_cEnergy_depth.fill(did, energy);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_depth.fill(did, timing);
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
			nChsHOCut++;
		}
		nChsHO++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 4), _currentLS,
		nChsHO);
	//	ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_cOccupancyCutvsBX_SubdetPM.fill(HcalDetId(HcalOuter, 1,1,1),
			bx, nChsHOCut);
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalOuter, 1, 1, 4), 
			_currentLS, nChsHOCut);
	}
	//	^^^ONLINE ONLY!

	int nChsHF = 0; int nChsHFCut = 0;
	double ehfp = 0; double ehfm = 0;
	for (HFRecHitCollection::const_iterator it=chf->begin();
		it!=chf->end(); ++it)
	{
		double energy = it->energy();
		double timing = it->time();
		HcalDetId did = it->id();
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
		_cEnergy_Subdet.fill(did, energy);
		_cTimingvsEnergy_SubdetPM.fill(did, energy, timing);
		_cOccupancy_depth.fill(did);
		did.ieta()>0?ehfp+=energy:ehfm+=energy;
		
		//	IMPORTANT:
		//	only for Online Processing
		//
		if (_ptype==fOnline)
		{
			_cOccupancyvsiphi_SubdetPM.fill(did);
			_cOccupancyvsieta_Subdet.fill(did);
		}
		//	ONLINE ONLY!

		if (eid.isVMEid())
		{
			_cOccupancy_FEDVME.fill(eid);
			_cOccupancy_ElectronicsVME.fill(eid);
		}
		else
		{
			_cOccupancy_FEDuTCA.fill(eid);
			_cOccupancy_ElectronicsuTCA.fill(eid);
		}

		if (energy>_cutE_HF)
		{
			//	ONLINE ONLY!
			if (_ptype==fOnline)
			{
				_cEnergyvsLS_SubdetPM.fill(did, _currentLS, energy);
				_cEnergyvsBX_SubdetPM.fill(did, bx, energy);
				_cEnergyvsieta_Subdet.fill(did, energy);
				_cEnergyvsiphi_SubdetPM.fill(did, energy);
				_cTimingCutvsieta_Subdet.fill(did, timing);
				_cTimingCutvsiphi_SubdetPM.fill(did, timing);
				_cTimingCutvsBX_SubdetPM.fill(did, bx, timing);
				_cOccupancyCutvsiphi_SubdetPM.fill(did);
				_cOccupancyCutvsieta_Subdet.fill(did);
				_cOccupancyCutvsiphivsLS_SubdetPM.fill(did, _currentLS);
				_xUniHF.get(eid)++;
			}
			//	^^^ONLINE ONLY!
			_cEnergy_depth.fill(did, energy);
			_cTimingCut_SubdetPM.fill(did, timing);
			_cTimingCutvsLS_FED.fill(eid, _currentLS, timing);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_depth.fill(did, timing);
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
			nChsHFCut++;
		}
		nChsHF++;
	}
	_cOccupancyvsLS_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), _currentLS,
		nChsHF);
	//	ONLINE ONLY!
	if (_ptype==fOnline)
	{
		_cOccupancyCutvsBX_SubdetPM.fill(HcalDetId(HcalForward, 1,1,1),
			bx, nChsHFCut);
		_cOccupancyCutvsLS_Subdet.fill(HcalDetId(HcalForward, 1, 1, 1), 
			_currentLS, nChsHFCut);
	}
	//	^^^ONLINE ONLY!
}

/* virtual */ void RecHitTask::beginLuminosityBlock(edm::LuminosityBlock const&
	lb, edm::EventSetup const& es)
{
	DQTask::beginLuminosityBlock(lb, es);

	/*
	_cTimingCutvsLS_FED.extendAxisRange(_currentLS);
	_cOccupancyvsLS_Subdet.extendAxisRange(_currentLS);

	//	ONLINE ONLY
	if (_ptype!=fOnline)
		return;
	_cEnergyvsLS_SubdetPM.extendAxisRange(_currentLS);
	_cOccupancyCutvsLS_Subdet.extendAxisRange(_currentLS);
	_cOccupancyCutvsiphivsLS_SubdetPM.extendAxisRange(_currentLS);
//	_cSummaryvsLS_FED.extendAxisRange(_currentLS);
//	_cSummaryvsLS.extendAxisRange(_currentLS);
//	*/
}

/* virtual */ void RecHitTask::endLuminosityBlock(edm::LuminosityBlock const& 
	lb, edm::EventSetup const& es)
{
	if (_ptype!=fOnline)
		return;

	//
	//	GENERATE STATUS ONLY FOR ONLINE
	//
//	for (std::vector<HcalGenericDetId>::const_iterator it=gids.begin();
//		it!=gids.end(); ++it)
//	{}

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
		flag::Flag fSum("RECO");
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

		//	FED is @cDAQ
		if (utilities::isFEDHF(eid) && (_runkeyVal==0 || _runkeyVal==4))
		{
			if (_xUni.get(eid)>0)
				_vflags[fUni]._state = flag::fBAD;
			else
				_vflags[fUni]._state = flag::fGOOD;
		}

		int iflag=0;
		for (std::vector<flag::Flag>::iterator ft=_vflags.begin();
			ft!=_vflags.end(); ++ft)
		{
			_cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag),
				int(ft->_state));
			fSum+=(*ft);
			iflag++;

			//	reset after using
			ft->reset();
		}
		_cSummaryvsLS.setBinContent(eid, _currentLS, fSum._state);
	}
	_xUniHF.reset(); _xUni.reset();

	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(RecHitTask);

