
#include "DQM/HcalTasks/interface/RecHitTask.h"

	using namespace hcaldqm;
	RecHitTask::RecHitTask(edm::ParameterSet const& ps) :
		DQTask(ps)
	{
		//	Energy
		_cEnergy_SubDet.initialize(_name+"/Energy/SubDet", "Energy", mapper::fSubDet, 
			new axis::ValueAxis(axis::fXaxis, axis::fEnergy),
			new axis::ValueAxis(axis::fYaxis, axis::fEntries, true), _debug);
		_cEnergyvsieta_SubDet.initialize(_name+"/Energy/vsieta_SubDet", "Energyvsieta",
			mapper::fSubDet, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::ValueAxis(axis::fYaxis, axis::fEnergy), _debug);
		_cEnergyvsiphi_SubDet.initialize(_name+"/Energy/vsiphi_SubDet", "Energyvsiphi",
			mapper::fSubDet, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fiphi), 
			new axis::ValueAxis(axis::fYaxis, axis::fEnergy), _debug);
		_cEnergy_depth.initialize(_name+"/Energy/depth", "Energy",
			mapper::fdepth, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
			new axis::ValueAxis(axis::fZaxis, axis::fEnergy, true), _debug);
		_cEnergyvsietaCut_SubDet.initialize(_name+"/Energy/vsieta_SubDet", "Energyvsieta",
			mapper::fSubDet, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::ValueAxis(axis::fYaxis, axis::fEnergy), _debug);
		_cEnergyvsiphiCut_SubDet.initialize(_name+"/Energy/vsiphi_SubDet", "Energyvsiphi",
			mapper::fSubDet, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fiphi), 
			new axis::ValueAxis(axis::fYaxis, axis::fEnergy), _debug);
		_cEnergyCut_depth.initialize(_name+"/Energy/depth", "Energy",
			mapper::fdepth, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
			new axis::ValueAxis(axis::fZaxis, axis::fEnergy, true), _debug);

		//	Timing
		_cTimingCut_SubDet.initialize(_name+"/Timing/SubDet", "Timing", mapper::fSubDet,
			new axis::ValueAxis(axis::fXaxis, axis::fTime), 
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);
		_cTimingCut_SubDetPM_iphi.initialize(_name+"/Timing/SubDetPM_iphi", "Timing", 
			mapper::fSubDetPM_iphi, 
			new axis::ValueAxis(axis::fXaxis, axis::fTime), 
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);
		_cTimingCut_depth.initialize(_name+"/Timing/depth", "Timing",
			mapper::fdepth, 
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
			new axis::ValueAxis(axis::fZaxis, axis::fTime), _debug);
		_cTimingCut_HBHEPrt.initialize(_name+"/Timing/HBHEPartition", "Timing",
			mapper::fHBHEPartition,
			new axis::ValueAxis(axis::fXaxis, axis::fTime), 
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);

		//	Occupancy
		_cOccupancy_depth.initialize(_name+"/Occupancy/depth", "Occupancy", mapper::fdepth,
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::CoordinateAxis(axis::fYaxis, axis::fiphi), 
			new axis::ValueAxis(axis::fZaxis, axis::fEntries), _debug);
		_cOccupancyvsLS_SubDet.initialize(_name+"/Occupancy/vsLS_SubDet", "Occupancy",
			mapper::fSubDet,
			new axis::ValueAxis(axis::fXaxis, axis::fLS),
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);
		_cOccupancyCutvsLS_SubDet.initialize(_name+"/Occupancy/vsLS_SubDet", "Occupancy",
			mapper::fSubDet,
			new axis::ValueAxis(axis::fXaxis, axis::fLS),
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);
		_cOccupancyCut_depth.initialize(_name+"/Occupancy/depth", "Occupancy", 
			mapper::fdepth,
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
			new axis::CoordinateAxis(axis::fYaxis, axis::fiphi), 
			new axis::ValueAxis(axis::fZaxis, axis::fEntries), _debug);
		_cOccupancyvsiphi_SubDetPM.initialize(_name+"/Occupancy/vsiphi_SubDetPM",
			"Occupancy", mapper::fSubDetPM,
			new axis::CoordinateAxis(axis::fXaxis, axis::fiphi),
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);
		_cOccupancyCutvsiphi_SubDetPM.initialize(_name+"/Occupancy/vsiphi_SubDetPM",
			"Occupancy", mapper::fSubDetPM,
			new axis::CoordinateAxis(axis::fXaxis, axis::fiphi),
			new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);

		//	Energy vs Timing
		if (this->_ptype!=fOffline)
		{
			//	these plots are too consuming and will only show up for 
			//	Online/Playback processing
			_cEnergy_SubDet_ieta.initialize(_name+"/Energy/SubDet_ieta", "Energy",
				mapper::fSubDet_ieta, 
				new axis::ValueAxis(axis::fXaxis, axis::fEnergy),
				new axis::ValueAxis(axis::fYaxis, axis::fEntries, true), _debug);
			_cEnergy_SubDetPM_iphi.initialize(_name+"/Energy/SubDetPM_iphi", 
				"Energy", mapper::fSubDetPM_iphi,
				new axis::ValueAxis(axis::fXaxis, axis::fEnergy),
				new axis::ValueAxis(axis::fYaxis, axis::fEntries, true), _debug);
			_cTimingCutvsLS_SubDetPM_iphi.initialize(_name+"/Timing/vsLS_SubDetPM_iphi",
				"Timing", mapper::fSubDetPM_iphi,
				new axis::ValueAxis(axis::fXaxis, axis::fLS),
				new axis::ValueAxis(axis::fYaxis, axis::fTime), _debug);
			_cTimingCut_SubDet_ieta.initialize(_name+"/Timing/SubDet_ieta", "Timing", 
				mapper::fSubDet_ieta, 
				new axis::ValueAxis(axis::fXaxis, axis::fTime), 
				new axis::ValueAxis(axis::fYaxis, axis::fEntries), _debug);
			_cTimingvsietaCut_SubDet_iphi.initialize(_name+"/Timing/vsieta_SubDet_iphi", 
				"Timing",
				mapper::fSubDet_iphi,
				new axis::CoordinateAxis(axis::fXaxis, axis::fieta),
				new axis::ValueAxis(axis::fYaxis, axis::fTime), _debug);
			_cTimingvsiphiCut_SubDet_ieta.initialize(_name+"/Timing/vsiphi_SubDet_ieta", 
				"Timing",
				mapper::fSubDet_ieta,
				new axis::CoordinateAxis(axis::fXaxis, axis::fiphi),
				new axis::ValueAxis(axis::fYaxis, axis::fTime), _debug);
			_cTimingvsEnergyCut_SubDetPM_iphi.initialize(
				_name+"/TimingvsEnergy/SubDetPM_iphi", 
				"TimingvsEnergy", mapper::fSubDetPM_iphi, 
				new axis::ValueAxis(axis::fXaxis, axis::fEnergy), 
				new axis::ValueAxis(axis::fYaxis, axis::fTime), 
				new axis::ValueAxis(axis::fZaxis, axis::fEntries), _debug);
		}

		//	Summary
		_cSummary.initialize(_name+"/Summary", "Summary",
			new axis::CoordinateAxis(axis::fXaxis, axis::fSubDet),
			new axis::FlagAxis(axis::fYaxis, "Flag", int(nRecHitFlag)));
		_cSummaryvsLS_SubDet.initialize(_name+"/Summary/vsLS_SubDet", "SummaryvsLS",
			mapper::fSubDet,
			new axis::ValueAxis(axis::fXaxis, axis::fLS),
			new axis::FlagAxis(axis::fYaxis, "Flag", int(nRecHitFlag)));
	
		//	tags
		_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
			edm::InputTag("hbhereco"));
		_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
			edm::InputTag("horeco"));
		_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
			edm::InputTag("hfreco"));
		_tokHBHE = consumes<HBHERecHitCollection>(_tagHBHE);
		_tokHO = consumes<HORecHitCollection>(_tagHO);
		_tokHF = consumes<HFRecHitCollection>(_tagHF);

		//	cuts
		_cutE_HBHE = ps.getUntrackedParameter<double>("cutE_HBHE", 5);
		_cutE_HO = ps.getUntrackedParameter<double>("cutE_HO", 5);
		_cutE_HF = ps.getUntrackedParameter<double>("cutE_HF", 5);

		//	load labels
		_fNames.push_back("Low Occupancy");
		_fNames.push_back("iphi Uniformity");
		_fNames.push_back("HBHE Partition Timing");
		_cSummary.loadLabels(_fNames);
		_cSummaryvsLS_SubDet.loadLabels(_fNames);
	}

	/* virtual */ void RecHitTask::bookHistograms(DQMStore::IBooker & ib,
		edm::Run const& r, edm::EventSetup const& es)
	{
		DQTask::bookHistograms(ib, r, es);
		char cutstr[200];
		sprintf(cutstr, "_EHBHE%dHO%dHF%d", int(_cutE_HBHE),
			int(_cutE_HO), int(_cutE_HF));

		_cEnergy_SubDet.book(ib);
		_cEnergyvsieta_SubDet.book(ib);
		_cEnergyvsiphi_SubDet.book(ib);
		_cEnergy_depth.book(ib);

		_cEnergyvsietaCut_SubDet.book(ib, _subsystem, std::string(cutstr));
		_cEnergyvsiphiCut_SubDet.book(ib, _subsystem, std::string(cutstr));
		_cEnergyCut_depth.book(ib, _subsystem, std::string(cutstr));

		_cTimingCut_SubDet.book(ib, _subsystem, std::string(cutstr));
		_cTimingCut_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
		_cTimingCut_depth.book(ib, _subsystem, std::string(cutstr));
		_cTimingCut_HBHEPrt.book(ib, _subsystem, std::string(cutstr));

		_cOccupancyCut_depth.book(ib, _subsystem, std::string(cutstr));
		_cOccupancy_depth.book(ib);
		_cOccupancyvsLS_SubDet.book(ib);
		_cOccupancyCutvsLS_SubDet.book(ib, _subsystem, std::string(cutstr));
		_cOccupancyvsiphi_SubDetPM.book(ib);
		_cOccupancyCutvsiphi_SubDetPM.book(ib, _subsystem, std::string(cutstr));

		if (this->_ptype!=fOffline)
		{
			//	Book the following histograms only when you are running not 
			//	Offline
			_cTimingvsEnergyCut_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
			_cTimingCutvsLS_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
			_cTimingCut_SubDet_ieta.book(ib, _subsystem, std::string(cutstr));
			_cTimingvsietaCut_SubDet_iphi.book(ib, _subsystem, std::string(cutstr));
			_cTimingvsiphiCut_SubDet_ieta.book(ib, _subsystem, std::string(cutstr));
			_cEnergy_SubDet_ieta.book(ib);
			_cEnergy_SubDetPM_iphi.book(ib);
		}

		_cSummary.book(ib);
		_cSummaryvsLS_SubDet.book(ib);
	}

	/* virtual */ void RecHitTask::_process(edm::Event const& e,
		edm::EventSetup const& es)
	{
		edm::Handle<HBHERecHitCollection>	chbhe;
		edm::Handle<HORecHitCollection>		cho;
		edm::Handle<HFRecHitCollection>		chf;

		if (!(e.getByToken(_tokHBHE, chbhe)))
			_logger.dqmthrow("Collection HBHERecHitCollection not available "
				+ _tagHBHE.label() + " " + _tagHBHE.instance());
		if (!(e.getByToken(_tokHO, cho)))
			_logger.dqmthrow("Collection HORecHitCollection not available "
				+ _tagHO.label() + " " + _tagHO.instance());
		if (!(e.getByToken(_tokHF, chf)))
			_logger.dqmthrow("Collection HFRecHitCollection not available "
				+ _tagHF.label() + " " + _tagHF.instance());

		//	Processing
		for (HBHERecHitCollection::const_iterator it=chbhe->begin();
			it!=chbhe->end(); ++it)
		{
			const HBHERecHit rh = (const HBHERecHit)(*it);
			double energy = rh.energy();
			double time = rh.time();
			const HcalDetId did = rh.id();

			_cEnergy_SubDet.fill(did, energy);
			_cEnergyvsieta_SubDet.fill(did, energy);
			_cEnergyvsiphi_SubDet.fill(did, energy);
			_cEnergy_depth.fill(did, energy);
			
			_cOccupancy_depth.fill(did);
			_cOccupancyvsiphi_SubDetPM.fill(did);
			_nRecHits[did.subdet()-1]++;

			if (this->_ptype!=fOffline)
			{
				_cEnergy_SubDet_ieta.fill(did, energy);
				_cEnergy_SubDetPM_iphi.fill(did, energy);
			}

			if (energy>_cutE_HBHE)
			{
				_cEnergyCut_depth.fill(did, energy);
				_cEnergyvsietaCut_SubDet.fill(did, energy);
				_cEnergyvsiphiCut_SubDet.fill(did, energy);
				_cEnergyCut_depth.fill(did, energy);
				_cTimingCut_SubDet.fill(did, time);
				_cTimingCut_SubDetPM_iphi.fill(did, time);
				_cTimingCut_depth.fill(did, time);
				_cOccupancyCut_depth.fill(did);
				_cOccupancyCutvsiphi_SubDetPM.fill(did);
				_nRecHitsCut[did.subdet()-1]++;
				_cTimingCut_HBHEPrt.fill(did, time);

				if (this->_ptype!=fOffline)
				{
					//	fill the following plots only when we aren't in Offline
					//
					_cTimingvsEnergyCut_SubDetPM_iphi.fill(did, energy, time);
					_cTimingCutvsLS_SubDetPM_iphi.fill(did, _currentLS, time);
					_cTimingCut_SubDet_ieta.fill(did, time);
					_cTimingvsietaCut_SubDet_iphi.fill(did, time);
					_cTimingvsiphiCut_SubDet_ieta.fill(did, time);
				}
			}
		}
		for (HORecHitCollection::const_iterator it=cho->begin();
			it!=cho->end(); ++it)
		{
			const HORecHit rh = (const HORecHit)(*it);
			double energy = rh.energy();
			double time = rh.time();
			const HcalDetId did = rh.id();

			_cEnergy_SubDet.fill(did, energy);
			_cEnergyvsieta_SubDet.fill(did, energy);
			_cEnergyvsiphi_SubDet.fill(did, energy);
			_cEnergy_depth.fill(did, energy);
			
			_cOccupancy_depth.fill(did);
			_cOccupancyvsiphi_SubDetPM.fill(did);
			_nRecHits[did.subdet()-1]++;

			if (this->_ptype!=fOffline)
			{
				_cEnergy_SubDet_ieta.fill(did, energy);
				_cEnergy_SubDetPM_iphi.fill(did, energy);
			}

			if (energy>_cutE_HO)
			{
				_cEnergyCut_depth.fill(did, energy);
				_cEnergyvsietaCut_SubDet.fill(did, energy);
				_cEnergyvsiphiCut_SubDet.fill(did, energy);
				_cEnergyCut_depth.fill(did, energy);
				_cTimingCut_SubDet.fill(did, time);
				_cTimingCut_SubDetPM_iphi.fill(did, time);
				_cTimingCut_depth.fill(did, time);
				_cOccupancyCut_depth.fill(did);
				_cOccupancyCutvsiphi_SubDetPM.fill(did);
				_nRecHitsCut[did.subdet()-1]++;

				if (this->_ptype!=fOffline)
				{
					//	Fill the following plots only when you are running not 
					//	Offline
					_cTimingvsEnergyCut_SubDetPM_iphi.fill(did, energy, time);
					_cTimingCutvsLS_SubDetPM_iphi.fill(did, _currentLS, time);
					_cTimingCut_SubDet_ieta.fill(did, time);
					_cTimingvsietaCut_SubDet_iphi.fill(did, time);
					_cTimingvsiphiCut_SubDet_ieta.fill(did, time);
				}
			}
		}
		for (HFRecHitCollection::const_iterator it=chf->begin();
			it!=chf->end(); ++it)
		{
			const HFRecHit rh = (const HFRecHit)(*it);
			double energy = rh.energy();
			double time = rh.time();
			const HcalDetId did = rh.id();

			_cEnergy_SubDet.fill(did, energy);
			_cEnergyvsieta_SubDet.fill(did, energy);
			_cEnergyvsiphi_SubDet.fill(did, energy);
			_cEnergy_depth.fill(did, energy);
			
			_cOccupancy_depth.fill(did);
			_cOccupancyvsiphi_SubDetPM.fill(did);
			_nRecHits[did.subdet()-1]++;

			if (this->_ptype!=fOffline)
			{
				_cEnergy_SubDet_ieta.fill(did, energy);
				_cEnergy_SubDetPM_iphi.fill(did, energy);
			}

			if (energy>_cutE_HF)
			{
				_cEnergyCut_depth.fill(did, energy);
				_cEnergyvsietaCut_SubDet.fill(did, energy);
				_cEnergyvsiphiCut_SubDet.fill(did, energy);
				_cEnergyCut_depth.fill(did, energy);
				_cTimingCut_SubDet.fill(did, time);
				_cTimingCut_SubDetPM_iphi.fill(did, time);
				_cTimingCut_depth.fill(did, time);
				_cOccupancyCut_depth.fill(did);
				_cOccupancyCutvsiphi_SubDetPM.fill(did);
				_nRecHitsCut[did.subdet()-1]++;

				if (this->_ptype!=fOffline)
				{
					//	Fill the following plots only when processing not 
					//	Offline
					_cTimingvsEnergyCut_SubDetPM_iphi.fill(did, energy, time);
					_cTimingCutvsLS_SubDetPM_iphi.fill(did, _currentLS, time);
					_cTimingCut_SubDet_ieta.fill(did, time);
					_cTimingvsietaCut_SubDet_iphi.fill(did, time);
					_cTimingvsiphiCut_SubDet_ieta.fill(did, time);
				}
			}
		}

		_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalBarrel, 5, 5, 1),
			_currentLS, _nRecHits[0]);
		_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalBarrel, 5, 5, 1),
			_currentLS, _nRecHitsCut[0]);
		_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalEndcap, 18, 5, 1),
			_currentLS, _nRecHits[1]);
		_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalEndcap, 18, 5, 1),
			_currentLS, _nRecHitsCut[1]);
		_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalOuter, 5, 5, 4),
			_currentLS, _nRecHits[2]);
		_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalOuter, 5, 5, 4),
			_currentLS, _nRecHitsCut[2]);
		_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalForward, 34, 5, 1),
			_currentLS, _nRecHits[3]);
		_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalForward, 34, 5, 1),
			_currentLS, _nRecHitsCut[3]);
	}

	/* virtual */ void RecHitTask::_resetMonitors(UpdateFreq uf)
	{
		switch(uf)
		{
			case fEvent:
				for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
				{
					_nRecHits[i] = 0;
					_nRecHitsCut[i] = 0;
				}
				break;
			default:
				break;
		}
		DQTask::_resetMonitors(uf);
	}

/* virtual */ void RecHitTask::endLuminosityBlock(edm::LuminosityBlock const& l,
	edm::EventSetup const& es)
{
	//	statuses
	//	By default all the flags are set as NOT APPLICABLE
	double status[constants::SUBDET_NUM][nRecHitFlag];
	for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
		for (int j=fLowOcp; j<nRecHitFlag; j++)
			status[i][j] = constants::NOT_APPLICABLE;

	/*
	 *	Do the checks here
	 *	->	HF Digi Occupancy
	 *	-> iphi Uniformity
	 */
	MonitorElement *meocpHF = _cOccupancyvsLS_SubDet.at(3);
	double numChs = meocpHF->getBinContent(_currentLS);
	if (constants::CHS_NUM[3] - numChs>=48)
		status[3][fLowOcp] = constants::VERY_LOW;
	else if (constants::CHS_NUM[3]-numChs>=24)
		status[3][fLowOcp] = constants::LOW;
	else if (constants::CHS_NUM[3]-numChs>=10)
		status[3][fLowOcp] = constants::LOW;
	else if (constants::CHS_NUM[3] - numChs>=1)
		status[3][fLowOcp] = constants::PROBLEMATIC;
	else if (constants::CHS_NUM[3]-numChs<0)
		status[3][fLowOcp] = constants::PROBLEMATIC;
	else if (constants::CHS_NUM[3]==numChs)
		status[3][fLowOcp] = constants::GOOD;

	//	Check HF uniformity vs iphi
	for (int i=0; i<IPHI_NUM; i+=4)
	{
		int i1 = (IPHI_NUM-1+i)%IPHI_NUM;
		int i2 = (IPHI_NUM-1+2+i)%IPHI_NUM;
		int j1 = (IPHI_NUM-1+4+i)%IPHI_NUM;
		int j2 = (IPHI_NUM-1+6+i)%IPHI_NUM;

		double occ1_m = _cOccupancyCutvsiphi_SubDetPM.getBinContent(6, i1) + 
			_cOccupancyCutvsiphi_SubDetPM.getBinContent(6, i2);
		double occ2_m = _cOccupancyCutvsiphi_SubDetPM.getBinContent(6, j1) + 
			_cOccupancyCutvsiphi_SubDetPM.getBinContent(6, j2);
		double ratio_m = std::min(occ1_m, occ2_m)/std::max(occ1_m, occ2_m);
		double occ1_p = _cOccupancyCutvsiphi_SubDetPM.getBinContent(7, i1) +
			_cOccupancyCutvsiphi_SubDetPM.getBinContent(7, i2);
		double occ2_p = _cOccupancyCutvsiphi_SubDetPM.getBinContent(7, j1) + 
			_cOccupancyCutvsiphi_SubDetPM.getBinContent(7, j2);
		double ratio_p = std::min(occ1_p, occ2_p)/std::max(occ1_p, occ2_p);

		if (ratio_m<0.8 || ratio_p<0.8)
		{
			//	set and exit
			status[3][fUniphi] = constants::VERY_LOW;
			break;
		}
		else 
			status[3][fUniphi] = constants::GOOD;
	}

	//	Check the shifts between HBHEabc partitions
	//	Currently, we use 1.5 ns threshold
	double mean_A = _cTimingCut_HBHEPrt.at(0)->getMean();
	double mean_B = _cTimingCut_HBHEPrt.at(1)->getMean();
	double mean_C = _cTimingCut_HBHEPrt.at(2)->getMean();

	double diff_AB = abs(mean_A - mean_B);
	double diff_AC = abs(mean_A - mean_C);
	double diff_BC = abs(mean_B - mean_C);

	if (diff_AB>=1.5 || diff_AC>=1.5 || diff_BC>=1.5)
	{
		status[0][fTCDS] = constants::LOW;
		status[1][fTCDS] = constants::LOW;
	}
	else 
	{
		status[0][fTCDS] = constants::GOOD;
		status[1][fTCDS] = constants::GOOD;
	}

	//	fill the statuses in the end
	for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
		for (int j=fLowOcp; j<nRecHitFlag; j++)
		{
			_cSummary.setBinContent(i, j, status[i][j]);
			_cSummaryvsLS_SubDet.setBinContent(i, _currentLS, j, status[i][j]);
		}

	DQTask::endLuminosityBlock(l, es);
}

DEFINE_FWK_MODULE(RecHitTask);




