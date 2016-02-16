
#include "DQM/HcalTasks/interface/DigiTask.h"

using namespace hcaldqm;
DigiTask::DigiTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	//	Signal, ADC, fC, SumQ
	_cfCperTS_SubDet.initialize(_name+"/Signal/fC_SubDet", "fCperTS",
		mapper::fSubDet,
		new axis::ValueAxis(axis::fXaxis, axis::fNomFC),
		new axis::ValueAxis(axis::fYaxis, axis::fEntries, true));
	_cADCperTS_SubDet.initialize(_name+"/Signal/ADC_SubDet", "ADCperTS",
		mapper::fSubDet,
		new axis::ValueAxis(axis::fXaxis, axis::fADC),
		new axis::ValueAxis(axis::fYaxis, axis::fEntries, true));
	_cSumQ_depth.initialize(_name+"/Signal/depth", "SumQ",
		mapper::fdepth,
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta),
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
		new axis::ValueAxis(axis::fZaxis, axis::fNomFC));

	//	Shape
	_cShapeCut_SubDetPM_iphi.initialize(_name+"/Shape/SubDetPM_iphi", "Shape",
		mapper::fSubDetPM_iphi,
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS),
		new axis::ValueAxis(axis::fYaxis, axis::fNomFC));
	_cShapeCut_p3e41d2.initialize(_name+"/Shape/iphi3ieta41d2", "Shape",
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS),
		new axis::ValueAxis(axis::fYaxis, axis::fNomFC));
	_cShapeCut_p3em41d2.initialize(_name+"/Shape/iphi3ieta-41d2", "Shape",
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS),
		new axis::ValueAxis(axis::fYaxis, axis::fNomFC));

	//	Timing
	_cTimingCut_SubDetPM_iphi.initialize(_name+"/Timing/SubDetPM_iphi", 
		"Timing", mapper::fSubDetPM_iphi,
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS_200));
	_cTimingCut_depth.initialize(_name+"/Timing/depth", "Timing",
		mapper::fdepth,
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta),
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
		new axis::ValueAxis(axis::fZaxis, axis::fTimeTS_200));

	//	Special
	_cQ2Q12CutvsLS_p3e41d2.initialize(_name+"/Q2Q12/vsLS_iphi3ieta41d2",
		"Q2Q12",
		new axis::ValueAxis(axis::fXaxis, axis::fLS),
		new axis::ValueAxis(axis::fYaxis, axis::fRatio));
	_cQ2Q12CutvsLS_p3em41d2.initialize(_name+"/Q2Q12/vsLS_iphi3ieta-41d2",
		"Q2Q12",
		new axis::ValueAxis(axis::fXaxis, axis::fLS),
		new axis::ValueAxis(axis::fYaxis, axis::fRatio));
	_cDigiSizevsLS_SubDet.initialize(_name+"/DigiSize/vsLS_SubDet", "DigiSize",
		mapper::fSubDet,
		new axis::ValueAxis(axis::fXaxis, axis::fLS),
		new axis::ValueAxis(axis::fYaxis, axis::fDigiSize));
	_cCapIdRots_depth.initialize(
		_name+"/CapIdRotations/depth", "CapIdRotations",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi));

	//	Occupancy
	_cOccupancyvsiphi_SubDetPM.initialize(_name+"/Occupancy/vsiphi_SubDetPM", "Occupancyvsiphi",
		mapper::fSubDetPM,
		new axis::CoordinateAxis(fXaxis, axis::fiphi));
	_cOccupancyCutvsiphi_SubDetPM.initialize(
		_name+"/Occupancy/vsiphi_SubDetPM", 
		"Occupancyvsiphi",
		mapper::fSubDetPM,
		new axis::CoordinateAxis(fXaxis, axis::fiphi));
	_cOccupancyvsLS_SubDet.initialize(_name+"/Occupancy/vsLS_SubDet",
		"OccupancyvsLS",
		mapper::fSubDet,
		new axis::ValueAxis(fXaxis, axis::fLS),
		new axis::ValueAxis(fYaxis, axis::fEntries));
	_cOccupancyCutvsLS_SubDet.initialize(_name+"/Occupancy/vsLS_SubDet",
		"OccupancyvsLS",
		mapper::fSubDet,
		new axis::ValueAxis(fXaxis, axis::fLS),
		new axis::ValueAxis(fYaxis, axis::fEntries));
	_cOccupancy_depth.initialize(_name+"/Occupancy/depth", "Occupancy",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi));
	_cOccupancyCut_depth.initialize(_name+"/Occupancy/depth", "Occupancy",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi));
	_cOccupancyOnce_depth.initialize(_name+"/Occupancy/Once_depth",
		"Occupancy", mapper::fdepth,
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta),
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi));
	_cMsn1LS_depth.initialize(_name+"/Missing/1LS_depth", "Missing",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi));
	_cMsn10LS_depth.initialize(_name+"/Missing/10LS_depth", "Missing",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi));
	_cMsn1LSvsLS_SubDet.initialize(_name+"/Missing/1LSvsLS_SubDet", "Missing",
		mapper::fSubDet,
		new axis::ValueAxis(axis::fXaxis, axis::fLS),
		new axis::ValueAxis(axis::fYaxis, axis::fEntries));

	//	Summaries
	_cSummary.initialize(_name+"/Summary", "Summary",
		new axis::CoordinateAxis(axis::fXaxis, axis::fSubDet),
		new axis::FlagAxis(axis::fYaxis, "Flag", int(nDigiFlag)));
	_cSummaryvsLS_SubDet.initialize(_name+"/Summary/vsLS_SubDet", "SummaryvsLS",
		mapper::fSubDet,
		new axis::ValueAxis(axis::fXaxis, axis::fLS),
		new axis::FlagAxis(axis::fYaxis, "Flag", int(nDigiFlag)));

	//	Initialize what should be present only for Online or Playback, not for 
	//	Offline
	if (this->_ptype!=fOffline)
	{
		_cSumQ_SubDetPM_iphi.initialize(_name+"/Signal/SubDetPM_iphi", "SumQ",
			mapper::fSubDetPM_iphi,
			new axis::ValueAxis(axis::fXaxis, axis::fNomFC),
			new axis::ValueAxis(axis::fYaxis, axis::fEntries, true));
		_cShape_SubDetPM_iphi.initialize(_name+"/Shape/SubDetPM_iphi", "Shape",
			mapper::fSubDetPM_iphi,
			new axis::ValueAxis(axis::fXaxis, axis::fTimeTS),
			new axis::ValueAxis(axis::fYaxis, axis::fNomFC));
		_cSumQvsLS_SubDetPM_iphi.initialize(_name+"/Signal/vsLS_SubDetPM_iphi", "SumQvsLS",
			mapper::fSubDetPM_iphi,
			new axis::ValueAxis(axis::fXaxis, axis::fLS),
			new axis::ValueAxis(axis::fYaxis, axis::fNomFC));
		_cTimingCutvsLS_SubDetPM_iphi.initialize(_name+"/Timing/vsLS_SubDetPM_iphi",
			"Timing", mapper::fSubDetPM_iphi,
			new axis::ValueAxis(axis::fXaxis, axis::fLS),
			new axis::ValueAxis(axis::fYaxis, axis::fTimeTS_200));
		_cTimingCutvsieta_SubDet_iphi.initialize(_name+"/Timing/vsieta_SubDet_iphi",
			"Timingvsieta", mapper::fSubDet_iphi,
			new axis::CoordinateAxis(axis::fXaxis, axis::fieta),
			new axis::ValueAxis(axis::fYaxis, axis::fTimeTS_200));
		_cTimingCutvsiphi_SubDet_ieta.initialize(_name+"/Timing/vsiphi_SubDet_ieta",
			"Timingvsiphi", mapper::fSubDet_ieta,
			new axis::CoordinateAxis(axis::fXaxis, axis::fiphi),
			new axis::ValueAxis(axis::fYaxis, axis::fTimeTS_200));
		_cQ2Q12CutvsLS_HFPM_iphi.initialize(_name+"/Q2Q12/vsLS_HFPM_iphi",
			"Q2Q12", mapper::fHFPM_iphi,
			new axis::ValueAxis(axis::fXaxis, axis::fLS),
			new axis::ValueAxis(axis::fYaxis, axis::fRatio));

	}

	//	tags and tokens
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));
	_tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<HFDigiCollection>(_tagHF);

	// cuts
	_cutSumQ_HBHE = ps.getUntrackedParameter<double>("cutSumQ_HBHE", 20);
	_cutSumQ_HO = ps.getUntrackedParameter<double>("cutSumQ_HO", 20);
	_cutSumQ_HF = ps.getUntrackedParameter<double>("cutSumQ_HF", 20);
	
	//	flags
	_fNames.push_back("Low Occupancy");
	_fNames.push_back("Digi Size Drift");
	_fNames.push_back("iphi Uniformity");
	_fNames.push_back("Missing for 1LS");
	_fNames.push_back("Cap Id Rotation");
	_cSummary.loadLabels(_fNames);
	_cSummaryvsLS_SubDet.loadLabels(_fNames);
}

/* virtual */ void DigiTask::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	char cutstr[200];
	sprintf(cutstr, "_SumQHBHE%dHO%dHF%d", int(_cutSumQ_HBHE),
		int(_cutSumQ_HO), int(_cutSumQ_HF));
	char cutstr2[200];
	sprintf(cutstr2, "_SumQHF%d", int(_cutSumQ_HF));

	DQTask::bookHistograms(ib, r, es);
	_cADCperTS_SubDet.book(ib);
	_cfCperTS_SubDet.book(ib);
	_cSumQ_depth.book(ib);

	_cShapeCut_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
	_cShapeCut_p3e41d2.book(ib, _subsystem, std::string(cutstr));
	_cShapeCut_p3em41d2.book(ib, _subsystem, std::string(cutstr));

	_cTimingCut_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
	_cQ2Q12CutvsLS_p3e41d2.book(ib, _subsystem, std::string(cutstr2));
	_cQ2Q12CutvsLS_p3em41d2.book(ib, _subsystem, std::string(cutstr2));
	_cTimingCut_depth.book(ib, _subsystem, std::string(cutstr));

	_cOccupancyvsiphi_SubDetPM.book(ib);
	_cOccupancyCutvsiphi_SubDetPM.book(ib, _subsystem, std::string(cutstr));
	_cOccupancyvsLS_SubDet.book(ib);
	_cOccupancyCutvsLS_SubDet.book(ib, _subsystem, std::string(cutstr));
	_cOccupancy_depth.book(ib);
	_cOccupancyOnce_depth.book(ib);
	_cOccupancyCut_depth.book(ib, _subsystem, std::string(cutstr));
	_cMsn1LS_depth.book(ib);
	_cMsn10LS_depth.book(ib);
	_cMsn1LSvsLS_SubDet.book(ib);

	_cDigiSizevsLS_SubDet.book(ib);
	_cCapIdRots_depth.book(ib);

	_cSummary.book(ib);
	_cSummaryvsLS_SubDet.book(ib);

	if (this->_ptype!=fOffline)
	{
		_cSumQ_SubDetPM_iphi.book(ib);
		_cShape_SubDetPM_iphi.book(ib);
		_cSumQvsLS_SubDetPM_iphi.book(ib);
		_cTimingCutvsLS_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
		_cTimingCutvsieta_SubDet_iphi.book(ib, _subsystem, 
			std::string(cutstr));
		_cTimingCutvsiphi_SubDet_ieta.book(ib, _subsystem, std::string(cutstr));
		_cQ2Q12CutvsLS_HFPM_iphi.book(ib, _subsystem, std::string(cutstr2));
	}
}

/* virtual */ void DigiTask::_resetMonitors(UpdateFreq uf)
{
	switch (uf)
	{
		case fEvent:
			for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
			{
				_numDigis[i]=0;
				_numDigisCut[i] = 0;
			}
			break;
		case hcaldqm::fLS:
			for (unsigned int idet=0; idet<constants::SUBDET_NUM; idet++)
			{
				for (int iiphi=0; iiphi<constants::IPHI_NUM; iiphi++)
					for (int iieta=0; iieta<constants::IETA_NUM; iieta++)
						for (int id=0; id<constants::DEPTH_NUM; id++)
						{
							_occ_1LS[idet][iiphi][iieta][id] = false;
							_error_1LS[idet][iiphi][iieta][id] = false;
						}
				_nMsn[idet] = 0;
				_nCapIdRots[idet] = 0;
			}
			break;
		case hcaldqm::f10LS:
			for (unsigned int idet=0; idet<constants::SUBDET_NUM; idet++)
				for (int iiphi=0; iiphi<constants::IPHI_NUM; iiphi++)
					for (int iieta=0; iieta<constants::IETA_NUM; iieta++)
						for (int id=0; id<constants::DEPTH_NUM; id++)
							_occ_10LS[idet][iiphi][iieta][id] = false;
			_cMsn1LS_depth.reset();
			_cCapIdRots_depth.reset();
			break;
		case hcaldqm::f50LS:
			_cMsn10LS_depth.reset();
			break;
		default:
			break;
	}
	DQTask::_resetMonitors(uf);
}

/* virtual */ void DigiTask::_process(edm::Event const& e,
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
		double sumQ = utilities::sumQ<HBHEDataFrame>(digi, 2.5, 0, 
			digi.size()-1);
		double timing = utilities::aveTS<HBHEDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		const HcalDetId did = digi.id();
		int iieta = did.ieta()<0 ? abs(did.ieta())-constants::IETA_MIN :
			did.ieta()-constants::IETA_MIN+constants::IETA_NUM/2;

		//	fill without a cut
		_cOccupancy_depth.fill(did);
		_cOccupancyvsiphi_SubDetPM.fill(did);
		_cSumQ_depth.fill(did, sumQ);
		_numDigis[did.subdet()-1]++;
		_cDigiSizevsLS_SubDet.fill(did, _currentLS, digi.size());
		_occ_1LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = true;
		_occ_10LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = true;
		_error_1LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = 
			utilities::isError<HBHEDataFrame>(digi);
		if (this->_ptype!=fOffline)
		{
			_cSumQvsLS_SubDetPM_iphi.fill(did, _currentLS, sumQ);
			_cSumQ_SubDetPM_iphi.fill(did, sumQ);
		}
		if (_occ_Always[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1]==false)
		{
			_cOccupancyOnce_depth.fill(did);
			_occ_Always[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1]=true;
		}

		//	fill with a cut
		if (sumQ>_cutSumQ_HBHE)
		{
			_cTimingCut_SubDetPM_iphi.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cOccupancyCutvsiphi_SubDetPM.fill(did);
			_cOccupancyCut_depth.fill(did);

			_numDigisCut[digi.id().subdet()-1]++;
			if (this->_ptype!=fOffline)
			{
				_cTimingCutvsLS_SubDetPM_iphi.fill(did, _currentLS, timing);
				_cTimingCutvsieta_SubDet_iphi.fill(did, timing);
				_cTimingCutvsiphi_SubDet_ieta.fill(did,	timing);
			}
		}
		
		//	per TS
		for (int i=0; i<digi.size(); i++)
		{
			//	without a cut
			_cADCperTS_SubDet.fill(did, digi.sample(i).adc());
			_cfCperTS_SubDet.fill(did, digi.sample(i).nominal_fC());
			if (this->_ptype!=fOffline)
			{
				_cShape_SubDetPM_iphi.fill(did, i, digi.sample(i).nominal_fC()-2.5);
			}

			//	with a cut
			if (sumQ>_cutSumQ_HBHE)
			{
				_cShapeCut_SubDetPM_iphi.fill(did, i,
					digi.sample(i).nominal_fC()-2.5);
			}
		}
	}
	for (HODigiCollection::const_iterator it=cho->begin();
		it!=cho->end(); ++it)
	{
		const HODataFrame digi = (const HODataFrame)(*it);
		double sumQ = utilities::sumQ<HODataFrame>(digi, 8.5, 0, 
			digi.size()-1);
		double timing = utilities::aveTS<HODataFrame>(digi, 8.5, 0,
			digi.size()-1);
		const HcalDetId did = digi.id();
		int iieta = did.ieta()<0 ? abs(did.ieta())-constants::IETA_MIN :
			did.ieta()-constants::IETA_MIN+constants::IETA_NUM/2;

		//	fill without a cut
		_cOccupancy_depth.fill(did);
		_cOccupancyvsiphi_SubDetPM.fill(did);
		_cSumQ_depth.fill(did, sumQ);
		_numDigis[did.subdet()-1]++;
		_cDigiSizevsLS_SubDet.fill(did, _currentLS, digi.size());
		_occ_1LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = true;
		_occ_10LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = true;
		_error_1LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = 
			utilities::isError<HODataFrame>(digi);
		if (this->_ptype!=fOffline)
		{
			_cSumQvsLS_SubDetPM_iphi.fill(did, _currentLS, sumQ);
			_cSumQ_SubDetPM_iphi.fill(did, sumQ);
		}
		if (_occ_Always[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1]==false)
		{
			_cOccupancyOnce_depth.fill(did);
			_occ_Always[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1]=true;
		}

		//	fill with a cut
		if (sumQ>_cutSumQ_HO)
		{
			_cTimingCut_SubDetPM_iphi.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cOccupancyCutvsiphi_SubDetPM.fill(did);
			_cOccupancyCut_depth.fill(did);
		
			_numDigisCut[digi.id().subdet()-1]++;

			if (this->_ptype!=fOffline)
			{
				_cTimingCutvsLS_SubDetPM_iphi.fill(did, _currentLS, timing);
				_cTimingCutvsieta_SubDet_iphi.fill(did, timing);
				_cTimingCutvsiphi_SubDet_ieta.fill(did,	timing);
			}
		}
		
		//	per TS
		for (int i=0; i<digi.size(); i++)
		{
			//	without a cut
			_cADCperTS_SubDet.fill(did, digi.sample(i).adc());
			_cfCperTS_SubDet.fill(did, digi.sample(i).nominal_fC());
			if (this->_ptype!=fOffline)
			{
				_cShape_SubDetPM_iphi.fill(did, i, digi.sample(i).nominal_fC()-8.5);
			}

			//	with a cut
			if (sumQ>_cutSumQ_HO)
			{
				_cShapeCut_SubDetPM_iphi.fill(did, i,
					digi.sample(i).nominal_fC()-8.5);
			}
		}
	}
	for (HFDigiCollection::const_iterator it=chf->begin();
		it!=chf->end(); ++it)
	{
		const HFDataFrame digi = (const HFDataFrame)(*it);
		double sumQ = utilities::sumQ<HFDataFrame>(digi, 2.5, 0, 
			digi.size()-1);
		double timing = utilities::aveTS<HFDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		const HcalDetId did = digi.id();
		int iieta = did.ieta()<0 ? abs(did.ieta())-constants::IETA_MIN :
			did.ieta()-constants::IETA_MIN+constants::IETA_NUM/2;

		//	fill without a cut
		_cOccupancy_depth.fill(did);
		_cOccupancyvsiphi_SubDetPM.fill(did);
		_cSumQ_depth.fill(did, sumQ);
		_numDigis[digi.id().subdet()-1]++;
		_cDigiSizevsLS_SubDet.fill(did, _currentLS, digi.size());
		_occ_1LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = true;
		_occ_10LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = true;
		_error_1LS[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1] = 
			utilities::isError<HFDataFrame>(digi);
		if (this->_ptype!=fOffline)
		{
			_cSumQvsLS_SubDetPM_iphi.fill(did, _currentLS, sumQ);
			_cSumQ_SubDetPM_iphi.fill(did, sumQ);
		}
		if (_occ_Always[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1]==false)
		{
			_cOccupancyOnce_depth.fill(did);
			_occ_Always[did.subdet()-1][did.iphi()-1][iieta][did.depth()-1]=true;
		}

		//	fill with a cut
		if (sumQ>_cutSumQ_HF)
		{
			_cTimingCut_SubDetPM_iphi.fill(did, timing);
			_cTimingCut_depth.fill(did, timing);
			_cOccupancyCutvsiphi_SubDetPM.fill(did);
			_cOccupancyCut_depth.fill(did);
		
			double q1 = digi.sample(1).nominal_fC()-2.5;
			double q2 = digi.sample(2).nominal_fC()-2.5;
			double q2q12 = q2/(q1+q2);
			if (did.iphi()==3 && did.ieta()==41 && did.depth()==2)
				_cQ2Q12CutvsLS_p3e41d2.fill(did, _currentLS, q2q12);
			if (did.iphi()==3 && did.ieta()==-41 && did.depth()==2)
				_cQ2Q12CutvsLS_p3em41d2.fill(did, _currentLS, q2q12);
			
		
			_numDigisCut[digi.id().subdet()-1]++;
			
			if (this->_ptype!=fOffline)
			{
				_cTimingCutvsLS_SubDetPM_iphi.fill(did, _currentLS, timing);
				_cTimingCutvsieta_SubDet_iphi.fill(did, timing);
				_cTimingCutvsiphi_SubDet_ieta.fill(did,	timing);
				_cQ2Q12CutvsLS_HFPM_iphi.fill(did, _currentLS, q2q12);
			}
		}
		
		//	per TS
		for (int i=0; i<digi.size(); i++)
		{
			//	without a cut
			_cADCperTS_SubDet.fill(did, digi.sample(i).adc());
			_cfCperTS_SubDet.fill(did, digi.sample(i).nominal_fC());
			if (this->_ptype!=fOffline)
			{
				_cShape_SubDetPM_iphi.fill(did, i, digi.sample(i).nominal_fC()-2.5);
			}

			//	with a cut
			if (sumQ>_cutSumQ_HF)
			{
				_cShapeCut_SubDetPM_iphi.fill(did, i,
					digi.sample(i).nominal_fC()-2.5);
				if (did.iphi()==3 && did.ieta()==41 && did.depth()==2)
					_cShapeCut_p3e41d2.fill(did, i,
						digi.sample(i).nominal_fC()-2.5);
				if (did.iphi()==3 && did.ieta()==-41 && did.depth()==2)
					_cShapeCut_p3em41d2.fill(did, i,
						digi.sample(i).nominal_fC()-2.5);
			}
		}
	}

	//	Fill the occupancy vs LS
	_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalBarrel, 5, 5, 1), _currentLS,
		_numDigis[0]);
	_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalBarrel, 5, 5, 1), _currentLS,
		_numDigisCut[0]);
	_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalEndcap, 18, 5, 1), _currentLS,
		_numDigis[1]);
	_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalEndcap, 18, 5, 1), _currentLS,
		_numDigisCut[1]);
	_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalOuter, 5, 5, 4), _currentLS,
		_numDigis[2]);
	_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalOuter, 5, 5, 4), _currentLS,
		_numDigisCut[2]);
	_cOccupancyvsLS_SubDet.fill(HcalDetId(HcalForward, 34, 5, 1), _currentLS,
		_numDigis[3]);
	_cOccupancyCutvsLS_SubDet.fill(HcalDetId(HcalForward, 34, 5, 1), _currentLS,
		_numDigisCut[3]);
}

/* virtual */ void DigiTask::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	//	statuses
	//	By default the flag is not applicable
	double status[constants::SUBDET_NUM][nDigiFlag]; 
	for (int j=fLowOcp; j<nDigiFlag; j++)
		for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
			status[i][j]=constants::NOT_APPLICABLE;

	/*
	 * Do the checks here.
	 * -> HF Digi Occupancy
	 * -> Digi Size Fluctuations
	 * -> Missing (or Dead Channels)
	 */

	//	HF Digi Occupancy Check
	MonitorElement *meocpHF = _cOccupancyvsLS_SubDet.at(3);
	double numChs = meocpHF->getBinContent(_currentLS);
	if (constants::CHS_NUM[3] - numChs>=48)
		status[3][fLowOcp] = constants::VERY_LOW;
	else if (constants::CHS_NUM[3] - numChs>=24)
		status[3][fLowOcp] = constants::LOW;
	else if (constants::CHS_NUM[3] - numChs>=10)
		status[3][fLowOcp] = constants::LOW;
	else if (constants::CHS_NUM[3] - numChs>=1)
		status[3][fLowOcp] = constants::PROBLEMATIC;
	else if (constants::CHS_NUM[3] - numChs<0)
		status[3][fLowOcp] = constants::PROBLEMATIC;
	else if (constants::CHS_NUM[3]==numChs)
		status[3][fLowOcp] = constants::GOOD;

	//	Digi Size Check
	for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
	{
		MonitorElement *meds = _cDigiSizevsLS_SubDet.at(i);
		double size = meds->getBinContent(_currentLS);
		double error = meds->getBinError(_currentLS);
		if (size==constants::TS_NUM[i] && error==0)
			status[i][fDigiSize] = constants::GOOD;
		else
			status[i][fDigiSize] = constants::PROBLEMATIC;

	}

	//	Check the HF uniformity vs iphi
	for (int i=0; i<IPHI_NUM; i+=4)
	{
		int i1 = (IPHI_NUM-1+i)%IPHI_NUM;
		int i2 = (IPHI_NUM-1+2+i)%IPHI_NUM;
		int j1 = (IPHI_NUM-1+4+i)%IPHI_NUM;
		int j2 = (IPHI_NUM-1+6+i)%IPHI_NUM;

		//	get HFM guys. For description of 6,7 see Mapper Class
		double occ1_m = _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			6, i1) + _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			6, i2);
		double occ2_m = _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			6, j1) + _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			6, j2);
		double ratio_m = std::min(occ1_m, occ2_m)/std::max(occ1_m, occ2_m);

		//	get HFP guys
		double occ1_p = _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			7, i1) + _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			7, i2);
		double occ2_p = _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			7, j1) + _cOccupancyCutvsiphi_SubDetPM.getBinContent(
			7, j2);
		double ratio_p = std::min(occ1_p, occ2_p)/std::max(occ1_p, occ2_p);

		if (ratio_m<0.8 || ratio_p<0.8)
		{
			//	set and exit the loop!
			status[3][fUniphi] = constants::VERY_LOW;
			break;
		}
		else
			status[3][fUniphi] = constants::GOOD;
	}

	/*
	 *	Generic all Hcal Loop.
	 *	-> Possible Missing Channels
	 *	-> Cap Id Rotations
	 */
	for (unsigned int idet=0; idet<constants::SUBDET_NUM; idet++)
	{
		HcalSubdetector subd = HcalEmpty;
		if (idet+1==HB)
			subd = HcalBarrel;
		else if (idet+1==HE)
			subd = HcalEndcap;
		else if (idet+1==HO)
			subd = HcalOuter;
		else
			subd = HcalForward;
		for (int iiphi=0; iiphi<constants::IPHI_NUM; iiphi++)
			for (int iieta=0; iieta<constants::IETA_NUM; iieta++)
				for (int id=0; id<constants::DEPTH_NUM; id++)
				{
					int ieta = iieta<constants::IETA_NUM/2 ? 
						-(iieta+constants::IETA_MIN) : 
						iieta-constants::IETA_NUM/2+constants::IETA_MIN;
					HcalDetId did(subd, ieta, iiphi+1, id+1);
					//	if not a valid Detector cell continue;
					if (!utilities::validDetId(did))
						continue;

					//	if absent for 1 full LS;
					if (!_occ_1LS[idet][iiphi][iieta][id])
					{
						_cMsn1LS_depth.fill(did);
						_nMsn[idet]++;
					}
					//	if absent for 10LSs 
					if (_procLSs>0 && _procLSs%10==0 && 
						!_occ_10LS[idet][iiphi][iieta][id])
						_cMsn10LS_depth.fill(did);

					//	capid rotations check
					if (_error_1LS[idet][iiphi][iieta][id])
					{
						_nCapIdRots[idet]++;
						_cCapIdRots_depth.fill(did);
					}
				}
	}
	_cMsn1LSvsLS_SubDet.fill(HcalDetId(HcalBarrel, 5, 5, 1), _currentLS, 
		_nMsn[0]);
	_cMsn1LSvsLS_SubDet.fill(HcalDetId(HcalEndcap, 18, 5, 1), _currentLS, 
		_nMsn[1]);
	_cMsn1LSvsLS_SubDet.fill(HcalDetId(HcalOuter, 5, 5, 4), _currentLS, 
		_nMsn[2]);
	_cMsn1LSvsLS_SubDet.fill(HcalDetId(HcalForward, 32, 5, 1), _currentLS, 
		_nMsn[3]);
	for (unsigned int idet=0; idet<constants::SUBDET_NUM; idet++)
	{
		//	deal with missing channels
		double ratio = 1-double(_nMsn[idet])/double(constants::CHS_NUM[idet]);
		if (ratio>=GOOD)
			status[idet][fMsn1LS] = constants::GOOD;
		else if (ratio>=constants::PROBLEMATIC)
			status[idet][fMsn1LS] = constants::PROBLEMATIC;
		else if (ratio>=constants::LOW)
			status[idet][fMsn1LS] = constants::LOW;
		else
			status[idet][fMsn1LS] = constants::VERY_LOW;

		//	deal with cap Id rotations
		ratio = 1-double(_nCapIdRots[idet])/
			double(constants::CHS_NUM[idet]);
		if (ratio>=GOOD)
			status[idet][fCapIdRot] = constants::GOOD;
		else if (ratio>=constants::PROBLEMATIC)
			status[idet][fCapIdRot] = constants::PROBLEMATIC;
		else if (ratio>=constants::LOW)
			status[idet][fCapIdRot] = constants::LOW;
		else
			status[idet][fCapIdRot] = constants::VERY_LOW;
	}

	//	finally set all the statuses!
	for (int j=fLowOcp; j<nDigiFlag; j++)
		for (unsigned int i=0; i<constants::SUBDET_NUM; i++)
		{
			_cSummary.setBinContent(i, j, status[i][j]);
			_cSummaryvsLS_SubDet.setBinContent(i,
				_currentLS, j, status[i][j]);
		}

	//	in the end always do the DQTask::endLumi
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(DigiTask);


