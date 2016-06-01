#include "DQM/HcalTasks/interface/NoCQTask.h"

NoCQTask::NoCQTask(edm::ParameterSet const& ps) : 
	DQTask(ps)
{
	_tagHBHE = ps.getUntrackedParameter<edm::InputTag>("tagHBHE",
		edm::InputTag("hcalDigis"));
	_tagHO = ps.getUntrackedParameter<edm::InputTag>("tagHO",
		 edm::InputTag("hcalDigis"));
	_tagHF = ps.getUntrackedParameter<edm::InputTag>("tagHF",
		edm::InputTag("hcalDigis"));
	_tagReport = ps.getUntrackedParameter<edm::InputTag>("tagReport",
		edm::InputTag("hcalDigis"));

	_tokHBHE = consumes<HBHEDigiCollection>(_tagHBHE);
	_tokHO = consumes<HODigiCollection>(_tagHO);
	_tokHF = consumes<HFDigiCollection>(_tagHF);
	_tokReport = consumes<HcalUnpackerReport>(_tagReport);

	_cutSumQ_HBHE = ps.getUntrackedParameter<double>("cutSumQ_HBHE", 20);
	_cutSumQ_HO = ps.getUntrackedParameter<double>("cutSumQ_HO", 20);
	_cutSumQ_HF = ps.getUntrackedParameter<double>("cutSumQ_HF", 20);
}

/* virtual */ void NoCQTask::bookHistograms(DQMStore::IBooker& ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	DQTask::bookHistograms(ib, r, es);

	edm::ESHandle<HcalDbService> dbs;
	es.get<HcalDbRecord>().get(dbs);
	_emap = dbs->getHcalMapping();

	_cTimingCut_depth.initialize(_name, "TimingCut", 
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fTiming_TS200));
	_cOccupancy_depth.initialize(_name, "Occupancy",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cOccupancyCut_depth.initialize(_name, "OccupancyCut",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));
	_cBadQuality_depth.initialize(_name, "BadQuality",
		hashfunctions::fdepth,
		new quantity::DetectorQuantity(quantity::fieta),
		new quantity::DetectorQuantity(quantity::fiphi),
		new quantity::ValueQuantity(quantity::fN));

	_cTimingCut_depth.book(ib, _emap, _subsystem);
	_cOccupancy_depth.book(ib, _emap, _subsystem);
	_cOccupancyCut_depth.book(ib, _emap, _subsystem);
	_cBadQuality_depth.book(ib, _emap, _subsystem);
}

/* virtual */ void NoCQTask::_resetMonitors(UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);
}

/* virtual */ void NoCQTask::_process(edm::Event const& e,
	edm::EventSetup const&)
{
	edm::Handle<HBHEDigiCollection> chbhe;
	edm::Handle<HODigiCollection> cho;
	edm::Handle<HFDigiCollection> chf;
	edm::Handle<HcalUnpackerReport> creport;

	if (!e.getByToken(_tokHBHE, chbhe))
		_logger.dqmthrow("Collection HBHEDigiCollection isn't available"
			+ _tagHBHE.label() + " " + _tagHBHE.instance());
	if (!e.getByToken(_tokHO, cho))
		_logger.dqmthrow("Collection HODigiCollection isn't available"
			 + _tagHO.label() + " " + _tagHO.instance());
	if (!e.getByToken(_tokHF, chf))
		_logger.dqmthrow("Collection HFDigiCollection isn't available"
			+ _tagHF.label() + " " + _tagHF.instance());
	if (!e.getByToken(_tokReport, creport))
		_logger.dqmthrow("Collection HcalUnpackerReport isn't available"+
			_tagReport.label()+" " +_tagReport.instance());

	//	RAW Bad Quality
	for (std::vector<DetId>::const_iterator it=creport->bad_quality_begin();
		it!=creport->bad_quality_end(); ++it)
	{
		if (!HcalGenericDetId(*it).isHcalDetId())
			continue;

		_cBadQuality_depth.fill(HcalDetId(*it));
	}

	//	DIGI HBH, HO, HF
	for (HBHEDigiCollection::const_iterator it=chbhe->begin(); it!=chbhe->end();
		++it)
	{
		double sumQ = utilities::sumQ<HBHEDataFrame>(*it, 2.5, 0, it->size()-1);
		HcalDetId const& did = it->id();

		_cOccupancy_depth.fill(did);
		if (sumQ>_cutSumQ_HBHE)
		{
			double timing = utilities::aveTS<HBHEDataFrame>(*it, 2.5, 0,
				it->size()-1);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_depth.fill(did, timing);
		}
	}

	for (HODigiCollection::const_iterator it=cho->begin(); it!=cho->end();
		++it)
	{
		double sumQ = utilities::sumQ<HODataFrame>(*it, 8.5, 0, it->size()-1);
		HcalDetId const& did = it->id();

		_cOccupancy_depth.fill(did);
		if (sumQ>_cutSumQ_HO)
		{
			double timing = utilities::aveTS<HODataFrame>(*it, 8.5, 0,
				it->size()-1);
			_cOccupancyCut_depth.fill(did);
			_cTimingCut_depth.fill(did, timing);
		}
	}

	for (HFDigiCollection::const_iterator it=chf->begin(); it!=chf->end();
		++it)
	{
		 double sumQ = utilities::sumQ<HFDataFrame>(*it, 2.5, 0, it->size()-1);
		 HcalDetId const& did = it->id();

		 _cOccupancy_depth.fill(did);
		 if (sumQ>_cutSumQ_HF)
		 {
			 double timing = utilities::aveTS<HFDataFrame>(*it, 2.5, 0, it->size()-1);
			 _cOccupancyCut_depth.fill(did);
			 _cTimingCut_depth.fill(did, timing);
		 }
	}
}

/* virtual */ void NoCQTask::beginLuminosityBlock(
	edm::LuminosityBlock const& lb, edm::EventSetup const& es)
{
	DQTask::beginLuminosityBlock(lb, es);
}

/* virtual */ void NoCQTask::endLuminosityBlock(edm::LuminosityBlock const& 
	lb, edm::EventSetup const& es)
{
	DQTask::endLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(NoCQTask);
