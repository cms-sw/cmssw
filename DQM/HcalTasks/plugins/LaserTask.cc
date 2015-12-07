
#include "DQM/HcalTasks/interface/LaserTask.h"

using namespace hcaldqm;
LaserTask::LaserTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	//	Containers
	_cSignalMeans_SubDet.initialize(_name+"/SignalMeans/SubDet", "SignalMeans",
		mapper::fSubDet, 
		new axis::ValueAxis(axis::fXaxis, axis::fNomFC_3000));
	_cSignalRMSs_SubDet.initialize(_name+"/SignalRMSs/SubDet", "SignalRMSs",
		mapper::fSubDet, 
		new axis::ValueAxis(axis::fXaxis, axis::fNomFC_1000));
	_cTimingMeans_SubDet.initialize(_name+"/TimingMeans/SubDet", "TimingMeans",
		mapper::fSubDet, 
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS_200));
	_cTimingRMSs_SubDet.initialize(_name+"/TimingRMSs/SubDet", "TimingRMSs",
		mapper::fSubDet, 
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS_200));
	_cShapeCut_SubDetPM_iphi.initialize(_name+"/Shape/SubDetPM_iphi", "Shape", 
		mapper::fSubDetPM_iphi,
		new axis::ValueAxis(axis::fXaxis, axis::fTimeTS),
		new axis::ValueAxis(axis::fYaxis, axis::fNomFC_3000));
	_cSignalMeans_depth.initialize(_name+"/SignalMeans/depth", "SignalMeans",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
		new axis::ValueAxis(axis::fZaxis, axis::fNomFC_3000));
	_cSignalRMSs_depth.initialize(_name+"/SignalRMSs/depth", "SignalRMSs",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
		new axis::ValueAxis(axis::fZaxis, axis::fNomFC_1000));
	_cTimingMeans_depth.initialize(_name+"/TimingMeans/depth", "TimingMeans",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
		new axis::ValueAxis(axis::fZaxis, axis::fTimeTS_200));
	_cTimingRMSs_depth.initialize(_name+"/TimingRMSs/depth", "TimingRMSs",
		mapper::fdepth, 
		new axis::CoordinateAxis(axis::fXaxis, axis::fieta), 
		new axis::CoordinateAxis(axis::fYaxis, axis::fiphi),
		new axis::ValueAxis(axis::fZaxis, axis::fTimeTS_200));

	//	tags and tokens
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

	//	constants
	_lowHBHE = ps.getUntrackedParameter<double>("lowHBHE",
		20);
	_lowHO = ps.getUntrackedParameter<double>("lowHO",
		20);
	_lowHF = ps.getUntrackedParameter<double>("lowHF",
		20);
}

/* virtual */ void LaserTask::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	char cutstr[20];
	sprintf(cutstr, "_sumQHBHE%dHO%dHF%d", int(_lowHBHE),
		int(_lowHO), int(_lowHF));

	DQTask::bookHistograms(ib, r, es);
	_cSignalMeans_SubDet.book(ib, _subsystem, std::string(cutstr));
	_cSignalRMSs_SubDet.book(ib, _subsystem, std::string(cutstr));
	_cTimingMeans_SubDet.book(ib, _subsystem, std::string(cutstr));
	_cTimingRMSs_SubDet.book(ib, _subsystem, std::string(cutstr));
	_cSignalMeans_depth.book(ib, _subsystem, std::string(cutstr));
	_cSignalRMSs_depth.book(ib, _subsystem, std::string(cutstr));
	_cTimingMeans_depth.book(ib, _subsystem, std::string(cutstr));
	_cTimingRMSs_depth.book(ib, _subsystem, std::string(cutstr));
	_cShapeCut_SubDetPM_iphi.book(ib, _subsystem, std::string(cutstr));
}

/* virtual */ void LaserTask::_resetMonitors(UpdateFreq uf)
{
	DQTask::_resetMonitors(uf);
}

/* virtual */ void LaserTask::_dump()
{
	_cSignalMeans_SubDet.reset();
	_cSignalRMSs_SubDet.reset();
	_cTimingMeans_SubDet.reset();
	_cTimingRMSs_SubDet.reset();
	_cSignalMeans_depth.reset();
	_cSignalRMSs_depth.reset();
	_cTimingMeans_depth.reset();
	_cTimingRMSs_depth.reset();
	_cSignals.dump(&_cSignalMeans_SubDet, true);
	_cSignals.dump(&_cSignalRMSs_SubDet, false);
	_cTiming.dump(&_cTimingMeans_SubDet, true);
	_cTiming.dump(&_cTimingRMSs_SubDet, false);
	_cSignals.dump(&_cSignalMeans_depth, true);
	_cSignals.dump(&_cSignalRMSs_depth, false);
	_cTiming.dump(&_cTimingMeans_depth, true);
	_cTiming.dump(&_cTimingRMSs_depth, false);
}

/* virtual */ void LaserTask::_process(edm::Event const& e,
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
		if (sumQ<_lowHBHE)
			continue;

		double aveTS = utilities::aveTS<HBHEDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		_cSignals.fill(digi.id(), sumQ>0 ? sumQ : GARBAGE_VALUE);
		_cTiming.fill(digi.id(), sumQ>0 ? aveTS : GARBAGE_VALUE);
		
		for (int i=0; i<digi.size(); i++)
			_cShapeCut_SubDetPM_iphi.fill(digi.id(), i, 
				digi.sample(i).nominal_fC()-2.5);
	}
	for (HODigiCollection::const_iterator it=cho->begin();
		it!=cho->end(); ++it)
	{
		const HODataFrame digi = (const HODataFrame)(*it);
		double sumQ = utilities::sumQ<HODataFrame>(digi, 8.5, 0,
			digi.size()-1);
		if (sumQ<_lowHO)
			continue;

		double aveTS = utilities::aveTS<HODataFrame>(digi, 8.5, 0,
			digi.size()-1);
		_cSignals.fill(digi.id(), sumQ>0 ? sumQ : GARBAGE_VALUE);
		_cTiming.fill(digi.id(), sumQ>0 ? aveTS : GARBAGE_VALUE);

		for (int i=0; i<digi.size(); i++)
			_cShapeCut_SubDetPM_iphi.fill(digi.id(), i, 
				digi.sample(i).nominal_fC()-8.5);
	}
	for (HFDigiCollection::const_iterator it=chf->begin();
		it!=chf->end(); ++it)
	{
		const HFDataFrame digi = (const HFDataFrame)(*it);
		double sumQ = utilities::sumQ<HFDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		if (sumQ<_lowHF)
			continue;

		double aveTS = utilities::aveTS<HFDataFrame>(digi, 2.5, 0,
			digi.size()-1);
		_cSignals.fill(digi.id(), sumQ>0 ? sumQ : GARBAGE_VALUE);
		_cTiming.fill(digi.id(), sumQ>0 ? aveTS : GARBAGE_VALUE);

		for (int i=0; i<digi.size(); i++)
			_cShapeCut_SubDetPM_iphi.fill(digi.id(), i, 
				digi.sample(i).nominal_fC()-2.5);
	}

	if (_ptype==fOnline && _evsTotal>0 &&
		_evsTotal%constants::CALIBEVENTS_MIN==0)
		this->_dump();
}

/* virtual */ bool LaserTask::_isApplicable(edm::Event const& e)
{
	if (_ptype==fOnline)
	{
		//	online-global
		int calibType =	this->_getCalibType(e);
		return (calibType==hc_HBHEHPD || calibType==hc_HFPMT);
	}
	else
	{
		//	local
		edm::Handle<HcalTBTriggerData> ctrigger;
		if (!e.getByToken(_tokTrigger, ctrigger))
			_logger.dqmthrow("Collection HcalTBTriggerData isn't available"
				+ _tagTrigger.label() + " " + _tagTrigger.instance());
		return ctrigger->wasLaserTrigger();
	}
}

DEFINE_FWK_MODULE(LaserTask);


