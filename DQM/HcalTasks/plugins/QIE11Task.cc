
#include "DQM/HcalTasks/interface/QIE11Task.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
QIE11Task::QIE11Task(edm::ParameterSet const& ps):
	DQTask(ps)
{
	
	//	tags
	_tagQIE11 = ps.getUntrackedParameter<edm::InputTag>("tagQIE11",
		edm::InputTag("hcalDigis"));
	_tokQIE11 = consumes<QIE11DigiCollection>(_tagQIE11);

	_taguMN = ps.getUntrackedParameter<edm::InputTag>("taguMN",
							  edm::InputTag("hcalDigis"));
	_tokuMN = consumes<HcalUMNioDigi>(_taguMN);

	//	cuts
	_cut = ps.getUntrackedParameter<double>("cut", 50.0);
	_ped = ps.getUntrackedParameter<int>("ped", 4);
	_laserType = ps.getUntrackedParameter<int32_t>("laserType", -1);
	_eventType = ps.getUntrackedParameter<int32_t>("eventType", -1);
}
/* virtual */ void QIE11Task::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	if (_ptype==fLocal)
		if (r.runAuxiliary().run()==1)
			return;

	DQTask::bookHistograms(ib, r, es);

	//	GET WHAT YOU NEED
	edm::ESHandle<HcalDbService> dbs;
	es.get<HcalDbRecord>().get(dbs);
	_emap = dbs->getHcalMapping();
	std::vector<uint32_t> vhashC36;
	vhashC36.push_back(HcalElectronicsId(36, 4,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_C36.initialize(filter::fPreserver, hcaldqm::hashfunctions::fCrateSlot,
		vhashC36);

	//	INITIALIZE what you need
	_cShapeCut_EChannel.initialize(_name,
		"ShapeCut", hcaldqm::hashfunctions::fEChannel,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_300000));
	_cShapeCut.initialize(_name,
		"ShapeCut", 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10fC_300000));
	_cTDCvsADC.initialize(_name, "TDCvsADC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cTDC.initialize(_name, "TDC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cADC.initialize(_name, "ADC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	for (unsigned int j=0; j<10; j++)
	{
		_cTDCvsADC_EChannel[j].initialize(_name,
			"TDCvsADC", hcaldqm::hashfunctions::fEChannel,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
		_cADC_EChannel[j].initialize(_name,
			"ADC", hcaldqm::hashfunctions::fEChannel,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
		_cTDC_EChannel[j].initialize(_name,
			"TDC", hcaldqm::hashfunctions::fEChannel,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	}

	_cShapeCut_EChannel.book(ib, _emap, _filter_C36, _subsystem);
	_cShapeCut.book(ib, _subsystem);
	_cTDCvsADC.book(ib, _subsystem);
	_cTDC.book(ib, _subsystem);
	_cADC.book(ib, _subsystem);
	for (unsigned int i=0; i<10; i++)
	{
		char aux[10];
		sprintf(aux, "TS%d", i);
		_cTDCvsADC_EChannel[i].book(ib, _emap, _filter_C36, _subsystem, aux);
		_cTDC_EChannel[i].book(ib, _emap, _filter_C36, _subsystem, aux);
		_cADC_EChannel[i].book(ib, _emap, _filter_C36, _subsystem, aux);
	}

	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap, _filter_C36);
}

/* virtual */ void QIE11Task::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	
	//	finish
	DQTask::endLuminosityBlock(lb, es);
}

/* virtual */ void QIE11Task::_process(edm::Event const& e, 
	edm::EventSetup const&)
{
	edm::Handle<QIE11DigiCollection> cqie10;
	if (!e.getByToken(_tokQIE11, cqie10))
		return;

	for (uint32_t i=0; i<cqie10->size(); i++)
	{
		QIE11DataFrame frame = static_cast<QIE11DataFrame>((*cqie10)[i]);
		DetId did = frame.detid();
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));

		//	compute the signal, ped subracted
//		double q = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(frame,
//			constants::adc2fC[_ped], 0, frame.samples()-1);


		//	iterate thru all TS and fill
		for (int j=0; j<frame.samples(); j++)
		{
			//	shapes are after the cut
			_cShapeCut_EChannel.fill(eid, j, adc2fC[frame[j].adc()]);	
			_cShapeCut.fill(eid, j, adc2fC[frame[j].adc()]);
			

			//	w/o a cut
			_cTDCvsADC_EChannel[j].fill(eid, frame[j].adc(), 
				frame[j].tdc());
			_cTDCvsADC.fill(frame[j].adc(), frame[j].tdc());
			_cTDC_EChannel[j].fill(eid, frame[j].tdc());
			_cTDC.fill(eid, frame[j].tdc());
			_cADC_EChannel[j].fill(eid, frame[j].adc());
			_cADC.fill(eid, frame[j].adc());

		}
	}
}


/* virtual */ bool QIE11Task::_isApplicable(edm::Event const& e)
{
  if (_ptype!=fOnline || (_laserType < 0 && _eventType < 0))
    return true;
  else
    {
      //      fOnline mode
      edm::Handle<HcalUMNioDigi> cumn;
      if (!e.getByToken(_tokuMN, cumn))
	return false;

      //      event type check first
      int eventType = cumn->eventType();
      if (eventType==_eventType)
	return true;

      //      check if this analysis task is of the right laser type
      int laserType = cumn->valueUserWord(0);
      if (laserType==_laserType)
	return true;
    }

  return false;
}


/* virtual */ void QIE11Task::_resetMonitors(hcaldqm::UpdateFreq)
{
}

DEFINE_FWK_MODULE(QIE11Task);
