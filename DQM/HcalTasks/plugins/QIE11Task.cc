
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
	std::vector<uint32_t> vhashC34;
	vhashC34.push_back(HcalElectronicsId(34, 11,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	vhashC34.push_back(HcalElectronicsId(34, 12,
		FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	_filter_C34.initialize(filter::fPreserver, hcaldqm::hashfunctions::fCrateSlot,
		vhashC34);

	//	INITIALIZE what you need
	unsigned int itr = 0;
	for (unsigned int crate = 34; crate <= 34; ++crate) {
		for (unsigned int slot = 11; slot <= 12; ++slot) {
			std::vector<uint32_t> vhashSlot;
			vhashSlot.push_back(HcalElectronicsId(crate, slot, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
			_filter_slot[itr].initialize(filter::fPreserver, hashfunctions::fCrateSlot, vhashSlot);
			_cShapeCut_EChannel[itr].initialize(_name,
				"ShapeCut", hcaldqm::hashfunctions::fEChannel,
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11fC_300000));
			_cTDCvsTS_EChannel[itr].initialize(_name, 
				"TDCvsTS", hcaldqm::hashfunctions::fEChannel, 
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11TDC_64),
                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true), 0);
			_cTDCTime_EChannel[itr].initialize(_name,
				"TDCTime", hcaldqm::hashfunctions::fEChannel,
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTime_ns_250),
				new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
			for (unsigned int j=0; j<10; j++) {
				_cTDCvsADC_EChannel[j][itr].initialize(_name,
					"TDCvsADC", hcaldqm::hashfunctions::fEChannel,
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11ADC_256),
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11TDC_64),
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
				_cADC_EChannel[j][itr].initialize(_name,
					"ADC", hcaldqm::hashfunctions::fEChannel,
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11ADC_256),
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
				_cTDC_EChannel[j][itr].initialize(_name,
					"TDC", hcaldqm::hashfunctions::fEChannel,
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11TDC_64),
					new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
			}
			++itr;
		}
	}
	_cShapeCut.initialize(_name,
		"ShapeCut", 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11fC_300000));
	_cTDCvsADC.initialize(_name, "TDCvsADC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11TDC_64),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cTDC.initialize(_name, "TDC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11TDC_64),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);
	_cADC.initialize(_name, "ADC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE11ADC_256),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true),0);

	itr = 0;
	std::map<std::pair<unsigned int, unsigned int>, unsigned int> itr_map;
	for(unsigned int crate = 34; crate <= 34; ++crate) {
		for(unsigned int slot=11; slot<=12; ++slot) {
			char aux[100];
			sprintf(aux, "/Crate%d_Slot%d", crate, slot);
			_cShapeCut_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
			_cTDCvsTS_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
			_cTDCTime_EChannel[itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux);
			for (unsigned int j=0; j<10; j++) {
				char aux2[100];
				sprintf(aux2, "/Crate%d_Slot%d/TS%d", crate, slot, j);
				_cTDCvsADC_EChannel[j][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux2);
				_cTDC_EChannel[j][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux2);
				_cADC_EChannel[j][itr].book(ib, _emap, _filter_slot[itr], _subsystem, aux2);
			}
			itr_map[std::make_pair(crate, slot)] = itr;
			++itr;
		}
	}
	_cShapeCut.book(ib, _subsystem);
	_cTDCvsADC.book(ib, _subsystem);
	_cTDC.book(ib, _subsystem);
	_cADC.book(ib, _subsystem);

	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap, _filter_C34);
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
	edm::Handle<QIE11DigiCollection> cqie11;
	if (!e.getByToken(_tokQIE11, cqie11))
		return;

	for (uint32_t i=0; i<cqie11->size(); i++)
	{
		QIE11DataFrame frame = static_cast<QIE11DataFrame>((*cqie11)[i]);
		DetId did = frame.detid();
		HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
		int fakecrate = -1;
		if (eid.crateId() == 34) fakecrate = 0;
		int index = fakecrate * 12 + (eid.slot() - 10) - 1;

		//	compute the signal, ped subracted
//		double q = hcaldqm::utilities::aveTS_v10<QIE11DataFrame>(frame,
//			constants::adc2fC[_ped], 0, frame.samples()-1);

		//	iterate thru all TS and fill
		for (int j=0; j<frame.samples(); j++)
		{
			if (index == 0 || index == 1) {
				//	shapes are after the cut
				_cShapeCut_EChannel[index].fill(eid, j, adc2fC[frame[j].adc()]);	
				_cTDCvsTS_EChannel[index].fill(eid, j, frame[j].tdc());	

				//	w/o a cut
				_cTDCvsADC_EChannel[j][index].fill(eid, frame[j].adc(), 
					frame[j].tdc());
				_cTDC_EChannel[j][index].fill(eid, frame[j].tdc());
				if (frame[j].tdc() < 50) {
					// Each TDC count is 0.5 ns. 
					// tdc == 62 or 63 means value was below or above threshold for whole time slice. 
					_cTDCTime_EChannel[index].fill(eid, j*25. + (frame[j].tdc() / 2.));
				}
				_cADC_EChannel[j][index].fill(eid, frame[j].adc());
			}
			_cShapeCut.fill(eid, j, adc2fC[frame[j].adc()]);

			_cTDCvsADC.fill(frame[j].adc(), frame[j].tdc());

			_cTDC.fill(eid, frame[j].tdc());

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
