
#include "DQM/HcalTasks/interface/ZDCTask.h"
#include <map>

using namespace hcaldqm;
using namespace hcaldqm::constants;
ZDCTask::ZDCTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	
	//	tags
	_tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10",
		edm::InputTag("hcalDigis"));
	_tokQIE10 = consumes<ZDCDigiCollection>(_tagQIE10);

	//	cuts
	_cut = ps.getUntrackedParameter<double>("cut", 50.0);
	_ped = ps.getUntrackedParameter<int>("ped", 4);
}

/* virtual */ void ZDCTask::bookHistograms(DQMStore::IBooker &ib,
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

	//  uhtr slot 6 is the uhtr to be used to read out qie10 coming from zdc
	vhashC36.push_back(HcalElectronicsId(18, 8, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
	// vhashC36.push_back(HcalElectronicsId(36, 6, 11, FIBERCH_MIN, false).rawId());
	// vhashC36.push_back(HcalElectronicsId(36, 6, 4, FIBERCH_MIN, false).rawId());
	// vhashC36.push_back(HcalElectronicsId(36, 6, 5, FIBERCH_MIN, false).rawId());

	_filter_C36.initialize(filter::fPreserver, hcaldqm::hashfunctions::fCrateSlot,
		vhashC36);

	//	INITIALIZE what you need
	_cShapeCut_EChannel.initialize(_name,
		"ShapeCut", hcaldqm::hashfunctions::fEChannel,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
	_cShapeCut.initialize(_name,
		"ShapeCut", 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
	_cShape_EChannel.initialize(_name,
		"Shape", hcaldqm::hashfunctions::fEChannel,
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));
	_cShape.initialize(_name,
		"Shape", 
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fTiming_TS),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::ffC_10000));

	// _cLETDCvsADC.initialize(_name, "LETDCvsADC",
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	// _cTETDCvsADC.initialize(_name, "TETDCvsADC",
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	// _cLETDC.initialize(_name, "LETDC",
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
	// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	_cADC.initialize(_name, "ADC",
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
		new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));

	unsigned int nTS = 10;
	for (unsigned int j=0; j<nTS; j++)
	{
		// _cLETDCvsADC_EChannel[j].initialize(_name,
		// 	"LETDCvsADC", hcaldqm::hashfunctions::fEChannel,
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
		// _cTETDCvsADC_EChannel[j].initialize(_name,
		// 	"TETDCvsADC", hcaldqm::hashfunctions::fEChannel,
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10ADC_256),
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
		// _cLETDC_EChannel[j].initialize(_name,
		// 	"LETDC", hcaldqm::hashfunctions::fEChannel,
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fQIE10TDC_64),
		// 	new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
		_cADC_EChannel[j].initialize(_name,
			"ADC", hcaldqm::hashfunctions::fEChannel,
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fADC_128),
			new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN, true));
	}

	_cShapeCut_EChannel.book(ib, _emap, _subsystem);
	_cShapeCut.book(ib, _subsystem);
	_cShape_EChannel.book(ib, _emap, _subsystem);
	_cShape.book(ib, _subsystem);
	// _cLETDCvsADC.book(ib, _subsystem);
	// _cTETDCvsADC.book(ib, _subsystem);
	// _cLETDC.book(ib, _subsystem);
	_cADC.book(ib, _subsystem);
	for (unsigned int i=0; i<nTS; i++)
	{
		char aux[10];
		sprintf(aux, "TS%d", i);
		// _cLETDCvsADC_EChannel[i].book(ib, _emap, _filter_C36, _subsystem, aux);
		// _cTETDCvsADC_EChannel[i].book(ib, _emap, _filter_C36, _subsystem, aux);
		// _cLETDC_EChannel[i].book(ib, _emap, _filter_C36, _subsystem, aux);
		_cADC_EChannel[i].book(ib, _emap, _subsystem, aux);
	}

	_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
}

/* virtual */ void ZDCTask::endLuminosityBlock(edm::LuminosityBlock const& lb,
	edm::EventSetup const& es)
{
	
	//	finish
	DQTask::endLuminosityBlock(lb, es);
}

/* virtual */ void ZDCTask::_process(edm::Event const& e, 
	edm::EventSetup const&)
{
	edm::Handle<ZDCDigiCollection> cqie10;
	if (!e.getByToken(_tokQIE10, cqie10))
		_logger.dqmthrow("Collection ZDCDigiCollection isn't available"
			+ _tagQIE10.label() + " " + _tagQIE10.instance());


	for (uint32_t i=0; i<cqie10->size(); i++)
	  {
	    ZDCDataFrame frame = static_cast<ZDCDataFrame>((*cqie10)[i]);
	    HcalElectronicsId eid = HcalElectronicsId(_emap->lookup(frame.id()));


	    //std::cout << eid.crateId() << " " << eid.slot() << " " << eid.fiberIndex() << " " << eid.fiberChanId() << std::endl;
	    //if (_filter_C36.filter(eid)) continue;
	    
		//	compute the signal, ped subracted
		double q = hcaldqm::utilities::sumQ_v10<ZDCDataFrame>(frame,
			constants::adc2fC[_ped], 0, frame.size()-1);

		//	iterate thru all TS and fill
		for (int j=0; j<frame.size(); j++)
		{
		  _cShape_EChannel.fill(eid, j, frame[j].nominal_fC());
		  _cShape.fill(j, frame[j].nominal_fC());

			//	shapes are after the cut
			if (q>_cut)
			{
			  _cShapeCut_EChannel.fill(eid, j, frame[j].nominal_fC());
			  _cShapeCut.fill(j, frame[j].nominal_fC());
			}

			//	w/o a cut
			// _cLETDCvsADC_EChannel[j].fill(eid, frame[j].adc(), 
			// 	frame[j].le_tdc());
			// _cLETDCvsADC.fill(frame[j].adc(), frame[j].le_tdc());
			// _cTETDCvsADC_EChannel[j].fill(eid, frame[j].adc(), 
			// 	frame[j].te_tdc());
			// _cTETDCvsADC.fill(frame[j].adc(), frame[j].te_tdc());
			// _cLETDC_EChannel[j].fill(eid, frame[j].le_tdc());
			// _cLETDC.fill(frame[j].le_tdc());
			_cADC_EChannel[j].fill(eid, frame[j].adc());
			_cADC.fill(frame[j].adc());
		}
	}
}

/* virtual */ void ZDCTask::_resetMonitors(hcaldqm::UpdateFreq)
{
}

DEFINE_FWK_MODULE(ZDCTask);
