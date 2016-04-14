#include "DQM/HcalTasks/interface/HcalCalibHarvesting.h"

HcalCalibHarvesting::HcalCalibHarvesting(edm::ParameterSet const& ps) :
	DQHarvester(ps), _reportSummaryMap(NULL)
{
	//	set labels for summaries
	_fpednames.push_back("Msn");
	_fpednames.push_back("BadMean");
	_fpednames.push_back("BadRMS");
}

/* virtual */ void HcalCalibHarvesting::_dqmEndLuminosityBlock(DQMStore::IBooker& ib,
	DQMStore::IGetter& ig, edm::LuminosityBlock const&, 
	edm::EventSetup const&)
{
	//	Create the reportSummaryMap if needed
	if (!_reportSummaryMap)
	{
		ig.setCurrentFolder("HcalCalib/EventInfo");
		_reportSummaryMap = ib.book2D("reportSummaryMap", "reportSummaryMap",
			_vFEDs.size(), 0, _vFEDs.size(), 1, 0, 1);
		_reportSummaryMap->setBinLabel(1, "PED", 2);
		for (uint32_t i=0; i<_vFEDs.size(); i++)
		{
			char name[5];
			sprintf(name, "%d", _vFEDs[i]);
			_reportSummaryMap->setBinLabel(i+1, name, 1);
		}
	}

	//	Initialize what you need
	ContainerSingle2D pedSummary;
	ContainerSingle2D pedSummaryCopy;

	//	book the new plots and load existing ones if needed
	char name[20];
	sprintf(name, "LS%d", _currentLS);
	pedSummary.initialize("PedestalTask", "Summary",
		new quantity::FEDQuantity(_vFEDs),
		new quantity::FlagQuantity(_fpednames),
		new quantity::QualityQuantity());
	pedSummaryCopy.initialize("PedestalTask", "SummaryvsLS",
		new quantity::FEDQuantity(_vFEDs),
		new quantity::FlagQuantity(_fpednames),
		new quantity::QualityQuantity());
	pedSummaryCopy.book(ib, _subsystem, name);
	pedSummary.load(ig, _subsystem);

	//	process: put the quality into the copy and set the reportSummaryMap
	//	contents. This is done only for those Summaries for which Tasks
	//	exist
	int ifed = 0;
	for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
		it!=_vhashFEDs.end(); ++it)
	{
		HcalElectronicsId eid(*it);
		//	RAW if set to harvest
		int counter = 0;
		for (uint32_t f=0; f<_fpednames.size(); f++)
		{
			quantity::Quality q = (quantity::Quality)
				((int)pedSummary.getBinContent(eid, (int)f));
			pedSummaryCopy.setBinContent(eid, (int)f, q);
			if (q>quantity::fGood)
				counter++;
		}
		counter>0?
			_reportSummaryMap->setBinContent(ifed+1, 1, quantity::fLow):
			_reportSummaryMap->setBinContent(ifed+1, 1, quantity::fGood);

		ifed++;
	}
}

/* virtual */ void HcalCalibHarvesting::_dqmEndJob(DQMStore::IBooker&,
	DQMStore::IGetter&)
{}

DEFINE_FWK_MODULE(HcalCalibHarvesting);
