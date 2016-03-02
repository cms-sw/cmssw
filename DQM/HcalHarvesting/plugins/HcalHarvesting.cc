#include "DQM/HcalHarvesting/interface/HcalHarvesting.h"

HcalHarvesting::HcalHarvesting(edm::ParameterSet const& ps) :
	DQHarvester(ps), _reportSummaryMap(NULL)
{
	//	get the flags 
	_digiHarvesting = ps.getUntrackedParameter<bool>("digiHarvesting");
	_rawHarvesting = ps.getUntrackedParameter<bool>("rawHarvesting");
	_recoHarvesting = ps.getUntrackedParameter<bool>("recoHarvesting");
	_tpHarvesting = ps.getUntrackedParameter<bool>("tpHarvesting");

	//	set labels for summaries
	_frawnames.push_back("EvnMsm");
	_frawnames.push_back("BcnMsm");
	_frawnames.push_back("BadQuality");

	_fdiginames.push_back("UniSlot");
	_fdiginames.push_back("Msn1LS");
	_fdiginames.push_back("CapIdRot");
	_fdiginames.push_back("DigiSize");

	_freconames.push_back("OcpUniSlot");
	_freconames.push_back("TimeUniSlot");
	_freconames.push_back("TCDS");
	_freconames.push_back("Msn1LS");

	_ftpnames.push_back("OcpUniSlotD");
	_ftpnames.push_back("OcpUniSlotE");
	_ftpnames.push_back("EtMsmUniSlot");
	_ftpnames.push_back("FGMsmUniSlot");
	_ftpnames.push_back("MsnUniSlotD");
	_ftpnames.push_back("MsnUniSlotE");
	_ftpnames.push_back("EtCorrRatio");
	_ftpnames.push_back("EtMsmRatio");
	_ftpnames.push_back("FGMsmNumber");
}

/* virtual */ void HcalHarvesting::_dqmEndLuminosityBlock(DQMStore::IBooker& ib,
	DQMStore::IGetter& ig, edm::LuminosityBlock const&, 
	edm::EventSetup const&)
{
	//	Create the reportSummaryMap if needed
	if (!_reportSummaryMap)
	{
		ig.setCurrentFolder("Hcal/EventInfo");
		_reportSummaryMap = ib.book2D("reportSummaryMap", "reportSummaryMap",
			_vFEDs.size(), 0, _vFEDs.size(), 4, 0, 4);
		_reportSummaryMap->setBinLabel(1, "RAW", 2);
		_reportSummaryMap->setBinLabel(2, "DIGI", 2);
		_reportSummaryMap->setBinLabel(3, "RECO", 2);
		_reportSummaryMap->setBinLabel(4, "TP", 2);
		for (uint32_t i=0; i<_vFEDs.size(); i++)
		{
			char name[5];
			sprintf(name, "%d", _vFEDs[i]);
			_reportSummaryMap->setBinLabel(i+1, name, 1);
		}
	}

	//	get the flags from DATA itself...
	ig.get("Hcal/DigiTask/Summary/Summary")!=NULL?
		_digiHarvesting = true:
		_digiHarvesting = false;
	ig.get("Hcal/RecHitTask/Summary/Summary")!=NULL?
		_recoHarvesting = true:
		_recoHarvesting = false;
	ig.get("Hcal/RawTask/Summary/Summary")!=NULL?
		_rawHarvesting = true:
		_rawHarvesting = false;
	ig.get("Hcal/TPTask/Summary/Summary")!=NULL?
		_tpHarvesting = true:
		_tpHarvesting = false;

	//	Initialize what you need
	ContainerSingle2D rawSummary;
	ContainerSingle2D digiSummary;
	ContainerSingle2D recoSummary;
	ContainerSingle2D tpSummary;
	ContainerSingle2D rawSummaryCopy;
	ContainerSingle2D digiSummaryCopy;
	ContainerSingle2D recoSummaryCopy;
	ContainerSingle2D tpSummaryCopy;

	//	book the new plots and load existing ones if needed
	char name[20];
	sprintf(name, "LS%d", _currentLS);
	if (_rawHarvesting)
	{
		rawSummary.initialize("RawTask", "Summary",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_frawnames),
			new quantity::QualityQuantity());
		rawSummaryCopy.initialize("RawTask", "SummaryvsLS",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_frawnames),
			new quantity::QualityQuantity());
		rawSummaryCopy.book(ib, "Hcal", name);
		rawSummary.load(ig);
	}
	if (_digiHarvesting)
	{
		digiSummary.initialize("DigiTask", "Summary",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_fdiginames),
			new quantity::QualityQuantity());
		digiSummaryCopy.initialize("DigiTask", "SummaryvsLS",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_fdiginames),
			new quantity::QualityQuantity());
		digiSummaryCopy.book(ib, "Hcal", name);
		digiSummary.load(ig);
	}
	if (_recoHarvesting)
	{
		recoSummary.initialize("RecHitTask", "Summary",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_freconames),
			new quantity::QualityQuantity());
		recoSummaryCopy.initialize("RecHitTask", "SummaryvsLS",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_freconames),
			new quantity::QualityQuantity());
		recoSummaryCopy.book(ib, "Hcal", name);
		recoSummary.load(ig);
	}
	if (_tpHarvesting)
	{
		tpSummary.initialize("TPTask", "Summary",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_freconames),
			new quantity::QualityQuantity());
		tpSummaryCopy.initialize("TPTask", "SummaryvsLS",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(_ftpnames),
			new quantity::QualityQuantity());
		tpSummaryCopy.book(ib, "Hcal", name);
		tpSummary.load(ig);
	}

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
		if (_rawHarvesting)
		{
			for (uint32_t f=0; f<_frawnames.size(); f++)
			{
				quantity::Quality q = (quantity::Quality)
					((int)rawSummary.getBinContent(eid, (int)f));
				rawSummaryCopy.setBinContent(eid, (int)f, q);
				if (q>quantity::fGood)
					counter++;
			}
		}
		counter>0?
			_reportSummaryMap->setBinContent(ifed+1, 1, quantity::fLow):
			_reportSummaryMap->setBinContent(ifed+1, 1, quantity::fGood);

		//	DIGI if set to harvest
		counter=0;
		if (_digiHarvesting)
		{
			for (uint32_t f=0; f<_fdiginames.size(); f++)
			{
				quantity::Quality q = (quantity::Quality)
					((int)digiSummary.getBinContent(eid, (int)f));
				digiSummaryCopy.setBinContent(eid, (int)f, q);
				if (q>quantity::fGood)
					counter++;
			}
		}
		counter>0?
			_reportSummaryMap->setBinContent(ifed+1, 2, quantity::fLow):
			_reportSummaryMap->setBinContent(ifed+1, 2, quantity::fGood);

		//	RECO
		counter=0;
		if (_recoHarvesting)
		{
			for (uint32_t f=0; f<_freconames.size(); f++)
			{
				quantity::Quality q = (quantity::Quality)
					((int)recoSummary.getBinContent(eid, (int)f));
				recoSummaryCopy.setBinContent(eid, (int)f, q);
				if (q>quantity::fGood)
					counter++;
			}
		}
		counter>0?
			_reportSummaryMap->setBinContent(ifed+1, 3, quantity::fLow):
			_reportSummaryMap->setBinContent(ifed+1, 3, quantity::fGood);

		//	TP
		counter=0;
		if (_tpHarvesting)
		{
			for (uint32_t f=0; f<_ftpnames.size(); f++)
			{
				quantity::Quality q = (quantity::Quality)
					((int)tpSummary.getBinContent(eid, (int)f));
				tpSummaryCopy.setBinContent(eid, (int)f, q);
				if (q>quantity::fGood)
					counter++;
			}
		}
		counter>0?
			_reportSummaryMap->setBinContent(ifed+1, 4, quantity::fLow):
			_reportSummaryMap->setBinContent(ifed+1, 4, quantity::fGood);

		ifed++;
	}
}

/* virtual */ void HcalHarvesting::_dqmEndJob(DQMStore::IBooker&,
	DQMStore::IGetter&)
{}

DEFINE_FWK_MODULE(HcalHarvesting);
