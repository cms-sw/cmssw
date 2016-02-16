#include "DQM/HcalTasks/interface/RawTask.h"

using namespace hcaldqm;
RawTask::RawTask(edm::ParameterSet const& ps):
	DQTask(ps)
{
	//	specify all the Containers
	_cVMEEvnMsm.initialize(_name+"/VME/EvnMsm", "EvnMismatch",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDVME),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSpigot));
	_cVMEBcnMsm.initialize(_name+"/VME/BcnMsm", "BcnMismatch",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDVME),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSpigot));
	_cVMEOrnMsm.initialize(_name+"/VME/OrnMsm", "OrnMismatch",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDVME),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSpigot));
	_cuTCAEvnMsm.initialize(_name+"/uTCA/EvnMsm", "EvnMismatch",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDuTCA),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSlotuTCA));
	_cuTCABcnMsm.initialize(_name+"/uTCA/BcnMsm", "BcnMismatch",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDuTCA),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSlotuTCA));
	_cuTCAOrnMsm.initialize(_name+"/uTCA/OrnMsm", "OrnMismatch",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDuTCA),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSlotuTCA));
	_cVMEOccupancy.initialize(_name+"/VME/Occupancy", "Occupancy",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDVME),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSpigot));
	_cuTCAOccupancy.initialize(_name+"/uTCA/Occupancy", "Occupancy",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDuTCA),
		new axis::CoordinateAxis(axis::fYaxis, axis::fSlotuTCA));

	//	Summary Containers
	_cSummary.initialize(_name+"/Summary", "Summary",
		new axis::CoordinateAxis(axis::fXaxis, axis::fFEDComb),
		new axis::FlagAxis(axis::fYaxis, "Flag", int(nRawFlag)));
	_cSummaryvsLS_FED.initialize(_name+"/Summary/vsLS_FED", "SummaryvsLS",
		mapper::fFED,
		new axis::ValueAxis(axis::fXaxis, axis::fLS),
		new axis::FlagAxis(axis::fYaxis, "Flag", int(nRawFlag)));

	//	tags
	_tagFEDs = ps.getUntrackedParameter<edm::InputTag>("tagFEDs",
		edm::InputTag("rawDataCollector"));
	_tokFEDs = consumes<FEDRawDataCollection>(_tagFEDs);

	//	Skip List
	_vSkipFEDList = ps.getUntrackedParameter<std::vector<int> >("skipFEDList");

	//	load labels 
	_fNames.push_back("EVN Mismatch");
	_fNames.push_back("ORN Mismatch");
	_fNames.push_back("BCN Mismatch");
	_cSummary.loadLabels(_fNames);
	_cSummaryvsLS_FED.loadLabels(_fNames);
}

/* virtual */ void RawTask::bookHistograms(DQMStore::IBooker &ib,
	edm::Run const& r, edm::EventSetup const& es)
{
	DQTask::bookHistograms(ib, r, es);
	_cVMEEvnMsm.book(ib);
	_cVMEBcnMsm.book(ib);
	_cVMEOrnMsm.book(ib);
	_cuTCAEvnMsm.book(ib);
	_cuTCABcnMsm.book(ib);
	_cuTCAOrnMsm.book(ib);
	_cVMEOccupancy.book(ib);
	_cuTCAOccupancy.book(ib);

	_cSummary.book(ib);
	_cSummaryvsLS_FED.book(ib);
}

/* virtual */ void RawTask::endLuminosityBlock(edm::LuminosityBlock const& l,
	edm::EventSetup const& es)
{
	//	statuses
	//	By default the flag is not applicable
	double status[constants::FED_TOTAL_NUM][nRawFlag];
	for (int i=0; i<constants::FED_TOTAL_NUM; i++)
		for (int j=fEvnMsm; j<nRawFlag; j++)
			status[i][j] = constants::NOT_APPLICABLE;

	/*
	 *	Do the checks here
	 *	-> Evn Mismatch
	 *	-> Bcn Mismatch
	 *	-> Orn Mismatch (Not Applicable)
	 */

	//	evn/bcn
	for (int i=0; i<constants::FED_TOTAL_NUM; i++)
	{
		//	skip status evaluation of empty FEDs
		bool q = false;
		for (std::vector<int>::const_iterator it=_vSkipFEDList.begin();
			it!=_vSkipFEDList.end(); ++it)
			if ((*it) == utilities::getFEDById(i))
			{
				q = true;
				break;
			}
		if (q==true)
			continue;

		if (_nEvnMsm[i]>0)
			status[i][fEvnMsm] = constants::LOW;
		else 
			status[i][fEvnMsm] = constants::GOOD;
		if (_nBcnMsm[i]>0)
			status[i][fBcnMsm] = constants::LOW;
		else 
			status[i][fBcnMsm] = constants::GOOD;
	}

	//	finally set all the statuses
	for (unsigned int i=0; i<constants::FED_TOTAL_NUM; i++)
		for (int j=fEvnMsm; j<nRawFlag; j++)
		{
			_cSummary.setBinContent(i, j, status[i][j]);
			_cSummaryvsLS_FED.setBinContent(i, _currentLS, 
				j, status[i][j]);
		}

	DQTask::endLuminosityBlock(l, es);
}

/* virtual */ void RawTask::_process(edm::Event const& e, 
	edm::EventSetup const&)
{
	edm::Handle<FEDRawDataCollection> craw;
	if (!e.getByToken(_tokFEDs, craw))
		_logger.dqmthrow("Collection FEDRawDataCollection isn't available"
			+ _tagFEDs.label() + " " + _tagFEDs.instance());
	
	for (int fed=FEDNumbering::MINHCALFEDID; 
		fed<+FEDNumbering::MAXHCALuTCAFEDID; fed++)
	{
		//	skip all non-HCAL FEDs
		if ((fed>FEDNumbering::MAXHCALFEDID &&
			fed<FEDNumbering::MINHCALuTCAFEDID) ||
			fed>FEDNumbering::MAXHCALuTCAFEDID)
			continue;

		FEDRawData const& raw = craw->FEDData(fed);
		if (raw.size() < RAW_EMPTY)// skip if empty
			continue;

		if (fed<=FEDNumbering::MAXHCALFEDID) // VME
		{
			HcalDCCHeader const* hdcc = (HcalDCCHeader const*)(raw.data());
			if (!hdcc)
				continue;;
			unsigned int bcn = hdcc->getBunchId();
			unsigned int orn = hdcc->getOrbitNumber();
			unsigned long evn = hdcc->getDCCEventNumber();
			int dccId = hdcc->getSourceId()-700; //	700 is the hard offset

			//	 Iterate over all the spigots
			HcalHTRData htr;
			for (int is=0; is<HcalDCCHeader::SPIGOT_COUNT; is++)
			{
				int r = hdcc->getSpigotData(is, htr, raw.size());

				//	invalid data
				if (r!=0)
					continue;

				unsigned int htr_evn = htr.getL1ANumber();
				unsigned int htr_orn = htr.getOrbitNumber();
				unsigned int htr_bcn = htr.getBunchNumber();
				bool qevn = htr_evn!=evn;
				bool qorn = htr_orn!=orn;
				bool qbcn = htr_bcn!=bcn;
				HcalElectronicsId eid(0, 1, is, dccId);
				_cVMEEvnMsm.fill(fed, eid, qevn ? 1 : 0);
				_cVMEBcnMsm.fill(fed, eid, qbcn ? 1 : 0);
				_cVMEOrnMsm.fill(fed, eid, qorn ? 1 : 0);
				_nEvnMsm[utilities::getIdByFED(fed)]+=
					qevn ? 1 : 0;
				_nBcnMsm[utilities::getIdByFED(fed)]+=
					qbcn ? 1 : 0;
				_cVMEOccupancy.fill(fed, eid);
			}
			
		}
		else //	uTCA
		{
			hcal::AMC13Header const* hamc13 = (hcal::AMC13Header const*)
				(raw.data());
			if (!hamc13)
				continue;

			unsigned int bcn = hamc13->bunchId();
			unsigned int orn = hamc13->orbitNumber();
			unsigned int evn = hamc13->l1aNumber();
			int namc = hamc13->NAMC();

			//	itearte over all AMC13
			for (int iamc=0; iamc<namc; iamc++)
			{
				int slot = hamc13->AMCSlot(iamc);
				int crate = hamc13->AMCId(iamc)&0xFF;
				HcalElectronicsId eid(crate, slot, 5, 0, false);
				HcalUHTRData uhtr(hamc13->AMCPayload(iamc), 
					hamc13->AMCSize(iamc));
				for (HcalUHTRData::const_iterator iuhtr=uhtr.begin(); 
					iuhtr!=uhtr.end(); ++iuhtr)
				{
					if (!iuhtr.isHeader())
						continue;

					_cuTCAOccupancy.fill(fed, eid);
					//	use data flavour - found in the unpacker
					if (iuhtr.flavor()==UTCA_DATAFLAVOR)
					{
						uint32_t uhtr_evn = uhtr.l1ANumber();
						uint32_t uhtr_bcn = uhtr.bunchNumber();
						uint32_t uhtr_orn = uhtr.orbitNumber();
						bool qevn = uhtr_evn!=evn;
						bool qorn = uhtr_orn!=orn;
						bool qbcn = uhtr_bcn!=bcn;
						_cuTCAEvnMsm.fill(fed, eid, 
							qevn ? 1 : 0);
						_cuTCABcnMsm.fill(fed, eid, 
							qbcn ? 1 : 0);
						_cuTCAOrnMsm.fill(fed, eid, 
							qorn ? 1 : 0);
						_nEvnMsm[utilities::getIdByFED(fed)]+=
							qevn ? 1 : 0;
						_nBcnMsm[utilities::getIdByFED(fed)]+=
							qbcn ? 1 : 0;
					}
				}
			}
		}
	}
}

/* virtual */ void RawTask::_resetMonitors(UpdateFreq uf)
{
	switch (uf)
	{
		case fEvent:
			break;
		case hcaldqm::fLS:
			for (int i=0; i<constants::FED_TOTAL_NUM; i++)
			{
				_nEvnMsm[i] = 0;
				_nOrnMsm[i] = 0;
				_nBcnMsm[i] = 0;
			}
			break;
		case hcaldqm::f10LS:
//			_cVMEEvnMsm.reset();
//			_cVMEOrnMsm.reset();
//			_cVMEBcnMsm.reset();
//			_cuTCAEvnMsm.reset();
//			_cuTCAOrnMsm.reset();
//			_cuTCABcnMsm.reset();
			break;
		default:
			break;
	}

	DQTask::_resetMonitors(uf);
}

DEFINE_FWK_MODULE(RawTask);
