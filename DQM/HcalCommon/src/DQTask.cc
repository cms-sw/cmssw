
#include "DQM/HcalCommon/interface/DQTask.h"

namespace hcaldqm
{	
	using namespace constants;
	DQTask::DQTask(edm::ParameterSet const& ps):
		DQModule(ps),
		_cEvsTotal(_name, "EventsTotal"),
		_cEvsPerLS(_name, "EventssPerLS"),
		_cRunKeyVal(_name, "RunKeyValue"),
		_cRunKeyName(_name, "RunKeyName"),
		_cProcessingTypeName(_name, "ProcessingType"),
		_procLSs(0)
	{
		//	tags and Tokens
		_tagRaw = ps.getUntrackedParameter<edm::InputTag>("tagRaw",
			edm::InputTag("rawDataCollector"));
		_tokRaw = consumes<FEDRawDataCollection>(_tagRaw);
	}

	/*
	 *	By design, all the sources will ahve this function inherited and will
	 *	never override. 
	 */
	/* virtual */ void DQTask::analyze(edm::Event const& e,
		edm::EventSetup const& es)
	{
		this->_resetMonitors(fEvent);
		_logger.debug(_name+" processing");
		if (!this->_isApplicable(e))
			return;

		_evsTotal++; _cEvsTotal.fill(_evsTotal);
		_evsPerLS++; _cEvsPerLS.fill(_evsPerLS);
		_currentLS = e.luminosityBlock();
		this->_process(e, es);
	}

	/* virtual */ void DQTask::bookHistograms(DQMStore::IBooker &ib,
		edm::Run const& r,
		edm::EventSetup const& es)
	{
		_cEvsTotal.book(ib);
		_cEvsPerLS.book(ib);
		_cRunKeyVal.book(ib);
		_cRunKeyName.book(ib);
		_cProcessingTypeName.book(ib);

		//	fill what you can now
		_cRunKeyVal.fill(_runkeyVal);
		_cRunKeyName.fill(_runkeyName);
		_cProcessingTypeName.fill(pTypeNames[_ptype]);
	}

	/* virtual */ void DQTask::dqmBeginRun(edm::Run const& r,
		edm::EventSetup const& es)
	{
		this->_resetMonitors(fEvent);
		this->_resetMonitors(fLS);
		this->_resetMonitors(f10LS);
		this->_resetMonitors(f50LS);
		this->_resetMonitors(f100LS);
	}

	/* virtual */ void DQTask::beginLuminosityBlock(
		edm::LuminosityBlock const& lb,
		edm::EventSetup const& es)
	{
		this->_resetMonitors(fLS);
		if (_procLSs%10==0)
			this->_resetMonitors(f10LS);
		if (_procLSs%50==0)
			this->_resetMonitors(f50LS);
		if (_procLSs%100==0)
			this->_resetMonitors(f100LS);
	}

	/* virtual */ void DQTask::endLuminosityBlock(
		edm::LuminosityBlock const& lb,
		edm::EventSetup const& es)
	{
		_procLSs++;
	}

	/* virtual */ void DQTask::_resetMonitors(UpdateFreq uf)
	{
		//	reset per event
		switch (uf)
		{
			case fEvent:
				break;
			case fLS:
				_evsPerLS = 0;
				break;
			case f10LS:
				break;
			case f50LS:
				break;
			case f100LS:
				break;
			default:
				break;
		}
	}

	/* virtual */ int DQTask::_getCalibType(edm::Event const&e)
	{
		int calibType = 0;

		edm::Handle<FEDRawDataCollection> craw;
		if (!e.getByToken(_tokRaw, craw))
			_logger.dqmthrow(
				"Collection FEDRawDataCollection isn't available " 
				+ _tagRaw.label() + " " + _tagRaw.instance());

		int badFEDs=0;
		std::vector<int> types(8,0);
		for (int i=FED_VME_MIN; i<=FED_VME_MAX; i++)
		{
			FEDRawData const& fd = craw->FEDData(i);
			if (fd.size()<24)
			{
				badFEDs++;
				continue;
			}
			int cval = (int)((HcalDCCHeader const*)(fd.data()))->getCalibType();
			if (cval>7)
				_logger.warn("Unexpected Calib Type in FED " + 
					boost::lexical_cast<std::string>(i));
			types[cval]++;
		}
		for (int i=FED_uTCA_MIN; i<=FED_uTCA_MAX; i++)
		{
			FEDRawData const& fd = craw->FEDData(i);
			if (fd.size()<24)
			{
				badFEDs++;
				continue;
			}
			int cval = (int)((HcalDCCHeader const*)(fd.data()))->getCalibType();
			if (cval>7)
				_logger.warn("Unexpected Calib Type in FED " + 
					boost::lexical_cast<std::string>(i));
			types[cval]++;
		}

		int max = 0;
		for (unsigned int ic=0; ic<8; ic++)
		{
			if (types[ic]>max)
			{
				max = types[ic];
				calibType = ic;
			}
		}
		if (max!=(FED_VME_NUM+(FED_uTCA_MAX-FED_uTCA_MIN+1)-badFEDs))
			_logger.warn("Conflicting Calibration Types found. Assigning " +
				boost::lexical_cast<std::string>(calibType));

		return calibType;
	}
}






