

#include "DQM/HcalCommon/interface/HcalDQSource.h"

namespace hcaldqm
{
	HcalDQSource::HcalDQSource(edm::ParameterSet const& ps) :
		HcalDQMonitor(ps.getUntrackedParameterSet("moduleParameters")), 
		_mes(ps.getUntrackedParameterSet("MEs"), _mi.debug)
	{
	}

	/* virtual */HcalDQSource::~HcalDQSource() 
	{
	}

	//	By design, all the Sources will have this function inherited
	//	They will not reimplement it, although it's possible
	//	This way, the try/catch block will be applied to all of them!
	/* virtual */ void HcalDQSource::analyze(edm::Event const &e,
			edm::EventSetup const& es)
	{
		try
		{
			//	Do event Reset and extract calibtype
			this->reset(0);
			this->extractCalibType(e);
			if (this->isAllowedCalibType()==false)
				return;

			//	Virtual Method that determines if we have to run this module 
			//	for this event, after extracting calibration type
			if (!(this->isApplicable(e)))
				return;

			this->debug_(_mi.name + " doing work");
			//	Update event counters;
			_mi.evsTotal++; _mes["EventsProcessed"].Fill(_mi.evsTotal);
			_mi.evsGood++;
			_mi.evsPerLS++; _mes["EventsProcessedPerLS"].Fill(_mi.evsPerLS);
			_mi.currentLS = e.luminosityBlock();
			this->doWork(e, es);
		}
		catch (cms::Exception& exc)
		{
			//	Catching cms Exceptions
			this->warn_(std::string("We have cms::Exception Triggered. ") 
					+ std::string(exc.what()));
		}
		catch (std::exception& exc)
		{
			this->warn_("We have STD Exception Triggered. " + 
					std::string(exc.what()));
		}
		catch (...)
		{
			this->warn_("UNKNOWN Exception Triggered. ");
		}
		
	}

	/* virtual */ void HcalDQSource::bookHistograms(DQMStore::IBooker &ib,
			edm::Run const& r, edm::EventSetup const& es)
	{
		if (this->shouldBook())
			_mes.book(ib, _mi.subsystem);
	}

	/* virtual */ void HcalDQSource::dqmBeginRun(edm::Run const& r,
			edm::EventSetup const& es)
	{
		this->reset(0);
		this->reset(1);
	}

	/* virtual */ void HcalDQSource::beginLuminosityBlock(
			edm::LuminosityBlock const& lb, edm::EventSetup const& es)
	{
		this->reset(1);
		//	Reset things per LS.
		//	But at least 100 events in LS
//		if (_mi.evsPerLS>100)
//			this->reset(1);
	}

	/* virtual */ void HcalDQSource::endLuminosityBlock(
			edm::LuminosityBlock const& lb, edm::EventSetup const& es)
	{}

	//	extract Event Calibration Type from FEDs
	void HcalDQSource::extractCalibType(edm::Event const&e)
	{
		edm::Handle<FEDRawDataCollection> craw;
		INITCOLL(_labels["RAW"], craw);

		//	for now
		int badFEDs = 0;
		std::vector<unsigned int> types(8,0);
		for (std::vector<int>::const_iterator it=_mi.feds.begin();
				it!=_mi.feds.end(); ++it)
		{
			FEDRawData const& fd = craw->FEDData(*it);
			if (fd.size()<hcaldqm::constants::RAWDATASIZE_CALIB)
			{badFEDs++; continue;}
			int cval = (int)((HcalDCCHeader const*)(fd.data()))->getCalibType();
			if (cval>hcaldqm::constants::MAXCALIBTYPE)
				warn_("Unexpected Calib Type in FED " + 
						boost::lexical_cast<std::string>(*it));
			types[cval]++;
		}

		unsigned int max = 0;
		for (unsigned int ic=0; ic<types.size(); ic++)
		{
			if (types[ic]>max)
			{_mi.currentCalibType=ic; max=types[ic];}
			if (max==_mi.feds.size())
				break;
		}

		if (max!=(_mi.feds.size()-badFEDs))
			warn_("Conflictings Calibration Types found. Assigning " + 
					boost::lexical_cast<std::string>(_mi.currentCalibType));
	}

	//	Check if calibration type set is among allowed
	bool HcalDQSource::isAllowedCalibType()
	{
		for (std::vector<int>::const_iterator it=_mi.calibTypesAllowed.begin();
				it!=_mi.calibTypesAllowed.end(); ++it)
			if (_mi.currentCalibType==*it)
				return true;

		return false;
	}

	//	reset
	/* virtual */ void HcalDQSource::reset(int const periodflag)
	{
		//	Collection Class will determine itself who needs a reset and when
		_mes.reset(periodflag);

		if (periodflag==0)
		{
			//	each event reset
		}
		else if (periodflag==1)
		{
			//	each LS reset
			_mi.evsPerLS=0;
		}
	}

}









