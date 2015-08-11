#include "DQM/HcalCommon/interface/HcalDQClient.h"

namespace hcaldqm
{
	HcalDQClient::HcalDQClient(edm::ParameterSet const& ps)
		: HcalDQMonitor(ps.getUntrackedParameterSet("moduleParameters")),
		_bmes(ps.getUntrackedParameterSet("bookMEs"), _mi.debug),
		_rmes(ps.getUntrackedParameterSet("retrieveMEs"), _mi.debug)
	{
	}

	/* virtual */HcalDQClient::~HcalDQClient()
	{
	}

	//	Function to be reimplemented from DQMEDAnalyzer
	//	Executed at the end of the job
	/* virtual */ void HcalDQClient::dqmEndJob(DQMStore::IBooker& ib,
			DQMStore::IGetter& ig)
	{
	//	_bmes.book(ib);	
	//	doWork(ib, ig);
	}

	//	beginJob
	/* virtual */ void HcalDQClient::beginJob()
	{
		this->debug_(_mi.name + " Begins Job");
	}

	//	Function to be reimplemented from the DQMEDAnalyzer
	//	Executed at the edn of LS
	/* virtual */ void HcalDQClient::dqmEndLuminosityBlock(DQMStore::IBooker& ib,
			DQMStore::IGetter& ig,
			edm::LuminosityBlock const& ls, edm::EventSetup const& es)
	{
		try
		{
			//	Retriver Histos you need and apply Resets
			_rmes.retrieve(ig, _mi.subsystem);
			this->reset(1);

			//	Do the Work
			this->debug_(_mi.name + " doing work");
			_mi.currentLS = ls.luminosityBlock();
			doWork(ig, ls, es);
		}
		catch (cms::Exception &exc)
		{
			//	Catching cms Exceptions
			this->warn_(std::string("We have cms::Exception Triggered. ") +
					std::string(exc.what()));
		}
		catch (std::exception &exc)
		{
			//	Catching STD Exceptions
			this->warn_("We have STD Exception Triggered. " +
					std::string(exc.what()));
		}
		catch(...)
		{
			this->warn_("UNKNOWN Exception Triggered. ");
		}
	}

	//	reset
	/* virtual */ void HcalDQClient::reset(int const periodflag)
	{
		//	Collection Class determines itself who needs a reset and when
		//	Do it only for Monitor Modules which have been booked in this client
		_bmes.reset(periodflag);

		if (periodflag==0)
		{
			//	each event 
		}
		else if (periodflag==1)
		{
			//	each LS
			_mi.evsPerLS = 0;
		}
	}
}












