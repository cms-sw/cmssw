#include "DQM/HcalTasks/interface/HcalOnlineHarvesting.h"

HcalOnlineHarvesting::HcalOnlineHarvesting(edm::ParameterSet const& ps) :
	DQHarvester(ps), _reportSummaryMap(NULL)
{

	//	NOTE: I will leave Run Summary Generators in place 
	//	just not triggering on endJob!
	_vsumgen.resize(nSummary);
	_vnames.resize(nSummary);
	_vmarks.resize(nSummary);
	for (uint32_t i=0; i<_vmarks.size(); i++)
		_vmarks[i]=false;
	_vnames[fRaw]="RawTask";
	_vnames[fDigi]="DigiTask";
	_vnames[fReco]="RecHitTask";
	_vnames[fTP]="TPTask";
	_vnames[fPedestal]="PedestalTask";

	_vsumgen[fRaw] = new RawRunSummary("RawRunHarvesting",
		_vnames[fRaw], ps);
	_vsumgen[fDigi] = new DigiRunSummary("DigiRunHarvesting", 
		_vnames[fDigi],ps);
	_vsumgen[fReco] = new RecoRunSummary("RecoRunHarvesting",
		_vnames[fReco], ps);
	_vsumgen[fTP] = new TPRunSummary("TPRunHarvesting",
		_vnames[fTP], ps);
	_vsumgen[fPedestal] = new PedestalRunSummary("PedestalRunHarvesting",
		_vnames[fPedestal], ps);
}

/* virtual */ void HcalOnlineHarvesting::beginRun(
	edm::Run const& r, edm::EventSetup const& es)
{
	DQHarvester::beginRun(r,es);
	for (std::vector<DQClient*>::const_iterator it=_vsumgen.begin();
		it!=_vsumgen.end(); ++it)
		(*it)->beginRun(r,es);
}

/* virtual */ void HcalOnlineHarvesting::_dqmEndLuminosityBlock(
	DQMStore::IBooker& ib,
	DQMStore::IGetter& ig, edm::LuminosityBlock const&, 
	edm::EventSetup const&)
{
	//	DETERMINE WHICH MODULES ARE PRESENT IN DATA
	if (ig.get(_subsystem+"/"+_vnames[fRaw]+"/EventsTotal")!=NULL)
		_vmarks[fRaw]=true;
	if (ig.get(_subsystem+"/"+_vnames[fDigi]+"/EventsTotal")!=NULL)
		_vmarks[fDigi]=true;
	if (ig.get(_subsystem+"/"+_vnames[fTP]+"/EventsTotal")!=NULL)
		_vmarks[fTP]=true;
	if (ig.get(_subsystem+"/"+_vnames[fReco]+"/EventsTotal")!=NULL)
		_vmarks[fReco]=true;
	if (ig.get(_subsystem+"/"+_vnames[fPedestal]+"/EventsTotal")!=NULL)
		_vmarks[fPedestal]=true;

	//	CREATE SUMMARY REPORT MAP FED vs LS and LOAD MODULE'S SUMMARIES
	//	NOTE: THIS STATEMENTS WILL BE EXECUTED ONLY ONCE!
	if (!_reportSummaryMap)
	{
		ig.setCurrentFolder(_subsystem+"/EventInfo");
		_reportSummaryMap = ib.book2D("reportSummaryMap", "reportSummaryMap",
			_maxLS, 1, _maxLS+1, _vFEDs.size(), 0, _vFEDs.size());
		for (uint32_t i=0; i<_vFEDs.size(); i++)
		{
			char name[5];
			sprintf(name, "%d", _vFEDs[i]);
			_reportSummaryMap->setBinLabel(i+1, name, 2);
		}
		//	set LS bit to mark Xaxis as LS axis
		_reportSummaryMap->getTH1()->SetBit(BIT(BIT_OFFSET+BIT_AXIS_LS));

		// INITIALIZE ALL THE MODULES
		for (uint32_t i=0; i<_vnames.size(); i++)
			_vcSummaryvsLS.push_back(ContainerSingle2D(_vnames[i],
				"SummaryvsLS",
				new quantity::LumiSection(_maxLS),
				new quantity::FEDQuantity(_vFEDs),
				new quantity::ValueQuantity(quantity::fState)));

		//	LOAD ONLY THOSE MODULES THAT ARE PRESENT IN DATA
		for (uint32_t i=0; i<_vmarks.size(); i++)
		{
			if (_vmarks[i])
				_vcSummaryvsLS[i].load(ig, _subsystem);
		}
	}

	int ifed=0;
	for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
		it!=_vhashFEDs.end(); ++it)
	{
		HcalElectronicsId eid(*it);
		flag::Flag fSum("Status", flag::fNCDAQ);
		for (uint32_t im=0; im<_vmarks.size(); im++)
			if (_vmarks[im])
			{
				int x = _vcSummaryvsLS[im].getBinContent(eid, _currentLS);
				flag::Flag flag("Status", (flag::State)x);
				fSum+=flag;
			}
		_reportSummaryMap->setBinContent(_currentLS, ifed+1, int(fSum._state));
		ifed++;
	}
}

/*
 *	NO END JOB PROCESSING FOR ONLINE!
 */
/* virtual */ void HcalOnlineHarvesting::_dqmEndJob(DQMStore::IBooker& ib,
	DQMStore::IGetter& ig)
{}

DEFINE_FWK_MODULE(HcalOnlineHarvesting);
