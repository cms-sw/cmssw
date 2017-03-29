#ifndef LaserTask_h
#define LaserTask_h

/*
 *	file:			LaserTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class LaserTask : public DQTask
{
	public:
		LaserTask(edm::ParameterSet const&);
		virtual ~LaserTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endRun(edm::Run const& r, edm::EventSetup const&)
		{
			if (_ptype==fLocal)
				if (r.runAuxiliary().run()==1)
					return;
			this->_dump();
		}

	protected:
		//	funcs
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);
		virtual bool _isApplicable(edm::Event const&);
		virtual void _dump();

		//	tags and tokens
		edm::InputTag	_tagHBHE;
		edm::InputTag	_tagHO;
		edm::InputTag	_tagHF;
		edm::InputTag	_tagTrigger;
		edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
		edm::EDGetTokenT<HODigiCollection> _tokHO;
		edm::EDGetTokenT<HFDigiCollection> _tokHF;
		edm::EDGetTokenT<HcalTBTriggerData> _tokTrigger;

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;
		HashFilter _filter_uTCA;
		HashFilter _filter_VME;

		//	Cuts and variables
		int _nevents;
		double _lowHBHE;
		double _lowHO;
		double _lowHF;

		//	Compact
		ContainerXXX<double> _xSignalSum;
		ContainerXXX<double> _xSignalSum2;
		ContainerXXX<int> _xEntries;
		ContainerXXX<double> _xTimingSum;
		ContainerXXX<double> _xTimingSum2;

		//	1D
		Container1D		_cSignalMean_Subdet;
		Container1D		_cSignalRMS_Subdet;
		Container1D		_cTimingMean_Subdet;
		Container1D		_cTimingRMS_Subdet;

		//	Prof1D
		ContainerProf1D	_cShapeCut_FEDSlot;
		ContainerProf1D _cTimingvsEvent_SubdetPM;
		ContainerProf1D _cSignalvsEvent_SubdetPM;
		ContainerProf1D _cTimingvsLS_SubdetPM;
		ContainerProf1D _cSignalvsLS_SubdetPM;

		//	2D timing/signals
		ContainerProf2D		_cSignalMean_depth;
		ContainerProf2D		_cSignalRMS_depth;
		ContainerProf2D		_cTimingMean_depth;
		ContainerProf2D		_cTimingRMS_depth;

		ContainerProf2D		_cSignalMean_FEDVME;
		ContainerProf2D		_cSignalMean_FEDuTCA;
		ContainerProf2D		_cTimingMean_FEDVME;
		ContainerProf2D		_cTimingMean_FEDuTCA;
		ContainerProf2D		_cSignalRMS_FEDVME;
		ContainerProf2D		_cSignalRMS_FEDuTCA;
		ContainerProf2D		_cTimingRMS_FEDVME;
		ContainerProf2D		_cTimingRMS_FEDuTCA;

		//	Bad Quality and Missing Channels
		Container2D		_cMissing_depth;
		Container2D		_cMissing_FEDVME;
		Container2D		_cMissing_FEDuTCA;
};

#endif







