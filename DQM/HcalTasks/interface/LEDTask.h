#ifndef LEDTask_h
#define LEDTask_h

/*
 *	file:			LEDTask.h
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

class LEDTask : public hcaldqm::DQTask
{
	public:
		LEDTask(edm::ParameterSet const&);
		virtual ~LEDTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endRun(edm::Run const& r, edm::EventSetup const&)
		{
			if (_ptype==hcaldqm::fLocal)
				if (r.runAuxiliary().run()==1)
					return;
			this->_dump();
		}

	protected:
		//	funcs
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(hcaldqm::UpdateFreq);
		virtual bool _isApplicable(edm::Event const&);
		virtual void _dump();

		//	tags and tokens
		edm::InputTag	_tagHBHE;
		edm::InputTag	_tagHEP17;
		edm::InputTag	_tagHO;
		edm::InputTag	_tagHF;
		edm::InputTag	_tagTrigger;
		edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
		edm::EDGetTokenT<QIE11DigiCollection> _tokHEP17;
		edm::EDGetTokenT<HODigiCollection> _tokHO;
		edm::EDGetTokenT<QIE10DigiCollection> _tokHF;
		edm::EDGetTokenT<HcalTBTriggerData> _tokTrigger;

		//	emap
		hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
		hcaldqm::filter::HashFilter _filter_uTCA;
		hcaldqm::filter::HashFilter _filter_VME;

		//	Cuts
		double _lowHBHE;
		double _lowHEP17;
		double _lowHO;
		double _lowHF;

		//	Compact
		hcaldqm::ContainerXXX<double> _xSignalSum;
		hcaldqm::ContainerXXX<double> _xSignalSum2;
		hcaldqm::ContainerXXX<int> _xEntries;
		hcaldqm::ContainerXXX<double> _xTimingSum;
		hcaldqm::ContainerXXX<double> _xTimingSum2;

		//	1D
		hcaldqm::Container1D		_cSignalMean_Subdet;
		hcaldqm::Container1D		_cSignalRMS_Subdet;
		hcaldqm::Container1D		_cTimingMean_Subdet;
		hcaldqm::Container1D		_cTimingRMS_Subdet;

		//	Prof1D
		hcaldqm::ContainerProf1D	_cShapeCut_FEDSlot;

		//	2D timing/signals
		hcaldqm::ContainerProf2D		_cSignalMean_depth;
		hcaldqm::ContainerProf2D		_cSignalRMS_depth;
		hcaldqm::ContainerProf2D		_cTimingMean_depth;
		hcaldqm::ContainerProf2D		_cTimingRMS_depth;

		hcaldqm::ContainerProf2D		_cSignalMean_FEDVME;
		hcaldqm::ContainerProf2D		_cSignalMean_FEDuTCA;
		hcaldqm::ContainerProf2D		_cTimingMean_FEDVME;
		hcaldqm::ContainerProf2D		_cTimingMean_FEDuTCA;
		hcaldqm::ContainerProf2D		_cSignalRMS_FEDVME;
		hcaldqm::ContainerProf2D		_cSignalRMS_FEDuTCA;
		hcaldqm::ContainerProf2D		_cTimingRMS_FEDVME;
		hcaldqm::ContainerProf2D		_cTimingRMS_FEDuTCA;

		//	Bad Quality and Missing Channels
		hcaldqm::Container2D		_cMissing_depth;
		hcaldqm::Container2D		_cMissing_FEDVME;
		hcaldqm::Container2D		_cMissing_FEDuTCA;
};

#endif







