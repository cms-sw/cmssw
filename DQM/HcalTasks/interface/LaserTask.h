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

class LaserTask : public hcaldqm::DQTask
{
	public:
		LaserTask(edm::ParameterSet const&);
		virtual ~LaserTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endRun(edm::Run const& r, edm::EventSetup const&)
		{
			if (_ptype==hcaldqm::fLocal)
			{
				if (r.runAuxiliary().run()==1)
					return;
				else 
					this->_dump();
			}
		}
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		//	funcs
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(hcaldqm::UpdateFreq);
		virtual bool _isApplicable(edm::Event const&);
		virtual void _dump();

		//	tags and tokens
		edm::InputTag	_tagHBHE;
		edm::InputTag	_tagHO;
		edm::InputTag	_tagHF;
		edm::InputTag	_taguMN;
		edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
		edm::EDGetTokenT<HODigiCollection> _tokHO;
		edm::EDGetTokenT<HFDigiCollection> _tokHF;
		edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

		//	emap
		HcalElectronicsMap const* _emap;
		hcaldqm::electronicsmap::ElectronicsMap _ehashmap;
		hcaldqm::filter::HashFilter _filter_uTCA;
		hcaldqm::filter::HashFilter _filter_VME;

		//	Cuts and variables
		int _nevents;
		double _lowHBHE;
		double _lowHO;
		double _lowHF;
		uint32_t _laserType;

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

		hcaldqm::Container1D _cADC_SubdetPM;

		//	Prof1D
		hcaldqm::ContainerProf1D	_cShapeCut_FEDSlot;
		hcaldqm::ContainerProf1D _cTimingvsEvent_SubdetPM;
		hcaldqm::ContainerProf1D _cSignalvsEvent_SubdetPM;
		hcaldqm::ContainerProf1D _cTimingvsLS_SubdetPM;
		hcaldqm::ContainerProf1D _cSignalvsLS_SubdetPM;
		hcaldqm::ContainerProf1D _cTimingvsBX_SubdetPM;
		hcaldqm::ContainerProf1D _cSignalvsBX_SubdetPM;

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
