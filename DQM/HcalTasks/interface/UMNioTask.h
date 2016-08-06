#ifndef DQM_HcalTasks_UMNioTask_h
#define DQM_HcalTasks_UMNioTask_h

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
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class UMNioTask : public DQTask
{
	public:
		UMNioTask(edm::ParameterSet const&);
		virtual ~UMNioTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endRun(edm::Run const& r, edm::EventSetup const&)
		{
			if (_ptype==fLocal)
			{
				if (r.runAuxiliary().run()==1)
					return;
			}
		}
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		//	funcs
		virtual void _process(edm::Event const&, edm::EventSetup const&);

		std::vector<uint32_t> _eventtypes;

		//	tags and tokens
		edm::InputTag	_taguMN;
		edm::InputTag   _tagHBHE;
		edm::InputTag   _tagHO;
		edm::InputTag   _tagHF;
		edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
		edm::EDGetTokenT<HODigiCollection> _tokHO;
		edm::EDGetTokenT<HFDigiCollection> _tokHF;
		edm::EDGetTokenT<HcalUMNioDigi> _tokuMN;

		//	cuts
		double _lowHBHE, _lowHO, _lowHF;

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;
		HashFilter _filter_uTCA;
		HashFilter _filter_VME;

		//	1D
		ContainerSingle2D		_cEventType;
		ContainerSingle2D		_cTotalCharge;
};
#endif
