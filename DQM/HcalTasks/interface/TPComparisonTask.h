#ifndef TPComparisonTask_h
#define TPComparisonTask_h

/**
 *	file:		TPComparisonTask.h
 *	Author:		Viktor Khristenko
 *	Date:		08.12.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class TPComparisonTask : public DQTask
{
	public: 
		TPComparisonTask(edm::ParameterSet const&);
		virtual ~TPComparisonTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		//	funcs
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		//	Tags and corresponding Tokens
		edm::InputTag	_tag1;
		edm::InputTag	_tag2;
		edm::EDGetTokenT<HcalTrigPrimDigiCollection>	_tok1;
		edm::EDGetTokenT<HcalTrigPrimDigiCollection>	_tok2;

		//	tmp flags
		bool _skip1x1;

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmapuTCA;
		electronicsmap::ElectronicsMap _ehashmapVME;

		//	hahses/FED vectors
		std::vector<uint32_t> _vhashFEDs;

		//	Filters
		HashFilter _filter_VME;
		HashFilter _filter_uTCA;

		/**
		 *	Containers
		 */

		//	Et
		Container2D			_cEt_TTSubdet[4];
		Container2D			_cEtall_TTSubdet;

		//	FG
		Container2D			_cFG_TTSubdet[4];

		//	Missing
		Container2D			_cMsn_FEDVME;
		Container2D			_cMsn_FEDuTCA;
		ContainerSingle2D	_cMsnVME;
		ContainerSingle2D	_cMsnuTCA;

		//	mismatches
		Container2D			_cEtMsm_FEDVME;
		Container2D			_cEtMsm_FEDuTCA;
		Container2D			_cFGMsm_FEDVME;
		Container2D			_cFGMsm_FEDuTCA;

		//	depth like
		ContainerSingle2D	_cEtMsm;
		ContainerSingle2D	_cFGMsm;
};

#endif
