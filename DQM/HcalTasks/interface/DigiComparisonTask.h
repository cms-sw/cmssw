#ifndef DigiComparisonTask_h
#define DigiComparisonTask_h

/**
 *	file:		DigiComparisonTask.h
 *	Author:		Viktor Khristenko
 *	Date:		08.12.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/HashFilter.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class DigiComparisonTask : public DQTask
{
	public: 
		DigiComparisonTask(edm::ParameterSet const&);
		virtual ~DigiComparisonTask()
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
		edm::InputTag	_tagHBHE1;
		edm::InputTag	_tagHBHE2;
		edm::EDGetTokenT<HBHEDigiCollection>	_tokHBHE1;
		edm::EDGetTokenT<HBHEDigiCollection>	_tokHBHE2;

		//	emap+hashmap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmapuTCA;
		electronicsmap::ElectronicsMap _ehashmapVME;

		//	hashes/FED vectors
		std::vector<uint32_t> _vhashFEDs;

		//	Filters
		HashFilter _filter_VME;
		HashFilter _filter_uTCA;

		/**
		 *	Containers
		 */

		//	ADC
		Container2D			_cADC_Subdet[10];
		Container2D			_cADCall_Subdet;

		//	Mismatched
		Container2D			_cMsm_FEDVME;
		Container2D			_cMsm_FEDuTCA;
		Container2D			_cMsm_depth;

		//	Missing Completely
		Container1D			_cADCMsnuTCA_Subdet;
		Container1D			_cADCMsnVME_Subdet;
		Container2D			_cMsnVME_depth;
		Container2D			_cMsnuTCA_depth;
		Container2D			_cMsn_FEDVME;
		Container2D			_cMsn_FEDuTCA;
};

#endif
