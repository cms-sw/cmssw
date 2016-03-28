#ifndef PedestalTask_h
#define PedestalTask_h

/*
 *	file:			PedestalTask.h
 *	Author:			Viktor Khristenko
 *	Date:			16.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerXXX.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class PedestalTask : public DQTask
{
	public:
		PedestalTask(edm::ParameterSet const&);
		virtual ~PedestalTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);
		virtual void endRun(edm::Run const&, edm::EventSetup const&);

		enum PedestalFlag
		{
			fMsn = 0,
			fBadMean = 1,
			fBadRMS = 2,
			nPedestalFlag = 3
		};

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
		HcalElectronicsMap const*	_emap;
		electronicsmap::ElectronicsMap _ehashmap;
		HashFilter _filter_uTCA;
		HashFilter _filter_VME;
		HashFilter _filter_C36;

		std::vector<uint32_t> _vhashFEDs;
		ContainerXXX<double> _xPedSum;
		ContainerXXX<double> _xPedSum2;
		ContainerXXX<int>	_xPedEntries;
		ContainerXXX<double> _xPedRefMean;
		ContainerXXX<double> _xPedRefRMS;
		ContainerXXX<int> _xNMsn;
		ContainerXXX<int> _xNBadMean;
		ContainerXXX<int> _xNBadRMS;

		//	1D Means/RMSs
		Container1D		_cMean_Subdet;
		Container1D		_cRMS_Subdet;

		//	1D Means/RMSs Conditions DB comparison
		Container1D		_cMeanDBRef_Subdet;
		Container1D		_cRMSDBRef_Subdet;

		//	2D
		ContainerProf2D		_cMean_depth;
		ContainerProf2D		_cRMS_depth;
		ContainerProf2D		_cMean_FEDVME;
		ContainerProf2D		_cMean_FEDuTCA;
		ContainerProf2D		_cRMS_FEDVME;
		ContainerProf2D		_cRMS_FEDuTCA;
		
		//	with DB Conditions comparison
		ContainerProf2D		_cMeanDBRef_depth;
		ContainerProf2D		_cRMSDBRef_depth;
		ContainerProf2D		_cMeanDBRef_FEDVME;
		ContainerProf2D		_cMeanDBRef_FEDuTCA;
		ContainerProf2D		_cRMSDBRef_FEDVME;
		ContainerProf2D		_cRMSDBRef_FEDuTCA;

		//	Missing + Bad Quality
		Container2D		_cMissing_depth;
		Container2D		_cMeanBad_depth;
		Container2D		_cRMSBad_depth;

		Container1D _cMissingvsLS_FED;
		Container1D _cOccupancyvsLS_Subdet;
		Container1D _cNBadMeanvsLS_FED;
		Container1D _cNBadRMSvsLS_FED;

		Container2D		_cMissing_FEDVME;
		Container2D		_cMissing_FEDuTCA;
		Container2D		_cMeanBad_FEDVME;
		Container2D		_cRMSBad_FEDuTCA;
		Container2D		_cRMSBad_FEDVME;
		Container2D		_cMeanBad_FEDuTCA;

		ContainerSingle2D _cSummary;
};

#endif







