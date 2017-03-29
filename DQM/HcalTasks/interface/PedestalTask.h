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

		std::vector<flag::Flag> _vflags;
		enum PedestalFlag
		{
			fMsn = 0,
			fBadM = 1,
			fBadR = 2,
			nPedestalFlag=3
		};

		//	emap
		HcalElectronicsMap const*	_emap;
		electronicsmap::ElectronicsMap _ehashmap;
		HashFilter _filter_uTCA;
		HashFilter _filter_VME;
		HashFilter _filter_C36;

		//	thresholds
		double _thresh_mean, _thresh_rms, _thresh_badm, _thresh_badr;

		//	hashed ids of FEDs
		std::vector<uint32_t> _vhashFEDs;

		//	need containers total over the run and per 1LS
		ContainerXXX<double> _xPedSum1LS;
		ContainerXXX<double> _xPedSum21LS;
		ContainerXXX<int>	_xPedEntries1LS;
		ContainerXXX<double> _xPedSumTotal;
		ContainerXXX<double> _xPedSum2Total;
		ContainerXXX<int>	_xPedEntriesTotal;
		ContainerXXX<int> _xNChs; // number of channels per FED as in emap
		ContainerXXX<int> _xNMsn1LS; // #missing for 1LS per FED
		ContainerXXX<int> _xNBadMean1LS,_xNBadRMS1LS;

		//	CondBD Reference
		ContainerXXX<double> _xPedRefMean;
		ContainerXXX<double> _xPedRefRMS;

		//	1D actual Means/RMSs
		Container1D		_cMeanTotal_Subdet;
		Container1D		_cRMSTotal_Subdet;
		Container1D		_cMean1LS_Subdet; // 1LS
		Container1D		_cRMS1LS_Subdet; // 1LS 

		//	2D actual values
		ContainerProf2D		_cMean1LS_depth; // 1LS
		ContainerProf2D		_cRMS1LS_depth; //  1lS
		ContainerProf2D		_cMean1LS_FEDVME; // 1ls
		ContainerProf2D		_cMean1LS_FEDuTCA; // 1ls
		ContainerProf2D		_cRMS1LS_FEDVME; // 1ls
		ContainerProf2D		_cRMS1LS_FEDuTCA; // 1ls
		
		ContainerProf2D		_cMeanTotal_depth;
		ContainerProf2D		_cRMSTotal_depth;
		ContainerProf2D		_cMeanTotal_FEDVME;
		ContainerProf2D		_cMeanTotal_FEDuTCA;
		ContainerProf2D		_cRMSTotal_FEDVME;
		ContainerProf2D		_cRMSTotal_FEDuTCA;
		
		//	Comparison with DB Conditions
		Container1D		_cMeanDBRef1LS_Subdet; // 1LS 
		Container1D		_cRMSDBRef1LS_Subdet; // 1LS
		Container1D		_cMeanDBRefTotal_Subdet;
		Container1D		_cRMSDBRefTotal_Subdet;
		ContainerProf2D		_cMeanDBRef1LS_depth;
		ContainerProf2D		_cRMSDBRef1LS_depth;
		ContainerProf2D		_cMeanDBRef1LS_FEDVME;
		ContainerProf2D		_cMeanDBRef1LS_FEDuTCA;
		ContainerProf2D		_cRMSDBRef1LS_FEDVME;
		ContainerProf2D		_cRMSDBRef1LS_FEDuTCA;
		
		ContainerProf2D		_cMeanDBRefTotal_depth;
		ContainerProf2D		_cRMSDBRefTotal_depth;
		ContainerProf2D		_cMeanDBRefTotal_FEDVME;
		ContainerProf2D		_cMeanDBRefTotal_FEDuTCA;
		ContainerProf2D		_cRMSDBRefTotal_FEDVME;
		ContainerProf2D		_cRMSDBRefTotal_FEDuTCA;

		//	vs LS
		Container1D _cMissingvsLS_Subdet;
		Container1D _cOccupancyvsLS_Subdet;
		Container1D _cNBadMeanvsLS_Subdet;
		Container1D _cNBadRMSvsLS_Subdet;

		//	map of missing channels
		Container2D	_cMissing1LS_depth;
		Container2D	_cMissing1LS_FEDVME;
		Container2D	_cMissing1LS_FEDuTCA;
		Container2D _cMissingTotal_depth;
		Container2D _cMissingTotal_FEDVME;
		Container2D _cMissingTotal_FEDuTCA;

		//	Mean/RMS Bad Maps
		Container2D	_cMeanBad1LS_depth;
		Container2D _cRMSBad1LS_depth;
		Container2D	_cMeanBad1LS_FEDVME;
		Container2D	_cRMSBad1LS_FEDuTCA;
		Container2D	_cRMSBad1LS_FEDVME;
		Container2D	_cMeanBad1LS_FEDuTCA;

		Container2D	_cMeanBadTotal_depth;
		Container2D _cRMSBadTotal_depth;
		Container2D	_cMeanBadTotal_FEDVME;
		Container2D	_cRMSBadTotal_FEDuTCA;
		Container2D	_cRMSBadTotal_FEDVME;
		Container2D	_cMeanBadTotal_FEDuTCA;
		
		//	Summaries
		Container2D _cSummaryvsLS_FED;
		ContainerSingle2D _cSummaryvsLS;
};

#endif







