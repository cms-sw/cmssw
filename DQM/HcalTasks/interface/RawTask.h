#ifndef RawTask_h
#define RawTask_h

/**
 *	file:
 *	Author:
 *	Description:
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Utilities.h"
#include "DQM/HcalCommon/interface/HashFilter.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"
#include "DQM/HcalCommon/interface/Flag.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class RawTask : public DQTask
{
	public:
		RawTask(edm::ParameterSet const&);
		virtual ~RawTask() {}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		edm::InputTag		_tagFEDs;
		edm::InputTag		_tagReport;
		edm::EDGetTokenT<FEDRawDataCollection>	_tokFEDs;
		edm::EDGetTokenT<HcalUnpackerReport> _tokReport;

		//	flag vector
		std::vector<flag::Flag> _vflags;
		enum RawFlag
		{
			fEvnMsm = 0,
			fBcnMsm = 1,
			fBadQ = 2,
			nRawFlag = 3
		};

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;

		//	vector of HcalElectronicsId for FEDs
		std::vector<uint32_t> _vhashFEDs;

		//	Filters
		HashFilter _filter_VME;
		HashFilter _filter_uTCA;
		HashFilter _filter_FEDsVME;
		HashFilter _filter_FEDsuTCA;

		//	Bad Quality
		Container2D _cBadQuality_FEDVME;
		Container2D _cBadQuality_FEDuTCA;
		Container2D _cBadQuality_depth;
		Container2D _cBadQualityLS_depth; // online only
		ContainerSingleProf1D _cBadQualityvsLS;
		ContainerSingleProf1D _cBadQualityvsBX;

		//	Mismatches
		Container2D _cEvnMsm_ElectronicsVME;
		Container2D _cBcnMsm_ElectronicsVME;
		Container2D _cOrnMsm_ElectronicsVME;
		Container2D _cEvnMsm_ElectronicsuTCA;
		Container2D _cBcnMsm_ElectronicsuTCA;
		Container2D _cOrnMsm_ElectronicsuTCA;
		ContainerXXX<uint32_t> _xEvnMsmLS, _xBcnMsmLS, _xBadQLS;
	
		Container2D	_cSummaryvsLS_FED; // online only
		ContainerSingle2D	_cSummaryvsLS; // online only
};

#endif
