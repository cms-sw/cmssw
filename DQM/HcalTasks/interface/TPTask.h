#ifndef TPTask_h
#define TPTask_h

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
#include "DQM/HcalCommon/interface/ContainerSingleProf2D.h"
#include "DQM/HcalCommon/interface/ElectronicsMap.h"

using namespace hcaldqm;
using namespace hcaldqm::filter;
class TPTask : public DQTask
{
	public:
		TPTask(edm::ParameterSet const&);
		virtual ~TPTask() {}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);
		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		edm::InputTag		_tagData;
		edm::InputTag		_tagEmul;
		edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokData;
		edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokEmul;

		//	flag vector
		std::vector<flag::Flag> _vflags;
		enum TPFlag
		{
			fEtMsm=0,
			fFGMsm=1,
			fDataMsn=2,
			fEmulMsn=3,
			nTPFlag=4
		};

		//	switches/cuts/etc...
		bool _skip1x1;
		int _cutEt;
		double _thresh_EtMsmRate,_thresh_FGMsmRate,_thresh_DataMsn,
			_thresh_EmulMsn;

		//	hashes/FEDs vectors
		std::vector<uint32_t> _vhashFEDs;

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;

		//	Filters
		HashFilter _filter_VME;
		HashFilter _filter_uTCA;
		HashFilter _filter_depth0;

		//	Et/FG
		Container1D _cEtData_TTSubdet;
		Container1D _cEtEmul_TTSubdet;
		Container2D	_cEtCorr_TTSubdet;
		Container2D _cEtCorr2x3_TTSubdet;	//	online only
		Container2D _cFGCorr_TTSubdet;
		ContainerProf1D _cEtCutDatavsLS_TTSubdet;	// online only!
		ContainerProf1D _cEtCutEmulvsLS_TTSubdet;	// online only!
		ContainerProf1D _cEtCutDatavsBX_TTSubdet;	// online only!
		ContainerProf1D _cEtCutEmulvsBX_TTSubdet;	// online only!

		ContainerProf2D _cEtData_ElectronicsVME;
		ContainerProf2D _cEtData_ElectronicsuTCA;
		ContainerProf2D _cEtEmul_ElectronicsVME;
		ContainerProf2D _cEtEmul_ElectronicsuTCA;

		//	depth like
		ContainerSingleProf2D _cEtData_depthlike;
		ContainerSingleProf2D _cEtEmul_depthlike;
		ContainerSingleProf2D _cEtCutData_depthlike;
		ContainerSingleProf2D _cEtCutEmul_depthlike;

		//	Et Correlation Ratio
		ContainerProf2D _cEtCorrRatio_ElectronicsVME;
		ContainerProf2D _cEtCorrRatio_ElectronicsuTCA;
		ContainerSingleProf2D _cEtCorrRatio_depthlike;
		ContainerProf1D _cEtCorrRatiovsLS_TTSubdet;	// online only!
		ContainerProf1D _cEtCorrRatiovsBX_TTSubdet; // online only!

		//	Occupancies
		Container2D _cOccupancyData_ElectronicsVME;
		Container2D _cOccupancyData_ElectronicsuTCA;
		Container2D _cOccupancyEmul_ElectronicsVME;
		Container2D _cOccupancyEmul_ElectronicsuTCA;
		
		Container2D _cOccupancyCutData_ElectronicsVME;
		Container2D _cOccupancyCutData_ElectronicsuTCA;
		Container2D _cOccupancyCutEmul_ElectronicsVME;
		Container2D _cOccupancyCutEmul_ElectronicsuTCA;

		//	depth like
		ContainerSingle2D _cOccupancyData_depthlike;
		ContainerSingle2D _cOccupancyEmul_depthlike;
		ContainerSingle2D _cOccupancyCutData_depthlike;
		ContainerSingle2D _cOccupancyCutEmul_depthlike;

		//	2x3 occupancies just in case
		ContainerSingle2D _cOccupancyData2x3_depthlike;	// online only!
		ContainerSingle2D _cOccupancyEmul2x3_depthlike; // online only!

		//	Mismatches: Et and FG
		Container2D _cEtMsm_ElectronicsVME;
		Container2D _cEtMsm_ElectronicsuTCA;
		Container2D _cFGMsm_ElectronicsVME;
		Container2D _cFGMsm_ElectronicsuTCA;
		ContainerSingle2D _cEtMsm_depthlike;
		ContainerSingle2D _cFGMsm_depthlike;
		ContainerProf1D _cEtMsmvsLS_TTSubdet;	// online only
		ContainerProf1D _cEtMsmRatiovsLS_TTSubdet; // online only
		ContainerProf1D _cEtMsmvsBX_TTSubdet;	// online only
		ContainerProf1D _cEtMsmRatiovsBX_TTSubdet; // online only

		//	Missing Data w.r.t. Emulator
		Container2D _cMsnData_ElectronicsVME;
		Container2D _cMsnData_ElectronicsuTCA;
		ContainerSingle2D _cMsnData_depthlike;
		ContainerProf1D _cMsnDatavsLS_TTSubdet;	//	online only
		ContainerProf1D _cMsnCutDatavsLS_TTSubdet;	// online only
		ContainerProf1D _cMsnDatavsBX_TTSubdet;	// online only
		ContainerProf1D _cMsnCutDatavsBX_TTSubdet; // online only

		//	Missing Emulator w.r.t. Data
		Container2D _cMsnEmul_ElectronicsVME;
		Container2D _cMsnEmul_ElectronicsuTCA;
		ContainerSingle2D _cMsnEmul_depthlike;
		ContainerProf1D _cMsnEmulvsLS_TTSubdet;	// online only
		ContainerProf1D _cMsnCutEmulvsLS_TTSubdet; //	online only
		ContainerProf1D _cMsnEmulvsBX_TTSubdet;	// online only
		ContainerProf1D _cMsnCutEmulvsBX_TTSubdet; // online only

		//	Occupancy vs BX and LS
		ContainerProf1D _cOccupancyDatavsBX_TTSubdet;	// online only
		ContainerProf1D _cOccupancyEmulvsBX_TTSubdet;	// online only
		ContainerProf1D _cOccupancyCutDatavsBX_TTSubdet; // online only
		ContainerProf1D _cOccupancyCutEmulvsBX_TTSubdet; // online only
		ContainerProf1D _cOccupancyDatavsLS_TTSubdet;	// online only
		ContainerProf1D _cOccupancyEmulvsLS_TTSubdet;	// online only
		ContainerProf1D _cOccupancyCutDatavsLS_TTSubdet;	// online only
		ContainerProf1D _cOccupancyCutEmulvsLS_TTSubdet; // online only


		Container2D _cSummaryvsLS_FED; // online only
		ContainerSingle2D _cSummaryvsLS; // online only
		ContainerXXX<uint32_t> _xEtMsm, _xFGMsm, _xNumCorr,
			_xDataMsn, _xDataTotal, _xEmulMsn, _xEmulTotal;
};

#endif
