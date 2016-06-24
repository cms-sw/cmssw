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

class TPTask : public hcaldqm::DQTask
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
		virtual void _resetMonitors(hcaldqm::UpdateFreq);

		edm::InputTag		_tagData;
		edm::InputTag		_tagEmul;
		edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokData;
		edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokEmul;

		//	flag vector
		std::vector<hcaldqm::flag::Flag> _vflags;
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
		hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

		//	Filters
		hcaldqm::filter::HashFilter _filter_VME;
		hcaldqm::filter::HashFilter _filter_uTCA;
		hcaldqm::filter::HashFilter _filter_depth0;

		//	Et/FG
		hcaldqm::Container1D _cEtData_TTSubdet;
		hcaldqm::Container1D _cEtEmul_TTSubdet;
		hcaldqm::Container2D	_cEtCorr_TTSubdet;
		hcaldqm::Container2D _cEtCorr2x3_TTSubdet;	//	online only
		hcaldqm::Container2D _cFGCorr_TTSubdet;
		hcaldqm::ContainerProf1D _cEtCutDatavsLS_TTSubdet;	// online only!
		hcaldqm::ContainerProf1D _cEtCutEmulvsLS_TTSubdet;	// online only!
		hcaldqm::ContainerProf1D _cEtCutDatavsBX_TTSubdet;	// online only!
		hcaldqm::ContainerProf1D _cEtCutEmulvsBX_TTSubdet;	// online only!
		hcaldqm::ContainerProf2D _cEtData_ElectronicsVME;
		hcaldqm::ContainerProf2D _cEtData_ElectronicsuTCA;
		hcaldqm::ContainerProf2D _cEtEmul_ElectronicsVME;
		hcaldqm::ContainerProf2D _cEtEmul_ElectronicsuTCA;

		//	depth like
		hcaldqm::ContainerSingleProf2D _cEtData_depthlike;
		hcaldqm::ContainerSingleProf2D _cEtEmul_depthlike;
		hcaldqm::ContainerSingleProf2D _cEtCutData_depthlike;
		hcaldqm::ContainerSingleProf2D _cEtCutEmul_depthlike;

		//	Et Correlation Ratio
		hcaldqm::ContainerProf2D _cEtCorrRatio_ElectronicsVME;
		hcaldqm::ContainerProf2D _cEtCorrRatio_ElectronicsuTCA;
		hcaldqm::ContainerSingleProf2D _cEtCorrRatio_depthlike;
		hcaldqm::ContainerProf1D _cEtCorrRatiovsLS_TTSubdet;	// online only!
		hcaldqm::ContainerProf1D _cEtCorrRatiovsBX_TTSubdet; // online only!

		//	Occupancies
		hcaldqm::Container2D _cOccupancyData_ElectronicsVME;
		hcaldqm::Container2D _cOccupancyData_ElectronicsuTCA;
		hcaldqm::Container2D _cOccupancyEmul_ElectronicsVME;
		hcaldqm::Container2D _cOccupancyEmul_ElectronicsuTCA;
		hcaldqm::Container2D _cOccupancyCutData_ElectronicsVME;
		hcaldqm::Container2D _cOccupancyCutData_ElectronicsuTCA;
		hcaldqm::Container2D _cOccupancyCutEmul_ElectronicsVME;
		hcaldqm::Container2D _cOccupancyCutEmul_ElectronicsuTCA;

		//	depth like
		hcaldqm::ContainerSingle2D _cOccupancyData_depthlike;
		hcaldqm::ContainerSingle2D _cOccupancyEmul_depthlike;
		hcaldqm::ContainerSingle2D _cOccupancyCutData_depthlike;
		hcaldqm::ContainerSingle2D _cOccupancyCutEmul_depthlike;

		//	2x3 occupancies just in case
		hcaldqm::ContainerSingle2D _cOccupancyData2x3_depthlike;	// online only!
		hcaldqm::ContainerSingle2D _cOccupancyEmul2x3_depthlike; // online only!

		//	Mismatches: Et and FG
		hcaldqm::Container2D _cEtMsm_ElectronicsVME;
		hcaldqm::Container2D _cEtMsm_ElectronicsuTCA;
		hcaldqm::Container2D _cFGMsm_ElectronicsVME;
		hcaldqm::Container2D _cFGMsm_ElectronicsuTCA;
		hcaldqm::ContainerSingle2D _cEtMsm_depthlike;
		hcaldqm::ContainerSingle2D _cFGMsm_depthlike;
		hcaldqm::ContainerProf1D _cEtMsmvsLS_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cEtMsmRatiovsLS_TTSubdet; // online only
		hcaldqm::ContainerProf1D _cEtMsmvsBX_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cEtMsmRatiovsBX_TTSubdet; // online only

		//	Missing Data w.r.t. Emulator
		hcaldqm::Container2D _cMsnData_ElectronicsVME;
		hcaldqm::Container2D _cMsnData_ElectronicsuTCA;
		hcaldqm::ContainerSingle2D _cMsnData_depthlike;
		hcaldqm::ContainerProf1D _cMsnDatavsLS_TTSubdet;	//	online only
		hcaldqm::ContainerProf1D _cMsnCutDatavsLS_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cMsnDatavsBX_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cMsnCutDatavsBX_TTSubdet; // online only

		//	Missing Emulator w.r.t. Data
		hcaldqm::Container2D _cMsnEmul_ElectronicsVME;
		hcaldqm::Container2D _cMsnEmul_ElectronicsuTCA;
		hcaldqm::ContainerSingle2D _cMsnEmul_depthlike;
		hcaldqm::ContainerProf1D _cMsnEmulvsLS_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cMsnCutEmulvsLS_TTSubdet; //	online only
		hcaldqm::ContainerProf1D _cMsnEmulvsBX_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cMsnCutEmulvsBX_TTSubdet; // online only

		//	Occupancy vs BX and LS
		hcaldqm::ContainerProf1D _cOccupancyDatavsBX_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cOccupancyEmulvsBX_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cOccupancyCutDatavsBX_TTSubdet; // online only
		hcaldqm::ContainerProf1D _cOccupancyCutEmulvsBX_TTSubdet; // online only
		hcaldqm::ContainerProf1D _cOccupancyDatavsLS_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cOccupancyEmulvsLS_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cOccupancyCutDatavsLS_TTSubdet;	// online only
		hcaldqm::ContainerProf1D _cOccupancyCutEmulvsLS_TTSubdet; // online only
		hcaldqm::Container2D _cSummaryvsLS_FED; // online only
		hcaldqm::ContainerSingle2D _cSummaryvsLS; // online only
		hcaldqm::ContainerXXX<uint32_t> _xEtMsm, _xFGMsm, _xNumCorr,
			_xDataMsn, _xDataTotal, _xEmulMsn, _xEmulTotal;
};

#endif
