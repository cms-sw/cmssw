#ifndef RecHitTask_h
#define RecHitTask_h

/**
 *	module:			RecHitTask.h
 *	Author:			VK
 *	Description:	
 *		HCAL RECO Data Tier Evaluation
 *
 *	Online:
 *	Offline:
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

class RecHitTask : public hcaldqm::DQTask
{
	public:
		RecHitTask(edm::ParameterSet const&);
		virtual ~RecHitTask() {}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

	protected:
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(hcaldqm::UpdateFreq);

		edm::InputTag		_tagHBHE;
		edm::InputTag		_tagHO;
		edm::InputTag		_tagHF;
		edm::EDGetTokenT<HBHERecHitCollection> _tokHBHE;
		edm::EDGetTokenT<HORecHitCollection>	 _tokHO;
		edm::EDGetTokenT<HFRecHitCollection>	_tokHF;

		double _cutE_HBHE, _cutE_HO, _cutE_HF;
		double _thresh_unihf;

		//	hashes/FED vectors
		std::vector<uint32_t> _vhashFEDs;

		//	flag vectors
		std::vector<hcaldqm::flag::Flag> _vflags;
		enum RecoFlag
		{
			fUni=0,
			fTCDS=1,
			fUnknownIds = 2,
			nRecoFlag=3
		};

		//	emap
		HcalElectronicsMap const* _emap;
		hcaldqm::electronicsmap::ElectronicsMap _ehashmap;

		//	Filters
		hcaldqm::filter::HashFilter _filter_VME;
		hcaldqm::filter::HashFilter _filter_uTCA;
		hcaldqm::filter::HashFilter _filter_FEDsVME;
		hcaldqm::filter::HashFilter _filter_FEDsuTCA;

		//	Energy. Just filling. No Summary Generation
		hcaldqm::Container1D _cEnergy_Subdet;
		hcaldqm::ContainerProf1D _cEnergyvsieta_Subdet;	//	online only!
		hcaldqm::ContainerProf1D _cEnergyvsiphi_SubdetPM;	// online only!
		hcaldqm::ContainerProf2D _cEnergy_depth;
		hcaldqm::ContainerProf1D _cEnergyvsLS_SubdetPM;	// online only!
		hcaldqm::ContainerProf1D _cEnergyvsBX_SubdetPM;	// online only

		//	Timing vs Energy. No Summary Generation
		hcaldqm::Container2D _cTimingvsEnergy_SubdetPM;

		//	Timing. HBHE Partition is used for TCDS shift monitoring
		hcaldqm::Container1D		_cTimingCut_SubdetPM;
		hcaldqm::Container1D		_cTimingCut_HBHEPartition;
		hcaldqm::ContainerProf2D _cTimingCut_FEDVME;
		hcaldqm::ContainerProf2D	_cTimingCut_FEDuTCA;
		hcaldqm::ContainerProf2D _cTimingCut_ElectronicsVME;
		hcaldqm::ContainerProf2D _cTimingCut_ElectronicsuTCA;
		hcaldqm::ContainerProf2D _cTimingCut_depth;
		hcaldqm::ContainerProf1D _cTimingCutvsLS_FED;
		hcaldqm::ContainerProf1D _cTimingCutvsieta_Subdet;	//	online only
		hcaldqm::ContainerProf1D _cTimingCutvsiphi_SubdetPM; //	online only
		hcaldqm::ContainerProf1D _cTimingCutvsBX_SubdetPM;	// online only

		//	Occupancy w/o a cut. Used for checking missing channels
		hcaldqm::Container2D _cOccupancy_depth;
		hcaldqm::Container2D _cOccupancy_FEDVME;
		hcaldqm::Container2D _cOccupancy_FEDuTCA;
		hcaldqm::Container2D _cOccupancy_ElectronicsVME;
		hcaldqm::Container2D _cOccupancy_ElectronicsuTCA;
		hcaldqm::ContainerProf1D _cOccupancyvsLS_Subdet;
		hcaldqm::Container1D _cOccupancyvsiphi_SubdetPM;	// online only
		hcaldqm::Container1D _cOccupancyvsieta_Subdet;	//	online only

		//	Occupancy w/ a Cut.
		hcaldqm::Container2D _cOccupancyCut_FEDVME;
		hcaldqm::Container2D _cOccupancyCut_FEDuTCA;
		hcaldqm::Container2D _cOccupancyCut_ElectronicsVME;
		hcaldqm::Container2D _cOccupancyCut_ElectronicsuTCA;
		hcaldqm::ContainerProf1D _cOccupancyCutvsLS_Subdet; // online only
		hcaldqm::Container2D _cOccupancyCut_depth;
		hcaldqm::Container1D _cOccupancyCutvsiphi_SubdetPM;	// online only
		hcaldqm::Container1D _cOccupancyCutvsieta_Subdet;	// online only
		hcaldqm::ContainerProf1D _cOccupancyCutvsBX_Subdet;	// online only!
		hcaldqm::Container2D _cOccupancyCutvsiphivsLS_SubdetPM; // online only
		hcaldqm::ContainerXXX<uint32_t> _xUniHF, _xUni;

		//	tracks the unknown ids
		MonitorElement *meUnknownIds1LS;
		bool _unknownIdsPresent;

		std::vector<HcalGenericDetId> _gids; // online only
		hcaldqm::Container2D _cSummaryvsLS_FED; // online only!
		hcaldqm::ContainerSingle2D _cSummaryvsLS;
};

#endif
