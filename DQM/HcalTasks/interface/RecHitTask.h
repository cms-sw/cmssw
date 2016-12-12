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

using namespace hcaldqm;
using namespace hcaldqm::filter;
class RecHitTask : public DQTask
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
		virtual void _resetMonitors(UpdateFreq);

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
		std::vector<flag::Flag> _vflags;
		enum RecoFlag
		{
			fUni=0,
			fTCDS=1,
			nRecoFlag=2
		};

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;

		//	Filters
		HashFilter _filter_VME;
		HashFilter _filter_uTCA;
		HashFilter _filter_FEDsVME;
		HashFilter _filter_FEDsuTCA;

		//	Energy. Just filling. No Summary Generation
		Container1D _cEnergy_Subdet;
		ContainerProf1D _cEnergyvsieta_Subdet;	//	online only!
		ContainerProf1D _cEnergyvsiphi_SubdetPM;	// online only!
		ContainerProf2D _cEnergy_depth;
		ContainerProf1D _cEnergyvsLS_SubdetPM;	// online only!
		ContainerProf1D _cEnergyvsBX_SubdetPM;	// online only

		//	Timing vs Energy. No Summary Generation
		Container2D _cTimingvsEnergy_SubdetPM;

		//	Timing. HBHE Partition is used for TCDS shift monitoring
		Container1D		_cTimingCut_SubdetPM;
		Container1D		_cTimingCut_HBHEPartition;
		ContainerProf2D _cTimingCut_FEDVME;
		ContainerProf2D	_cTimingCut_FEDuTCA;
		ContainerProf2D _cTimingCut_ElectronicsVME;
		ContainerProf2D _cTimingCut_ElectronicsuTCA;
		ContainerProf2D _cTimingCut_depth;
		ContainerProf1D _cTimingCutvsLS_FED;
		ContainerProf1D _cTimingCutvsieta_Subdet;	//	online only
		ContainerProf1D _cTimingCutvsiphi_SubdetPM; //	online only
		ContainerProf1D _cTimingCutvsBX_SubdetPM;	// online only

		//	Occupancy w/o a cut. Used for checking missing channels
		Container2D _cOccupancy_depth;
		Container2D _cOccupancy_FEDVME;
		Container2D _cOccupancy_FEDuTCA;
		Container2D _cOccupancy_ElectronicsVME;
		Container2D _cOccupancy_ElectronicsuTCA;
		ContainerProf1D _cOccupancyvsLS_Subdet;
		Container1D _cOccupancyvsiphi_SubdetPM;	// online only
		Container1D _cOccupancyvsieta_Subdet;	//	online only

		//	Occupancy w/ a Cut.
		Container2D _cOccupancyCut_FEDVME;
		Container2D _cOccupancyCut_FEDuTCA;
		Container2D _cOccupancyCut_ElectronicsVME;
		Container2D _cOccupancyCut_ElectronicsuTCA;
		ContainerProf1D _cOccupancyCutvsLS_Subdet; // online only
		Container2D _cOccupancyCut_depth;
		Container1D _cOccupancyCutvsiphi_SubdetPM;	// online only
		Container1D _cOccupancyCutvsieta_Subdet;	// online only
		ContainerProf1D _cOccupancyCutvsBX_SubdetPM;	// online only!
		Container2D _cOccupancyCutvsiphivsLS_SubdetPM; // online only
		ContainerXXX<uint32_t> _xUniHF, _xUni;

		std::vector<HcalGenericDetId> _gids; // online only
		Container2D _cSummaryvsLS_FED; // online only!
		ContainerSingle2D _cSummaryvsLS;
};

#endif
