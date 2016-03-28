#ifndef RecHitTask_h
#define RecHitTask_h

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
class RecHitTask : public DQTask
{
	public:
		RecHitTask(edm::ParameterSet const&);
		virtual ~RecHitTask() {}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

		enum RecoFlag
		{
			fOcpUniSlot = 0,
			fTimeUniSlot = 1,
			fTCDS = 2,
			fMsn1LS = 3,
			nRecoFlag = 4
		};

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

		//	hashes/FED vectors
		std::vector<uint32_t> _vhashFEDs;

		//	emap
		HcalElectronicsMap const* _emap;
		electronicsmap::ElectronicsMap _ehashmap;

		//	Filters
		HashFilter _filter_VME;
		HashFilter _filter_uTCA;
		HashFilter _filter_FEDsVME;
		HashFilter _filter_FEDsuTCA;

		//	Energy
		Container1D _cEnergy_Subdet;
		ContainerProf2D _cEnergy_depth;
		ContainerProf2D _cEnergy_FEDVME;
		ContainerProf2D _cEnergy_FEDuTCA;
		ContainerProf2D _cEnergy_ElectronicsVME;
		ContainerProf2D _cEnergy_ElectronicsuTCA;

		//	Timing vs Energy
		Container2D _cTimingvsEnergy_SubdetPM;

		//	Timing
		Container1D		_cTimingCut_SubdetPM;
		Container1D		_cTimingCut_HBHEPartition;
		ContainerProf2D _cTimingCut_FEDVME;
		ContainerProf2D	_cTimingCut_FEDuTCA;
		ContainerProf2D _cTimingCut_ElectronicsVME;
		ContainerProf2D _cTimingCut_ElectronicsuTCA;
		ContainerProf2D _cTimingCut_depth;
		ContainerProf1D _cTimingCutvsLS_FED;

		Container2D _cOccupancy_depth;
		Container2D _cOccupancy_FEDVME;
		Container2D _cOccupancy_FEDuTCA;
		Container2D _cOccupancy_ElectronicsVME;
		Container2D _cOccupancy_ElectronicsuTCA;
		ContainerProf1D _cOccupancyvsLS_Subdet;

		Container2D _cOccupancyCut_FEDVME;
		Container2D _cOccupancyCut_FEDuTCA;
		Container2D _cOccupancyCut_ElectronicsVME;
		Container2D _cOccupancyCut_ElectronicsuTCA;
		ContainerProf1D _cOccupancyCutvsLS_Subdet;
		Container2D _cOccupancyCut_depth;

		Container2D _cMissing1LS_FEDVME;
		Container2D _cMissing1LS_FEDuTCA;

		ContainerSingle2D _cSummary;
};

#endif
