#ifndef RecHitTask_h
#define RecHitTask_h

/*
 *	file:		RecHitTask.h
 *	Author:		Viktor Khristenko
 *	Date:		13.10.2015
 */

#include "DQM/HcalCommon/interface/DQTask.h"
#include "DQM/HcalCommon/interface/Container1D.h"
#include "DQM/HcalCommon/interface/ContainerProf1D.h"
#include "DQM/HcalCommon/interface/ContainerProf2D.h"
#include "DQM/HcalCommon/interface/Container2D.h"
#include "DQM/HcalCommon/interface/ContainerSingle2D.h"

	using namespace hcaldqm;
	class RecHitTask : public DQTask
	{
		public:
			RecHitTask(edm::ParameterSet const& ps);
			virtual ~RecHitTask()
			{}

			virtual void bookHistograms(DQMStore::IBooker &,
				edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);

			enum RecHitFlag
			{
				fLowOcp = 0,
				fUniphi = 1,
				fTCDS = 2,

				nRecHitFlag = 3
			};

		protected:
			//	protected funcs
			virtual void _process(edm::Event const&, edm::EventSetup const&);
			virtual void _resetMonitors(UpdateFreq);

			//	tags and tokens
			edm::InputTag	_tagHBHE;
			edm::InputTag	_tagHO;
			edm::InputTag	_tagHF;
			edm::EDGetTokenT<HBHERecHitCollection> _tokHBHE;
			edm::EDGetTokenT<HORecHitCollection> _tokHO;
			edm::EDGetTokenT<HFRecHitCollection> _tokHF;

			//	counters
			int		_nRecHits[constants::SUBDET_NUM];
			int		_nRecHitsCut[constants::SUBDET_NUM];
//			bool	_nDups[constants::SUBDET_NUM][constants::IPHI_NUM][constants::IETA_NUM][constants::DEPTH_NUM];

			//	Flag Names
			std::vector<std::string>	_fNames;

			//	cuts
			double _cutE_HBHE, _cutE_HO, _cutE_HF;

			//	Energy
			Container1D		_cEnergy_SubDet;
			Container1D		_cEnergy_SubDet_ieta;
			Container1D		_cEnergy_SubDetPM_iphi;
			ContainerProf1D _cEnergyvsieta_SubDet;
			ContainerProf1D _cEnergyvsiphi_SubDet;
			ContainerProf2D	_cEnergy_depth;

			ContainerProf1D _cEnergyvsietaCut_SubDet;
			ContainerProf1D _cEnergyvsiphiCut_SubDet;
			ContainerProf2D	_cEnergyCut_depth;

			//	Timing
			Container1D		_cTimingCut_SubDet;
			Container1D		_cTimingCut_SubDetPM_iphi;
			ContainerProf1D _cTimingCutvsLS_SubDetPM_iphi;
			Container1D		_cTimingCut_SubDet_ieta;
			ContainerProf1D _cTimingvsietaCut_SubDet_iphi;
			ContainerProf1D	_cTimingvsiphiCut_SubDet_ieta;
			ContainerProf2D _cTimingCut_depth;
			Container1D		_cTimingCut_HBHEPrt;

			//	Occupancy
			Container2D		_cOccupancy_depth;
			ContainerProf1D _cOccupancyvsLS_SubDet;
			ContainerProf1D	_cOccupancyCutvsLS_SubDet;
			Container2D		_cOccupancyCut_depth;
			Container1D		_cOccupancyvsiphi_SubDetPM;
			Container1D		_cOccupancyCutvsiphi_SubDetPM;

			//	Energy vs Timing
			Container2D		_cTimingvsEnergyCut_SubDetPM_iphi;

			//	Summaries
			ContainerSingle2D		_cSummary;
			Container2D				_cSummaryvsLS_SubDet;
	};

#endif




