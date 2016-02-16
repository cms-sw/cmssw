#ifndef DigiTask_h
#define DigiTask_h

/*
 *	file:			DigiTask.h
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
#include "DQM/HcalCommon/interface/ContainerSingle1D.h"
#include "DQM/HcalCommon/interface/ContainerSingleProf1D.h"

using namespace hcaldqm;
class DigiTask : public DQTask
{
	public:
		DigiTask(edm::ParameterSet const&);
		virtual ~DigiTask()
		{}

		virtual void bookHistograms(DQMStore::IBooker&,
			edm::Run const&, edm::EventSetup const&);
		virtual void endLuminosityBlock(edm::LuminosityBlock const&,
			edm::EventSetup const&);

		enum DigiFlag
		{
			fLowOcp = 0,
			fDigiSize = 1,
			fUniphi = 2,
			fMsn1LS = 3,
			fCapIdRot = 4,

			nDigiFlag = 5
		};

	protected:
		//	funcs
		virtual void _process(edm::Event const&, edm::EventSetup const&);
		virtual void _resetMonitors(UpdateFreq);

		//	Tags and corresponding Tokens
		edm::InputTag	_tagHBHE;
		edm::InputTag	_tagHO;
		edm::InputTag	_tagHF;
		edm::EDGetTokenT<HBHEDigiCollection> _tokHBHE;
		edm::EDGetTokenT<HODigiCollection> _tokHO;
		edm::EDGetTokenT<HFDigiCollection> _tokHF;

		//	Flag Names
		std::vector<std::string> _fNames;

		//	Counters
		int				_numDigis[constants::SUBDET_NUM];
		int				_numDigisCut[constants::SUBDET_NUM];
		int				_nMsn[constants::SUBDET_NUM];
		int				_nCapIdRots[constants::SUBDET_NUM];
		bool			_occ_1LS[constants::SUBDET_NUM][constants::IPHI_NUM][constants::IETA_NUM][constants::DEPTH_NUM];
		bool			_error_1LS[constants::SUBDET_NUM][constants::IPHI_NUM][constants::IETA_NUM][constants::DEPTH_NUM];
		bool			_occ_10LS[constants::SUBDET_NUM][constants::IPHI_NUM][constants::IETA_NUM][constants::DEPTH_NUM];
		bool			_occ_Always[constants::SUBDET_NUM][constants::IPHI_NUM][constants::IETA_NUM][constants::DEPTH_NUM];

		//	Cuts
		double _cutSumQ_HBHE, _cutSumQ_HO, _cutSumQ_HF;

		// Containers by quantities

		//	Signal, ADC, fC, SumQ
		Container1D		_cfCperTS_SubDet;
		Container1D		_cADCperTS_SubDet;
		Container1D		_cSumQ_SubDetPM_iphi;
		ContainerProf2D	_cSumQ_depth;
		ContainerProf1D	_cSumQvsLS_SubDetPM_iphi;

		//	Shape
		Container1D		_cShape_SubDetPM_iphi;
		Container1D		_cShapeCut_SubDetPM_iphi;
		ContainerSingle1D		_cShapeCut_p3e41d2;
		ContainerSingle1D		_cShapeCut_p3em41d2;

		//	Timing
		Container1D		_cTimingCut_SubDetPM_iphi;
		ContainerProf1D	_cTimingCutvsieta_SubDet_iphi;
		ContainerProf1D _cTimingCutvsiphi_SubDet_ieta;
		ContainerProf1D _cTimingCutvsLS_SubDetPM_iphi;
		ContainerProf2D	_cTimingCut_depth;

		//	Specific
		ContainerProf1D _cQ2Q12CutvsLS_HFPM_iphi;
		ContainerSingleProf1D _cQ2Q12CutvsLS_p3e41d2;
		ContainerSingleProf1D _cQ2Q12CutvsLS_p3em41d2;
		ContainerProf1D	_cDigiSizevsLS_SubDet;
		Container2D		_cCapIdRots_depth;

		//	Occupancy
		Container1D		_cOccupancyvsiphi_SubDetPM;
		Container1D		_cOccupancyCutvsiphi_SubDetPM;
		ContainerProf1D _cOccupancyvsLS_SubDet;
		ContainerProf1D	_cOccupancyCutvsLS_SubDet;
		Container2D		_cOccupancy_depth;
		Container2D		_cOccupancyCut_depth;
//		ContainerProf2D	_cOccupancyCutiphivsLS_SubDet;
		Container2D		_cOccupancyOnce_depth;
		Container2D		_cMsn1LS_depth;
		Container2D		_cMsn10LS_depth;
		ContainerProf1D	_cMsn1LSvsLS_SubDet;

		//	Summaries
		ContainerSingle2D		_cSummary;
		Container2D		_cSummaryvsLS_SubDet;
};

#endif







