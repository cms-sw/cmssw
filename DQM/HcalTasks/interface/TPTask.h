#ifndef TPTask_h
#define TPTask_h

/*
 *	file:		TPTask.h
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
	class TPTask : public DQTask
	{
		public:
			TPTask(edm::ParameterSet const& ps);
			virtual ~TPTask()
			{}

			virtual void bookHistograms(DQMStore::IBooker &,
				edm::Run const&, edm::EventSetup const&);
			virtual void endLuminosityBlock(edm::LuminosityBlock const&,
				edm::EventSetup const&);

			enum TPFlag
			{
				fOccUniphi_Data = 0,
				fOccUniphi_Emul = 1,
				fLowOcp_Emul = 2,
				fCorrRatio = 3,
				fCorrUniphi = 4,
				fMsmEtUniphi = 5,
				fMsmEtNum = 6,
				fMsnUniphi_Data = 7,

				nTPFlag = 9
			};

		protected:
			//	protected funcs
			virtual void _process(edm::Event const&, edm::EventSetup const&);
			virtual void _resetMonitors(UpdateFreq);

			//	tags and tokens
			edm::InputTag	_tagData;
			edm::InputTag	_tagEmul;
			edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokData;
			edm::EDGetTokenT<HcalTrigPrimDigiCollection> _tokEmul;

			//	counters
			int _nMsmEt[constants::TPSUBDET_NUM];
			int _nMsmFG[constants::TPSUBDET_NUM];
			int _nTPs_Data[constants::TPSUBDET_NUM];
			int _nTPs_Emul[constants::TPSUBDET_NUM];

			// some tmp flags
			bool	_skip1x1;

			//	dqm flags
			std::vector<std::string> _fNames;

			//	Et
			Container1D		_cEtData_SubDet;
			Container1D		_cEtEmul_SubDet;	
			Container2D		_cEtCorr_TPSubDet;
			ContainerProf1D	_cEtCorrRatiovsLS_TPSubDet;
			Container1D		_cEtCorrRatio_TPSubDet;
			ContainerProf1D _cEtCorrRatiovsiphi_TPSubDetPM;
			ContainerSingle2D	_cEtMsm;
			ContainerProf1D	_cNumEtMsmvsLS_TPSubDet;
			Container1D		_cNumEtMsm_TPSubDet;
			Container1D		_cNumEtMsmvsiphi_TPSubDetPM;
			ContainerProf1D _cSumdEtvsLS_TPSubDet;
			Container1D		_cSumdEt_TPSubDet;

			Container1D		_cEtData_SubDetPM_iphi;
			Container1D		_cEtData_SubDet_ieta;

			//	FG
			Container2D		_cFGCorr_SubDet;
			ContainerSingle2D	_cFGMsm;

			//	Occupancy
			ContainerSingle2D		_cOccupancyData;
			ContainerSingle2D		_cOccupancyEmul;
			Container1D				_cOccupancyDatavsiphi_TPSubDetPM;
			Container1D				_cOccupancyEmulvsiphi_TPSubDetPM;
			ContainerProf1D			_cOccupancyDatavsLS_TPSubDet;
			ContainerProf1D			_cOccupancyEmulvsLS_TPSubDet;
			ContainerSingle2D		_cMsData;
			Container1D				_cMsDatavsiphi_TPSubDetPM;
			ContainerSingle2D		_cMsEmul;

			//	Special
			ContainerProf1D			_cDigiSizeDatavsLS_TPSubDet;
			ContainerProf1D			_cDigiSizeEmulvsLS_TPSubDet;

			//	Summaries
			ContainerSingle2D		_cSummary;
			Container2D				_cSummaryvsLS_TPSubDet;
	};

#endif




