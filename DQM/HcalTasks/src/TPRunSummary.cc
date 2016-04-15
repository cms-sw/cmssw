#include "DQM/HcalTasks/interface/TPRunSummary.h"

namespace hcaldqm
{
	TPRunSummary::TPRunSummary(std::string const& name, 
		std::string const& taskname, edm::ParameterSet const& ps) :
		DQClient(name, taskname, ps)
	{
		_thresh_fgmsm = ps.getUntrackedParameter<double>("thresh_fgmsm",
			0.1);
		_thresh_etmsm = ps.getUntrackedParameter<double>("thresh_etmsm",
			0.1);
	}

	/* virtual */ void TPRunSummary::beginRun(edm::Run const& r,
		edm::EventSetup const& es)
	{
		DQClient::beginRun(r,es);
	}

	/* virtual */ void TPRunSummary::endLuminosityBlock(DQMStore::IBooker& ib,
		DQMStore::IGetter& ig, edm::LuminosityBlock const& lb,
		edm::EventSetup const& es)
	{
		DQClient::endLuminosityBlock(ib, ig, lb, es);
	}

	/* virtual */ std::vector<flag::Flag> TPRunSummary::endJob(
		DQMStore::IBooker& ib, DQMStore::IGetter& ig)
	{
		//	hahs maps
		electronicsmap::ElectronicsMap ehashmap;
		ehashmap.initialize(_emap, electronicsmap::fT2EHashMap);
		std::vector<flag::Flag> vflags; vflags.resize(nTPFlag);
		vflags[fEtMsm]=flag::Flag("EtMsm");
		vflags[fFGMsm]=flag::Flag("FGMsm");

		//	INITIALIZE
		ContainerSingle2D cOccupancyData_depthlike, cOccupancyEmul_depthlike;
		ContainerSingle2D cEtMsm_depthlike, cFGMsm_depthlike,
			cEtCorrRatio_depthlike;
		ContainerSingle2D cSummary;
		ContainerXXX<double> xDeadD, xDeadE, xEtMsm, xFGMsm;
		xDeadD.initialize(hashfunctions::fFED);
		xDeadE.initialize(hashfunctions::fFED);
		xEtMsm.initialize(hashfunctions::fFED);
		xFGMsm.initialize(hashfunctions::fFED);
		cOccupancyData_depthlike.initialize(_taskname, "OccupancyData",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fN, true));
		cOccupancyEmul_depthlike.initialize(_taskname, "OccupancyEmul",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fN, true));
		cEtMsm_depthlike.initialize(_taskname, "EtMsm",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fN));
		cFGMsm_depthlike.initialize(_taskname, "FGMsm",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fN));
		cEtCorrRatio_depthlike.initialize(_taskname, "EtCorrRatio",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fRatio_0to2));
		_cEtMsmFraction_depthlike.initialize(_taskname, "EtMsmFraction",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fRatio_0to2));
		_cFGMsmFraction_depthlike.initialize(_taskname, "FGMsmFraction",
			new quantity::TrigTowerQuantity(quantity::fTTieta),
			new quantity::TrigTowerQuantity(quantity::fTTiphi),
			new quantity::ValueQuantity(quantity::fRatio_0to2));
		cSummary.initialize(_name, "Summary",
			new quantity::FEDQuantity(_vFEDs),
			new quantity::FlagQuantity(vflags),
			new quantity::ValueQuantity(quantity::fState));

		//	BOOK
		xDeadD.book(_emap); xDeadE.book(_emap); xEtMsm.book(_emap); 
		xFGMsm.book(_emap);

		//	LOAD
		cOccupancyData_depthlike.load(ig, _subsystem);
		cOccupancyEmul_depthlike.load(ig, _subsystem);
		cEtMsm_depthlike.load(ig, _subsystem);
		cFGMsm_depthlike.load(ig, _subsystem);
		cEtCorrRatio_depthlike.load(ig, _subsystem);
		_cEtMsmFraction_depthlike.book(ib, _subsystem);
		_cFGMsmFraction_depthlike.book(ib, _subsystem);
		cSummary.book(ib, _subsystem);

		//	iterate
		std::vector<HcalTrigTowerDetId> tids = _emap->allTriggerId();
		for (std::vector<HcalTrigTowerDetId>::const_iterator it=tids.begin();
			it!=tids.end(); ++it)
		{
			//	skip 2x3
			HcalTrigTowerDetId tid = HcalTrigTowerDetId(*it);
			if (tid.version()==0 && tid.ietaAbs()>=29)
				continue;
			HcalElectronicsId eid=HcalElectronicsId(ehashmap.lookup(*it));
			double etmsmfr = double(cEtMsm_depthlike.getBinContent(tid))/
				double(cEtCorrRatio_depthlike.getBinEntries(tid));	
			etmsmfr>=0.1?xEtMsm.get(eid)++:xEtMsm.get(eid)+=0;
			double fgmsmfr = double(cFGMsm_depthlike.getBinContent(tid))/
				double(cEtCorrRatio_depthlike.getBinEntries(tid));
			fgmsmfr>=0.1?xFGMsm.get(eid)++:xFGMsm.get(eid)+=0;

			_cEtMsmFraction_depthlike.setBinContent(tid, etmsmfr);
			_cFGMsmFraction_depthlike.setBinContent(tid, fgmsmfr);
		}

		std::vector<flag::Flag> sumflags;
		for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
			it!=_vhashFEDs.end(); ++it)
		{
			flag::Flag fSum("TP");
			HcalElectronicsId eid(*it);

			std::vector<uint32_t>::const_iterator cit=std::find(
				_vcdaqEids.begin(), _vcdaqEids.end(), *it);
			if (cit==_vcdaqEids.end())
			{
				//	not @cDAQ
				sumflags.push_back(flag::Flag("TP", flag::fNCDAQ));
				continue;
			}

			//	@cDAQ
			if (utilities::isFEDHBHE(eid) || utilities::isFEDHF(eid))
			{
				if (xEtMsm.get(eid)>0)
					vflags[fEtMsm]._state = flag::fBAD;
				else
					vflags[fEtMsm]._state = flag::fGOOD;
				if (xFGMsm.get(eid)>0)
					vflags[fFGMsm]._state = flag::fBAD;
				else
					vflags[fFGMsm]._state = flag::fGOOD;
			}

			//	combine
			int iflag=0;
			for (std::vector<flag::Flag>::iterator ft=vflags.begin();
				ft!=vflags.end(); ++ft)
			{
				cSummary.setBinContent(eid, iflag, ft->_state);
				fSum+=(*ft);
				iflag++;
				ft->reset();
			}
			sumflags.push_back(fSum);
		}

		return sumflags;
	}
}
