#include "DQM/HcalTasks/interface/RawRunSummary.h"

namespace hcaldqm
{
	RawRunSummary::RawRunSummary(std::string const& name, 
		std::string const& taskname, edm::ParameterSet const& ps) :
		DQClient(name, taskname, ps), _booked(false)
	{}

	/* virtual */ void RawRunSummary::beginRun(edm::Run const& r,
		edm::EventSetup const& es)
	{
		DQClient::beginRun(r,es);
		
		if (_ptype!=fOffline)
			return;

		//	INITIALIZE WHAT NEEDS TO BE INITIALIZE ONLY ONCE!
		_ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
		_vhashVME.push_back(HcalElectronicsId(constants::FIBERCH_MIN,
			constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
		_vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN,
			FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
		_filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
			_vhashVME);	// filter out VME 
		_filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
			_vhashuTCA); // filter out uTCA

		//	INTIALIZE CONTAINERS ACTING AS HOLDERS OF RUN INFORAMTION
		_cEvnMsm_ElectronicsVME.initialize(_name, "EvnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsVME),
			new quantity::ElectronicsQuantity(quantity::fSpigot),
			new quantity::ValueQuantity(quantity::fN));
		_cBcnMsm_ElectronicsVME.initialize(_name, "BcnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsVME),
			new quantity::ElectronicsQuantity(quantity::fSpigot),
			new quantity::ValueQuantity(quantity::fN));
		_cEvnMsm_ElectronicsuTCA.initialize(_name, "EvnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsuTCA),
			new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
			new quantity::ValueQuantity(quantity::fN));
		_cBcnMsm_ElectronicsuTCA.initialize(_name, "BcnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsuTCA),
			new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
			new quantity::ValueQuantity(quantity::fN));
		_cBadQuality_depth.initialize(_name, "BadQuality",
			 hashfunctions::fdepth,
			 new quantity::DetectorQuantity(quantity::fieta),
			 new quantity::DetectorQuantity(quantity::fiphi),
			 new quantity::ValueQuantity(quantity::fN));

		_xEvn.initialize(hashfunctions::fFED);
		_xBcn.initialize(hashfunctions::fFED);
		_xBadQ.initialize(hashfunctions::fFED);
		//	BOOK CONTAINERSXXX
		_xEvn.book(_emap); _xBcn.book(_emap); _xBadQ.book(_emap);
	}

	/*
	 *	END OF LUMINOSITY HARVESTING 
	 *	RAW FORMAT HAS ONLY LUMI BASED FLAGS!
	 *	THEREFORE STEPS ARE:
	 *	1) LOAD CONTAINERS YOU NEED (MUST BE LUMI BASED)
	 *	2) ANALYZE 
	 *	3) GENERATE SUMMARY FLAGS AND PUSH THEM
	 */
	/* virtual */ void RawRunSummary::endLuminosityBlock(DQMStore::IBooker& ib,
		DQMStore::IGetter& ig, edm::LuminosityBlock const& lb,
		edm::EventSetup const& es)
	{
		DQClient::endLuminosityBlock(ib, ig, lb, es);
		
		if (_ptype!=fOffline)
			return;

		//	INITIALIZE WHAT YOU NEED
		LSSummary lssum; // summary for this LS
		lssum._LS = _currentLS; // set the LS

		//	RESET CONTAINERS USED FOR ANALYSIS OF THIS LS
		_xEvn.reset(); _xBcn.reset(); _xBadQ.reset();
		
		//	INITIALIZE LUMI BASED HISTOGRAMS
		Container2D cEvnMsm_ElectronicsVME,cEvnMsm_ElectronicsuTCA;
		Container2D cBcnMsm_ElectronicsVME,cBcnMsm_ElectronicsuTCA;
		Container2D cBadQuality_depth;
		cEvnMsm_ElectronicsVME.initialize(_taskname, "EvnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsVME),
			new quantity::ElectronicsQuantity(quantity::fSpigot),
			new quantity::ValueQuantity(quantity::fN));
		cBcnMsm_ElectronicsVME.initialize(_taskname, "BcnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsVME),
			new quantity::ElectronicsQuantity(quantity::fSpigot),
			new quantity::ValueQuantity(quantity::fN));
		cEvnMsm_ElectronicsuTCA.initialize(_taskname, "EvnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsuTCA),
			new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
			new quantity::ValueQuantity(quantity::fN));
		cBcnMsm_ElectronicsuTCA.initialize(_taskname, "BcnMsm",
			hashfunctions::fElectronics,
			new quantity::FEDQuantity(_vFEDsuTCA),
			new quantity::ElectronicsQuantity(quantity::fSlotuTCA),
			new quantity::ValueQuantity(quantity::fN));
		cBadQuality_depth.initialize(_taskname, "BadQuality",
			 hashfunctions::fdepth,
			 new quantity::DetectorQuantity(quantity::fieta),
			 new quantity::DetectorQuantity(quantity::fiphi),
			 new quantity::ValueQuantity(quantity::fN));

		//	LOAD LUMI BASED HISTOGRAMS
		cEvnMsm_ElectronicsVME.load(ig, _emap, _filter_uTCA, _subsystem);
		cBcnMsm_ElectronicsVME.load(ig, _emap, _filter_uTCA, _subsystem);
		cEvnMsm_ElectronicsuTCA.load(ig, _emap, _filter_VME, _subsystem);
		cBcnMsm_ElectronicsuTCA.load(ig, _emap, _filter_VME, _subsystem);
		cBadQuality_depth.load(ig, _emap, _subsystem);

		//	BOOK for the very first time
		if (!_booked)
		{
			_cEvnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
			_cBcnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
			_cEvnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
			_cBcnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
			_cBadQuality_depth.book(ib, _emap, _subsystem);
			_booked=true;
		}

		// ANALYZE THIS LS
		// iterate over all channels	
		std::vector<HcalGenericDetId> gids = _emap->allPrecisionId();
		for (std::vector<HcalGenericDetId>::const_iterator it=gids.begin();
			it!=gids.end(); ++it)
		{
			if (!it->isHcalDetId())
				continue;
			HcalDetId did = HcalDetId(it->rawId());
			HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));

			_xBadQ.get(eid)+=cBadQuality_depth.getBinContent(did);
			_cBadQuality_depth.fill(did, cBadQuality_depth.getBinContent(did));
			if (eid.isVMEid())
			{
				_xEvn.get(eid)+=cEvnMsm_ElectronicsVME.getBinContent(eid);
				_xBcn.get(eid)+=cBcnMsm_ElectronicsVME.getBinContent(eid);

				_cEvnMsm_ElectronicsVME.fill(eid, 
					cEvnMsm_ElectronicsVME.getBinContent(eid));
				_cBcnMsm_ElectronicsVME.fill(eid, 
					cBcnMsm_ElectronicsVME.getBinContent(eid));
			}
			else
			{
				_xEvn.get(eid)+=cEvnMsm_ElectronicsuTCA.getBinContent(eid);
				_xBcn.get(eid)+=cBcnMsm_ElectronicsuTCA.getBinContent(eid);

				_cEvnMsm_ElectronicsuTCA.fill(eid, 
					cEvnMsm_ElectronicsuTCA.getBinContent(eid));
				_cBcnMsm_ElectronicsuTCA.fill(eid, 
					cBcnMsm_ElectronicsuTCA.getBinContent(eid));
			}
		}
		

		//	GENERATE THE SUMMARY FOR THIS LS AND STORE IT
		std::vector<flag::Flag> vtmpflags; // tmp summary flags vector
		vtmpflags.resize(nRawFlag);
		vtmpflags[fEvnMsm]=flag::Flag("EvnMsm");
		vtmpflags[fBcnMsm]=flag::Flag("BcnMsm");
		vtmpflags[fBadQ]=flag::Flag("BadQ");
		for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
			it!=_vhashFEDs.end(); ++it)
		{
			HcalElectronicsId eid(*it);
			
			//	reset all the tmp flags to fNA
			//	MUST DO IT NOW! AS NCDAQ MIGHT OVERWRITE IT!
			for (std::vector<flag::Flag>::iterator ft=vtmpflags.begin();
				ft!=vtmpflags.end(); ++ft)
				ft->reset();
			
			//	check if this FED was @cDAQ
			std::vector<uint32_t>::const_iterator cit=std::find(
				_vcdaqEids.begin(), _vcdaqEids.end(), *it);
			if (cit==_vcdaqEids.end())
			{
				//	was not @cDAQ, set all the flags for this FED as fNCDAQ
				for (std::vector<flag::Flag>::iterator ft=vtmpflags.begin();
					ft!=vtmpflags.end(); ++ft)
					ft->_state = flag::fNCDAQ;

				// push all the flags for this FED
				// IMPORTANT!!!
				lssum._vflags.push_back(vtmpflags);
				continue;
			}

			//	here only if was registered at cDAQ
			if (utilities::isFEDHBHE(eid) || utilities::isFEDHF(eid) ||
				utilities::isFEDHO(eid))
			{
				if (_xEvn.get(eid)>0)
					vtmpflags[fEvnMsm]._state = flag::fBAD;
				else
					vtmpflags[fEvnMsm]._state = flag::fGOOD;
				if (_xBcn.get(eid)>0)
					vtmpflags[fBcnMsm]._state = flag::fBAD;
				else
					vtmpflags[fBcnMsm]._state = flag::fGOOD;
				if (_xBadQ.get(eid)>0)
					vtmpflags[fBadQ]._state = flag::fBAD;
				else
					vtmpflags[fBadQ]._state = flag::fGOOD;
			}

			// push all the flags for this FED
			lssum._vflags.push_back(vtmpflags);
		}

		//	push all flags for all FEDs for this LS
		_vflagsLS.push_back(lssum);
	}

	/*
	 * END JOB
	 * BOOK THE SUMMARY CONTAINERS, SET THE FLAGS
	 * RETURN THE LIST OF FLAGS FOR THIS DATATIER
	 */
	/* virtual */ std::vector<flag::Flag> RawRunSummary::endJob(
		DQMStore::IBooker& ib, DQMStore::IGetter& ig)
	{

		if (_ptype!=fOffline)
			return std::vector<flag::Flag>();


		//	PREPARE LS BASED FLAGS to use it for booking
		std::vector<flag::Flag> vflagsLS;
		vflagsLS.resize(nRawFlag);
		vflagsLS[fEvnMsm]=flag::Flag("EvnMsm");
		vflagsLS[fBcnMsm]=flag::Flag("BcnMsm");
		vflagsLS[fBadQ]=flag::Flag("BadQ");


		//	INITIALIZE AND BOOK SUMMARY CONTAINERS
		ContainerSingle2D cSummaryvsLS; // summary per FED: flag vs LS
		Container2D cSummaryvsLS_FED; // LS based flags vs LS for each FED
		cSummaryvsLS.initialize(_name, "SummaryvsLS",
			new quantity::LumiSection(_maxProcessedLS),
			new quantity::FEDQuantity(_vFEDs),
			new quantity::ValueQuantity(quantity::fState));
		cSummaryvsLS_FED.initialize(_name, "SummaryvsLS",
			hashfunctions::fFED,
			new quantity::LumiSection(_maxProcessedLS),
			new quantity::FlagQuantity(vflagsLS),
			new quantity::ValueQuantity(quantity::fState));
		cSummaryvsLS_FED.book(ib, _emap, _subsystem);
		cSummaryvsLS.book(ib, _subsystem);

		/*
		 *	Iterate over each FED
		 *		Iterate over each LS SUmmary
		 *			Iterate over all flags
		 *				set...
		 */

		std::vector<flag::Flag> sumflags; // flag per FED
		int ifed=0;
		for (std::vector<uint32_t>::const_iterator it=_vhashFEDs.begin();
			it!=_vhashFEDs.end(); ++it)
		{
			flag::Flag fSumRun("RAW"); // summary flag for this FED
			HcalElectronicsId eid(*it);

			//	ITERATE OVER EACH LS
			for (std::vector<LSSummary>::const_iterator itls=_vflagsLS.begin();
				itls!=_vflagsLS.end(); ++itls)
			{
				//	fill histograms per LS
				int iflag=0;
				flag::Flag fSumLS("RAW");
				for (std::vector<flag::Flag>::const_iterator ft=
					itls->_vflags[ifed].begin(); ft!=itls->_vflags[ifed].end();
					++ft)
				{
					//	Flag vs LS per FEd
					cSummaryvsLS_FED.setBinContent(eid, itls->_LS, int(iflag),
						ft->_state);
					fSumLS+=(*ft);
					iflag++;
				}
				//	FED vs LS
				cSummaryvsLS.setBinContent(eid, itls->_LS, fSumLS._state);
				fSumRun+=fSumLS;
			}
			
			//	push the summary flag for this FED for the whole RUN
			sumflags.push_back(fSumRun);
			
			//	increment the fed counter
			ifed++;
		}
	
		return sumflags;
	}
}
