#include "DQM/HcalTasks/interface/DigiRunSummary.h"

namespace hcaldqm {
  using namespace constants;

  DigiRunSummary::DigiRunSummary(std::string const& name,
                                 std::string const& taskname,
                                 edm::ParameterSet const& ps,
                                 edm::ConsumesCollector& iC)
      : DQClient(name, taskname, ps, iC), _booked(false) {
    _thresh_unihf = ps.getUntrackedParameter<double>("thresh_unihf", 0.2);
    _thresh_pindiode = ps.getUntrackedParameter<double>("thresh_pindiode", 2000);

    std::vector<uint32_t> vrefDigiSize = ps.getUntrackedParameter<std::vector<uint32_t>>("refDigiSize");
    _refDigiSize[HcalBarrel] = vrefDigiSize[0];
    _refDigiSize[HcalEndcap] = vrefDigiSize[1];
    _refDigiSize[HcalOuter] = vrefDigiSize[2];
    _refDigiSize[HcalForward] = vrefDigiSize[3];
  }

  /* virtual */ void DigiRunSummary::beginRun(edm::Run const& r, edm::EventSetup const& es) {
    DQClient::beginRun(r, es);

    if (_ptype != fOffline)
      return;

    //  INITIALIZE WHAT NEEDS TO BE INITIALIZE ONLY ONCE!
    _ehashmap.initialize(_emap, electronicsmap::fD2EHashMap);
    _vhashVME.push_back(
        HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
    _vhashuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _filter_VME.initialize(filter::fFilter, hashfunctions::fElectronics,
                           _vhashVME);  // filter out VME
    _filter_uTCA.initialize(filter::fFilter, hashfunctions::fElectronics,
                            _vhashuTCA);  // filter out uTCA
    _vhashFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _vhashFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _vhashFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _vhashFEDHF.push_back(HcalElectronicsId(22, SLOT_uTCA_MIN + 6, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _vhashFEDHF.push_back(HcalElectronicsId(29, SLOT_uTCA_MIN + 6, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _vhashFEDHF.push_back(HcalElectronicsId(32, SLOT_uTCA_MIN + 6, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    _filter_FEDHF.initialize(filter::fPreserver, hashfunctions::fFED,
                             _vhashFEDHF);  // preserve only HF FEDs

    _xDead.initialize(hashfunctions::fFED);
    _xDigiSize.initialize(hashfunctions::fFED);
    _xUni.initialize(hashfunctions::fFED);
    _xUniHF.initialize(hashfunctions::fFEDSlot);

    _xDead.book(_emap);
    _xDigiSize.book(_emap);
    _xUniHF.book(_emap);
    _xUni.book(_emap, _filter_FEDHF);
    _xNChs.initialize(hashfunctions::fFED);
    _xNChsNominal.initialize(hashfunctions::fFED);
    _xNChs.book(_emap);
    _xNChsNominal.book(_emap);

    _cOccupancy_depth.initialize(_name,
                                 "Occupancy",
                                 hashfunctions::fdepth,
                                 new quantity::DetectorQuantity(quantity::fieta),
                                 new quantity::DetectorQuantity(quantity::fiphi),
                                 new quantity::ValueQuantity(quantity::fN),
                                 0);

    //	GET THE NOMINAL NUMBER OF CHANNELS PER FED
    std::vector<HcalGenericDetId> gids = _emap->allPrecisionId();
    for (std::vector<HcalGenericDetId>::const_iterator it = gids.begin(); it != gids.end(); ++it) {
      if (!it->isHcalDetId())
        continue;
      HcalDetId did(it->rawId());
      HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));
      _xNChsNominal.get(eid)++;
    }
  }

  /*
	 *	END LUMI. EVALUATE LUMI BASED FLAGS
	 */
  /* virtual */ void DigiRunSummary::endLuminosityBlock(DQMStore::IBooker& ib,
                                                        DQMStore::IGetter& ig,
                                                        edm::LuminosityBlock const& lb,
                                                        edm::EventSetup const& es) {
    DQClient::endLuminosityBlock(ib, ig, lb, es);

    if (_ptype != fOffline)
      return;

    LSSummary lssum;
    lssum._LS = _currentLS;

    _xDigiSize.reset();
    _xNChs.reset();

    //	INITIALIZE LUMI BASED HISTOGRAMS
    Container2D cDigiSize_FED, cOccupancy_depth;
    cDigiSize_FED.initialize(_taskname,
                             "DigiSize",
                             hashfunctions::fFED,
                             new quantity::ValueQuantity(quantity::fDigiSize),
                             new quantity::ValueQuantity(quantity::fN),
                             0);
    cOccupancy_depth.initialize(_taskname,
                                "Occupancy",
                                hashfunctions::fdepth,
                                new quantity::DetectorQuantity(quantity::fieta),
                                new quantity::DetectorQuantity(quantity::fiphi),
                                new quantity::ValueQuantity(quantity::fN),
                                0);

    //	LOAD LUMI BASED HISTOGRAMS
    cOccupancy_depth.load(ig, _emap, _subsystem);
    cDigiSize_FED.load(ig, _emap, _subsystem);
    MonitorElement* meNumEvents = ig.get(_subsystem + "/RunInfo/NumberOfEvents");
    int numEvents = meNumEvents->getBinContent(1);
    bool unknownIdsPresent = ig.get(_subsystem + "/" + _taskname + "/UnknownIds")->getBinContent(1) > 0;

    //	book the Numer of Events - set axis extendable
    if (!_booked) {
      ib.setCurrentFolder(_subsystem + "/" + _taskname);
      _meNumEvents = ib.book1DD("NumberOfEvents", "NumberOfEvents", 1000, 1, 1001);  // 1000 to start with
      _meNumEvents->getTH1()->SetCanExtend(TH1::kXaxis);

      _cOccupancy_depth.book(ib, _emap, _subsystem);
      _booked = true;
    }
    _meNumEvents->setBinContent(_currentLS, numEvents);

    //	ANALYZE THIS LS for LS BASED FLAGS
    std::vector<HcalGenericDetId> gids = _emap->allPrecisionId();
    for (std::vector<HcalGenericDetId>::const_iterator it = gids.begin(); it != gids.end(); ++it) {
      if (!it->isHcalDetId())
        continue;

      HcalDetId did = HcalDetId(it->rawId());
      HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));

      cOccupancy_depth.getBinContent(did) > 0 ? _xNChs.get(eid)++ : _xNChs.get(eid) += 0;
      _cOccupancy_depth.fill(did, cOccupancy_depth.getBinContent(did));
      //	digi size
      cDigiSize_FED.getMean(eid) != _refDigiSize[did.subdet()] ? _xDigiSize.get(eid)++ : _xDigiSize.get(eid) += 0;
      cDigiSize_FED.getRMS(eid) != 0 ? _xDigiSize.get(eid)++ : _xDigiSize.get(eid) += 0;
    }

    //	GENERATE SUMMARY AND STORE IT
    std::vector<flag::Flag> vtmpflags;
    vtmpflags.resize(nLSFlags);
    vtmpflags[fDigiSize] = flag::Flag("DigiSize");
    vtmpflags[fNChsHF] = flag::Flag("NChsHF");
    vtmpflags[fUnknownIds] = flag::Flag("UnknownIds");
    vtmpflags[fLED] = flag::Flag("LedMonCU");
    vtmpflags[fRADDAM] = flag::Flag("RaddamMon");
    vtmpflags[fLASER] = flag::Flag("LaserMonCU");
    vtmpflags[fPinDiode] = flag::Flag("LaserMon");

    // Push FED-based flags for this LS
    for (std::vector<uint32_t>::const_iterator it = _vhashFEDs.begin(); it != _vhashFEDs.end(); ++it) {
      HcalElectronicsId eid(*it);

      // ZDC: skip detailed monitoring entirely (crate2fed-based, no emap lookup)
      if (utilities::isFEDZDC(eid)) {
        for (std::vector<flag::Flag>::iterator ft = vtmpflags.begin(); ft != vtmpflags.end(); ++ft)
          ft->reset();
        lssum._vflags.push_back(vtmpflags);
        continue;
      }

      // reset flags to NA
      for (std::vector<flag::Flag>::iterator ft = vtmpflags.begin(); ft != vtmpflags.end(); ++ft)
        ft->reset();

      if (_xDigiSize.get(eid) > 0)
        vtmpflags[fDigiSize]._state = flag::fBAD;
      else
        vtmpflags[fDigiSize]._state = flag::fGOOD;

      // NChsHF is only relevant for HF FEDs
      if (utilities::isFEDHF(eid)) {
        if (_xNChs.get(eid) != _xNChsNominal.get(eid))
          vtmpflags[fNChsHF]._state = flag::fBAD;
        else
          vtmpflags[fNChsHF]._state = flag::fGOOD;
      } else {
        vtmpflags[fNChsHF]._state = flag::fNA;
      }
      if (unknownIdsPresent)
        vtmpflags[fUnknownIds]._state = flag::fBAD;
      else
        vtmpflags[fUnknownIds]._state = flag::fGOOD;

      // Determine subdetector category directly from FED number - no emap lookup.
      // crateListHF = {22,29,32}, crateListHO = {23,26,27,38}, VME crates are HO.
      // HBHE crates serve both HB and HE: check both subdet histograms and OR the results.
      const bool isHBHE = utilities::isFEDHBHE(eid);
      const bool isHF = utilities::isFEDHF(eid);
      const bool isHO = utilities::isFEDHO(eid) || eid.isVMEid();

      if (isHBHE || isHF || isHO) {
        int fed =
            eid.isVMEid() ? eid.dccid() + constants::FED_VME_MIN : utilities::crate2fed(eid.crateId(), eid.slot());
        std::string const fedName = "FED" + std::to_string(fed);
        std::string ledFEDPath = _subsystem + "/" + _taskname + "/CU_LED/CU_LED_CUCountvsLS/FED/" + fedName;
        std::string laserFEDPath = _subsystem + "/" + _taskname + "/CU_Laser/CU_LASER_CUCountvsLS/FED/" + fedName;

        // LED CU - per-FED
        MonitorElement* ledFED = ig.get(ledFEDPath);
        vtmpflags[fLED]._state =
            ledFED ? ((ledFED->getBinContent(_currentLS) > 0) ? flag::fBAD : flag::fGOOD) : flag::fNA;

        // Laser CU - per-FED
        MonitorElement* laserFED = ig.get(laserFEDPath);
        vtmpflags[fLASER]._state =
            laserFED ? ((laserFED->getBinContent(_currentLS) > 0) ? flag::fBAD : flag::fGOOD) : flag::fNA;

        // Pin Diode (LaserMon) - HBHE only
        if (isHBHE) {
          MonitorElement* pinDiodeHist = ig.get(_subsystem + "/" + _taskname + "/PinDiodeMon/sumQvsLS/sumQvsLS");
          vtmpflags[fPinDiode]._state =
              pinDiodeHist ? ((pinDiodeHist->getBinContent(_currentLS) > _thresh_pindiode) ? flag::fBAD : flag::fGOOD)
                           : flag::fNA;
        } else {
          vtmpflags[fPinDiode]._state = flag::fNA;
        }

        // Raddam CU - HF only
        if (isHF) {
          MonitorElement* raddamHist =
              ig.get(_subsystem + "/" + _taskname + "/CU_Raddam/CU_Raddam_CUCountvsLS/CU_Raddam_CUCountvsLS");
          vtmpflags[fRADDAM]._state =
              raddamHist ? ((raddamHist->getBinContent(_currentLS) > 0) ? flag::fBAD : flag::fGOOD) : flag::fNA;
        } else {
          vtmpflags[fRADDAM]._state = flag::fNA;
        }
      } else {
        vtmpflags[fLED]._state = flag::fNA;
        vtmpflags[fLASER]._state = flag::fNA;
        vtmpflags[fPinDiode]._state = flag::fNA;
        vtmpflags[fRADDAM]._state = flag::fNA;
      }

      // push all the flags for this FED
      lssum._vflags.push_back(vtmpflags);
    }

    //	push all the flags for all FEDs for this LS
    _vflagsLS.push_back(lssum);
    cDigiSize_FED.reset();
    cOccupancy_depth.reset();
  }

  /*
	 *	End Job
	 */
  /* virtual */ std::vector<flag::Flag> DigiRunSummary::endJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
    if (_ptype != fOffline)
      return std::vector<flag::Flag>();

    _xDead.reset();
    _xUniHF.reset();
    _xUni.reset();

    //	PREPARE LS-BASED FLAGS FOR BOOKING
    std::vector<flag::Flag> vflagsPerLS;
    vflagsPerLS.resize(nLSFlags);
    vflagsPerLS[fDigiSize] = flag::Flag("DigiSize");
    vflagsPerLS[fNChsHF] = flag::Flag("NChsHF");
    vflagsPerLS[fUnknownIds] = flag::Flag("UnknownIds");
    vflagsPerLS[fLED] = flag::Flag("LedMonCU");
    vflagsPerLS[fRADDAM] = flag::Flag("RaddamMon");
    vflagsPerLS[fLASER] = flag::Flag("LaserMonCU");
    vflagsPerLS[fPinDiode] = flag::Flag("LaserMon");

    //	INITIALIZE SUMMARY CONTAINERS (FED-based)
    ContainerSingle2D cSummaryvsLS_FEDSummary;
    Container2D cSummaryvsLS_FED;
    cSummaryvsLS_FEDSummary.initialize(_name,
                                       "SummaryvsLS_FED",
                                       new quantity::LumiSection(_maxProcessedLS),
                                       new quantity::FEDQuantity(_vFEDs),
                                       new quantity::ValueQuantity(quantity::fState),
                                       0);
    cSummaryvsLS_FED.initialize(_name,
                                "SummaryvsLS_FED",
                                hashfunctions::fFED,
                                new quantity::LumiSection(_maxProcessedLS),
                                new quantity::FlagQuantity(vflagsPerLS),
                                new quantity::ValueQuantity(quantity::fState),
                                0);
    cSummaryvsLS_FED.book(ib, _emap, _subsystem);
    cSummaryvsLS_FEDSummary.book(ib, _subsystem);

    // INITIALIZE CONTAINERS WE NEED TO LOAD or BOOK
    Container2D cOccupancyCut_depth;
    Container2D cDead_depth, cDead_Crate;
    cOccupancyCut_depth.initialize(_taskname,
                                   "OccupancyCut",
                                   hashfunctions::fdepth,
                                   new quantity::DetectorQuantity(quantity::fieta),
                                   new quantity::DetectorQuantity(quantity::fiphi),
                                   new quantity::ValueQuantity(quantity::fN),
                                   0);
    cDead_depth.initialize(_name,
                           "Dead",
                           hashfunctions::fdepth,
                           new quantity::DetectorQuantity(quantity::fieta),
                           new quantity::DetectorQuantity(quantity::fiphi),
                           new quantity::ValueQuantity(quantity::fN),
                           0);
    cDead_Crate.initialize(_name,
                           "Dead",
                           hashfunctions::fCrate,
                           new quantity::ElectronicsQuantity(quantity::fSpigot),
                           new quantity::ElectronicsQuantity(quantity::fFiberVMEFiberCh),
                           new quantity::ValueQuantity(quantity::fN),
                           0);

    //	LOAD
    cOccupancyCut_depth.load(ig, _emap, _subsystem);
    cDead_depth.book(ib, _emap, _subsystem);
    cDead_Crate.book(ib, _emap, _subsystem);

    //	ANALYZE RUN BASED QUANTITIES
    std::vector<HcalGenericDetId> gids = _emap->allPrecisionId();
    for (std::vector<HcalGenericDetId>::const_iterator it = gids.begin(); it != gids.end(); ++it) {
      if (!it->isHcalDetId())
        continue;

      HcalDetId did = HcalDetId(it->rawId());
      HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(did));

      if (_cOccupancy_depth.getBinContent(did) < 1) {
        _xDead.get(eid)++;
        cDead_depth.fill(did);
        cDead_Crate.fill(eid);
      }
      if (did.subdet() == HcalForward)
        _xUniHF.get(eid) += cOccupancyCut_depth.getBinContent(did);
    }
    //	ANALYZE FOR HF SLOT UNIFORMITY
    for (uintCompactMap::const_iterator it = _xUniHF.begin(); it != _xUniHF.end(); ++it) {
      uint32_t hash1 = it->first;
      HcalElectronicsId eid1(hash1);
      double x1 = it->second;

      for (uintCompactMap::const_iterator jt = _xUniHF.begin(); jt != _xUniHF.end(); ++jt) {
        if (jt == it)
          continue;

        double x2 = jt->second;
        if (x2 == 0)
          continue;
        if (x1 / x2 < _thresh_unihf)
          _xUni.get(eid1)++;
      }
    }

    // Iterate over each FED: fill per-LS histograms and accumulate the per-run summary flag
    std::vector<flag::Flag> sumflags;
    int ifed = 0;
    for (auto& it_fed : _vhashFEDs) {
      HcalElectronicsId eid(it_fed);
      flag::Flag fSumRun("DIGI");
      flag::Flag ffDead("Dead");
      flag::Flag ffUniSlotHF("UniSlotHF");

      if (utilities::isFEDZDC(eid)) {
        sumflags.push_back(fSumRun);
        ifed++;
        continue;
      }

      // Per-LS flag histograms
      for (std::vector<LSSummary>::const_iterator itls = _vflagsLS.begin(); itls != _vflagsLS.end(); ++itls) {
        int iflag = 0;
        flag::Flag fSumLS("DIGI");
        for (std::vector<flag::Flag>::const_iterator ft = itls->_vflags[ifed].begin(); ft != itls->_vflags[ifed].end();
             ++ft) {
          cSummaryvsLS_FED.setBinContent(eid, itls->_LS, static_cast<int>(iflag), ft->_state);
          fSumLS += (*ft);
          iflag++;
        }
        cSummaryvsLS_FEDSummary.setBinContent(eid, itls->_LS, fSumLS._state);
        fSumRun += fSumLS;
      }

      // Run-based flags: dead channels and HF slot uniformity
      if (_xDead.get(eid) > 0)
        ffDead._state = flag::fBAD;
      else
        ffDead._state = flag::fGOOD;

      if (utilities::isFEDHF(eid)) {
        if (_xUni.get(eid) > 0)
          ffUniSlotHF._state = flag::fBAD;
        else
          ffUniSlotHF._state = flag::fGOOD;
      }
      fSumRun += ffDead + ffUniSlotHF;

      sumflags.push_back(fSumRun);
      ifed++;
    }

    return sumflags;
  }
}  // namespace hcaldqm
