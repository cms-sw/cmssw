#include "DQM/HcalTasks/interface/RawTask.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;

RawTask::RawTask(edm::ParameterSet const& ps) : DQTask(ps) {
  _tagFEDs = ps.getUntrackedParameter<edm::InputTag>("tagFEDs", edm::InputTag("rawDataCollector"));
  _tagReport = ps.getUntrackedParameter<edm::InputTag>("tagReport", edm::InputTag("hcalDigis"));
  _calibProcessing = ps.getUntrackedParameter<bool>("calibProcessing", false);
  _thresh_calib_nbadq = ps.getUntrackedParameter<int>("thresh_calib_nbadq", 5000);

  _tokFEDs = consumes<FEDRawDataCollection>(_tagFEDs);
  _tokReport = consumes<HcalUnpackerReport>(_tagReport);

  _vflags.resize(nRawFlag);
  _vflags[fEvnMsm] = flag::Flag("EvnMsm");
  _vflags[fBcnMsm] = flag::Flag("BcnMsm");
  _vflags[fBadQ] = flag::Flag("BadQ");
  _vflags[fOrnMsm] = flag::Flag("OrnMsm");
}

/* virtual */ void RawTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  DQTask::bookHistograms(ib, r, es);

  //	GET WHAT YOU NEED
  edm::ESHandle<HcalDbService> dbs;
  es.get<HcalDbRecord>().get(dbs);
  _emap = dbs->getHcalMapping();
  std::vector<uint32_t> vVME;
  std::vector<uint32_t> vuTCA;
  vVME.push_back(
      HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, CRATE_VME_MIN).rawId());
  vuTCA.push_back(HcalElectronicsId(CRATE_uTCA_MIN, SLOT_uTCA_MIN, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
  _filter_VME.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vVME);
  _filter_uTCA.initialize(filter::fFilter, hcaldqm::hashfunctions::fElectronics, vuTCA);

  _cBadQualityvsLS.initialize(_name,
                              "BadQualityvsLS",
                              new hcaldqm::quantity::LumiSection(_maxLS),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_m0to10000),
                              0);
  _cBadQualityvsBX.initialize(_name,
                              "BadQualityvsBX",
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fBX),
                              new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN_m0to10000),
                              0);
  _cBadQuality_depth.initialize(_name,
                                "BadQuality",
                                hcaldqm::hashfunctions::fdepth,
                                new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                                new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                                new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                0);

  // FED-based plots
  if (_ptype != fOffline) {  // hidefed2crate
    std::vector<int> vFEDs = hcaldqm::utilities::getFEDList(_emap);
    std::vector<int> vFEDsVME = hcaldqm::utilities::getFEDVMEList(_emap);
    std::vector<int> vFEDsuTCA = hcaldqm::utilities::getFEDuTCAList(_emap);

    std::vector<uint32_t> vhashFEDsVME;
    std::vector<uint32_t> vhashFEDsuTCA;

    for (std::vector<int>::const_iterator it = vFEDsVME.begin(); it != vFEDsVME.end(); ++it) {
      vhashFEDsVME.push_back(
          HcalElectronicsId(
              constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, (*it) - constants::FED_VME_MIN)
              .rawId());
      _vhashFEDs.push_back(
          HcalElectronicsId(
              constants::FIBERCH_MIN, constants::FIBER_VME_MIN, SPIGOT_MIN, (*it) - constants::FED_VME_MIN)
              .rawId());
    }
    for (std::vector<int>::const_iterator it = vFEDsuTCA.begin(); it != vFEDsuTCA.end(); ++it) {
      std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(*it);
      vhashFEDsuTCA.push_back(
          HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
      _vhashFEDs.push_back(HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false).rawId());
    }
    _filter_FEDsVME.initialize(filter::fPreserver, hcaldqm::hashfunctions::fFED, vhashFEDsVME);
    _filter_FEDsuTCA.initialize(filter::fPreserver, hcaldqm::hashfunctions::fFED, vhashFEDsuTCA);

    //	INITIALIZE FIRST
    _cEvnMsm_ElectronicsVME.initialize(_name,
                                       "EvnMsm",
                                       hcaldqm::hashfunctions::fElectronics,
                                       new hcaldqm::quantity::FEDQuantity(vFEDsVME),
                                       new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                       0);
    _cBcnMsm_ElectronicsVME.initialize(_name,
                                       "BcnMsm",
                                       hcaldqm::hashfunctions::fElectronics,
                                       new hcaldqm::quantity::FEDQuantity(vFEDsVME),
                                       new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                       0);
    _cOrnMsm_ElectronicsVME.initialize(_name,
                                       "OrnMsm",
                                       hcaldqm::hashfunctions::fElectronics,
                                       new hcaldqm::quantity::FEDQuantity(vFEDsVME),
                                       new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                       new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                       0);
    _cEvnMsm_ElectronicsuTCA.initialize(_name,
                                        "EvnMsm",
                                        hcaldqm::hashfunctions::fElectronics,
                                        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                        0);
    _cBcnMsm_ElectronicsuTCA.initialize(_name,
                                        "BcnMsm",
                                        hcaldqm::hashfunctions::fElectronics,
                                        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                        0);
    _cOrnMsm_ElectronicsuTCA.initialize(_name,
                                        "OrnMsm",
                                        hcaldqm::hashfunctions::fElectronics,
                                        new hcaldqm::quantity::FEDQuantity(vFEDsuTCA),
                                        new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                        0);

    //	Bad Quality
    _cBadQuality_FEDVME.initialize(_name,
                                   "BadQuality",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSpigot),
                                   new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberVMEFiberCh),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                   0);
    _cBadQuality_FEDuTCA.initialize(_name,
                                    "BadQuality",
                                    hcaldqm::hashfunctions::fFED,
                                    new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fSlotuTCA),
                                    new hcaldqm::quantity::ElectronicsQuantity(hcaldqm::quantity::fFiberuTCAFiberCh),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                    0);

    //	Online only
    if (_ptype == fOnline) {
      _xEvnMsmLS.initialize(hcaldqm::hashfunctions::fFED);
      _xBcnMsmLS.initialize(hcaldqm::hashfunctions::fFED);
      _xOrnMsmLS.initialize(hcaldqm::hashfunctions::fFED);
      _xBadQLS.initialize(hcaldqm::hashfunctions::fFED);
      _cSummaryvsLS_FED.initialize(_name,
                                   "SummaryvsLS",
                                   hcaldqm::hashfunctions::fFED,
                                   new hcaldqm::quantity::LumiSection(_maxLS),
                                   new hcaldqm::quantity::FlagQuantity(_vflags),
                                   new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState),
                                   0);
      _cSummaryvsLS.initialize(_name,
                               "SummaryvsLS",
                               new hcaldqm::quantity::LumiSection(_maxLS),
                               new hcaldqm::quantity::FEDQuantity(vFEDs),
                               new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState),
                               0);
      //	FED Size vs LS
      _cDataSizevsLS_FED.initialize(_name,
                                    "DataSizevsLS",
                                    hcaldqm::hashfunctions::fFED,
                                    new hcaldqm::quantity::LumiSection(_maxLS),
                                    new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fDataSize),
                                    0);
    }
  }

  //	BOOK HISTOGRAMS
  if (_ptype != fOffline) {
    _cEvnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cBcnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cOrnMsm_ElectronicsVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cEvnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cBcnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);
    _cOrnMsm_ElectronicsuTCA.book(ib, _emap, _filter_VME, _subsystem);

    _cBadQuality_FEDVME.book(ib, _emap, _filter_uTCA, _subsystem);
    _cBadQuality_FEDuTCA.book(ib, _emap, _filter_VME, _subsystem);
  }
  if (_ptype == fOffline) {
    auto scope = DQMStore::IBooker::UseLumiScope(ib);
    _cBadQuality_depth.book(ib, _emap, _subsystem);
  } else {
    _cBadQuality_depth.book(ib, _emap, _subsystem);
  }
  _cBadQualityvsLS.book(ib, _subsystem);
  _cBadQualityvsBX.book(ib, _subsystem);

  // BOOK HISTOGRAMS to be used in ONLINE ONLY!
  if (_ptype == fOnline) {
    _xEvnMsmLS.book(_emap);
    _xBcnMsmLS.book(_emap);
    _xOrnMsmLS.book(_emap);
    _xBadQLS.book(_emap);
    _cSummaryvsLS_FED.book(ib, _emap, _subsystem);
    _cSummaryvsLS.book(ib, _subsystem);
    _cDataSizevsLS_FED.book(ib, _emap, _subsystem);
  }

  //	FOR OFFLINE PROCESSING MARK THESE HISTOGRAMS AS LUMI BASED
  //if (_ptype == fOffline) {
  //if (_ptype != fOffline) {  // hidefed2crate
  // Note that this is deliberately contradictory for the fed2crate fix, so it can be reversed if fed2crate is ever fixed properly,
  // TODO: set LUMI scope while booking.
  // _cEvnMsm_ElectronicsVME.setLumiFlag();
  // _cBcnMsm_ElectronicsVME.setLumiFlag();
  // _cEvnMsm_ElectronicsuTCA.setLumiFlag();
  // _cBcnMsm_ElectronicsuTCA.setLumiFlag();
  //}
  //}

  //	initialize hash map
  _ehashmap.initialize(_emap, hcaldqm::electronicsmap::fD2EHashMap);
}

/* virtual */ void RawTask::_resetMonitors(hcaldqm::UpdateFreq uf) {
  //	base reset
  DQTask::_resetMonitors(uf);
}

/* virtual */ void RawTask::_process(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<FEDRawDataCollection> craw;
  edm::Handle<HcalUnpackerReport> creport;
  if (!e.getByToken(_tokFEDs, craw))
    _logger.dqmthrow("Collection FEDRawDataCollection isn't available" + _tagFEDs.label() + " " + _tagFEDs.instance());
  if (!e.getByToken(_tokReport, creport))
    _logger.dqmthrow("Collection HcalUnpackerReport isn't available" + _tagReport.label() + " " +
                     _tagReport.instance());

  //	extract some info
  int bx = e.bunchCrossing();

  /*
	 *	For Calibration/Abort Gap Processing
	 *	check if the #channels taht are bad from the unpacker 
	 *	is > 5000. If it is skip...
	 */
  if (_calibProcessing) {
    int nbadq = creport->badQualityDigis();
    if (nbadq >= _thresh_calib_nbadq)
      return;
  }

  int nn = 0;
  //	loop thru and fill the detIds with bad quality
  //	NOTE: Calibration Channels are skipped!
  //	TODO: Include for Online Calibration Channels marked as bad
  //	a comment below is left on purpose!
  //_cBadQualityvsBX.fill(bx, creport->badQualityDigis());
  for (std::vector<DetId>::const_iterator it = creport->bad_quality_begin(); it != creport->bad_quality_end(); ++it) {
    //	skip non HCAL det ids
    if (!HcalGenericDetId(*it).isHcalDetId())
      continue;

    //	skip those that are of bad quality from conditions
    //	Masked or Dead
    if (_xQuality.exists(HcalDetId(*it))) {
      HcalChannelStatus cs(it->rawId(), _xQuality.get(HcalDetId(*it)));
      if (cs.isBitSet(HcalChannelStatus::HcalCellMask) || cs.isBitSet(HcalChannelStatus::HcalCellDead))
        continue;
    }

    nn++;
    HcalElectronicsId eid = HcalElectronicsId(_ehashmap.lookup(*it));
    _cBadQuality_depth.fill(HcalDetId(*it));
    //	ONLINE ONLY!
    if (_ptype == fOnline)
      _xBadQLS.get(eid)++;
    if (_ptype != fOffline) {  // hidefed2crate
      if (eid.isVMEid()) {
        if (_filter_FEDsVME.filter(eid))
          continue;
        _cBadQuality_FEDVME.fill(eid);
      } else {
        if (_filter_FEDsuTCA.filter(eid))
          continue;
        _cBadQuality_FEDuTCA.fill(eid);
      }
    }
  }
  _cBadQualityvsLS.fill(_currentLS, nn);
  _cBadQualityvsBX.fill(bx, nn);

  if (_ptype != fOffline) {  // hidefed2crate
    for (int fed = FEDNumbering::MINHCALFEDID; fed <= FEDNumbering::MAXHCALuTCAFEDID; fed++) {
      //	skip nonHCAL FEDs
      if ((fed > FEDNumbering::MAXHCALFEDID && fed < FEDNumbering::MINHCALuTCAFEDID) ||
          fed > FEDNumbering::MAXHCALuTCAFEDID)
        continue;
      FEDRawData const& raw = craw->FEDData(fed);
      if (raw.size() < constants::RAW_EMPTY)
        continue;

      if (fed <= FEDNumbering::MAXHCALFEDID)  // VME
      {
        HcalDCCHeader const* hdcc = (HcalDCCHeader const*)(raw.data());
        if (!hdcc)
          continue;

        uint32_t bcn = hdcc->getBunchId();
        uint32_t orn = hdcc->getOrbitNumber() & 0x1F;  // LS 5 bits only
        uint32_t evn = hdcc->getDCCEventNumber();
        int dccId = hdcc->getSourceId() - constants::FED_VME_MIN;

        /* online only */
        if (_ptype == fOnline) {
          HcalElectronicsId eid =
              HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, constants::SPIGOT_MIN, dccId);
          if (_filter_FEDsVME.filter(eid))
            continue;
          _cDataSizevsLS_FED.fill(eid, _currentLS, double(raw.size()) / 1024.);
        }

        //	iterate over spigots
        HcalHTRData htr;
        for (int is = 0; is < HcalDCCHeader::SPIGOT_COUNT; is++) {
          int r = hdcc->getSpigotData(is, htr, raw.size());
          if (r != 0 || !htr.check())
            continue;
          HcalElectronicsId eid = HcalElectronicsId(constants::FIBERCH_MIN, constants::FIBER_VME_MIN, is, dccId);
          if (_filter_FEDsVME.filter(eid))
            continue;

          uint32_t htr_evn = htr.getL1ANumber();
          uint32_t htr_orn = htr.getOrbitNumber();
          uint32_t htr_bcn = htr.getBunchNumber();
          bool qevn = (htr_evn != evn);
          bool qbcn = (htr_bcn != bcn);
          bool qorn = (htr_orn != orn);
          if (qevn) {
            _cEvnMsm_ElectronicsVME.fill(eid);

            if (_ptype == fOnline && is <= constants::SPIGOT_MAX)
              _xEvnMsmLS.get(eid)++;
          }
          if (qorn) {
            _cOrnMsm_ElectronicsVME.fill(eid);

            if (_ptype == fOnline && is <= constants::SPIGOT_MAX)
              _xOrnMsmLS.get(eid)++;
          }
          if (qbcn) {
            _cBcnMsm_ElectronicsVME.fill(eid);

            if (_ptype == fOnline && is <= constants::SPIGOT_MAX)
              _xBcnMsmLS.get(eid)++;
          }
        }
      } else  // uTCA
      {
        hcal::AMC13Header const* hamc13 = (hcal::AMC13Header const*)raw.data();
        if (!hamc13)
          continue;

        /* online only */
        if (_ptype == fOnline) {
          std::pair<uint16_t, uint16_t> cspair = utilities::fed2crate(fed);
          HcalElectronicsId eid = HcalElectronicsId(cspair.first, cspair.second, FIBER_uTCA_MIN1, FIBERCH_MIN, false);
          if (_filter_FEDsuTCA.filter(eid))
            continue;
          _cDataSizevsLS_FED.fill(eid, _currentLS, double(raw.size()) / 1024.);
        }

        uint32_t bcn = hamc13->bunchId();
        uint32_t orn = hamc13->orbitNumber() & 0xFFFF;  // LS 16bits only
        uint32_t evn = hamc13->l1aNumber();
        int namc = hamc13->NAMC();

        for (int iamc = 0; iamc < namc; iamc++) {
          int slot = hamc13->AMCSlot(iamc);
          int crate = hamc13->AMCId(iamc) & 0xFF;
          HcalElectronicsId eid(crate, slot, FIBER_uTCA_MIN1, FIBERCH_MIN, false);
          if (_filter_FEDsuTCA.filter(eid))
            continue;
          HcalUHTRData uhtr(hamc13->AMCPayload(iamc), hamc13->AMCSize(iamc));

          uint32_t uhtr_evn = uhtr.l1ANumber();
          uint32_t uhtr_bcn = uhtr.bunchNumber();
          uint32_t uhtr_orn = uhtr.orbitNumber();
          bool qevn = (uhtr_evn != evn);
          bool qbcn = (uhtr_bcn != bcn);
          bool qorn = (uhtr_orn != orn);
          if (qevn) {
            _cEvnMsm_ElectronicsuTCA.fill(eid);

            if (_ptype == fOnline)
              _xEvnMsmLS.get(eid)++;
          }
          if (qorn) {
            _cOrnMsm_ElectronicsuTCA.fill(eid);

            if (_ptype == fOnline)
              _xOrnMsmLS.get(eid)++;
          }
          if (qbcn) {
            _cBcnMsm_ElectronicsuTCA.fill(eid);

            if (_ptype == fOnline)
              _xBcnMsmLS.get(eid)++;
          }
        }
      }
    }
  }
}

/* virtual */ void RawTask::dqmBeginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  DQTask::dqmBeginLuminosityBlock(lb, es);

  //	_cBadQualityvsLS.extendAxisRange(_currentLS);

  //	ONLINE ONLY!
  if (_ptype != fOnline)
    return;
  //	_cSummaryvsLS_FED.extendAxisRange(_currentLS);
  //	_cSummaryvsLS.extendAxisRange(_currentLS);
}

/* virtual */ void RawTask::dqmEndLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const& es) {
  if (_ptype != fOnline)
    return;

  //
  //	GENERATE STATUS ONLY FOR ONLINE!
  //
  for (std::vector<uint32_t>::const_iterator it = _vhashFEDs.begin(); it != _vhashFEDs.end(); ++it) {
    flag::Flag fSum("RAW");
    HcalElectronicsId eid = HcalElectronicsId(*it);

    std::vector<uint32_t>::const_iterator cit = std::find(_vcdaqEids.begin(), _vcdaqEids.end(), *it);
    if (cit == _vcdaqEids.end()) {
      // not @cDAQ
      for (uint32_t iflag = 0; iflag < _vflags.size(); iflag++)
        _cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag), int(flag::fNCDAQ));
      _cSummaryvsLS.setBinContent(eid, _currentLS, int(flag::fNCDAQ));
      continue;
    }

    //	FED is @cDAQ
    if (hcaldqm::utilities::isFEDHBHE(eid) || hcaldqm::utilities::isFEDHF(eid) || hcaldqm::utilities::isFEDHO(eid)) {
      if (_xEvnMsmLS.get(eid) > 0)
        _vflags[fEvnMsm]._state = flag::fBAD;
      else
        _vflags[fEvnMsm]._state = flag::fGOOD;
      if (_xBcnMsmLS.get(eid) > 0)
        _vflags[fBcnMsm]._state = flag::fBAD;
      else
        _vflags[fBcnMsm]._state = flag::fGOOD;
      if (_xOrnMsmLS.get(eid) > 0)
        _vflags[fOrnMsm]._state = flag::fBAD;
      else
        _vflags[fOrnMsm]._state = flag::fGOOD;
      if (double(_xBadQLS.get(eid)) > double(12 * _evsPerLS))
        _vflags[fBadQ]._state = flag::fBAD;
      else if (_xBadQLS.get(eid) > 0)
        _vflags[fBadQ]._state = flag::fPROBLEMATIC;
      else
        _vflags[fBadQ]._state = flag::fGOOD;
    }

    int iflag = 0;
    //	iterate over all flags:
    //	- sum them all up in summary flag for this FED
    //	- reset each flag right after using it
    for (std::vector<flag::Flag>::iterator ft = _vflags.begin(); ft != _vflags.end(); ++ft) {
      _cSummaryvsLS_FED.setBinContent(eid, _currentLS, int(iflag), ft->_state);
      fSum += (*ft);
      iflag++;

      //	this is the MUST! We don't keep flags per FED, reset
      //	each one of them after using
      ft->reset();
    }
    _cSummaryvsLS.setBinContent(eid, _currentLS, fSum._state);
  }

  //	reset...
  _xOrnMsmLS.reset();
  _xEvnMsmLS.reset();
  _xBcnMsmLS.reset();
  _xBadQLS.reset();

  //	in the end always do the DQTask::endLumi
  DQTask::dqmEndLuminosityBlock(lb, es);
}

DEFINE_FWK_MODULE(RawTask);
