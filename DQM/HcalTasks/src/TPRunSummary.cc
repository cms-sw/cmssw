#include "DQM/HcalTasks/interface/TPRunSummary.h"

namespace hcaldqm {
  using namespace constants;
  TPRunSummary::TPRunSummary(std::string const& name,
                             std::string const& taskname,
                             edm::ParameterSet const& ps,
                             edm::ConsumesCollector& iC)
      : DQClient(name, taskname, ps, iC) {
    _thresh_EtMsmRate_high = ps.getUntrackedParameter<double>("thresh_EtMsmRate_high", 0.2);
    _thresh_EtMsmRate_low = ps.getUntrackedParameter<double>("thresh_EtMsmRate_low", 0.05);
    _thresh_FGMsmRate_high = ps.getUntrackedParameter<double>("thresh_FGMsmRate_high", 0.2);
    _thresh_FGMsmRate_low = ps.getUntrackedParameter<double>("thresh_FGMsmRate_low", 0.05);
  }

  /* virtual */ void TPRunSummary::beginRun(edm::Run const& r, edm::EventSetup const& es) { DQClient::beginRun(r, es); }

  /* virtual */ void TPRunSummary::endLuminosityBlock(DQMStore::IBooker& ib,
                                                      DQMStore::IGetter& ig,
                                                      edm::LuminosityBlock const& lb,
                                                      edm::EventSetup const& es) {
    DQClient::endLuminosityBlock(ib, ig, lb, es);
  }

  /* virtual */ std::vector<flag::Flag> TPRunSummary::endJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
    //	hahs maps
    electronicsmap::ElectronicsMap ehashmap;
    ehashmap.initialize(_emap, electronicsmap::fT2EHashMap);
    std::vector<flag::Flag> vflags;
    vflags.resize(nTPFlag);
    vflags[fEtMsm] = flag::Flag("EtMsm");
    vflags[fFGMsm] = flag::Flag("FGMsm");

    //	INITIALIZE
    ContainerSingle2D cOccupancyData_depthlike, cOccupancyEmul_depthlike;
    ContainerSingle2D cEtMsm_depthlike, cFGMsm_depthlike, cEtCorrRatio_depthlike;
    ContainerSingle2D cSummary;
    ContainerXXX<double> xDeadD, xDeadE, xEtMsm, xFGMsm;
    ContainerXXX<double> xNumCorr;
    xDeadD.initialize(hashfunctions::fCrate);
    xDeadE.initialize(hashfunctions::fCrate);
    xEtMsm.initialize(hashfunctions::fCrate);
    xFGMsm.initialize(hashfunctions::fCrate);
    xNumCorr.initialize(hashfunctions::fCrate);
    cOccupancyData_depthlike.initialize(_taskname,
                                        "OccupancyData",
                                        new quantity::TrigTowerQuantity(quantity::fTTieta),
                                        new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                        new quantity::ValueQuantity(quantity::fN, true),
                                        0);
    cOccupancyEmul_depthlike.initialize(_taskname,
                                        "OccupancyEmul",
                                        new quantity::TrigTowerQuantity(quantity::fTTieta),
                                        new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                        new quantity::ValueQuantity(quantity::fN, true),
                                        0);
    cEtMsm_depthlike.initialize(_taskname,
                                "EtMsm",
                                new quantity::TrigTowerQuantity(quantity::fTTieta),
                                new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                new quantity::ValueQuantity(quantity::fN),
                                0);
    cFGMsm_depthlike.initialize(_taskname,
                                "FGMsm",
                                new quantity::TrigTowerQuantity(quantity::fTTieta),
                                new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                new quantity::ValueQuantity(quantity::fN),
                                0);
    cEtCorrRatio_depthlike.initialize(_taskname,
                                      "EtCorrRatio",
                                      new quantity::TrigTowerQuantity(quantity::fTTieta),
                                      new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                      new quantity::ValueQuantity(quantity::fRatio_0to2),
                                      0);
    _cEtMsmFraction_depthlike.initialize(_taskname,
                                         "EtMsmFraction",
                                         new quantity::TrigTowerQuantity(quantity::fTTieta),
                                         new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                         new quantity::ValueQuantity(quantity::fRatio_0to2),
                                         0);
    _cFGMsmFraction_depthlike.initialize(_taskname,
                                         "FGMsmFraction",
                                         new quantity::TrigTowerQuantity(quantity::fTTieta),
                                         new quantity::TrigTowerQuantity(quantity::fTTiphi),
                                         new quantity::ValueQuantity(quantity::fRatio_0to2),
                                         0);
    cSummary.initialize(_name,
                        "Summary",
                        new quantity::CrateQuantity(_emap),
                        new quantity::FlagQuantity(vflags),
                        new quantity::ValueQuantity(quantity::fState),
                        0);

    //	BOOK
    xDeadD.book(_emap);
    xDeadE.book(_emap);
    xEtMsm.book(_emap);
    xFGMsm.book(_emap);
    xNumCorr.book(_emap);

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
    for (std::vector<HcalTrigTowerDetId>::const_iterator it = tids.begin(); it != tids.end(); ++it) {
      //	skip 2x3
      HcalTrigTowerDetId tid = HcalTrigTowerDetId(*it);
      if (tid.version() == 0 && tid.ietaAbs() >= 29)
        continue;

      //	do the comparison if there are channels that were correlated
      //	both had emul and data tps
      if (cEtCorrRatio_depthlike.getBinEntries(tid) > 0) {
        HcalElectronicsId eid = HcalElectronicsId(ehashmap.lookup(*it));

        double numetmsm = cEtMsm_depthlike.getBinContent(tid);
        double numfgmsm = cFGMsm_depthlike.getBinContent(tid);
        double numcorr = cEtCorrRatio_depthlike.getBinEntries(tid);

        xEtMsm.get(eid) += numetmsm;
        xFGMsm.get(eid) += numfgmsm;
        xNumCorr.get(eid) += numcorr;

        _cEtMsmFraction_depthlike.setBinContent(tid, numetmsm / numcorr);
        _cFGMsmFraction_depthlike.setBinContent(tid, numfgmsm / numcorr);
      }
    }

    std::vector<flag::Flag> sumflags;
    for (auto& it_hashcrate : _vhashCrates) {
      flag::Flag fSum("TP");
      HcalElectronicsId eid(it_hashcrate);
      HcalDetId did = HcalDetId(_emap->lookup(eid));

      if (did.subdet() == HcalBarrel || did.subdet() == HcalEndcap || did.subdet() == HcalForward) {
        double etmsmfr = xNumCorr.get(eid) > 0 ? double(xEtMsm.get(eid)) / double(xNumCorr.get(eid)) : 0;
        double fgmsmfr = xNumCorr.get(eid) > 0 ? double(xFGMsm.get(eid)) / double(xNumCorr.get(eid)) : 0;

        if (etmsmfr >= _thresh_EtMsmRate_high)
          vflags[fEtMsm]._state = flag::fBAD;
        else if (etmsmfr >= _thresh_EtMsmRate_low)
          vflags[fEtMsm]._state = flag::fPROBLEMATIC;
        else
          vflags[fEtMsm]._state = flag::fGOOD;
        if (fgmsmfr >= _thresh_FGMsmRate_high)
          vflags[fFGMsm]._state = flag::fBAD;
        else if (fgmsmfr >= _thresh_FGMsmRate_low)
          vflags[fFGMsm]._state = flag::fPROBLEMATIC;
        else
          vflags[fFGMsm]._state = flag::fGOOD;
      }

      //	combine
      int iflag = 0;
      for (std::vector<flag::Flag>::iterator ft = vflags.begin(); ft != vflags.end(); ++ft) {
        cSummary.setBinContent(eid, iflag, ft->_state);
        fSum += (*ft);
        iflag++;
        ft->reset();
      }
      sumflags.push_back(fSum);
    }

    return sumflags;
  }
}  // namespace hcaldqm
