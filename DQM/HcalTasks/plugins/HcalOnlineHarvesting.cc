#include "DQM/HcalTasks/interface/HcalOnlineHarvesting.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;

HcalOnlineHarvesting::HcalOnlineHarvesting(edm::ParameterSet const& ps)
    : DQHarvester(ps), _nBad(0), _nTotal(0), _reportSummaryMap(nullptr) {
  //	NOTE: I will leave Run Summary Generators in place
  //	just not triggering on endJob!
  _vsumgen.resize(nSummary);
  _vnames.resize(nSummary);
  _vmarks.resize(nSummary);
  for (uint32_t i = 0; i < _vmarks.size(); i++)
    _vmarks[i] = false;
  _vnames[fRaw] = "RawTask";
  _vnames[fDigi] = "DigiTask";
  _vnames[fReco] = "RecHitTask";
  _vnames[fTP] = "TPTask";
  _vnames[fPedestal] = "PedestalTask";

  auto iC = consumesCollector();
  _vsumgen[fRaw] = new hcaldqm::RawRunSummary("RawRunHarvesting", _vnames[fRaw], ps, iC);
  _vsumgen[fDigi] = new hcaldqm::DigiRunSummary("DigiRunHarvesting", _vnames[fDigi], ps, iC);
  _vsumgen[fReco] = new hcaldqm::RecoRunSummary("RecoRunHarvesting", _vnames[fReco], ps, iC);
  _vsumgen[fTP] = new hcaldqm::TPRunSummary("TPRunHarvesting", _vnames[fTP], ps, iC);
  _vsumgen[fPedestal] = new hcaldqm::PedestalRunSummary("PedestalRunHarvesting", _vnames[fPedestal], ps, iC);

  _thresh_bad_bad = ps.getUntrackedParameter("thresh_bad_bad", 0.05);
}

/* virtual */ void HcalOnlineHarvesting::beginRun(edm::Run const& r, edm::EventSetup const& es) {
  DQHarvester::beginRun(r, es);
  for (std::vector<DQClient*>::const_iterator it = _vsumgen.begin(); it != _vsumgen.end(); ++it)
    (*it)->beginRun(r, es);
}

/* virtual */ void HcalOnlineHarvesting::_dqmEndLuminosityBlock(DQMStore::IBooker& ib,
                                                                DQMStore::IGetter& ig,
                                                                edm::LuminosityBlock const&,
                                                                edm::EventSetup const&) {
  //	DETERMINE WHICH MODULES ARE PRESENT IN DATA
  if (ig.get(_subsystem + "/" + _vnames[fRaw] + "/EventsTotal") != nullptr)
    _vmarks[fRaw] = true;
  if (ig.get(_subsystem + "/" + _vnames[fDigi] + "/EventsTotal") != nullptr)
    _vmarks[fDigi] = true;
  if (ig.get(_subsystem + "/" + _vnames[fTP] + "/EventsTotal") != nullptr)
    _vmarks[fTP] = true;
  if (ig.get(_subsystem + "/" + _vnames[fReco] + "/EventsTotal") != nullptr)
    _vmarks[fReco] = true;
  if (ig.get(_subsystem + "/" + _vnames[fPedestal] + "/EventsTotal") != nullptr)
    _vmarks[fPedestal] = true;

  //	CREATE SUMMARY REPORT MAP FED vs LS and LOAD MODULE'S SUMMARIES
  //	NOTE: THIS STATEMENTS WILL BE EXECUTED ONLY ONCE!
  if (!_reportSummaryMap) {
    ig.setCurrentFolder(_subsystem + "/EventInfo");
    _reportSummaryMap =
        ib.book2D("reportSummaryMap", "reportSummaryMap", _maxLS, 1, _maxLS + 1, _vFEDs.size(), 0, _vFEDs.size());
    for (uint32_t i = 0; i < _vFEDs.size(); i++) {
      char name[5];
      sprintf(name, "%d", _vFEDs[i]);
      _reportSummaryMap->setBinLabel(i + 1, name, 2);
    }
    //	set LS bit to mark Xaxis as LS axis
    _reportSummaryMap->getTH1()->SetBit(BIT(BIT_OFFSET + BIT_AXIS_LS));

    // INITIALIZE ALL THE MODULES
    for (uint32_t i = 0; i < _vnames.size(); i++)
      _vcSummaryvsLS.push_back(ContainerSingle2D(_vnames[i],
                                                 "SummaryvsLS",
                                                 new hcaldqm::quantity::LumiSection(_maxLS),
                                                 new hcaldqm::quantity::FEDQuantity(_vFEDs),
                                                 new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fState)));

    //	LOAD ONLY THOSE MODULES THAT ARE PRESENT IN DATA
    for (uint32_t i = 0; i < _vmarks.size(); i++) {
      if (_vmarks[i])
        _vcSummaryvsLS[i].load(ig, _subsystem);
    }

    //	Create a map of bad channels and fill
    _cKnownBadChannels_depth.initialize("RunInfo",
                                        "KnownBadChannels",
                                        hcaldqm::hashfunctions::fdepth,
                                        new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fieta),
                                        new hcaldqm::quantity::DetectorQuantity(hcaldqm::quantity::fiphi),
                                        new hcaldqm::quantity::ValueQuantity(hcaldqm::quantity::fN),
                                        0);
    _cKnownBadChannels_depth.book(ib, _emap, _subsystem);
    for (uintCompactMap::const_iterator it = _xQuality.begin(); it != _xQuality.end(); ++it)
      _cKnownBadChannels_depth.fill(HcalDetId(it->first));

    ig.setCurrentFolder(_subsystem + "/EventInfo");
    _runSummary = ib.book2D("runSummary", "runSummary", 1, 0, 1, 1, 0, 1);
  }

  int ifed = 0;
  hcaldqm::flag::Flag fTotal("Status", hcaldqm::flag::fNCDAQ);
  if (_ptype != fOffline) {  // hidefed2crate
    for (std::vector<uint32_t>::const_iterator it = _vhashFEDs.begin(); it != _vhashFEDs.end(); ++it) {
      HcalElectronicsId eid(*it);
      hcaldqm::flag::Flag fSum("Status", hcaldqm::flag::fNCDAQ);
      for (uint32_t im = 0; im < _vmarks.size(); im++)
        if (_vmarks[im]) {
          int x = _vcSummaryvsLS[im].getBinContent(eid, _currentLS);
          hcaldqm::flag::Flag flag("Status", (hcaldqm::flag::State)x);
          fSum += flag;
        }
      _reportSummaryMap->setBinContent(_currentLS, ifed + 1, int(fSum._state));
      ifed++;
      fTotal += fSum;
    }
  }

  // update the Run Summary
  // ^^^TEMPORARY AT THIS POINT!
  if (fTotal._state == hcaldqm::flag::fBAD)
    _nBad++;
  _nTotal++;
  if (double(_nBad) / double(_nTotal) >= _thresh_bad_bad)
    _runSummary->setBinContent(1, 1, int(hcaldqm::flag::fBAD));
  else if (fTotal._state == hcaldqm::flag::fNCDAQ)
    _runSummary->setBinContent(1, 1, int(hcaldqm::flag::fNCDAQ));
  else
    _runSummary->setBinContent(1, 1, int(hcaldqm::flag::fGOOD));

  // HF TDC TP efficiency
  if (_vmarks[fTP]) {
    MonitorElement* meOccupancy_HF_depth = ig.get("Hcal/TPTask/OccupancyDataHF_depth/OccupancyDataHF_depth");
    MonitorElement* meOccupancyNoTDC_HF_depth =
        ig.get("Hcal/TPTask/OccupancyEmulHFNoTDC_depth/OccupancyEmulHFNoTDC_depth");
    MonitorElement* meOccupancy_HF_ieta = ig.get("Hcal/TPTask/OccupancyDataHF_ieta/OccupancyDataHF_ieta");
    MonitorElement* meOccupancyNoTDC_HF_ieta =
        ig.get("Hcal/TPTask/OccupancyEmulHFNoTDC_ieta/OccupancyEmulHFNoTDC_ieta");

    if (meOccupancy_HF_depth && meOccupancyNoTDC_HF_depth && meOccupancy_HF_ieta && meOccupancyNoTDC_HF_ieta) {
      TH2F* hOccupancy_HF_depth = meOccupancy_HF_depth->getTH2F();
      TH2F* hOccupancyNoTDC_HF_depth = meOccupancyNoTDC_HF_depth->getTH2F();
      TH1D* hOccupancy_HF_ieta = meOccupancy_HF_ieta->getTH1D();
      TH1D* hOccupancyNoTDC_HF_ieta = meOccupancyNoTDC_HF_ieta->getTH1D();

      TH2F* hEfficiency_HF_depth = (TH2F*)hOccupancy_HF_depth->Clone();
      hEfficiency_HF_depth->Divide(hOccupancyNoTDC_HF_depth);
      TH1D* hEfficiency_HF_ieta = (TH1D*)hOccupancy_HF_ieta->Clone();
      hEfficiency_HF_ieta->Divide(hOccupancyNoTDC_HF_ieta);

      ib.setCurrentFolder("Hcal/TPTask");

      MonitorElement* meEfficiency_HF_depth = ib.book2D("TDCCutEfficiency_depth", hEfficiency_HF_depth);
      meEfficiency_HF_depth->setEfficiencyFlag();
      MonitorElement* meEfficiency_HF_ieta = ib.book1DD("TDCCutEfficiency_ieta", hEfficiency_HF_ieta);
      meEfficiency_HF_ieta->setEfficiencyFlag();

      delete hEfficiency_HF_depth;
      delete hEfficiency_HF_ieta;
    }
  }
}

/*
 *	NO END JOB PROCESSING FOR ONLINE!
 */
/* virtual */ void HcalOnlineHarvesting::_dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {}

DEFINE_FWK_MODULE(HcalOnlineHarvesting);
