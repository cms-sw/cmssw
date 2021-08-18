#include "DQM/HcalTasks/interface/HcalOfflineHarvesting.h"

using namespace hcaldqm;
using namespace hcaldqm::constants;
using namespace hcaldqm::filter;

HcalOfflineHarvesting::HcalOfflineHarvesting(edm::ParameterSet const& ps)
    : DQHarvester(ps), _reportSummaryMap(nullptr) {
  _summaryList.push_back(fTP);
  _summaryList.push_back(fDigi);
  _summaryList.push_back(fReco);
  _sumnames[fRaw] = "RawTask";
  _sumnames[fDigi] = "DigiTask";
  _sumnames[fReco] = "RecHitTask";
  _sumnames[fTP] = "TPTask";
  for (auto& it_sum : _summaryList) {
    _summarks[it_sum] = false;
  }

  auto iC = consumesCollector();
  if (std::find(_summaryList.begin(), _summaryList.end(), fRaw) != _summaryList.end()) {
    _sumgen[fRaw] = new hcaldqm::RawRunSummary("RawRunHarvesting", _sumnames[fRaw], ps, iC);
  }
  if (std::find(_summaryList.begin(), _summaryList.end(), fDigi) != _summaryList.end()) {
    _sumgen[fDigi] = new hcaldqm::DigiRunSummary("DigiRunHarvesting", _sumnames[fDigi], ps, iC);
  }
  if (std::find(_summaryList.begin(), _summaryList.end(), fReco) != _summaryList.end()) {
    _sumgen[fReco] = new hcaldqm::RecoRunSummary("RecoRunHarvesting", _sumnames[fReco], ps, iC);
  }
  if (std::find(_summaryList.begin(), _summaryList.end(), fTP) != _summaryList.end()) {
    _sumgen[fTP] = new hcaldqm::TPRunSummary("TPRunHarvesting", _sumnames[fTP], ps, iC);
  }
}

/* virtual */ void HcalOfflineHarvesting::beginRun(edm::Run const& r, edm::EventSetup const& es) {
  DQHarvester::beginRun(r, es);

  for (auto& it_sum : _summaryList) {
    _sumgen[it_sum]->beginRun(r, es);
  }
}

//
//	For OFFLINE there is no per LS evaluation
//
/* virtual */ void HcalOfflineHarvesting::_dqmEndLuminosityBlock(DQMStore::IBooker& ib,
                                                                 DQMStore::IGetter& ig,
                                                                 edm::LuminosityBlock const& lb,
                                                                 edm::EventSetup const& es) {
  for (auto& it_sum : _summaryList) {
    if (ig.get(_subsystem + "/" + _sumnames[it_sum] + "/EventsTotal") != nullptr) {
      _summarks[it_sum] = true;
    }
  }

  //	CALL ALL THE HARVESTERS
  for (auto& it_sum : _summaryList) {
    //	run only if have to
    if (_summarks[it_sum]) {
      (_sumgen[it_sum])->endLuminosityBlock(ib, ig, lb, es);
    }
  }
}

//
//	Evaluate and Generate Run Summary
//
/* virtual */ void HcalOfflineHarvesting::_dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
  //	OBTAIN/SET WHICH MODULES ARE PRESENT
  std::map<Summary, std::string> datatier_names;
  datatier_names[fRaw] = "RAW";
  datatier_names[fDigi] = "DIGI";
  datatier_names[fReco] = "RECO";
  datatier_names[fTP] = "TP";

  int num = 0;
  std::map<std::string, int> datatiers;
  for (auto& it_sum : _summaryList) {
    if (_summarks[it_sum]) {
      datatiers.insert(std::pair<std::string, int>(datatier_names[it_sum], num));
      ++num;
    }
  }

  //	CREATE THE REPORT SUMMARY MAP
  //	num is #modules
  //	datatiers - std map [DATATIER_NAME] -> [positional value [0,num-1]]
  //	-> bin wise +1 should be
  if (!_reportSummaryMap) {
    ib.setCurrentFolder(_subsystem + "/EventInfo");
    _reportSummaryMap =
        ib.book2D("reportSummaryMap", "reportSummaryMap", _vCrates.size(), 0, _vCrates.size(), num, 0, num);
    //	x axis labels

    for (uint32_t i = 0; i < _vCrates.size(); i++) {
      char name[5];
      sprintf(name, "%d", _vCrates[i]);
      _reportSummaryMap->setBinLabel(i + 1, name, 1);
    }
    //	y axis lables
    for (std::map<std::string, int>::const_iterator it = datatiers.begin(); it != datatiers.end(); ++it) {
      std::string name = it->first;
      int value = it->second;
      _reportSummaryMap->setBinLabel(value + 1, name, 2);
    }
  }

  //	iterate over all summary generators and get the flags
  for (auto& it_sum : _summaryList) {
    //	IF MODULE IS NOT PRESENT IN DATA SKIP
    if (!_summarks[it_sum]) {
      continue;
    }

    //	OBTAIN ALL THE FLAGS FOR THIS MODULE
    //	AND SET THE REPORT STATUS MAP
    //	NOTE AGAIN: datatiers map [DATATIER]->[value not bin!]+1 therefore
    if (_debug > 0) {
      std::cout << _sumnames[it_sum] << std::endl;
    }
    std::vector<hcaldqm::flag::Flag> flags = (_sumgen[it_sum])->endJob(ib, ig);
    if (_debug > 0) {
      std::cout << "********************" << std::endl;
      std::cout << "SUMMARY" << std::endl;
    }
    for (uint32_t icrate = 0; icrate < _vCrates.size(); icrate++) {
      _reportSummaryMap->setBinContent(icrate + 1, datatiers[flags[icrate]._name] + 1, (int)flags[icrate]._state);
      if (_debug > 0) {
        std::cout << "Crate=" << _vCrates[icrate] << std::endl;
        std::cout << flags[icrate]._name << "  " << flags[icrate]._state << std::endl;
      }
    }
  }
}

DEFINE_FWK_MODULE(HcalOfflineHarvesting);
