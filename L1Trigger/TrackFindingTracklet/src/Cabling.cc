#include "L1Trigger/TrackFindingTracklet/interface/Cabling.h"
#include "L1Trigger/TrackFindingTracklet/interface/DTCLink.h"
#include "L1Trigger/TrackFindingTracklet/interface/DTC.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace trklet;

Cabling::Cabling(string dtcconfig, string moduleconfig, Settings const& settings) : settings_(settings) {
  ifstream indtc(dtcconfig.c_str());
  assert(indtc.good());

  string dtc;
  int isec;

  while (indtc.good()) {
    indtc >> dtc >> isec;

    if (!indtc.good())
      continue;

    if (dtcs_.find(dtc) == dtcs_.end()) {
      dtcs_[dtc].setName(dtc);
    }

    dtcs_[dtc].addSec(isec);

    string dtcbase = dtc.substr(2, dtc.size() - 2);
    if (dtc[0] == 'n') {
      dtcbase = "neg_" + dtc.substr(6, dtc.size() - 6);
    }
    if (dtcranges_.find(dtcbase) == dtcranges_.end()) {
      dtcranges_[dtcbase].setName(dtcbase);
    }
  }

  ifstream inmodules(moduleconfig.c_str());

  int layer, ladder, module;

  while (inmodules.good()) {
    inmodules >> layer >> ladder >> module >> dtc;

    // in the cabling module map, module# 300+ is flat part of barrel, 200-299 is tilted z-, 100-199 is tilted z+
    if (module > 300) {
      if (layer > 0 && layer <= (int)N_PSLAYER) {
        module = (module - 300) + N_TILTED_RINGS;
      } else if (layer > (int)N_PSLAYER) {
        module = (module - 300);
      }
    }
    if (module > 200) {
      module = (module - 200);
    }
    if ((module > 100) && (layer > 0 && layer <= (int)N_PSLAYER)) {
      module = (module - 100) + N_TILTED_RINGS + N_MOD_PLANK.at(layer - 1);
    }
    if (!inmodules.good())
      break;
    modules_[layer][ladder][module] = dtc;
  }
}

const string& Cabling::dtc(int layer, int ladder, int module) const {
  auto it1 = modules_.find(layer);
  assert(it1 != modules_.end());
  auto it2 = it1->second.find(ladder);
  assert(it2 != it1->second.end());
  auto it3 = it2->second.find(module);
  if (it3 == it2->second.end()) {
    throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << "Could not find stub " << layer << " "
                                       << ladder << " " << module;
  }
  return it3->second;
}

void Cabling::addphi(const string& dtc, double phi, int layer, int module) {
  unsigned int layerdisk = layer - 1;

  if (layer > 1000)
    layerdisk = module + N_DISK;

  assert(layerdisk < N_LAYERDISK);

  int isec = dtc[0] - '0';

  string dtcbase = dtc.substr(2, dtc.size() - 2);
  if (dtc[0] == 'n') {
    dtcbase = "neg_" + dtc.substr(6, dtc.size() - 6);
    isec = dtc[4] - '0';
  }

  double phisec = reco::reduceRange(phi - isec * settings_.dphisector());

  assert(dtcranges_.find(dtcbase) != dtcranges_.end());

  dtcranges_[dtcbase].addphi(phisec, layerdisk);
}

void Cabling::writephirange() const {
  ofstream out("dtcphirange.txt");

  for (auto&& it : dtcranges_) {
    for (unsigned int i = 0; i < N_LAYERDISK; i++) {
      double min = it.second.min(i);
      double max = it.second.max(i);
      if (min < max) {
        out << it.first << " " << i + 1 << " " << min << " " << max << endl;
      }
    }
  }
}

std::vector<string> Cabling::DTCs() const {
  std::vector<string> tmp;

  for (const auto& it : dtcs_) {
    tmp.push_back(it.first);
  }

  return tmp;
}
