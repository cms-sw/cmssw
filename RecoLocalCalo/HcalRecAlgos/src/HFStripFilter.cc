#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HFStripFilter.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <cmath>

HFStripFilter::HFStripFilter(const double stripThreshold,
                             const double maxThreshold,
                             const double timeMax,
                             const double maxStripTime,
                             const double wedgeCut,
                             const int seedHitIetaMax,
                             const int gap,
                             const int lstrips,
                             const int verboseLevel)
    : stripThreshold_(stripThreshold),
      maxThreshold_(maxThreshold),
      timeMax_(timeMax),
      maxStripTime_(maxStripTime),
      wedgeCut_(wedgeCut),
      seedHitIetaMax_(seedHitIetaMax),
      gap_(gap),
      lstrips_(lstrips),
      verboseLevel_(verboseLevel) {
  // For the description of CMSSW message logging, see
  // https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMessageLogger
  if (verboseLevel_ >= 20)
    edm::LogInfo("HFStripFilter") << "constructor called";
}

HFStripFilter::~HFStripFilter() {
  if (verboseLevel_ >= 20)
    edm::LogInfo("HFStripFilter") << "destructor called";
}

void HFStripFilter::runFilter(HFRecHitCollection& rec, const HcalChannelQuality* myqual) const {
  if (verboseLevel_ >= 20)
    edm::LogInfo("HFStripFilter") << "runFilter called";

  std::vector<HFRecHit> d1strip;
  std::vector<HFRecHit> d2strip;

  HFRecHit d1max;
  HFRecHit d2max;

  d1max.setEnergy(-10);
  // find d1 and d2 seed hits with max energy and time < timeMax_
  for (auto& it : rec) {
    if (it.time() > timeMax_ || it.time() < 0 || it.energy() < stripThreshold_)
      continue;
    // find HF hit with maximum signal in depth = 1
    if (it.id().depth() == 1) {
      if (it.energy() > d1max.energy() && std::abs(it.id().ieta()) < seedHitIetaMax_) {
        d1max = it;
      }
    }
  }

  bool signStripIeta = false;
  int stripIphiMax = 0;
  int stripIetaMax = 0;
  int stripDepthMax = 0;

  if (d1max.energy() > 0) {
    d1strip.push_back(d1max);
    signStripIeta = std::signbit(d1max.id().ieta());
    stripIphiMax = d1max.id().iphi();
    stripIetaMax = d1max.id().ieta();
    stripDepthMax = d1max.id().depth();
  }

  d2max.setEnergy(-10);
  for (auto& it : rec) {
    if (it.time() > timeMax_ || it.time() < 0 || it.energy() < stripThreshold_)
      continue;
    // find HFhit with maximum signal in depth = 2
    if (it.id().depth() == 2) {
      if (it.energy() > d2max.energy() && std::abs(it.id().ieta()) < seedHitIetaMax_) {
        if (d1max.energy() > 0) {
          bool signIeta = std::signbit(it.id().ieta());
          if (it.id().iphi() == stripIphiMax && signIeta == signStripIeta) {
            d2max = it;
          }
        } else {
          d2max = it;
        }
      }
    }
  }

  if (d2max.energy() > 0)
    d2strip.push_back(d2max);

  // check if at least one of possible seed hits has enough energy
  if (d1max.energy() < maxThreshold_ && d2max.energy() < maxThreshold_)
    return;

  if (d1max.energy() <= 0 && d2max.energy() > 0) {
    signStripIeta = std::signbit(d2max.id().ieta());
    stripIphiMax = d2max.id().iphi();
    stripIetaMax = d2max.id().ieta();
    stripDepthMax = d2max.id().depth();
  }

  std::stringstream ss;
  if (verboseLevel_ >= 30) {
    if (d1max.energy() > 0) {
      ss << "  MaxHit in Depth 1: ieta = " << d1max.id().ieta() << " iphi = " << stripIphiMax
         << " energy = " << d1max.energy() << " time = " << d1max.time() << std::endl;
    }
    if (d2max.energy() > 0) {
      ss << "  MaxHit in Depth 2: ieta = " << d2max.id().ieta() << " iphi = " << stripIphiMax
         << " energy = " << d2max.energy() << " time = " << d2max.time() << std::endl;
    }
    ss << "  stripThreshold_ = " << stripThreshold_ << std::endl;
  }

  // prepare the strips: all hits along given ieta in one wedge (d1strip and d2strip)
  //---------------------------------------------------------------------------------
  for (auto& it : rec) {
    if (it.energy() < stripThreshold_)
      continue;
    bool signIeta = std::signbit(it.id().ieta());

    if (verboseLevel_ >= 30) {
      ss << " HF hit: ieta = " << it.id().ieta() << "\t iphi = " << it.id().iphi() << "\t depth = " << it.id().depth()
         << "\t time = " << it.time() << "\t energy = " << it.energy() << "\t flags = " << it.flags() << std::endl;
    }

    // collect hits with the same iphi but different ieta into strips
    if (it.id().iphi() == stripIphiMax && signIeta == signStripIeta && it.time() < maxStripTime_) {
      if (it.id().depth() == 1) {
        // check if hit = it is already in d1strip
        bool pass = false;
        if (d1strip.empty()) {
          if (std::abs(it.id().iphi() - stripIetaMax) <= gap_) {
            d1strip.push_back(it);
            pass = true;
          }
        } else {
          for (auto& it1 : d1strip) {
            if (it.id().ieta() == it1.id().ieta() && it.energy() == it1.energy()) {
              pass = true;
              break;
            }
          }
        }
        if (pass)
          continue;
        // add hit = it to d1strip if distance to the closest hit in d1strip <= gap_
        for (auto& it1 : d1strip) {
          // check distance along Ieta to the closest hit
          if (std::abs(it1.id().ieta() - it.id().ieta()) <= gap_) {
            d1strip.push_back(it);
            break;
          }
        }
      } else if (it.id().depth() == 2) {
        // check if hit = it is already in d2strip
        bool pass = false;
        if (d2strip.empty()) {
          if (std::abs(it.id().ieta() - stripIetaMax) <= gap_) {
            d2strip.push_back(it);
            pass = true;
          }
        } else {
          for (auto& it1 : d2strip) {
            if (it.id().ieta() == it1.id().ieta() && it.energy() == it1.energy()) {
              pass = true;
              break;
            }
          }
        }
        if (pass)
          continue;

        // add hit = it to d2strip if distance to the closest hit in d1strip <= gap_
        for (auto& it1 : d2strip) {
          // check distance along Ieta to the closest hit
          if (std::abs(it1.id().ieta() - it.id().ieta()) <= gap_) {
            d2strip.push_back(it);
            break;
          }
        }
      }
    }
  }

  if (verboseLevel_ >= 30) {
    ss << " Lstrip1 = " << (int)d1strip.size() << " (iphi = " << stripIphiMax << ")  Lstrip2 = " << (int)d2strip.size()
       << std::endl
       << " Strip1: ";
    for (auto& it1 : d1strip) {
      ss << it1.energy() << " (" << it1.id().ieta() << ") ";
    }
    ss << std::endl << " Strip2: ";
    for (auto& it1 : d2strip) {
      ss << it1.energy() << " (" << it1.id().ieta() << ") ";
    }
    ss << std::endl;
  }

  // check if at least one of strips in depth1 or depth2 is not less than lstrips_
  if ((int)d1strip.size() < lstrips_ && (int)d2strip.size() < lstrips_) {
    if (verboseLevel_ >= 30) {
      LogDebug("HFSFilter") << ss.str();
    }
    return;
  }

  // define range of strips in ieta
  int ietaMin1 = 1000;  // for d1strip
  int ietaMax1 = -1000;
  for (auto& it1 : d1strip) {
    if (it1.id().ieta() < ietaMin1)
      ietaMin1 = it1.id().ieta();
    if (it1.id().ieta() > ietaMax1)
      ietaMax1 = it1.id().ieta();
  }
  int ietaMin2 = 1000;  // for d2strip
  int ietaMax2 = -1000;
  for (auto& it1 : d2strip) {
    if (it1.id().ieta() < ietaMin2)
      ietaMin2 = it1.id().ieta();
    if (it1.id().ieta() > ietaMax2)
      ietaMax2 = it1.id().ieta();
  }

  // define ietamin and ietamax - common area for d1strip and d2strip
  int ietaMin = ietaMin1;
  int ietaMax = ietaMax1;

  if (ietaMin2 >= ietaMin1 && ietaMin2 <= ietaMax1) {
    if (ietaMax2 > ietaMax1)
      ietaMax = ietaMax2;
  } else if (ietaMin2 <= ietaMin1 && ietaMax2 >= ietaMin1) {
    if (ietaMin2 < ietaMin1)
      ietaMin = ietaMin2;
    if (ietaMax2 > ietaMax1)
      ietaMax = ietaMax2;
  }

  // calculate the total energy in strips
  double eStrip = 0;
  for (auto& it1 : d1strip) {
    if (it1.id().ieta() < ietaMin || it1.id().ieta() > ietaMax)
      continue;
    eStrip += it1.energy();
  }
  for (auto& it1 : d2strip) {
    if (it1.id().ieta() < ietaMin || it1.id().ieta() > ietaMax)
      continue;
    eStrip += it1.energy();
  }

  if (verboseLevel_ >= 30) {
    ss << " ietaMin1 = " << ietaMin1 << "  ietaMax1 = " << ietaMax1 << std::endl
       << " ietaMin2 = " << ietaMin2 << "  ietaMax2 = " << ietaMax2 << std::endl
       << " Common strip:  ietaMin = " << ietaMin << "  ietaMax = " << ietaMax << std::endl;
  }

  // energies in the neighboring wedges
  double energyIphi1 = 0;
  double energyIphi2 = 0;
  int iphi1 = 0, iphi2 = 0;

  // get information about the neighboring wedges from HcalTopology
  HcalDetId neighbour;

  for (auto& it : rec) {
    if (it.energy() < stripThreshold_)
      continue;
    if (it.id().ieta() < ietaMin || it.id().ieta() > ietaMax)
      continue;
    HcalDetId id(HcalForward, it.id().ieta(), stripIphiMax, stripDepthMax);
    bool neigh = myqual->topo()->decIPhi(id, neighbour);
    iphi1 = (neigh) ? neighbour.iphi() : 0;
    neigh = myqual->topo()->incIPhi(id, neighbour);
    iphi2 = (neigh) ? neighbour.iphi() : 0;
    if (it.id().iphi() == iphi1)
      energyIphi1 += it.energy();  // Energy iphi1
    else if (it.id().iphi() == iphi2)
      energyIphi2 += it.energy();  // Energy iphi2
  }

  double ratio1 = eStrip > 0 ? energyIphi1 / eStrip : 0;
  double ratio2 = eStrip > 0 ? energyIphi2 / eStrip : 0;

  if (verboseLevel_ >= 30) {
    ss << "  iphi = " << stripIphiMax << "  iphi1 = " << iphi1 << "  iphi2 = " << iphi2 << std::endl
       << "  Estrip = " << eStrip << "  EnergyIphi1 = " << energyIphi1 << "  Ratio = " << ratio1 << std::endl
       << "                  "
       << "  EnergyIphi2 = " << energyIphi2 << "  Ratio = " << ratio2 << std::endl;
  }

  // check if our wedge does not have substantial leak into adjacent wedges
  if (ratio1 < wedgeCut_ && ratio2 < wedgeCut_) {  // noise event with strips (d1 and/or d2)

    if (verboseLevel_ >= 30) {
      ss << "  stripPass = false" << std::endl << "  mark hits in strips now " << std::endl;
    }

    // Figure out which rechits need to be tagged (d1strip)
    for (auto& it1 : d1strip) {
      if (it1.id().ieta() < ietaMin || it1.id().ieta() > ietaMax)
        continue;
      HFRecHitCollection::iterator hit = rec.find(it1.id());
      if (hit != rec.end()) {
        // tag a rechit with the anomalous hit flag
        if (verboseLevel_ >= 30) {
          ss << " d1strip: marked hit = " << (*hit) << std::endl;
        }
        hit->setFlagField(1U, HcalPhase1FlagLabels::HFAnomalousHit);
      }
    }

    // Figure out which rechits need to be tagged (dstrip)
    for (auto& it1 : d2strip) {
      if (it1.id().ieta() < ietaMin || it1.id().ieta() > ietaMax)
        continue;
      HFRecHitCollection::iterator hit = rec.find(it1.id());
      if (hit != rec.end()) {
        // tag a rechit with the anomalous hit flag
        if (verboseLevel_ >= 30) {
          ss << " d2strip: marked hit = " << (*hit) << std::endl;
        }
        hit->setFlagField(1U, HcalPhase1FlagLabels::HFAnomalousHit);
      }
    }
    if (verboseLevel_ >= 30) {
      LogDebug("HFSFilter") << ss.str();
    }
  } else {
    if (verboseLevel_ >= 30) {
      ss << "  stripPass = true" << std::endl;
      LogDebug("HFSFilter") << ss.str();
    }
  }
}

std::unique_ptr<HFStripFilter> HFStripFilter::parseParameterSet(const edm::ParameterSet& ps) {
  return std::make_unique<HFStripFilter>(ps.getParameter<double>("stripThreshold"),
                                         ps.getParameter<double>("maxThreshold"),
                                         ps.getParameter<double>("timeMax"),
                                         ps.getParameter<double>("maxStripTime"),
                                         ps.getParameter<double>("wedgeCut"),
                                         ps.getParameter<int>("seedHitIetaMax"),
                                         ps.getParameter<int>("gap"),
                                         ps.getParameter<int>("lstrips"),
                                         ps.getUntrackedParameter<int>("verboseLevel"));
}

edm::ParameterSetDescription HFStripFilter::fillDescription() {
  edm::ParameterSetDescription desc;

  desc.add<double>("stripThreshold", 40.0)->setComment(" threshold to include hits into strips");
  desc.add<double>("maxThreshold", 100.0)->setComment(" threshold for seed hits in the strips (depth1 and depth2)");
  desc.add<double>("timeMax", 6.0)->setComment(" seed hits should have time < timeMax");
  desc.add<double>("maxStripTime", 10.0)->setComment(" maximum time for hits in the strips");
  desc.add<double>("wedgeCut", 0.05)->setComment(" the possible level of energy leak into adjacent wedges");
  desc.add<int>("seedHitIetaMax", 35)->setComment(" maximum possible Ieta value for seed hit");
  desc.add<int>("gap", 2)->setComment(" maximum distance between hits in the strip (along Ieta direction)");
  desc.add<int>("lstrips", 2)->setComment(" at least one of strips in depth1 or depth2 is not less than lstrips");
  desc.addUntracked<int>("verboseLevel", 0)
      ->setComment(" verboseLevel for debugging printouts, should be > 20 to get output");

  return desc;
}
