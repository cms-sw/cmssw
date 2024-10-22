// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripDelay
// Implementation:
//     <Notes on implementation>
// Original Author:  M. De Mattia
//         Created:  26/10/2010

#include "CalibFormats/SiStripObjects/interface/SiStripDelay.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include <cassert>
#include <sstream>

void SiStripDelay::fillNewDelay(const SiStripBaseDelay &baseDelay,
                                const int sumSign,
                                const std::pair<std::string, std::string> &recordLabelPair) {
  baseDelayVector_.push_back(&baseDelay);
  sumSignVector_.push_back(sumSign);
  recordLabelPair_.push_back(recordLabelPair);
}

float SiStripDelay::getDelay(const uint32_t detId) const {
  std::unordered_map<uint32_t, double>::const_iterator it = delays_.find(detId);
  if (it != delays_.end()) {
    return it->second;
  }
  return 0.;
}

bool SiStripDelay::makeDelay() {
  if (baseDelayVector_.empty()) {
    return false;
  }
  std::vector<const SiStripBaseDelay *>::const_iterator it = baseDelayVector_.begin();
  // Check for consistent size in all baseDelays
  if (baseDelayVector_.size() > 1) {
    for (; it != baseDelayVector_.end() - 1; ++it) {
      if ((*it)->delaysSize() != (*(it + 1))->delaysSize()) {
        std::cout << "makeDelay: Error, size of base delays is different!!" << std::endl;
        return false;
      }
    }
  }

  //   // All checks done, fill the boost::unoredered_map with the first one
  //   (initialization) int sumSignIndex = 0; int sumSign = 0;
  //   std::vector<uint32_t>::const_iterator detIdIt;
  //   for( it = baseDelayVector_.begin(); it != baseDelayVector_.end(); ++it,
  //   ++sumSignIndex ) {
  //     std::vector<uint32_t> detIds;
  //     (*it)->detIds(detIds);
  //     sumSign = sumSignVector_[sumSignIndex];
  //     for( detIdIt = detIds.begin(); detIdIt != detIds.end(); ++detIdIt ) {
  //       // Check if is alread there so that we never rely on the default
  //       initialization boost::unordered_map<uint32_t, double>::iterator
  //       delayIt = delays_.find(*detIdIt); if( delayIt != delays_.end() ) {
  // 	std::cout << "second delay = " << (*it)->delay(*detIdIt)*sumSign <<
  // std::endl; 	delays_[*detIdIt] += (*it)->delay(*detIdIt)*sumSign;
  // std::cout
  // << "Full delay = " << delays_[*detIdIt] << std::endl;
  //       }
  //       else {
  // 	std::cout << "first delay = " << (*it)->delay(*detIdIt)*sumSign <<
  // std::endl; 	delays_[*detIdIt] = (*it)->delay(*detIdIt)*sumSign;
  //       }
  //     }
  //   }

  // All checks done, fill the boost::unoredered_map with the first one
  // (initialization)
  int sumSignIndex = 0;
  int sumSign = sumSignVector_[sumSignIndex];
  it = baseDelayVector_.begin();
  std::vector<uint32_t> detIds;
  (*it)->detIds(detIds);
  std::vector<uint32_t>::const_iterator detIdIt = detIds.begin();
  for (; detIdIt != detIds.end(); ++detIdIt) {
    delays_[*detIdIt] = (*it)->delay(*detIdIt) * sumSign;
  }
  ++it;
  ++sumSignIndex;
  // Fill all the others
  for (; it != baseDelayVector_.end(); ++it, ++sumSignIndex) {
    std::vector<uint32_t> detIds;
    (*it)->detIds(detIds);
    detIdIt = detIds.begin();
    sumSign = sumSignVector_[sumSignIndex];
    for (; detIdIt != detIds.end(); ++detIdIt) {
      // The same detIds should be in both maps, if not don't rely on the
      // default initialization
      std::unordered_map<uint32_t, double>::iterator delayIt = delays_.find(*detIdIt);
      if (delayIt != delays_.end()) {
        delays_[*detIdIt] += (*it)->delay(*detIdIt) * sumSign;
      } else {
        std::cout << "makeDelay: Warning, detId = " << *detIdIt << " not present, summing to 0..." << std::endl;
        std::cout << "This means that the two baseDelay tags have different "
                     "detIds. PLEASE, CHECK THAT THIS IS EXPECTED."
                  << std::endl;
        delays_[*detIdIt] = (*it)->delay(*detIdIt) * sumSign;
      }
    }
  }

  return true;
}

void SiStripDelay::clear() {
  baseDelayVector_.clear();
  sumSignVector_.clear();
  recordLabelPair_.clear();
  delays_.clear();
}

void SiStripDelay::printDebug(std::stringstream &ss, const TrackerTopology * /*trackerTopo*/) const {
  std::unordered_map<uint32_t, double>::const_iterator it = delays_.begin();
  for (; it != delays_.end(); ++it) {
    ss << "detId = " << it->first << " delay = " << it->second << std::endl;
  }
}

void SiStripDelay::printSummary(std::stringstream &ss, const TrackerTopology *trackerTopo) const {
  SiStripDetSummary summaryDelays{trackerTopo};
  std::unordered_map<uint32_t, double>::const_iterator it = delays_.begin();
  for (; it != delays_.end(); ++it) {
    summaryDelays.add(it->first, it->second);
  }
  summaryDelays.print(ss);
}
