#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDetIdBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <bitset>

CmsTrackerDetIdBuilder::CmsTrackerDetIdBuilder(const std::vector<int>& detidShifts) : m_detidshifts() {
  if (detidShifts.size() != nSubDet * maxLevels)
    edm::LogError("WrongConfiguration") << "Wrong configuration of TrackerGeometricDetESModule. Vector of "
                                        << detidShifts.size() << " elements provided";
  else {
    for (unsigned int i = 0; i < nSubDet * maxLevels; ++i) {
      m_detidshifts[i] = detidShifts[i];
    }
  }
}

void CmsTrackerDetIdBuilder::buildId(GeometricDet& tracker) {
  LogDebug("BuildingTrackerDetId") << "Starting to build Tracker DetIds";

  DetId t(DetId::Tracker, 0);
  tracker.setGeographicalID(t);
  iterate(tracker, 0, tracker.geographicalId().rawId());
}

void CmsTrackerDetIdBuilder::iterate(GeometricDet& in, int level, unsigned int ID) {
  std::bitset<32> binary_ID(ID);

  // SubDetector (useful to know fron now on, valid only after level 0, where SubDetector is assigned)
  uint32_t mask = (7 << 25);
  uint32_t iSubDet = ID & mask;
  iSubDet = iSubDet >> 25;
  //

  LogTrace("BuildingTrackerDetId") << std::string(2 * level, '-') << "+" << ID << " " << iSubDet << " " << level;

  switch (level) {
      // level 0: special case because it is used to assign the proper detid bits based on the endcap-like subdetector position: +z or -z
    case 0: {
      for (uint32_t i = 0; i < in.components().size(); i++) {
        GeometricDet* component = in.component(i);
        uint32_t iSubDet = component->geographicalId().rawId();
        uint32_t temp = ID;
        temp |= (iSubDet << 25);
        component->setGeographicalID(temp);

        if (iSubDet > 0 && iSubDet <= nSubDet && m_detidshifts[level * nSubDet + iSubDet - 1] >= 0) {
          if (m_detidshifts[level * nSubDet + iSubDet - 1] + 2 < 25)
            temp |= (0 << (m_detidshifts[level * nSubDet + iSubDet - 1] + 2));
          bool negside = component->translation().z() < 0.;
          if (std::abs(component->translation().z()) < 1.)
            negside = component->components().front()->translation().z() <
                      0.;  // needed for subdet like TID which are NOT translated
          LogTrace("BuildingTrackerDetId")
              << "Is negative endcap? " << negside << ", because z translation is " << component->translation().z()
              << " and component z translation is " << component->components().front()->translation().z();
          if (negside) {
            temp |= (1 << m_detidshifts[level * nSubDet + iSubDet - 1]);
          } else {
            temp |= (2 << m_detidshifts[level * nSubDet + iSubDet - 1]);
          }
        }
        component->setGeographicalID(DetId(temp));

        // next level
        iterate(*component, level + 1, (in.components())[i]->geographicalId().rawId());
      }
      break;
    }
      // level 1 to 5
    default: {
      for (uint32_t i = 0; i < in.components().size(); i++) {
        auto component = in.component(i);
        uint32_t temp = ID;

        if (level < maxLevels) {
          if (iSubDet > 0 && iSubDet <= nSubDet && m_detidshifts[level * nSubDet + iSubDet - 1] >= 0) {
            temp |= (component->geographicalId().rawId() << m_detidshifts[level * nSubDet + iSubDet - 1]);
          }
          component->setGeographicalID(temp);
          // next level
          iterate(*component, level + 1, (in.components())[i]->geographicalId().rawId());
        }
      }

      break;
    }
      // level switch ends
  }

  return;
}
