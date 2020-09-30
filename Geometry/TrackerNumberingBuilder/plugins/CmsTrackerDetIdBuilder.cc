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
#include <deque>

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

//std::ofstream outfile("DetIdOLD.log", std::ios::out);
  std::ofstream outfile("DetIdDD4hep.log", std::ios::out);

  std::deque<const GeometricDet*> queue;
  queue.emplace_back(&tracker);

  while (!queue.empty()) {
    const GeometricDet* myDet = queue.front();
    queue.pop_front();
    for (auto& child : myDet->components()) {
      queue.emplace_back(child);
    }

    outfile << " " << std::endl;
    outfile << " " << std::endl;
    outfile << "............................." << std::endl;
    outfile << "myDet->geographicalID() = " << myDet->geographicalId() << std::endl;

    //const auto& found = myDet->name().find(":");
    //outfile << "myDet->name() = " << (found != std::string::npos ? myDet->name().substr(found + 1) : myDet->name()) << std::endl;
    outfile << "myDet->name() = " << myDet->name() << std::endl;
    outfile << "myDet->module->type() = " << std::fixed << std::setprecision(7) << myDet->type() << std::endl;

    outfile << "myDet->module->translation() = " << std::fixed << std::setprecision(7) << myDet->translation()
            << std::endl;
    outfile << "myDet->module->rho() = " << std::fixed << std::setprecision(7) << myDet->rho() << std::endl;

    if (fabs(myDet->rho()) > 0.00001) {
      outfile << "myDet->module->phi() = " << std::fixed << std::setprecision(7) << myDet->phi() << std::endl;
    }

    outfile << "myDet->module->rotation() = " << std::fixed << std::setprecision(7) << myDet->rotation() << std::endl;
    outfile << "myDet->module->shape() = " << std::fixed << std::setprecision(7) << myDet->shape() << std::endl;

    if (myDet->shape_dd4hep() == cms::DDSolidShape::ddbox || myDet->shape_dd4hep() == cms::DDSolidShape::ddtrap ||
        myDet->shape_dd4hep() == cms::DDSolidShape::ddtubs) {
      outfile << "myDet->params() = " << std::fixed << std::setprecision(7);
      for (const auto& para : myDet->params()) {
        outfile << para << "  ";
      }
      outfile << " " << std::endl;
    }

    //outfile << "myDet->radLength() = " << myDet->radLength() << std::endl;
    //outfile << "myDet->xi() = " << myDet->xi() << std::endl;
    outfile << "myDet->pixROCRows() = " << myDet->pixROCRows() << std::endl;
    outfile << "myDet->pixROCCols() = " << myDet->pixROCCols() << std::endl;
    outfile << "myDet->pixROCx() = " << myDet->pixROCx() << std::endl;
    outfile << "myDet->pixROCy() = " << myDet->pixROCy() << std::endl;
    outfile << "myDet->stereo() = " << myDet->stereo() << std::endl;
    outfile << "myDet->isLowerSensor() = " << myDet->isLowerSensor() << std::endl;
    outfile << "myDet->isUpperSensor() = " << myDet->isUpperSensor() << std::endl;
    outfile << "myDet->siliconAPVNum() = " << myDet->siliconAPVNum() << std::endl;
  }
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
