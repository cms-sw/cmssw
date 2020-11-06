/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "CalibPPS/AlignmentRelative/interface/AlignmentGeometry.h"
#include "CalibPPS/AlignmentRelative/interface/Utilities.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

const DetGeometry &AlignmentGeometry::get(unsigned int id) const {
  auto it = sensorGeometry.find(id);
  if (it == sensorGeometry.end())
    throw cms::Exception("PPS") << "No geometry available for sensor " << id << ".";

  return it->second;
}

//----------------------------------------------------------------------------------------------------

void AlignmentGeometry::insert(unsigned int id, const DetGeometry &g) { sensorGeometry[id] = g; }

//----------------------------------------------------------------------------------------------------

void AlignmentGeometry::print() const {
  for (const auto &it : sensorGeometry) {
    printId(it.first);

    printf(" z = %+10.4f mm │ shift: x = %+7.3f mm, y = %+7.3f mm │ ", it.second.z, it.second.sx, it.second.sy);

    for (const auto &dit : it.second.directionData) {
      printf("dir%u: %+.3f, %+.3f, %+.3f │ ", dit.first, dit.second.dx, dit.second.dy, dit.second.dz);
    }

    if (CTPPSDetId(it.first).subdetId() == CTPPSDetId::sdTrackingStrip)
      printf("%s", (it.second.isU) ? "U-det" : "V-det");

    printf("\n");
  }
}
