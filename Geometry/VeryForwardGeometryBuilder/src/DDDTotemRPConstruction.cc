/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPConstruction.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"

// this might be useful one day
//.#include "Geometry/TrackerNumberingBuilder/interface/ExtractStringFromDDD.h"
//.#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerBuilder.h"
//.#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDetIdBuilder.h"

#include <iostream>

using namespace std;

//----------------------------------------------------------------------------------------------------

DDDTotemRPContruction::DDDTotemRPContruction()
{
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc* DDDTotemRPContruction::construct(const DDCompactView* cpv)
{
  // create DDFilteredView and apply the filter
  DDPassAllFilter filter;
  DDFilteredView fv(*cpv, filter);

  // conversion to DetGeomDesc structure
  // create the root node and recursively propagates through the tree
  // adds IDs
  DetGeomDesc* tracker = new DetGeomDesc(&fv);
  buildDetGeomDesc(&fv, tracker);

  // return the root of the structure
  return tracker;
}

//----------------------------------------------------------------------------------------------------

void DDDTotemRPContruction::buildDetGeomDesc(DDFilteredView *fv, DetGeomDesc *gd)
{
  // try to dive into next level
  if (! fv->firstChild())
    return;

  // loop over siblings in the level
  do {
    // create new DetGeomDesc node and add it to the parent's (gd) list
    DetGeomDesc* newGD = new DetGeomDesc(fv);

    // add ID (only for detectors/sensors)
    if (fv->logicalPart().name().name().compare(DDD_TOTEM_RP_DETECTOR_NAME) == 0)
    {
      const vector<int> &cN = fv->copyNumbers();
      // check size of copy numubers array
      if (cN.size() < 3)
        throw cms::Exception("DDDTotemRPContruction") << "size of copyNumbers for RP_Silicon_Detector is "
          << cN.size() << ". It must be >= 3." << endl;

      // extract information
      const unsigned int A = cN[cN.size() - 3];
      const unsigned int arm = A / 100;
      const unsigned int station = (A % 100) / 10;
      const unsigned int rp = A % 10;
      const unsigned int detector = cN[cN.size() - 1];
      newGD->setGeographicalID(TotemRPDetId(arm, station, rp, detector));
    }

    if (fv->logicalPart().name().name().compare(DDD_TOTEM_RP_PRIMARY_VACUUM_NAME) == 0)
    {
      const uint32_t decRPId = fv->copyno();
      const uint32_t armIdx = (decRPId / 100) % 10;
      const uint32_t stIdx = (decRPId / 10) % 10;
      const uint32_t rpIdx = decRPId % 10;
      
      newGD->setGeographicalID(TotemRPDetId(armIdx, stIdx, rpIdx));
    }

    if (fv->logicalPart().name().name().compare(DDD_CTPPS_DIAMONDS_DETECTOR_NAME) == 0)
    {
      const vector<int>& copy_num = fv->copyNumbers();
      const unsigned int id = copy_num[copy_num.size()-1],
                         arm = copy_num[1]-1,
                         station = 1,
                         rp = 6,
                         plane = ( id / 100 ),
                         channel = id % 100;
      newGD->setGeographicalID( CTPPSDiamondDetId( arm, station, rp, plane, channel ) );
    }

    gd->addComponent(newGD);

    // recursion
    buildDetGeomDesc(fv, newGD);
  } while (fv->nextSibling());

  // go a level up
  fv->parent();
}
