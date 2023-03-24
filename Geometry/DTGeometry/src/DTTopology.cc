/** \file
 *
 *  \author R. Bellan  - INFN Torino
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"
#include <FWCore/Utilities/interface/Exception.h>

#include <iostream>
//#define EDM_ML_DEBUG

// FIXME now put by hand, check the number!

const float DTTopology::theWidth = 4.2;   // cm
const float DTTopology::theHeight = 1.3;  // cm ...

const float DTTopology::IBeamWingThickness = 0.13;  // cm
const float DTTopology::IBeamWingLength = 0.635;    // cm

const float DTTopology::plateThickness = 0.15;  // aluminium plate:  1.5 mm
const float DTTopology::IBeamThickness = 0.13;  // I-beam thickness: 1.3 mm

DTTopology::DTTopology(int firstWire, int nChannels, float semilenght)
    : theFirstChannel(firstWire), theNChannels(nChannels), theLength(semilenght * 2) {
  theOffSet = Local2DPoint(-theNChannels / 2. * theWidth, -theLength / 2.);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DTGeometry") << "Constructing DTTopology with:" << std::endl << "number of wires = " << theNChannels << ", first wire number = " << theFirstChannel << std::endl << ", width = " << theWidth << ", height = " << theHeight << ", length = " << theLength;
#endif
}

float DTTopology::sensibleWidth() const { return theWidth - IBeamThickness; }

float DTTopology::sensibleHeight() const { return theHeight - plateThickness; }

LocalPoint DTTopology::localPosition(const MeasurementPoint& mp) const {
  return LocalPoint((mp.x() - theFirstChannel) * theWidth + theOffSet.x(), (1 - mp.y()) * theLength + theOffSet.y());
}

LocalError DTTopology::localError(const MeasurementPoint& /*mp*/, const MeasurementError& me) const {
  return LocalError(me.uu() * (theWidth * theWidth), 0, me.vv() * (theLength * theLength));
}

MeasurementPoint DTTopology::measurementPosition(const LocalPoint& lp) const {
  return MeasurementPoint(static_cast<int>((lp.x() - theOffSet.x()) / theWidth + theFirstChannel),
                          1 - (lp.y() - theOffSet.y()) / theLength);
}

MeasurementError DTTopology::measurementError(const LocalPoint& /*lp*/, const LocalError& le) const {
  return MeasurementError(le.xx() / (theWidth * theWidth), 0, le.yy() / (theLength * theLength));
}

int DTTopology::channel(const LocalPoint& lp) const {
  return static_cast<int>((lp.x() - theOffSet.x()) / theWidth + theFirstChannel);
}

// return the x wire position in the layer, starting from its wire number.
float DTTopology::wirePosition(int wireNumber) const {
  if (!isWireValid(wireNumber))  //- (theFirstChannel-1) <= 0. || wireNumber > lastChannel() )
    throw cms::Exception("InvalidWireNumber")
        << "DTTopology::wirePosition:"
        << " Requested wire number: " << wireNumber << " ,but the first wire number is " << theFirstChannel
        << " and the last wire number is " << lastChannel() << std::endl;
  else
    return (wireNumber - (theFirstChannel - 1) - 0.5) * theWidth + theOffSet.x();
}

/*
// return the x wire position in the layer r.f., starting from its wire number.
float DTTopology::wirePosition(int wireNumber) const{
  int layerCoord = theNChannels - wireNumber + theFirstChannel;
  return  (layerCoord - 0.5)*theWidth + theOffSet.x();
}
*/

//New cell geometry
DTTopology::Side DTTopology::onWhichBorder(float x, float /*y*/, float z) const {
  // epsilon = Tolerance to determine if a hit starts/ends on the cell border.
  // Current value comes from CMSIM, where hit position is
  // always ~10um far from surface. For OSCAR the discrepancy is < 1um.
  const float epsilon = 0.0015;  // 15 um

  // with new geometry the cell shape is not rectangular, but is a
  // rectangular with the I-beam "Wing" subtracted.
  // The height of the Wing is 1.0 mm and the length is 6.35 mm: these 4
  // volumens must be taken into account when the border is computed

  Side side = none;

  if (fabs(z) > (sensibleHeight() / 2. - epsilon) ||
      (fabs(x) > (sensibleWidth() / 2. - IBeamWingLength - epsilon) &&
       fabs(z) > (sensibleHeight() / 2. - IBeamWingThickness - epsilon))) {  //FIXME

    if (z > 0.)
      side = zMax;  // This is currently the INNER surface.
    else
      side = zMin;
  }

  else if (fabs(x) > (sensibleWidth() / 2. - epsilon)) {
    if (x > 0.)
      side = xMax;
    else
      side = xMin;
  }  // FIXME: else if ymax, ymin...

  return side;
}

//Old geometry of the DT
DTTopology::Side DTTopology::onWhichBorder_old(float x, float /*y*/, float z) const {
  // epsilon = Tolerance to determine if a hit starts/ends on the cell border.
  // Current value comes from CMSIM, where hit position is
  // always ~10um far from surface. For OSCAR the discrepancy is < 1um.
  const float epsilon = 0.0015;  // 15 um

  Side side = none;

  if (fabs(z) > (sensibleHeight() / 2. - epsilon)) {
    if (z > 0.) {
      side = zMax;  // This is currently the INNER surface.
    } else {
      side = zMin;
    }
  } else if (fabs(x) > (sensibleWidth() / 2. - epsilon)) {
    if (x > 0.) {
      side = xMax;
    } else {
      side = xMin;
    }
  }  // FIXME: else if ymax, ymin...

  return side;
}
