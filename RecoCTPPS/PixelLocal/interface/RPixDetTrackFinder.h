/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/
#ifndef RecoCTPPS_PixelLocal_RPixDetTrackFinder_H
#define RecoCTPPS_PixelLocal_RPixDetTrackFinder_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "RecoCTPPS/PixelLocal/interface/RPixDetPatternFinder.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include <vector>
#include <map>

class RPixDetTrackFinder {
public:
  RPixDetTrackFinder(edm::ParameterSet const &parameterSet) : romanPotId_(CTPPSPixelDetId(0, 2, 3, 0)) {}

  virtual ~RPixDetTrackFinder(){};

  void setHits(std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane> > *hitMap) {
    hitMap_ = hitMap;
  }
  virtual void findTracks(int run) = 0;
  virtual void initialize() = 0;
  void clear() { localTrackVector_.clear(); }
  std::vector<CTPPSPixelLocalTrack> const &getLocalTracks() const { return localTrackVector_; }
  void setRomanPotId(CTPPSPixelDetId rpId) { romanPotId_ = rpId; };
  void setGeometry(const CTPPSGeometry *geometry) { geometry_ = geometry; }
  void setListOfPlanes(std::vector<uint32_t> listOfAllPlanes) { listOfAllPlanes_ = listOfAllPlanes; }
  void setZ0(double z0) { z0_ = z0; }

protected:
  std::map<CTPPSPixelDetId, std::vector<RPixDetPatternFinder::PointInPlane> > *hitMap_;
  std::vector<CTPPSPixelLocalTrack> localTrackVector_;
  CTPPSPixelDetId romanPotId_;
  const CTPPSGeometry *geometry_;
  uint32_t numberOfPlanesPerPot_;
  std::vector<uint32_t> listOfAllPlanes_;
  double z0_;
};

#endif
