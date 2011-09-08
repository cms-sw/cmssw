#ifndef CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H
#define CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H 1

#include <vector>
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

/** \class HcalPulseShapes
  *  
  * $Date: 2011/07/26 21:05:40 $
  * $Revision: 1.4 $
  * \author J. Mans - Minnesota
  */
class HcalMCParams;

class HcalPulseShapes {
public:
  typedef HcalPulseShape Shape;
  HcalPulseShapes();
  ~HcalPulseShapes();
  // only needed if you'll be geting shapes by DetId
  void beginRun(edm::EventSetup const & es);
  void endRun();

  const Shape& hbShape() const { return hpdShape_; }
  const Shape& heShape() const { return hpdShape_; }
  const Shape& hfShape() const { return hfShape_; }
  const Shape& hoShape(bool sipm=false) const { return sipm ? siPMShape_ : hpdShape_; }
  /// automatically figures out which shape to return
  const Shape& shape(const HcalDetId & detId) const;
  /// in case of conditions problems
  const Shape& defaultShape(const HcalDetId & detId) const;
private:
  void computeHPDShape();
  void computeHFShape();
  void computeSiPMShape();
  Shape hpdShape_, hfShape_, siPMShape_;
  const HcalMCParams * theMCParams;
  typedef std::map<int, const Shape *> ShapeMap;
  ShapeMap theShapes;

};
#endif
