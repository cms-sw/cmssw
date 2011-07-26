#ifndef CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H
#define CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H 1

#include <vector>
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HcalPulseShapes
  *  
  * $Date: 2011/06/23 01:29:25 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
class HcalPulseShapes {
public:
  typedef HcalPulseShape Shape;
  HcalPulseShapes();

  const Shape& hbShape() const { return hpdShape_; }
  const Shape& heShape() const { return hpdShape_; }
  const Shape& hfShape() const { return hfShape_; }
  const Shape& hoShape(bool sipm=false) const { return hpdShape_; }
  /// automatically figures out which shape to return
  const Shape& shape(const HcalDetId & detId) const;
private:
  Shape hpdShape_, hfShape_;
  void computeHPDShape(Shape& s);
  void computeHFShape(Shape& s);
};
#endif
