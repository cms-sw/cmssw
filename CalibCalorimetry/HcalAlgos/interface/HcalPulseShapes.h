#ifndef CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H
#define CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H 1

#include <vector>
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

/** \class HcalPulseShapes
  *  
  * $Date: 2011/11/23 13:48:27 $
  * $Revision: 1.6 $
  * \author J. Mans - Minnesota
  */
class HcalMCParams;
class HcalRecoParams;
class HcalTopology;

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
  //  return Shpape for given shapeType.
  const Shape& getShape(int shapeType) const;
  /// automatically figures out which shape to return
  const Shape& shape(const HcalDetId & detId) const;
  const Shape& shapeForReco(const HcalDetId & detId) const;
  /// in case of conditions problems
  const Shape& defaultShape(const HcalDetId & detId) const;
private:
  void computeHPDShape(float, float, float, float, float ,
                       float, float, float, Shape&);
  // void computeHPDShape();
  void computeHFShape();
  void computeSiPMShape();
  Shape hpdShape_, hfShape_, siPMShape_;
  Shape hpdShape_v2, hpdShapeMC_v2;
  Shape hpdShape_v3, hpdShapeMC_v3;
  Shape hpdBV30Shape_v2, hpdBV30ShapeMC_v2;
  const HcalMCParams * theMCParams;
  const HcalTopology * theTopology;
  const HcalRecoParams * theRecoParams;
  typedef std::map<int, const Shape *> ShapeMap;
  ShapeMap theShapes;

};
#endif
