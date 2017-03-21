#ifndef CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H
#define CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H 1

#include <map>
#include <vector>
#include <cmath>
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

/** \class HcalPulseShapes
  *  
  * \author J. Mans - Minnesota
  */
class HcalMCParams;
class HcalRecoParams;
class HcalTopology;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalPulseShapes {
public:
  typedef HcalPulseShape Shape;
  HcalPulseShapes();
  ~HcalPulseShapes();
  // only needed if you'll be getting shapes by DetId
  void beginRun(edm::EventSetup const & es);
  void endRun();

  const Shape& hbShape() const { return hpdShape_; }
  const Shape& heShape() const { return hpdShape_; }
  const Shape& hfShape() const { return hfShape_; }
  const Shape& hoShape(bool sipm=false) const { return sipm ? siPMShape_ : hpdShape_; }
  //  return Shape for given shapeType.
  const Shape& getShape(int shapeType) const;
  /// automatically figures out which shape to return
  const Shape& shape(const HcalDetId & detId) const;
  const Shape& shapeForReco(const HcalDetId & detId) const;
  /// in case of conditions problems
  const Shape& defaultShape(const HcalDetId & detId) const;
  //public static helpers
  static const int nBinsSiPM_ = 250;
  static constexpr float deltaTSiPM_ = 0.5;
  static constexpr float invDeltaTSiPM_ = 2.0;
  static double analyticPulseShapeSiPMHO(double t);
  static double analyticPulseShapeSiPMHE(double t);
  static constexpr float Y11RANGE_ = nBinsSiPM_;
  static constexpr float Y11MAX_ = 0.04;
  static double Y11TimePDF(double t);
  static double generatePhotonTime(CLHEP::HepRandomEngine* engine);
  //this function can take function pointers *or* functors!
  template <class F1, class F2>
  static std::vector<double> convolve(unsigned nbin, F1 f1, F2 f2){
    std::vector<double> result(2*nbin-1,0.);
    for(unsigned i = 0; i < 2*nbin-1; ++i){
      for(unsigned j = 0; j < std::min(i+1,nbin); ++j){
        double tmp = f1(j)*f2(i-j);
        if(std::isnan(tmp) or std::isinf(tmp)) continue;
        result[i] += tmp;
      }
    }
    return result;
  }

private:
  void computeHPDShape(float, float, float, float, float ,
                       float, float, float, Shape&);
  void computeHFShape();
  void computeSiPMShape();
  void computeSiPMShape2017();
  Shape hpdShape_, hfShape_, siPMShape_, siPMShape2017_;
  Shape hpdShape_v2, hpdShapeMC_v2;
  Shape hpdShape_v3, hpdShapeMC_v3;
  Shape hpdBV30Shape_v2, hpdBV30ShapeMC_v2;
  HcalMCParams * theMCParams;
  const HcalTopology * theTopology;
  HcalRecoParams * theRecoParams;
  typedef std::map<int, const Shape *> ShapeMap;
  ShapeMap theShapes;

};
#endif
