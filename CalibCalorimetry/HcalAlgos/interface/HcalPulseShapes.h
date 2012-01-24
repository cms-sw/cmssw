#ifndef CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H
#define CALIBCALORIMETRY_HCALALGOS_HCALPULSESHAPES_H 1

#include <vector>

/** \class HcalPulseShapes
  *  
  * $Date: 2006/10/27 19:46:53 $
  * $Revision: 1.2 $
  * \author J. Mans - Minnesota
  */
class HcalPulseShapes {
public:
  HcalPulseShapes();

  class Shape {
  public:
    Shape();
    void setNBin(int n);
    void setShapeBin(int i, float f);
    float getTpeak() const { return tpeak_; }
    float operator()(double time) const;
    float at(double time) const;
    float integrate(double tmin, double tmax) const;
    int nbins() const {return nbin_;}
  private:
    std::vector<float> shape_;
    int nbin_;
    float tpeak_;
  };


  const Shape& hbShape() const { return hpdShape_; }
  const Shape& heShape() const { return hpdShape_; }
  const Shape& hfShape() const { return hfShape_; }
  const Shape& hoShape(bool sipm=false) const { return hpdShape_; }

private:
  Shape hpdShape_, hfShape_;
  void computeHPDShape(Shape& s);
  void computeHFShape(Shape& s);
};
#endif
