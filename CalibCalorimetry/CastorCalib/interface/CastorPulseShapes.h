#ifndef CALIBCALORIMETRY_CASTORALGOS_CASTORPULSESHAPES_H
#define CALIBCALORIMETRY_CASTORALGOS_CASTORPULSESHAPES_H 1

#include <vector>

/** \class CastorPulseShapes
  *  
  * \author P. Katsas - Univ. of Athens
  */
class CastorPulseShapes {
public:
  CastorPulseShapes();

  class Shape {
  public:
    Shape();
    void setNBin(int n);
    void setShapeBin(int i, float f);
    float getTpeak() const { return tpeak_; }
    float operator()(double time) const;
    float at(double time) const;
    float integrate(double tmin, double tmax) const;
  private:
    std::vector<float> shape_;
    int nbin_;
    float tpeak_;
  };

  const Shape& castorShape() const { return castorShape_; }

private:
  Shape castorShape_;
  void computeCastorShape(Shape& s);
};
#endif
