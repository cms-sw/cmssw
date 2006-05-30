#ifndef Geometry_EcalTestBeam_EcalTBHodoscopeGeometry_HH
#define Geometry_EcalTestBeam_EcalTBHodoscopeGeometry_HH

#include <vector>

class EcalTBHodoscopeGeometry {

 public:

  EcalTBHodoscopeGeometry() {};
  ~EcalTBHodoscopeGeometry(){};

  float getFibreLp(const int& plane, const int& fibre) const
    { 
      if (plane < nPlanes_ && fibre < nFibres_ )
	return fibrePos_[plane][fibre].lp;
      else
	return -99999.;
    }
	  
  float getFibreRp(const int& plane, const int& fibre) const
    {
      if (plane < nPlanes_ && fibre < nFibres_ )
	return fibrePos_[plane][fibre].rp;
      else
	return -99999.;
    }
  
  std::vector<int> getFiredFibresInPlane(const float& xtr, const int& plane) const;
  
  int getNPlanes() const 
    {
      return nPlanes_;
    }

  int getNFibres() const 
    {
      return nFibres_;
    }

 private:

  struct fibre_pos {
    float lp, rp;
  };

  static const int nPlanes_=4;
  static const int nFibres_=64;
  static const fibre_pos fibrePos_[nPlanes_][nFibres_];
  
};

#endif
