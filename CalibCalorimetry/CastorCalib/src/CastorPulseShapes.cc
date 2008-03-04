#include "CalibCalorimetry/CastorCalib/interface/CastorPulseShapes.h"
#include <cmath>

CastorPulseShapes::CastorPulseShapes() {
  //  computeHPDShape(hpdShape_);
  computeCastorShape(castorShape_);
}


void CastorPulseShapes::computeCastorShape(CastorPulseShapes::Shape& sh) {

  //  cout << endl << " ===== computeShapeHF  !!! " << endl << endl;

  const float ts = 3.0;           // time constant in   t * exp(-(t/ts)**2)

  // first create pulse shape over a range of time 0 ns to 255 ns in 1 ns steps
  int nbin = 256;
  sh.setNBin(nbin);
  std::vector<float> ntmp(nbin,0.0);  // 

  int j;
  float norm;

  // CASTOR SHAPE
  norm = 0.0;
  for( j = 0; j < 3 * ts && j < nbin; j++){
    ntmp[j] = ((float)j)*exp(-((float)(j*j))/(ts*ts));
    norm += ntmp[j];
  }
  // normalize pulse area to 1.0
  for( j = 0; j < 3 * ts && j < nbin; j++){
    ntmp[j] /= norm;

    //    cout << " nt [" << j << "] = " <<  ntmp[j] << endl;
    sh.setShapeBin(j,ntmp[j]);
  }
}

CastorPulseShapes::Shape::Shape() {
  nbin_=0;
  tpeak_=0;
}

void CastorPulseShapes::Shape::setNBin(int n) {
  nbin_=n;
  shape_=std::vector<float>(n,0.0f);
}

void CastorPulseShapes::Shape::setShapeBin(int i, float f) {
  if (i>=0 && i<nbin_) shape_[i]=f;
}

float CastorPulseShapes::Shape::operator()(double t) const {
  // shape is in 1 ns steps
  int i=(int)(t+0.5);
  float rv=0;
  if (i>=0 && i<nbin_) rv=shape_[i];
  return rv;
}

float CastorPulseShapes::Shape::at(double t) const {
  // shape is in 1 ns steps
  int i=(int)(t+0.5);
  float rv=0;
  if (i>=0 && i<nbin_) rv=shape_[i];
  return rv;
}

float CastorPulseShapes::Shape::integrate(double t1, double t2) const {
  static const float int_delta_ns = 0.05f; 
  double intval = 0.0;

  for (double t = t1; t < t2; t+= int_delta_ns) {
    float loedge = at(t);
    float hiedge = at(t+int_delta_ns);
    intval += (loedge+hiedge)*int_delta_ns/2.0;
  }

  return (float)intval;
}
