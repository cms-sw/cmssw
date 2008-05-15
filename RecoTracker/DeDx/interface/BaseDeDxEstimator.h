#ifndef BaseDeDxEstimator_h
#define BaseDeDxEstimator_h
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

class BaseDeDxEstimator
{
public: 
 virtual Measurement1D  dedx(std::vector<Measurement1D> ChargeMeasurements) = 0;

 class LessFunc {
 public:
    bool operator () (const Measurement1D& A, const Measurement1D& B) {
              return (A.value() < B.value()); }
 };

};

#endif
