#ifndef SiStripRecHitConverter_InverseCrosstalkMatrix_h
#define SiStripRecHitConverter_InverseCrosstalkMatrix_h

#include <vector>
#include <cmath>

class InverseCrosstalkMatrix {
 public:
  InverseCrosstalkMatrix(const unsigned N, const float x);
  float operator()(const unsigned i, const unsigned j) const;
  
 private:
  float element(const unsigned, const unsigned) const;
  const unsigned N;
  const double sq, lambdaP, lambdaM, denominator;

 public:
  static std::vector<float> unfold(const std::vector<uint8_t>& q, const float x);

};
#endif
