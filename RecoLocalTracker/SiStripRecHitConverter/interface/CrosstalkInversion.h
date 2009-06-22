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
  float mu(const unsigned) const;
  const float r;
  const unsigned N;
  float lambda1, lambda2, rmu0;    

 public:
  static std::vector<float> unfold(const std::vector<uint8_t>& q, const float x);

};
#endif
