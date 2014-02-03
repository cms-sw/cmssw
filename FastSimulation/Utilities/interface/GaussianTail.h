#ifndef FastSimulation_Utilities_GaussianTail_H
#define FastSimulation_Utilities_GaussianTail_H

// Florian Beaudette (LLR).
// 11/09/06 
// Gaussian tail generator. Copied from the GNU Scientific library

class RandomEngine;

class GaussianTail
{
 public:
  GaussianTail(const RandomEngine* engine, 
	       double sigma=1., double threshold=2.);  
  ~GaussianTail();
  inline void setParameters(double sigma, double threshold) 
    {
      sigma_= sigma; 
      threshold_ = threshold;
      s_=threshold_/sigma_;
      ssquare_ = s_ * s_;
    };
  double shoot() const;

 private:

  const RandomEngine* random;

  double sigma_;
  double threshold_;
  double s_;
  double ssquare_;

};

#endif
