#ifndef RecoLocalTracker_SiPixelRecHits_EtaCorrection_H
#define RecoLocalTracker_SiPixelRecHits_EtaCorrection_H 1

/*
 define 'eta'-corrections for the pixel hit position determination 
 depending on the charge ratio qFirst/(qFirst-qLast) 
*/

// &&& Question from Petar: maybe the methods should be static, as
// &&& the class does not carry any state...

class EtaCorrection 
{
public:
  EtaCorrection(){};
  ~EtaCorrection(){};

  float xEtaShift(const int& size, const float& pitch, 
		  const float& charatio, const float& alpha) const;
  float yEtaShift(const int& size, const float& pitch, 
		  const float& charatio, const float& beta) const;

  // private:
};

#endif
