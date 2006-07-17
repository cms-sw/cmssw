#ifndef RecoLocalCalo_EcalRecAlgos_ESRecHitSimAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_ESRecHitSimAlgo_HH

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/Math/interface/Point3D.h"

// ESRecHitSimAlgo author : Chia-Ming, Kuo

class ESRecHitSimAlgo {

 public:

  ESRecHitSimAlgo(int gain, int pedestal, double MIPADC, double MIPkeV);
  ~ESRecHitSimAlgo(){}
  double EvalAmplitude(const ESDataFrame& digi, bool corr) const;
  EcalRecHit reconstruct(const ESDataFrame& digi, bool corr) const;

  void setGeometry(const CaloGeometry * geometry) { theGeometry = geometry; }

 private:

  int gain_;
  double ped_;
  float pw[3];
  double MIPADC_;
  double MIPkeV_;

 protected:

  const CaloGeometry * theGeometry;
 
};

#endif
