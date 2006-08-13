#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "CLHEP/Units/PhysicalConstants.h"

// ESRecHitSimAlgo author : Chia-Ming, Kuo

ESRecHitSimAlgo::ESRecHitSimAlgo(int gain, int pedestal, double MIPADC, double MIPkeV) :
  gain_(gain), ped_(pedestal), MIPADC_(MIPADC), MIPkeV_(MIPkeV) 
{

  // pulse height parametrization
  // 0 : old gain in ORCA
  // 1 : low gain for data taking
  // 2 : high gain for calibration
  if (gain_ == 0) {
    pw[0] = -1.12521;
    pw[1] =  0.877968;
    pw[2] =  0.247238;
  }
  else if (gain_ == 1) {
    pw[0] = -0.0772417;
    pw[1] =  0.8168024;
    pw[2] =  0.3857636;
  }
  else if (gain_ == 2) {
    pw[0] = -0.01687177;
    pw[1] =  0.77676196;
    pw[2] =  0.416363;
  }

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Gain "<<gain_<<" Weights : "<<pw[0]<<" "<<pw[1]<<" "<<pw[2];
}

double ESRecHitSimAlgo::EvalAmplitude(const ESDataFrame& digi, bool corr) const {
  
  float energy = 0;
  float adc[3];  

  for (int i=0; i<digi.size(); i++) {
    energy += pw[i]*(digi.sample(i).adc()-ped_);
    LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Digi "<<i<<" ADC counts "<<digi.sample(i).adc()<<" Ped "<<ped_;
    adc[i] = digi.sample(i).adc();
  }
  if (gain_>0) energy *= MIPkeV_/MIPADC_;

  // convert to GeV
  energy /= 1000000.;

  if (corr) {
    DetId detId = digi.id();
    
    const CaloCellGeometry *this_cell = theGeometry->getSubdetectorGeometry(detId)->getGeometry(detId);
    double theta = this_cell->getPosition().theta();

    return energy*fabs(cos(theta));
  }
  else {
    return energy;
  }
}

EcalRecHit ESRecHitSimAlgo::reconstruct(const ESDataFrame& digi, bool corr) const {

  float energy = 0;
  float time = 0;

  energy = EvalAmplitude(digi, corr);

  DetId detId = digi.id();
  const CaloCellGeometry *this_cell = theGeometry->getSubdetectorGeometry(detId)->getGeometry(detId);
  double distance = this_cell->getPosition().mag();
  time = distance * cm / c_light; 

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  return EcalRecHit(digi.id(), energy, time); 
}

