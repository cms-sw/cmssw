#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <vdt/vdtMath.h>

#include<cstdlib>
#include<cstring>
#include<cassert>
#include <iostream>

EcalRecHit::ESFlags ESRecHitSimAlgo::evalAmplitude(float * results, const ESDataFrame& digi, float ped) const {
  
  float energy = 0;
  float adc[3];
  float pw[3];
  pw[0] = w0_;
  pw[1] = w1_;
  pw[2] = w2_;

  for (int i=0; i<digi.size(); i++) {
    energy += pw[i]*(digi.sample(i).adc()-ped);
    LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Digi "<<i<<" ADC counts "<<digi.sample(i).adc()<<" Ped "<<ped;
    //std::cout<<i<<" "<<digi.sample(i).adc()<<" "<<ped<<" "<<pw[i]<<std::endl;
    adc[i] = digi.sample(i).adc() - ped;
  }
  
  EcalRecHit::ESFlags status = EcalRecHit::kESGood;
  if (adc[0] > 20.f) status = EcalRecHit::kESTS13Sigmas; // 14;
  if (adc[1] <= 0 || adc[2] <= 0) status = EcalRecHit::kESTS3Negative; // 10;
  if (adc[0] > adc[1] && adc[0] > adc[2]) status = EcalRecHit::kESTS1Largest; // 8;
  if (adc[2] > adc[1] && adc[2] > adc[0]) status = EcalRecHit::kESTS3Largest; // 9;
  auto r12 = (adc[1] != 0) ? adc[0]/adc[1] : 99.f;
  auto r23 = (adc[2] != 0) ? adc[1]/adc[2] : 99.f;
  if (r12 > ratioCuts_->getR12High()) status = EcalRecHit::kESBadRatioFor12;// 5;
  if (r23 > ratioCuts_->getR23High()) status = EcalRecHit::kESBadRatioFor23Upper; // 6;
  if (r23 < ratioCuts_->getR23Low()) status = EcalRecHit::kESBadRatioFor23Lower; // 7

  auto A1 = adc[1];
  auto A2 = adc[2];

  // t0 from analytical formula:
  constexpr float n = 1.798;
  constexpr float w = 0.07291;
  constexpr float DeltaT = 25.;
  auto aaa = (A2 > 0 && A1 > 0) ? std::log(A2/A1)/n : 20.f; // if A1=0, t0=20
  constexpr float bbb = w/n*DeltaT;
  auto ccc= std::exp(aaa+bbb);

  auto t0 = (2.f-ccc)/(1.f-ccc) * DeltaT - 5.f;

  // A from analytical formula:
  constexpr float t1 = 20.;
  #if defined(__clang__) || defined(__INTEL_COMPILER)
  const float A_1 = 1./( std::pow(w/n*(t1),n) * std::exp(n-w*(t1)) );
  #else
  constexpr float A_1 = 1./( std::pow(w/n*(t1),n) * std::exp(n-w*(t1)) );
  #endif
  auto AA1 = A1 * A_1 ;

 if (adc[1] > 2800.f && adc[2] > 2800.f) status = EcalRecHit::kESSaturated;
  else if (adc[1] > 2800.f) status = EcalRecHit::kESTS2Saturated;
  else if (adc[2] > 2800.f) status = EcalRecHit::kESTS3Saturated;

  results[0] = energy; // energy with weight method
  results[1] = t0;     // timing
  results[2] = AA1;    // energy with analytic method

  return status;

}

EcalRecHit ESRecHitSimAlgo::reconstruct(const ESDataFrame& digi) const {


  auto ind = digi.id().hashedIndex();

  auto const & ped = peds_->preshower(ind);
  auto const & mip = mips_->getMap().preshower(ind);
  auto const & ang = ang_->getMap().preshower(ind);
  auto const & statusCh = channelStatus_->getMap().preshower(ind);

  float results[3];

  auto status = evalAmplitude(results, digi, ped.getMean());

  auto energy   = results[0];
  auto t0       = results[1];
  auto otenergy = results[2] * 1000000.f; // set out-of-time energy to keV
  

  auto mipCalib = (mip != 0.f) ? MIPGeV_*std::abs(vdt::fast_cosf(ang))/(mip) : 0.f;
  energy *= mipCalib;
  otenergy *= mipCalib;

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  EcalRecHit rechit(digi.id(), energy, t0);
  // edm: this is just a placeholder for alternative energy reconstruction,
  // so put it in the same float, with different name
  // rechit.setOutOfTimeEnergy(otenergy);
  rechit.setEnergyError(otenergy);

  rechit.setFlag(statusCh.getStatusCode() == 1 ? EcalRecHit::kESDead : status);

  return rechit;

}

/*

  auto oldHit = oldreconstruct(digi);
  
  assert(rechit.recoFlag()==oldHit.recoFlag());

  if (oldHit.energy()>0) 
  std::cout <<  "ESd " << digi.id() <<" : "
	    << rechit.energy()<<"," << oldHit.energy() << " "
	    << rechit.time()<<"," << oldHit.time()  << " "
	    << rechit.outOfTimeEnergy()<<"," << oldHit.outOfTimeEnergy() << std::endl;
 

  auto bitdiff = [](float a, float b)->int {
    int ia, ib;
    memcpy(&ia,&a,4);
    memcpy(&ib,&b,4);
    return std::abs(ia-ib);
  };

  auto d0 = bitdiff( rechit.energy(), rechit.energy() );
  auto d1 = bitdiff( rechit.time(), rechit.time() );
  auto d2 = bitdiff(rechit.outOfTimeEnergy(), oldHit.outOfTimeEnergy());

  struct aMax {
    ~aMax() { std::cout << "\n\nmax deviation " << m << " " << mp << std::endl; }
    int m=0;
    int mp=0;
  };

  static aMax am;
  am.m = std::max(am.m,std::max(d0,std::max(d1,d2)));
  if (oldHit.energy()>0) am.mp = std::max(am.m,std::max(d0,std::max(d1,d2)));


  return rechit;
}
*/

//

double* ESRecHitSimAlgo::oldEvalAmplitude(const ESDataFrame& digi, const double& ped, const double& w0, const double& w1, const double& w2) const {
  
  double *results = new double[4];
  float energy = 0;
  double adc[3];
  float pw[3];
  pw[0] = w0;
  pw[1] = w1;
  pw[2] = w2;

  for (int i=0; i<digi.size(); i++) {
    energy += pw[i]*(digi.sample(i).adc()-ped);
    LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Digi "<<i<<" ADC counts "<<digi.sample(i).adc()<<" Ped "<<ped;
    //std::cout<<i<<" "<<digi.sample(i).adc()<<" "<<ped<<" "<<pw[i]<<std::endl;
    adc[i] = digi.sample(i).adc() - ped;
  }
  
  double status = 0;
  if (adc[0] > 20) status = 14;
  if (adc[1] <= 0 || adc[2] <= 0) status = 10;
  if (adc[0] > adc[1] && adc[0] > adc[2]) status = 8;
  if (adc[2] > adc[1] && adc[2] > adc[0]) status = 9;
  double r12 = (adc[1] != 0) ? adc[0]/adc[1] : 99;
  double r23 = (adc[2] != 0) ? adc[1]/adc[2] : 99;
  if (r12 > ratioCuts_->getR12High()) status = 5;
  if (r23 > ratioCuts_->getR23High()) status = 6;
  if (r23 < ratioCuts_->getR23Low()) status = 7;

  double A1 = adc[1];
  double A2 = adc[2];

  // t0 from analytical formula:
  double n = 1.798;
  double w = 0.07291;
  double DeltaT = 25.;
  double aaa = (A2 > 0 && A1 > 0) ? log(A2/A1)/n : 20.; // if A1=0, t0=20
  double bbb = w/n*DeltaT;
  double ccc= exp(aaa+bbb);

  double t0 = (2.-ccc)/(1.-ccc) * DeltaT - 5;

  // A from analytical formula:
  double t1 = 20.;
  double A_1 =  pow(w/n*(t1),n) * exp(n-w*(t1));
  double AA1 = (A_1 != 0.) ? A1 / A_1 : 0.;

  if (adc[1] > 2800 && adc[2] > 2800) status = 11;
  else if (adc[1] > 2800) status = 12;
  else if (adc[2] > 2800) status = 13;

  results[0] = energy; // energy with weight method
  results[1] = t0;     // timing
  results[2] = status; // hit status
  results[3] = AA1;    // energy with analytic method

  return results;
}

EcalRecHit ESRecHitSimAlgo::oldreconstruct(const ESDataFrame& digi) const {

  ESPedestals::const_iterator it_ped = peds_->find(digi.id());

  ESIntercalibConstantMap::const_iterator it_mip = mips_->getMap().find(digi.id());
  ESAngleCorrectionFactors::const_iterator it_ang = ang_->getMap().find(digi.id());

  ESChannelStatusMap::const_iterator it_status = channelStatus_->getMap().find(digi.id());

  double* results;

  results = oldEvalAmplitude(digi, it_ped->getMean(), w0_, w1_, w2_);

  double energy   = results[0];
  double t0       = results[1];
  int status      = (int) results[2];
  double otenergy = results[3] * 1000000.; // set out-of-time energy to keV
  delete[] results;

  double mipCalib = (fabs(cos(*it_ang)) != 0.) ? (*it_mip)/fabs(cos(*it_ang)) : 0.;
  energy *= (mipCalib != 0.) ? MIPGeV_/mipCalib : 0.;
  otenergy *= (mipCalib != 0.) ? MIPGeV_/mipCalib : 0.;

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  EcalRecHit rechit(digi.id(), energy, t0);
  // edm: this is just a placeholder for alternative energy reconstruction,
  // so put it in the same float, with different name
  // rechit.setOutOfTimeEnergy(otenergy);
  rechit.setEnergyError(otenergy);

  if (it_status->getStatusCode() == 1) {
    rechit.setFlag(EcalRecHit::kESDead);
  } else {
    if (status == 0)
      rechit.setFlag(EcalRecHit::kESGood);
    else if (status == 5)
      rechit.setFlag(EcalRecHit::kESBadRatioFor12);
    else if (status == 6)
      rechit.setFlag(EcalRecHit::kESBadRatioFor23Upper);
    else if (status == 7)
      rechit.setFlag(EcalRecHit::kESBadRatioFor23Lower);
    else if (status == 8)
      rechit.setFlag(EcalRecHit::kESTS1Largest);
    else if (status == 9)
      rechit.setFlag(EcalRecHit::kESTS3Largest);
    else if (status == 10)
      rechit.setFlag(EcalRecHit::kESTS3Negative);
    else if (status == 11)
      rechit.setFlag(EcalRecHit::kESSaturated);
    else if (status == 12)
      rechit.setFlag(EcalRecHit::kESTS2Saturated);
    else if (status == 13)
      rechit.setFlag(EcalRecHit::kESTS3Saturated);
    else if (status == 14)
      rechit.setFlag(EcalRecHit::kESTS13Sigmas);
  }

  return rechit;
}

