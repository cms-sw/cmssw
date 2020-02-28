#ifndef HDRShower_H
#define HDRShower_H

//FastSimulation Headers
#include "FastSimulation/ShowerDevelopment/interface/HDShowerParametrization.h"

#include <vector>

/** 
 * \date: 09-Feb-2005
 * new hadronic shower simulation by V.Popov
 */

#define NEnergyScan 7
// 7 lambda
#define R_range 100

class EcalHitMaker;
class HcalHitMaker;
class RandomEngineAndDistribution;

class HDRShower {
public:
  HDRShower(const RandomEngineAndDistribution* engine,
            HDShowerParametrization* myParam,
            EcalHitMaker* myGrid,
            HcalHitMaker* myHcalHitMaker,
            int onECAL,
            double epart);

  virtual ~HDRShower() { ; }

  bool computeShower();
  bool setHit(float espot, float theta);
  void thetaFunction(int nthetaStep);
  float getR();
  void setFuncParam();

private:
  // Input
  HDShowerParametrization* theParam;
  EcalHitMaker* theGrid;
  HcalHitMaker* theHcalHitMaker;
  int onEcal;
  double e;  // Input energy to distribute

private:
  //  const ECALProperties* theECALproperties;
  //  const HCALProperties* theHCALproperties;

  double lambdaEM, lambdaHD, x0EM, x0HD;
  double depthStart;
  float eHDspot;
  float EsCut;
  float EcalShift;
  int nthetaStep;

  float thetaStep;
  float depthECAL, depthGAP, maxDepth;
  std::vector<int> thetaSpots;
  std::vector<float> elastspot;
  float rpdf[R_range];
  bool qstatus;
  float decal;

  float EgridTable[NEnergyScan];
  float Theta1amp[NEnergyScan];
  float Theta1ampSig[NEnergyScan];
  float Theta1Lambda[NEnergyScan];
  float Theta1LambdaSig[NEnergyScan];
  float ThetaLam21[NEnergyScan];
  float ThetaLam21Sig[NEnergyScan];

  // The famos random engine
  const RandomEngineAndDistribution* random;
};

#endif
