//FastSimulation Headers
#include "FastSimulation/ShowerDevelopment/interface/HDRShower.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"

//CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

////////////////////////////////////////////////////////////////////////////////
// What's this? Doesn't seem to be needed. Maybe Geometry/CaloGeometry/interface/CaloCellGeometry.h?
//#include "Calorimetry/CaloDetector/interface/CellGeometry.h"

// number attempts for transverse distribution if exit on a spec. condition
#define infinity 5000
// debugging flag ( 0, 1, 2, 3)
#define debug 0

using namespace std;

HDRShower::HDRShower(const RandomEngineAndDistribution* engine,
                     HDShowerParametrization* myParam,
                     EcalHitMaker* myGrid,
                     HcalHitMaker* myHcalHitMaker,
                     int onECAL,
                     double epart)
    : theParam(myParam), theGrid(myGrid), theHcalHitMaker(myHcalHitMaker), onEcal(onECAL), e(epart), random(engine) {
  eHDspot = 0.2;
  EsCut = 0.050;
  EcalShift = 0.12;
  nthetaStep = 10;
  thetaStep = 0.5 * M_PI / nthetaStep;

  if (e < 0)
    e = 0.;
  setFuncParam();
}

bool HDRShower::computeShower() {
  if (onEcal) {
    depthECAL = theGrid->ecalTotalL0();        // ECAL depth segment
    depthGAP = theGrid->ecalHcalGapTotalL0();  // GAP  depth segment
  } else
    depthECAL = depthGAP = 0;

  float depthHCAL = theGrid->hcalTotalL0();  // HCAL depth segment

  //  maxDepth   = depthECAL + depthGAP + depthHCAL - 1.0;
  maxDepth = depthECAL + depthHCAL - 0.5;
  depthStart = log(1. / random->flatShoot());  // starting point lambda unts

  if (depthStart > maxDepth) {
    depthStart = maxDepth * random->flatShoot();
    if (depthStart < 0.)
      depthStart = 0.;
  }

  if (depthStart < EcalShift)
    depthStart = EcalShift;

  decal = (depthECAL + depthStart) * 0.5;
  qstatus = false;
  if (decal < depthECAL) {
    qstatus = theGrid->getPads(decal);
    //    if(!qstatus)
    //      cout<<" depth rejected by getQuads(decal="<<decal<<") status="<<qstatus
    //	  <<" depthECAL="<<depthECAL<<endl;
  }

  thetaFunction(nthetaStep);
  int maxLoops = 10000;
  for (int itheta = 0; itheta < nthetaStep; itheta++) {
    float theta, es;
    for (int i = 0; i <= thetaSpots[itheta]; i++) {
      if (i == thetaSpots[itheta])
        es = elastspot[itheta];
      else
        es = eHDspot;
      float loops = 0;
      for (int j = 0; j < maxLoops; j++) {
        theta = (itheta + random->flatShoot()) * thetaStep;
        if (setHit(es, theta))
          break;
        loops++;
      }
    }
  }
  return (true);
}

bool HDRShower::setHit(float espot, float theta) {
  float phi = 2. * M_PI * random->flatShoot();  // temporary: 1st approximation
  float rshower = getR();                       // temporary: 1st approximation

  float d = depthStart + rshower * cos(theta);
  if (d + depthGAP > maxDepth)
    return (false);

  // Commented (F.B) to remove a warning. Not used anywhere ?
  //  bool inHcal = !onEcal || d>depthECAL || !qstatus;
  bool result = false;
  if (!onEcal || d > depthECAL || !qstatus) {  // in HCAL (HF or HB, HE)
    d += depthGAP;
    bool setHDdepth = theHcalHitMaker->setDepth(d);
    if (setHDdepth) {
      theHcalHitMaker->setSpotEnergy(espot);
      result = theHcalHitMaker->addHit(rshower * sin(theta), phi, 0);
    } else
      LogWarning("FastCalorimetry") << " setHit in HCAL failed d=" << d << " maxDepth=" << maxDepth << " onEcal'"
                                    << onEcal << endl;
  } else {
    //    bool status = theGrid->getQuads(d);
    theGrid->setSpotEnergy(espot);
    result = theGrid->addHit(rshower * sin(theta), phi, 0);
  }
  return (result);
}

float HDRShower::getR() {
  float p = random->flatShoot();
  unsigned int i = 1;
  while (rpdf[i] < p && i < R_range - 1) {
    i++;
  }
  float r;
  float dr = rpdf[i] - rpdf[i - 1];
  if (dr != 0.0)
    r = (float(i) + (p - rpdf[i - 1]) / dr) / lambdaHD;
  else
    r = float(i) / lambdaHD;
  return (r);
}

void HDRShower::thetaFunction(int nthetaStep) {
  unsigned int i = 0;
  while (EgridTable[i] < e && i < NEnergyScan - 1) {
    i++;
  }

  float amean, asig, lambda1, lambda1sig, lam21, lam21sig;
  amean = Theta1amp[i];
  asig = Theta1ampSig[i];
  lambda1 = Theta1Lambda[i];
  lambda1sig = Theta1LambdaSig[i];
  lam21 = ThetaLam21[i];
  lam21sig = ThetaLam21Sig[i];
  if (i == 0)
    i = 1;  //extrapolation to the left
  float c = (e - EgridTable[i - 1]) / (EgridTable[i] - EgridTable[i - 1]);

  amean += (Theta1amp[i] - Theta1amp[i - 1]) * c;
  asig += (Theta1ampSig[i] - Theta1ampSig[i - 1]) * c;
  lambda1 += (Theta1Lambda[i] - Theta1Lambda[i - 1]) * c;
  lambda1sig += (Theta1LambdaSig[i] - Theta1LambdaSig[i - 1]) * c;
  lam21 += (ThetaLam21[i] - ThetaLam21[i - 1]) * c;
  lam21sig += (ThetaLam21Sig[i] - ThetaLam21Sig[i - 1]) * c;

  float a = exp(amean + asig * random->gaussShoot());
  float L1 = lambda1 + lambda1sig * random->gaussShoot();
  if (L1 < 0.02)
    L1 = 0.02;
  float L2 = L1 * (lam21 + lam21sig * random->gaussShoot());

  vector<double> pdf;
  pdf.erase(pdf.begin(), pdf.end());
  thetaSpots.erase(thetaSpots.begin(), thetaSpots.end());
  elastspot.erase(elastspot.begin(), elastspot.end());
  double sum = 0;
  for (int it = 0; it < nthetaStep; it++) {
    float theta = it * thetaStep;
    float p = a * exp(L1 * theta) + exp(L2 * theta);
    sum += p;
    pdf.push_back(p);
  }
  float ntot = e / eHDspot;
  float esum = 0;
  for (int it = 0; it < nthetaStep; it++) {
    float fn = ntot * pdf[it] / sum;
    thetaSpots.push_back(int(fn));
    elastspot.push_back((fn - int(fn)) * eHDspot);
  }

  for (int it = 0; it < nthetaStep; it++)
    if (elastspot[it] < EsCut) {
      esum += elastspot[it];
      elastspot[it] = 0;
    }

  float en = esum / EsCut;
  int n = int(en);
  en = esum - n * EsCut;

  for (int ie = 0; ie <= n; ie++) {
    int k = int(nthetaStep * random->flatShoot());
    if (k < 0 || k > nthetaStep - 1)
      k = k % nthetaStep;
    if (ie == n)
      elastspot[k] += en;
    else
      elastspot[k] += EsCut;
  }
}

void HDRShower::setFuncParam() {
  lambdaHD = theParam->hcalProperties()->interactionLength();
  x0HD = theParam->hcalProperties()->radLenIncm();
  if (onEcal)
    lambdaEM = theParam->ecalProperties()->interactionLength();
  else
    lambdaEM = lambdaHD;

  if (debug)
    LogDebug("FastCalorimetry") << "setFuncParam-> lambdaEM=" << lambdaEM << " lambdaHD=" << lambdaHD << endl;

  float _EgridTable[NEnergyScan] = {10, 20, 30, 50, 100, 300, 500};
  float _Theta1amp[NEnergyScan] = {1.57, 2.05, 2.27, 2.52, 2.66, 2.76, 2.76};
  float _Theta1ampSig[NEnergyScan] = {2.40, 1.50, 1.25, 1.0, 0.8, 0.52, 0.52};

  float _Theta1Lambda[NEnergyScan] = {0.086, 0.092, 0.88, 0.80, 0.0713, 0.0536, 0.0536};
  float _Theta1LambdaSig[NEnergyScan] = {0.038, 0.037, 0.027, 0.03, 0.023, 0.018, 0.018};

  float _ThetaLam21[NEnergyScan] = {2.8, 2.44, 2.6, 2.77, 3.16, 3.56, 3.56};
  float _ThetaLam21Sig[NEnergyScan] = {1.8, 0.97, 0.87, 0.77, 0.7, 0.49, 0.49};

  for (int i = 0; i < NEnergyScan; i++) {
    EgridTable[i] = _EgridTable[i];
    Theta1amp[i] = _Theta1amp[i];
    Theta1ampSig[i] = _Theta1ampSig[i];
    Theta1Lambda[i] = _Theta1Lambda[i];
    Theta1LambdaSig[i] = _Theta1LambdaSig[i];
    ThetaLam21[i] = _ThetaLam21[i];
    ThetaLam21Sig[i] = _ThetaLam21Sig[i];
  }

#define lambdafit 15.05
  float R_alfa = -0.0993 + 0.1114 * log(e);
  float R_p = 0.589191 + 0.0463392 * log(e);
  float R_beta_lam = (0.54134 - 0.00011148 * e) / 4.0 * lambdafit;  //was fitted in 4cmbin
  float LamOverX0 = lambdaHD / x0HD;                                // 10.52
  //  int R_range = 100; // 7 lambda
  //  rpdf.erase(rpdf.begin(),rpdf.end());

  rpdf[0] = 0.;
  for (int i = 1; i < R_range; i++) {
    float x = (float(i)) / lambdaHD;
    float r = pow(x, R_alfa) * (R_p * exp(-R_beta_lam * x) + (1 - R_p) * exp(-LamOverX0 * R_beta_lam * x));
    rpdf[i] = r;
    //    rpdf.push_back(r);
  }

  for (int i = 1; i < R_range; i++)
    rpdf[i] += rpdf[i - 1];
  for (int i = 0; i < R_range; i++)
    rpdf[i] /= rpdf[R_range - 1];
}
