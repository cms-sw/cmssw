//FastSimulation Headers
#include "FastSimulation/ShowerDevelopment/interface/HFShower.h"
//#include "FastSimulation/Utilities/interface/Histos.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

//////////////////////////////////////////////////////////////////////
// What's this?
//#include "FastSimulation/FamosCalorimeters/interface/FASTCalorimeter.h"

#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///////////////////////////////////////////////////////////////
// And This???? Doesn't seem to be needed
// #include "Calorimetry/CaloDetector/interface/CellGeometry.h"

#include <cmath>

// number attempts for transverse distribution if exit on a spec. condition
#define infinity 5000
// debugging flag ( 0, 1, 2, 3)
#define debug 0

using namespace edm;

HFShower::HFShower(const RandomEngineAndDistribution* engine,
                   HDShowerParametrization* myParam,
                   EcalHitMaker* myGrid,
                   HcalHitMaker* myHcalHitMaker,
                   int onECAL,
                   double epart)
    : theParam(myParam), theGrid(myGrid), theHcalHitMaker(myHcalHitMaker), onEcal(onECAL), e(epart), random(engine) {
  // To get an access to constants read in FASTCalorimeter
  //  FASTCalorimeter * myCalorimeter= FASTCalorimeter::instance();

  // Values taken from FamosGeneric/FamosCalorimeter/src/FASTCalorimeter.cc
  lossesOpt = myParam->hsParameters()->getHDlossesOpt();
  nDepthSteps = myParam->hsParameters()->getHDnDepthSteps();
  nTRsteps = myParam->hsParameters()->getHDnTRsteps();
  transParam = myParam->hsParameters()->getHDtransParam();
  eSpotSize = myParam->hsParameters()->getHDeSpotSize();
  depthStep = myParam->hsParameters()->getHDdepthStep();
  criticalEnergy = myParam->hsParameters()->getHDcriticalEnergy();
  maxTRfactor = myParam->hsParameters()->getHDmaxTRfactor();
  balanceEH = myParam->hsParameters()->getHDbalanceEH();
  hcalDepthFactor = myParam->hsParameters()->getHDhcalDepthFactor();

  // Special tr.size fluctuations
  transParam *= (1. + random->flatShoot());

  // Special ad hoc long. extension + some fluctuations
  double depthExt;
  if (e < 50.)
    depthExt = 0.8 * (50. - e) / 50. + 0.3;
  else {
    if (e < 500.)
      depthExt = (500. - e) / 500. * 0.4 - 0.1;
    else
      depthExt = -0.1;
  }
  hcalDepthFactor += depthExt + 0.05 * (2. * random->flatShoot() - 1.);

  // normally 1, in HF - might be smaller to take into account
  // a narrowness of the HF shower (Cherenkov light)
  if (e < 50.)
    transFactor = 0.5 - (50. - e) / 50. * 0.2;
  else
    transFactor = 0.7 - (1000. - e) / 1000. * 0.2;

  // simple protection ...
  if (e < 0)
    e = 0.;

  // Get the Famos Histos pointer
  //  myHistos = FamosHistos::instance();
  //  std::cout << " Hello FamosShower " << std::endl;

  theECALproperties = theParam->ecalProperties();
  theHCALproperties = theParam->hcalProperties();

  double emax = theParam->emax();
  double emid = theParam->emid();
  double emin = theParam->emin();
  double effective = e;

  if (e < emid) {
    theParam->setCase(1);
    // avoid "underflow" below Emin (for parameters calculation only)
    if (e < emin)
      effective = emin;
  } else
    theParam->setCase(2);

  // A bit coarse espot size for HF...
  eSpotSize *= 2.5;
  if (effective > 0.5 * emax) {
    eSpotSize *= 2.;
    if (effective > emax) {
      effective = emax;
      eSpotSize *= 3.;
      depthStep *= 2.;
    }
  }

  if (debug == 2)
    LogDebug("FastCalorimetry") << " HFShower : " << std::endl
                                << "       Energy   " << e << std::endl
                                << "      lossesOpt " << lossesOpt << std::endl
                                << "    nDepthSteps " << nDepthSteps << std::endl
                                << "       nTRsteps " << nTRsteps << std::endl
                                << "     transParam " << transParam << std::endl
                                << "      eSpotSize " << eSpotSize << std::endl
                                << " criticalEnergy " << criticalEnergy << std::endl
                                << "    maxTRfactor " << maxTRfactor << std::endl
                                << "      balanceEH " << balanceEH << std::endl
                                << "hcalDepthFactor " << hcalDepthFactor << std::endl;

  double alpEM1 = theParam->alpe1();
  double alpEM2 = theParam->alpe2();

  double betEM1 = theParam->bete1();
  double betEM2 = theParam->bete2();

  double alpHD1 = theParam->alph1();
  double alpHD2 = theParam->alph2();

  double betHD1 = theParam->beth1();
  double betHD2 = theParam->beth2();

  double part1 = theParam->part1();
  double part2 = theParam->part2();

  aloge = std::log(effective);

  double edpar = (theParam->e1() + aloge * theParam->e2()) * effective;
  double aedep = std::log(edpar);

  if (debug == 2)
    LogDebug("FastCalorimetry") << " HFShower : " << std::endl
                                << "     edpar " << edpar << "   aedep " << aedep << std::endl
                                << "    alpEM1 " << alpEM1 << std::endl
                                << "    alpEM2 " << alpEM2 << std::endl
                                << "    betEM1 " << betEM1 << std::endl
                                << "    betEM2 " << betEM2 << std::endl
                                << "    alpHD1 " << alpHD1 << std::endl
                                << "    alpHD2 " << alpHD2 << std::endl
                                << "    betHD1 " << betHD1 << std::endl
                                << "    betHD2 " << betHD2 << std::endl
                                << "     part1 " << part1 << std::endl
                                << "     part2 " << part2 << std::endl;

  // private members to set
  theR1 = theParam->r1();
  theR2 = theParam->r2();
  theR3 = theParam->r3();

  alpEM = alpEM1 + alpEM2 * aedep;
  tgamEM = tgamma(alpEM);
  betEM = betEM1 - betEM2 * aedep;
  alpHD = alpHD1 + alpHD2 * aedep;
  tgamHD = tgamma(alpHD);
  betHD = betHD1 - betHD2 * aedep;
  part = part1 - part2 * aedep;
  if (part > 1.)
    part = 1.;  // protection - just in case of

  if (debug == 2)
    LogDebug("FastCalorimetry") << " HFShower : " << std::endl
                                << "    alpEM " << alpEM << std::endl
                                << "   tgamEM " << tgamEM << std::endl
                                << "    betEM " << betEM << std::endl
                                << "    alpHD " << alpHD << std::endl
                                << "   tgamHD " << tgamHD << std::endl
                                << "    betHD " << betHD << std::endl
                                << "     part " << part << std::endl;

  if (onECAL) {
    lambdaEM = theParam->ecalProperties()->interactionLength();
    x0EM = theParam->ecalProperties()->radLenIncm();
  } else {
    lambdaEM = 0.;
    x0EM = 0.;
  }
  lambdaHD = theParam->hcalProperties()->interactionLength();
  x0HD = theParam->hcalProperties()->radLenIncm();

  if (debug == 2)
    LogDebug("FastCalorimetry") << " HFShower e " << e << std::endl
                                << "          x0EM = " << x0EM << std::endl
                                << "          x0HD = " << x0HD << std::endl
                                << "         lamEM = " << lambdaEM << std::endl
                                << "         lamHD = " << lambdaHD << std::endl;

  // Starting point of the shower
  // try first with ECAL lambda

  double sum1 = 0.;  // lambda path from the ECAL/HF entrance;
  double sum2 = 0.;  // lambda path from the interaction point;
  double sum3 = 0.;  // x0     path from the interaction point;
  int nsteps = 0;    // full number of longitudinal steps (counter);

  int nmoresteps;  // how many longitudinal steps in addition to
                   // one (if interaction happens there) in ECAL

  if (e < criticalEnergy)
    nmoresteps = 1;
  else
    nmoresteps = nDepthSteps;

  double depthECAL = 0.;
  double depthGAP = 0.;
  double depthGAPx0 = 0.;
  if (onECAL) {
    depthECAL = theGrid->ecalTotalL0();          // ECAL depth segment
    depthGAP = theGrid->ecalHcalGapTotalL0();    // GAP  depth segment
    depthGAPx0 = theGrid->ecalHcalGapTotalX0();  // GAP  depth x0
  }

  double depthHCAL = theGrid->hcalTotalL0();  // HCAL depth segment
  double depthToHCAL = depthECAL + depthGAP;

  //---------------------------------------------------------------------------
  // Depth simulation & various protections, among them
  // if too deep - get flat random in the allowed region
  // if no HCAL material behind - force to deposit in ECAL
  double maxDepth = depthToHCAL + depthHCAL - 1.1 * depthStep;

  double depthStart = std::log(1. / random->flatShoot());  // starting point lambda unts

  if (e < emin) {
    if (debug)
      LogDebug("FastCalorimetry") << " FamosHFShower : e <emin ->  depthStart = 0" << std::endl;
    depthStart = 0.;
  }

  if (depthStart > maxDepth) {
    if (debug)
      LogDebug("FastCalorimetry") << " FamosHFShower : depthStart too big ...   = " << depthStart << std::endl;

    depthStart = maxDepth * random->flatShoot();
    if (depthStart < 0.)
      depthStart = 0.;
    if (debug)
      LogDebug("FastCalorimetry") << " FamosHFShower : depthStart re-calculated = " << depthStart << std::endl;
  }

  if (onECAL && e < emid) {
    if ((depthECAL - depthStart) / depthECAL > 0.2 && depthECAL > depthStep) {
      depthStart = 0.5 * depthECAL * random->flatShoot();
      if (debug)
        LogDebug("FastCalorimetry") << " FamosHFShower : small energy, "
                                    << " depthStart reduced to = " << depthStart << std::endl;
    }
  }

  if (depthHCAL < depthStep) {
    if (debug)
      LogDebug("FastCalorimetry") << " FamosHFShower : depthHCAL  too small ... = " << depthHCAL
                                  << " depthStart -> forced to 0 !!!" << std::endl;
    depthStart = 0.;
    nmoresteps = 0;

    if (depthECAL < depthStep) {
      nsteps = -1;
      LogInfo("FastCalorimetry") << " FamosHFShower : too small ECAL and HCAL depths - "
                                 << " particle is lost !!! " << std::endl;
    }
  }

  if (debug)
    LogDebug("FastCalorimetry") << " FamosHFShower  depths(lam) - " << std::endl
                                << "          ECAL = " << depthECAL << std::endl
                                << "           GAP = " << depthGAP << std::endl
                                << "          HCAL = " << depthHCAL << std::endl
                                << "  starting point = " << depthStart << std::endl;

  if (onEcal) {
    if (debug)
      LogDebug("FastCalorimetry") << " FamosHFShower : onECAL" << std::endl;
    if (depthStart < depthECAL) {
      if (debug)
        LogDebug("FastCalorimetry") << " FamosHFShower : depthStart < depthECAL" << std::endl;
      if ((depthECAL - depthStart) / depthECAL > 0.25 && depthECAL > depthStep) {
        if (debug)
          LogDebug("FastCalorimetry") << " FamosHFShower : enough space to make ECAL step" << std::endl;
        //  ECAL - one step
        nsteps++;
        sum1 += depthECAL;  // at the end of step
        sum2 += depthECAL - depthStart;
        sum3 += sum2 * lambdaEM / x0EM;
        lamtotal.push_back(sum1);
        lamdepth.push_back(sum2);
        lamcurr.push_back(lambdaEM);
        lamstep.push_back(depthECAL - depthStart);
        x0depth.push_back(sum3);
        x0curr.push_back(x0EM);
        detector.push_back(1);

        if (debug)
          LogDebug("FastCalorimetry") << " FamosHFShower : "
                                      << " in ECAL sum1, sum2 " << sum1 << " " << sum2 << std::endl;

        //                           // Gap - no additional step after ECAL
        //                           // just move further to HCAL over the gap
        sum1 += depthGAP;
        sum2 += depthGAP;
        sum3 += depthGAPx0;
      }
      // Just shift starting point to HCAL
      else {
        //	cout << " FamosHFShower : not enough space to make ECAL step" << std::endl;
        if (debug)
          LogDebug("FastCalorimetry") << " FamosHFShower : goto HCAL" << std::endl;

        depthStart = depthToHCAL;
        sum1 += depthStart;
      }
    } else {  // GAP or HCAL

      if (depthStart >= depthECAL && depthStart < depthToHCAL) {
        depthStart = depthToHCAL;  // just a shift to HCAL for simplicity
      }
      sum1 += depthStart;

      if (debug)
        LogDebug("FastCalorimetry") << " FamosHFShower : goto HCAL" << std::endl;
    }
  } else {  // Forward
    if (debug)
      LogDebug("FastCalorimetry") << " FamosHFShower : forward" << std::endl;
    sum1 += depthStart;
  }

  for (int i = 0; i < nmoresteps; i++) {
    sum1 += depthStep;
    if (sum1 > (depthECAL + depthGAP + depthHCAL))
      break;
    sum2 += depthStep;
    sum3 += sum2 * lambdaHD / x0HD;
    lamtotal.push_back(sum1);
    lamdepth.push_back(sum2);
    lamcurr.push_back(lambdaHD);
    lamstep.push_back(depthStep);
    x0depth.push_back(sum3);
    x0curr.push_back(x0HD);
    detector.push_back(3);
    nsteps++;
  }

  // Make fractions of energy and transverse radii at each step

  // PV
  //  std::cout << "HFShower::HFShower() : Nsteps = " << nsteps << std::endl;

  if (nsteps > 0) {
    makeSteps(nsteps);
  }
}

void HFShower::makeSteps(int nsteps) {
  double sumes = 0.;
  double sum = 0.;
  std::vector<double> temp;

  if (debug)
    LogDebug("FastCalorimetry") << " FamosHFShower::makeSteps - "
                                << " nsteps required : " << nsteps << std::endl;

  int count = 0;
  for (int i = 0; i < nsteps; i++) {
    double deplam = lamdepth[i] - 0.5 * lamstep[i];
    double depx0 = x0depth[i] - 0.5 * lamstep[i] / x0curr[i];
    double x = betEM * depx0;
    double y = betHD * deplam;

    if (debug == 2)
      LogDebug("FastCalorimetry") << " FamosHFShower::makeSteps "
                                  << " - step " << i << "   depx0, x = " << depx0 << ", " << x
                                  << "   deplam, y = " << deplam << ", " << y << std::endl;

    double est = (part * betEM * gam(x, alpEM) * lamcurr[i] / (x0curr[i] * tgamEM) +
                  (1. - part) * betHD * gam(y, alpHD) / tgamHD) *
                 lamstep[i];

    // protection ...
    if (est < 0.) {
      LogDebug("FastCalorimetry") << "*** FamosHFShower::makeSteps "
                                  << " - negative step energy !!!" << std::endl;
      est = 0.;
      break;
    }

    // for estimates only
    sum += est;
    int nPest = (int)(est * e / sum / eSpotSize);

    if (debug == 2)
      LogDebug("FastCalorimetry") << " FamosHFShower::makeSteps - nPoints estimate = " << nPest << std::endl;

    if (nPest <= 1 && count != 0)
      break;

    // good step - to proceed

    temp.push_back(est);
    sumes += est;

    rlamStep.push_back(transParam * (theR1 + (theR2 - theR3 * aloge)) * deplam * transFactor);
    count++;
  }

  // fluctuations in ECAL and re-distribution of remaining energy in HCAL
  if (detector[0] == 1 && count > 1) {
    double oldECALenergy = temp[0];
    double oldHCALenergy = sumes - oldECALenergy;
    double newECALenergy = 2. * sumes;
    for (int i = 0; newECALenergy > sumes && i < infinity; i++)
      newECALenergy = 2. * balanceEH * random->flatShoot() * oldECALenergy;

    if (debug == 2)
      LogDebug("FastCalorimetry") << "*** FamosHFShower::makeSteps "
                                  << " ECAL fraction : old/new - " << oldECALenergy / sumes << "/"
                                  << newECALenergy / sumes << std::endl;

    temp[0] = newECALenergy;
    double newHCALenergy = sumes - newECALenergy;
    double newHCALreweight = newHCALenergy / oldHCALenergy;

    for (int i = 1; i < count; i++) {
      temp[i] *= newHCALreweight;
    }
  }

  // final re-normalization of the energy fractions
  for (int i = 0; i < count; i++) {
    eStep.push_back(temp[i] * e / sumes);
    nspots.push_back((int)(eStep[i] / eSpotSize) + 1);

    if (debug)
      LogDebug("FastCalorimetry") << i << "  xO and lamdepth at the end of step = " << x0depth[i] << " " << lamdepth[i]
                                  << "   Estep func = " << eStep[i] << "   Rstep = " << rlamStep[i]
                                  << "  Nspots = " << nspots[i] << std::endl;
  }

  // The only step is in ECAL - let's make the size bigger ...
  if (count == 1 and detector[0] == 1)
    rlamStep[0] *= 2.;

  if (debug) {
    if (eStep[0] > 0.95 * e && detector[0] == 1)
      LogDebug("FastCalorimetry") << " FamosHFShower::makeSteps - "
                                  << "ECAL energy = " << eStep[0] << " out of total = " << e << std::endl;
  }
}

bool HFShower::compute() {
  //  TimeMe theT("FamosHFShower::compute");

  bool status = false;
  int numLongit = eStep.size();
  if (debug)
    LogDebug("FastCalorimetry") << " FamosHFShower::compute - "
                                << " N_long.steps required : " << numLongit << std::endl;

  if (numLongit > 0) {
    status = true;
    // Prepare the trsanverse probability function
    std::vector<double> Fhist;
    std::vector<double> rhist;
    for (int j = 0; j < nTRsteps + 1; j++) {
      rhist.push_back(maxTRfactor * j / nTRsteps);
      Fhist.push_back(transProb(maxTRfactor, 1., rhist[j]));
      if (debug == 3)
        LogDebug("FastCalorimetry") << "indexFinder - i, Fhist[i] = " << j << " " << Fhist[j] << std::endl;
    }

    //================================================================
    // Longitudinal steps
    //================================================================
    for (int i = 0; i < numLongit; i++) {
      double currentDepthL0 = lamtotal[i] - 0.5 * lamstep[i];
      // vary the longitudinal profile if needed
      if (detector[i] != 1)
        currentDepthL0 *= hcalDepthFactor;
      if (debug)
        LogDebug("FastCalorimetry") << " FamosHFShower::compute - detector = " << detector[i]
                                    << "  currentDepthL0 = " << currentDepthL0 << std::endl;

      double maxTRsize = maxTRfactor * rlamStep[i];  // in lambda units
      double rbinsize = maxTRsize / nTRsteps;
      double espot = eStep[i] / (double)nspots[i];  // re-adjust espot

      if (espot > 4. || espot < 0.)
        LogDebug("FastCalorimetry") << " FamosHFShower::compute - unphysical espot = " << espot << std::endl;

      int ecal = 0;
      if (detector[i] != 1) {
        bool setHDdepth = theHcalHitMaker->setDepth(currentDepthL0);

        if (debug)
          LogDebug("FastCalorimetry") << " FamosHFShower::compute - status of "
                                      << " theHcalHitMaker->setDepth(currentDepthL0) is " << setHDdepth << std::endl;

        if (!setHDdepth) {
          currentDepthL0 -= lamstep[i];
          setHDdepth = theHcalHitMaker->setDepth(currentDepthL0);
        }
        if (!setHDdepth)
          continue;

        theHcalHitMaker->setSpotEnergy(espot);
      } else {
        ecal = 1;
        bool status = theGrid->getPads(currentDepthL0);

        if (debug)
          LogDebug("FastCalorimetry") << " FamosHFShower::compute - status of Grid = " << status << std::endl;

        if (!status)
          continue;

        theGrid->setSpotEnergy(espot);
      }

      //------------------------------------------------------------
      // Transverse distribution
      //------------------------------------------------------------
      int nok = 0;  // counter of OK
      int count = 0;
      int inf = infinity;
      if (lossesOpt)
        inf = nspots[i];  // losses are enabled, otherwise
      // only OK points are counted ...

      // Total energy in this layer
      double eremaining = eStep[i];
      bool converged = false;

      while (eremaining > 0. && !converged && count < inf) {
        ++count;

        // energy spot  (HFL)
        double newespot = espot;

        // We need to know a priori if this energy spot if for
        // a long (1) or short (2) fiber

        unsigned layer = 1;
        if (currentDepthL0 < 1.3)  // first 22 cm = 1.3 lambda - only HFL
          layer = 1;
        else
          layer = random->flatShoot() < 0.5 ? 1 : 2;

        if (layer == 2)
          newespot = 2. * espot;

        if (eremaining - newespot < 0.)
          newespot = eremaining;

        // process transverse distribution

        double prob = random->flatShoot();
        int index = indexFinder(prob, Fhist);
        double radius = rlamStep[i] * rhist[index] + random->flatShoot() * rbinsize;  // in-bin
        double phi = 2. * M_PI * random->flatShoot();

        if (debug == 2)
          LogDebug("FastCalorimetry") << std::endl
                                      << " FamosHFShower::compute "
                                      << " r = " << radius << "    phi = " << phi << std::endl;

        // add hit

        theHcalHitMaker->setSpotEnergy(newespot);
        theGrid->setSpotEnergy(newespot);

        bool result;
        if (ecal) {
          result = theGrid->addHit(radius, phi, 0);  // shouldn't get here !

          if (debug == 2)
            LogDebug("FastCalorimetry") << " FamosHFShower::compute - "
                                        << " theGrid->addHit result = " << result << std::endl;

        } else {
          // PV assign espot to long/short fibers
          result = theHcalHitMaker->addHit(radius, phi, layer);

          if (debug == 2)
            LogDebug("FastCalorimetry") << " FamosHFShower::compute - "
                                        << " theHcalHitMaker->addHit result = " << result << std::endl;
        }

        if (result) {
          ++nok;
          eremaining -= newespot;
          if (eremaining <= 0.)
            converged = true;
          //	  std::cout << "Transverse : " << nok << " "
          //		    << " , E= "     << newespot
          //		    << " , Erem = " << eremaining
          //		    << std::endl;
        } else {
          //	  std::cout << "WARNING : hit not added" << std::endl;
        }
      }

      // end of tranverse simulation
      //-----------------------------------------------------

      if (count == infinity) {
        status = false;
        if (debug)
          LogDebug("FastCalorimetry") << "*** FamosHFShower::compute "
                                      << " maximum number of"
                                      << " transverse points " << count << " is used !!!" << std::endl;
        break;
      }

      if (debug)
        LogDebug("FastCalorimetry") << " FamosHFShower::compute "
                                    << " long.step No." << i << "   Ntry, Nok = " << count << " " << nok << std::endl;

    }  // end of longitudinal steps
  }    // end of no steps
  return status;
}

int HFShower::indexFinder(double x, const std::vector<double>& Fhist) {
  // binary search in the vector of doubles
  int size = Fhist.size();

  int curr = size / 2;
  int step = size / 4;
  int iter;
  int prevdir = 0;
  int actudir = 0;

  for (iter = 0; iter < size; iter++) {
    if (curr >= size || curr < 1)
      LogWarning("FastCalorimetry") << " FamosHFShower::indexFinder - wrong current index = " << curr << " !!!"
                                    << std::endl;

    if ((x <= Fhist[curr]) && (x > Fhist[curr - 1]))
      break;
    prevdir = actudir;
    if (x > Fhist[curr]) {
      actudir = 1;
    } else {
      actudir = -1;
    }
    if (prevdir * actudir < 0) {
      if (step > 1)
        step /= 2;
    }
    curr += actudir * step;
    if (curr > size)
      curr = size;
    else {
      if (curr < 1) {
        curr = 1;
      }
    }

    if (debug == 3)
      LogDebug("FastCalorimetry") << " indexFinder - end of iter." << iter << " curr, F[curr-1], F[curr] = " << curr
                                  << " " << Fhist[curr - 1] << " " << Fhist[curr] << std::endl;
  }

  if (debug == 3)
    LogDebug("FastCalorimetry") << " indexFinder x = " << x << "  found index = " << curr - 1 << std::endl;

  return curr - 1;
}
