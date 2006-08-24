//FastSimulation Headers
#include "FastSimulation/ShowerDevelopment/interface/HDShower.h"
#include "FastSimulation/Utilities/interface/Histos.h"

//////////////////////////////////////////////////////////////////////
// What's this?
//#include "FastSimulation/FamosCalorimeters/interface/FASTCalorimeter.h"

#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"

//Anaphe headers
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandLandau.h"

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///////////////////////////////////////////////////////////////
// And This???? Doesn't seem to be needed
// #include "Calorimetry/CaloDetector/interface/CellGeometry.h"

#include <math.h>

// number attempts for transverse distribution if exit on a spec. condition
#define infinity 5000
// debugging flag ( 0, 1, 2, 3)
#define debug 0

using namespace std;
using namespace edm;

HDShower::HDShower(HDShowerParametrization* myParam, 
		   EcalHitMaker* myGrid,
		   HcalHitMaker* myHcalHitMaker,
		   int onECAL,
		   double epart)
  : theParam(myParam), 
    theGrid(myGrid),
    theHcalHitMaker(myHcalHitMaker),
    onEcal(onECAL),
    e(epart)
{ 
  // To get an access to constants read in FASTCalorimeter
  //  FASTCalorimeter * myCalorimeter= FASTCalorimeter::instance();

  // Values taken from FamosGeneric/FamosCalorimeter/src/FASTCalorimeter.cc
  lossesOpt      = 0;//myCalorimeter->getHDlossesOpt();
  nDepthSteps    = 10;//myCalorimeter->getHDnDepthSteps();
  nTRsteps       = 40;//myCalorimeter->getHDnTRsteps();
  transParam     = 102;//myCalorimeter->getHDtransParam();
  eSpotSize      = 0.2;//myCalorimeter->getHDeSpotSize();
  depthStep      = 0.5;//myCalorimeter->getHDdepthStep();
  criticalEnergy = 3.0;//myCalorimeter->getHDcriticalEnergy();
  maxTRfactor    = 4;//myCalorimeter->getHDmaxTRfactor();
  balanceEH      = 0.9;//myCalorimeter->getHDbalanceEH();

  // Special tr.size fluctuations 
  transParam *= (1. + RandFlat::shoot());

  transFactor = 1.;   // normally 1, in HF - might be smaller 
                     // to take into account
                     // a narrowness of the HF shower (Cherenkov light) 

  // simple protection ...
  if(e < 0) e = 0.;

  // Get the Famos Histos pointer
  //  myHistos = FamosHistos::instance();
  //  std::cout << " Hello FamosShower " << std::endl;
  
  theECALproperties = theParam->ecalProperties();
  theHCALproperties = theParam->hcalProperties();

  double emax = theParam->emax(); 
  double emid = theParam->emid(); 
  double emin = theParam->emin(); 
  double effective = e;
    

  if( e < emid ) {
    theParam->setCase(1);
    // avoid "underflow" below Emin (for parameters calculation only)
    if(e < emin) effective = emin; 
  } 
  else 
    theParam->setCase(2);
 
  // Avoid "overflow" beyond Emax (for parameters)
  if(effective > 0.5 * emax) {
    eSpotSize *= 2.5;
    if(effective > emax) {
      effective = emax; 
      eSpotSize *= 2.; 
      depthStep *= 2.;
    }
  }


  if(debug == 2 )
    LogDebug("FastCalorimetry") << " HDShower : " << endl 
         << "       Energy   " <<              e << endl     
         << "      lossesOpt " <<      lossesOpt << endl  
         << "    nDepthSteps " <<    nDepthSteps << endl
         << "       nTRsteps " <<       nTRsteps << endl
         << "     transParam " <<     transParam << endl
         << "      eSpotSize " <<      eSpotSize << endl
         << " criticalEnergy " << criticalEnergy << endl
         << "    maxTRfactor " <<    maxTRfactor << endl
         << "      balanceEH " <<      balanceEH << endl;


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

  aloge = log(effective);
 
  double edpar = (theParam->e1() + aloge * theParam->e2()) * effective;
  double aedep = log(edpar);

  if(debug == 2)
    LogDebug("FastCalorimetry") << " HDShower : " << endl
         << "     edpar " <<   edpar << "   aedep " << aedep << endl 
         << "    alpEM1 " <<  alpEM1 << endl  
         << "    alpEM2 " <<  alpEM2 << endl  
         << "    betEM1 " <<  betEM1 << endl  
         << "    betEM2 " <<  betEM2 << endl  
         << "    alpHD1 " <<  alpHD1 << endl  
         << "    alpHD2 " <<  alpHD2 << endl  
         << "    betHD1 " <<  betHD1 << endl  
         << "    betHD2 " <<  betHD2 << endl  
         << "     part1 " <<   part1 << endl  
         << "     part2 " <<   part2 << endl; 

  // private members to set
  theR1  = theParam->r1();
  theR2  = theParam->r2();
  theR3  = theParam->r3();

  alpEM  = alpEM1 + alpEM2 * aedep;
  tgamEM = tgamma(alpEM);
  betEM  = betEM1 - betEM2 * aedep;
  alpHD  = alpHD1 + alpHD2 * aedep;
  tgamHD = tgamma(alpHD);
  betHD  = betHD1 - betHD2 * aedep;
  part   = part1  -  part2 * aedep;
  if(part > 1.) part = 1.;          // protection - just in case of 

  if(debug  == 2 )
    LogDebug("FastCalorimetry") << " HDShower : " << endl 
         << "    alpEM " <<  alpEM << endl  
         << "   tgamEM " << tgamEM << endl
         << "    betEM " <<  betEM << endl
         << "    alpHD " <<  alpHD << endl
         << "   tgamHD " << tgamHD << endl
         << "    betHD " <<  betHD << endl
         << "     part " <<   part << endl;
       

  if(onECAL){
    lambdaEM = theParam->ecalProperties()->interactionLength();
    x0EM     = theParam->ecalProperties()->radLenIncm();
  } 
  else {
    lambdaEM = 0.;
    x0EM     = 0.;
  }
  lambdaHD = theParam->hcalProperties()->interactionLength();
  x0HD     = theParam->hcalProperties()->radLenIncm();

  if(debug == 2)
    LogDebug("FastCalorimetry") << " HDShower e " << e        << endl
         << "          x0EM = " << x0EM     << endl 
         << "          x0HD = " << x0HD     << endl 
         << "         lamEM = " << lambdaEM << endl
         << "         lamHD = " << lambdaHD << endl;


  // Starting point of the shower
  // try first with ECAL lambda  

  double sum1   = 0.;  // lambda path from the ECAL/HF entrance;  
  double sum2   = 0.;  // lambda path from the interaction point;
  double sum3   = 0.;  // x0     path from the interaction point;
  int  nsteps   = 0;   // full number of longitudinal steps (counter);

  int nmoresteps;      // how many longitudinal steps in addition to 
                       // one (if interaction happens there) in ECAL
 

  if(e < criticalEnergy ) nmoresteps = 1;  
  else                    nmoresteps = nDepthSteps;
  
  double depthECAL  = 0.;
  double depthGAP   = 0.;
  double depthGAPx0 = 0.; 
  if(onECAL ) {
    depthECAL  = theGrid->ecalTotalL0();         // ECAL depth segment
    depthGAP   = theGrid->ecalHcalGapTotalL0();  // GAP  depth segment
    depthGAPx0 = theGrid->ecalHcalGapTotalX0();  // GAP  depth x0
  }
  
  double depthHCAL   = theGrid->hcalTotalL0();    // HCAL depth segment
  double depthToHCAL = depthECAL + depthGAP;    

  //---------------------------------------------------------------------------
  // Depth simulation & various protections, among them
  // if too deep - get flat random in the allowed region
  // if no HCAL material behind - force to deposit in ECAL
  double maxDepth    = depthToHCAL + depthHCAL - 1.1 * depthStep;
  double depthStart  = log(1./RandFlat::shoot()); // starting point lambda unts

  if(e < emin) {
    if(debug)
      LogDebug("FastCalorimetry") << " FamosHDShower : e <emin ->  depthStart = 0" << endl; 
    depthStart = 0.;
  }
 
  if(depthStart > maxDepth) {
    if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : depthStart too big ...   = " 
		   << depthStart << endl; 
    
    depthStart = maxDepth *  RandFlat::shoot();
    if( depthStart < 0.) depthStart = 0.;
    if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : depthStart re-calculated = " 
		   << depthStart << endl; 
  }
  
  if( onECAL && e < emid ) {
    if((depthECAL - depthStart)/depthECAL > 0.2 && depthECAL > depthStep ) {
      
      depthStart = 0.5 * depthECAL * RandFlat::shoot();
      if(debug) 
 	LogDebug("FastCalorimetry") << " FamosHDShower : small energy, "
	     << " depthStart reduced to = " << depthStart << endl; 
      
    }
  }

  if( depthHCAL < depthStep) {
    if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : depthHCAL  too small ... = " 
		   << depthHCAL << " depthStart -> forced to 0 !!!" 
		   << endl;
    depthStart = 0.;    
    nmoresteps = 0;
    
    if(depthECAL < depthStep) {
      nsteps = -1;
      LogInfo("FastCalorimetry") << " FamosHDShower : too small ECAL and HCAL depths - " 
	   << " particle is lost !!! " << endl; 
    }
  }



  if(debug)
    LogDebug("FastCalorimetry") << " FamosHDShower  depths(lam) - "  << endl 
         << "          ECAL = " << depthECAL  << endl
         << "           GAP = " << depthGAP   << endl
         << "          HCAL = " << depthHCAL  << endl
         << "starting point = " << depthStart << endl; 

  if( onEcal ) {
    if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : onECAL" << endl;
    if(depthStart < depthECAL) {
      if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : depthStart < depthECAL" << endl;
      if((depthECAL - depthStart)/depthECAL > 0.25 && depthECAL > depthStep) {
	if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : enough space to make ECAL step"
		       << endl;
	//  ECAL - one step
	nsteps++; 
	sum1   += depthECAL;             // at the end of step
	sum2   += depthECAL-depthStart; 
	sum3   += sum2 * lambdaEM / x0EM;
	lamtotal.push_back(sum1);
	lamdepth.push_back(sum2);
	lamcurr.push_back(lambdaEM);
	lamstep.push_back(depthECAL-depthStart);
	x0depth.push_back(sum3);
	x0curr.push_back(x0EM);
	detector.push_back(1);
	
	if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : " << " in ECAL sum1, sum2 "
		       << sum1 << " " << sum2 << endl;
	
	//                           // Gap - no additional step after ECAL
	//                           // just move further to HCAL over the gap
	sum1   += depthGAP;          
	sum2   += depthGAP; 
	sum3   += depthGAPx0;
      }
      // Just shift starting point to HCAL
      else { 
	//	cout << " FamosHDShower : not enough space to make ECAL step" << endl;
	if(debug)  LogDebug("FastCalorimetry") << " FamosHDShower : goto HCAL" << endl;

	depthStart = depthToHCAL;
	sum1 += depthStart;     
      }
    }
    else { // GAP or HCAL   
      
      if(depthStart >= depthECAL &&  depthStart < depthToHCAL ) {
      depthStart = depthToHCAL; // just a shift to HCAL for simplicity
      }
      sum1 += depthStart;

      if(debug) LogDebug("FastCalorimetry") << " FamosHDShower : goto HCAL" << endl;
    }
  }
  else {   // Forward 
    if(debug)  LogDebug("FastCalorimetry") << " FamosHDShower : forward" << endl;
    sum1 += depthStart;
    //    transFactor = 0.5;   // makes narower tresverse size of shower     
  }
 
  for (int i = 0; i < nmoresteps ; i++) {
    sum1 += depthStep;
    if (sum1 > (depthECAL + depthGAP + depthHCAL)) break; 
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

  if(nsteps > 0) { makeSteps(nsteps); }

}

void HDShower::makeSteps(int nsteps) {

  double sumes = 0.;
  double sum   = 0.;
  vector<double> temp;


  if(debug)
    LogDebug("FastCalorimetry") << " FamosHDShower::makeSteps - " 
       << " nsteps required : " << nsteps << endl;

  int count = 0;
  for (int i = 0; i < nsteps; i++) {    

    double deplam = lamdepth[i] - 0.5 * lamstep[i];
    double depx0  = x0depth[i]  - 0.5 * lamstep[i] / x0curr[i]; 
    double     x = betEM * depx0;
    double     y = betHD * deplam;

   if(debug == 2)
     LogDebug("FastCalorimetry") << " FamosHDShower::makeSteps " << " - step " << i
	  << "   depx0, x = " << depx0 << ", " << x 
	  << "   deplam, y = " << deplam << ", " << y << endl;
    
    double est = (part * betEM * gam(x,alpEM) * lamcurr[i] /
		  (x0curr[i] * tgamEM) + 
		  (1.-part) * betHD * gam(y,alpHD) / tgamHD) * lamstep[i];

    // protection ...
    if(est < 0.) {
      LogDebug("FastCalorimetry") << "*** FamosHDShower::makeSteps " << " - negative step energy !!!" 
	   << endl;
      est = 0.;
      break ; 
    }

    // for estimates only
    sum += est;
    int nPest = (int) (est * e / sum / eSpotSize) ;

    if(debug == 2)
      LogDebug("FastCalorimetry") << " FamosHDShower::makeSteps - nPoints estimate = " 
	   <<  nPest << endl;

    if(nPest <= 1 && count !=0 ) break;

    // good step - to proceed

    temp.push_back(est);
    sumes += est;
    
    rlamStep.push_back(transParam * (theR1 + (theR2 - theR3 * aloge))
		       * deplam * transFactor); 
    count ++;
  }

  // fluctuations in ECAL and re-distribution of remaining energy in HCAL
  if(detector[0] == 1 && count > 1) {
    double oldECALenergy = temp[0];
    double oldHCALenergy = sumes - oldECALenergy ;
    double newECALenergy = 2. * sumes;
    for (int i = 0; newECALenergy > sumes && i < infinity; i++)
      newECALenergy = 2.* balanceEH * RandFlat::shoot() * oldECALenergy; 
     
    if(debug == 2)
      LogDebug("FastCalorimetry") << "*** FamosHDShower::makeSteps " << " ECAL fraction : old/new - "
	   << oldECALenergy/sumes << "/" << newECALenergy/sumes << endl;

    temp[0] = newECALenergy;
    double newHCALenergy = sumes - newECALenergy;
    double newHCALreweight =  newHCALenergy / oldHCALenergy;

    for (int i = 1; i < count; i++) {
      temp[i] *= newHCALreweight;
    }
    
  }

  
  // final re-normalization of the energy fractions  
  for (int i = 0; i < count ; i++) {
    eStep.push_back(temp[i] * e / sumes );
    nspots.push_back((int)(eStep[i]/eSpotSize)+1);

   if(debug)
     LogDebug("FastCalorimetry") << i << "  xO and lamdepth at the end of step = " 
	  << x0depth[i] << " " 
	  << lamdepth[i] << "   Estep func = " <<  eStep[i] 
	  << "   Rstep = " << rlamStep[i] << "  Nspots = " <<  nspots[i]
	  << endl; 

  }

  // The only step is in ECAL - let's make the size bigger ...  
  if(count == 1 and detector[0] == 1) rlamStep[0] *= 2.;


  if(debug) {
    if(eStep[0] > 0.95 * e && detector[0] == 1) 
      LogDebug("FastCalorimetry") << " FamosHDShower::makeSteps - " << "ECAL energy = " << eStep[0]
	   << " out of total = " << e << endl;  
  }

}

bool HDShower::compute() {
  
  //  TimeMe theT("FamosHDShower::compute");

  bool status = false;
  int numLongit = eStep.size();
  if(debug)
    LogDebug("FastCalorimetry") << " FamosHDShower::compute - " 
	    << " N_long.steps required : " << numLongit << endl;

  if(numLongit > 0) {

    status = true;    
    // Prepare the trsanverse probability function
    vector<double> Fhist;
    vector<double> rhist; 
    for (int j = 0; j < nTRsteps + 1; j++) {
      rhist.push_back(maxTRfactor * j / nTRsteps );  
      Fhist.push_back(transProb(maxTRfactor,1.,rhist[j]));
      if(debug == 3) 
	LogDebug("FastCalorimetry") << "indexFinder - i, Fhist[i] = " << j << " " << Fhist[j] << endl;
    }
    
    // Longitudinal steps
    for (int i = 0; i < numLongit ; i++) {
      
      double currentDepthL0 = lamtotal[i] - 0.5 * lamstep[i];
      if(debug)
	LogDebug("FastCalorimetry") << " FamosHDShower::compute - detector = " << detector[i]
	     << "    currentDepthL0 = " << currentDepthL0 << endl;
      
      double maxTRsize   = maxTRfactor * rlamStep[i];     // in lambda units
      double rbinsize    = maxTRsize / nTRsteps; 
      double espot       = eStep[i] / (double)nspots[i];  // re-adjust espot

      if(espot > 2. || espot < 0. ) 
	LogDebug("FastCalorimetry") << " FamosHDShower::compute - unphysical espot = " 
	     << espot << endl;

      int ecal = 0;
      if(detector[i] != 1) { 
	bool setHDdepth = theHcalHitMaker->setDepth(currentDepthL0);
	
	if(debug)
	  LogDebug("FastCalorimetry") << " FamosHDShower::compute - status of " 
	       << " theHcalHitMaker->setDepth(currentDepthL0) is " 
	       << setHDdepth << endl;
	
	if(!setHDdepth) continue;    

	theHcalHitMaker->setSpotEnergy(espot);        
      }
      else {
	ecal = 1;
	bool status = theGrid->getQuads(currentDepthL0);   
	
	if(debug)
	  LogDebug("FastCalorimetry") << " FamosHDShower::compute - status of Grid = " 
	       << status << endl;
	
	if(!status) continue; 

	theGrid->setSpotEnergy(espot);
      }  

      
      // Transverse distribition
      int nok   = 0;                          // counter of OK  
      int count = 0;
      int inf   = infinity;
      if(lossesOpt) inf = nspots[i];          // losses are enabled, otherwise
      // only OK points are counted ...
      for (int j = 0; j < inf; j++) {
	if(nok == nspots[i]) break;
	count ++;
	
	double prob   = RandFlat::shoot();
	int index     = indexFinder(prob,Fhist);
	double radius = rlamStep[i] * rhist[index] +
	  RandFlat::shoot() * rbinsize; // in-bin  
	double phi = 2.*M_PI*RandFlat::shoot();
	
	if(debug == 2)
	  LogDebug("FastCalorimetry") << endl << " FamosHDShower::compute " << " r = " << radius 
	       << "    phi = " << phi << endl;
	
	bool result;
	if(ecal) {
	  result = theGrid->addHit(radius,phi,0);
	  
	  if(debug == 2)
	    LogDebug("FastCalorimetry") << " FamosHDShower::compute - " 
		 << " theGrid->addHit result = " 
		 << result << endl;
	}
	else {
	  result = theHcalHitMaker->addHit(radius,phi,0); 
	  
	  if(debug == 2)
	    LogDebug("FastCalorimetry") << " FamosHDShower::compute - " 
		 << " theHcalHitMaker->addHit result = " 
		 << result << endl;
	}    
	if(result) nok ++; 
	
      } // end of tranverse simulation
      if(count == infinity) { 
        status = false; 
	if(debug)
	  LogDebug("FastCalorimetry") << "*** FamosHDShower::compute " << " maximum number of" 
	       << " transverse points " << count << " is used !!!" << endl; 
        break;
      }

      if(debug)
	LogDebug("FastCalorimetry") << " FamosHDShower::compute " << " long.step No." << i 
	     << "   Ntry, Nok = " << count
	     << " " << nok << endl; 
      
    } // end of longitudinal steps
  } // end of no steps
  return status;

}

int HDShower::indexFinder(double x, const vector<double> & Fhist) {
  // binary search in the vector of doubles
  int size = Fhist.size();

  int curr = size / 2;
  int step = size / 4;
  int iter;
  int prevdir = 0; 
  int actudir = 0; 

  for (iter = 0; iter < size ; iter++) {

    if( curr >= size || curr < 1 )
      LogError("FastCalorimetry") << " FamosHDShower::indexFinder - wrong current index = " 
	   << curr << " !!!" << endl;

    if ((x <= Fhist[curr]) && (x > Fhist[curr-1])) break;
    prevdir = actudir;
    if(x > Fhist[curr]) {actudir =  1;}
    else                {actudir = -1;}
    if(prevdir * actudir < 0) { if(step > 1) step /= 2;}
    curr += actudir * step;
    if(curr > size) curr = size;
    else { if(curr < 1) {curr = 1;}}
 
    if(debug == 3)
      LogDebug("FastCalorimetry") << " indexFinder - end of iter." << iter 
	   << " curr, F[curr-1], F[curr] = "
	   << curr << " " << Fhist[curr-1] << " " << Fhist[curr] << endl;
    
  }

  if(debug == 3)
    LogDebug("FastCalorimetry") << " indexFinder x = " << x << "  found index = " << curr-1
         << endl;


  return curr-1;
}
