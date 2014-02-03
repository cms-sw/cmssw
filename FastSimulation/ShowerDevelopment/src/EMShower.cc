//FAMOS Headers
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"

#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/GammaFunctionGenerator.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include <cmath>

//#include "FastSimulation/Utilities/interface/Histos.h"

using std::vector;


EMShower::EMShower(const RandomEngine* engine,
		   GammaFunctionGenerator* gamma,
		   EMECALShowerParametrization* const myParam, 
		   vector<const RawParticle*>* const myPart, 
		   DQMStore * const dbeIn,
		   EcalHitMaker * const myGrid,
		   PreshowerHitMaker * const myPresh,
		   bool bFixedLength)
			 
  : theParam(myParam), 
    thePart(myPart), 
    theGrid(myGrid),
    thePreshower(myPresh),
    random(engine),
    myGammaGenerator(gamma),
    bFixedLength_(bFixedLength)
{ 

  // Get the Famos Histos pointer
  //  myHistos = Histos::instance();
  //  myGammaGenerator = GammaFunctionGenerator::instance();
  stepsCalculated=false;
  hasPreshower = myPresh!=NULL;
  theECAL = myParam->ecalProperties();
  theHCAL = myParam->hcalProperties();
  theLayer1 = myParam->layer1Properties();
  theLayer2 = myParam->layer2Properties();

  
  double fotos = theECAL->photoStatistics() 
               * theECAL->lightCollectionEfficiency();

  dbe = dbeIn;

  nPart = thePart->size();
  totalEnergy = 0.;
  globalMaximum = 0.;
  double meanDepth=0.;
  // Initialize the shower parameters for each particle

  if (dbe) {
    dbe->cd();             
    if (!dbe->get("EMShower/NumberOfParticles")) {}//std::cout << "NOT FOUND IN Shower.cc" << std::endl;}
	else {
	  dbe->get("EMShower/NumberOfParticles")->Fill(nPart);
	}
    }


  for ( unsigned int i=0; i<nPart; ++i ) {
    //    std::cout << " AAA " << *(*thePart)[i] << std::endl;
    // The particle and the shower energy
    Etot.push_back(0.);
    E.push_back(((*thePart)[i])->e());
    totalEnergy+=E[i];
    


    if (dbe) {
	dbe->cd();             
	if (!dbe->get("EMShower/ParticlesEnergy")) {}//std::cout << "NOT FOUND IN Shower.cc" << std::endl;}
	else {
	  dbe->get("EMShower/ParticlesEnergy")->Fill(log10(E[i]));
	}
    }







    double lny = std::log ( E[i] / theECAL->criticalEnergy() );

    // Average and Sigma for T and alpha
    double theMeanT        = myParam->meanT(lny);
    double theMeanAlpha    = myParam->meanAlpha(lny);
    double theMeanLnT      = myParam->meanLnT(lny);
    double theMeanLnAlpha  = myParam->meanLnAlpha(lny);
    double theSigmaLnT     = myParam->sigmaLnT(lny);
    double theSigmaLnAlpha = myParam->sigmaLnAlpha(lny);


    // The correlation matrix
    double theCorrelation = myParam->correlationAlphaT(lny);
    double rhop = std::sqrt( (1.+theCorrelation)/2. );
    double rhom = std::sqrt( (1.-theCorrelation)/2. );

    // The number of spots in ECAL / HCAL
    theNumberOfSpots.push_back(myParam->nSpots(E[i]));
    //    theNumberOfSpots.push_back(myParam->nSpots(E[i])*spotFraction);
    //theNumberOfSpots = random->poissonShoot(myParam->nSpots(myPart->e()));

    // Photo-statistics
    photos.push_back(E[i] * fotos);
    
    // The longitudinal shower development parameters
    // Fluctuations of alpha, T and beta
    double z1=0.;
    double z2=0.;
    double aa=0.;

    // Protect against too large fluctuations (a < 1) for small energies
    while ( aa <= 1. ) {
      z1 = random->gaussShoot(0.,1.);
      z2 = random->gaussShoot(0.,1.);
      aa = std::exp(theMeanLnAlpha + theSigmaLnAlpha * (z1*rhop-z2*rhom));
    }

    a.push_back(aa);
    T.push_back(std::exp(theMeanLnT + theSigmaLnT * (z1*rhop+z2*rhom)));
    b.push_back((a[i]-1.)/T[i]);
    maximumOfShower.push_back((a[i]-1.)/b[i]);
    globalMaximum += maximumOfShower[i]*E[i];
    meanDepth += a[i]/b[i]*E[i];
    //    std::cout << " Adding max " << maximumOfShower[i] << " " << E[i] << " " <<maximumOfShower[i]*E[i] << std::endl; 
    //    std::cout << std::setw(8) << std::setprecision(5) << " a /b " << a[i] << " " << b[i] << std::endl;
    Ti.push_back(
		 a[i]/b[i] * (std::exp(theMeanLnAlpha)-1.) / std::exp(theMeanLnAlpha));
  
    // The parameters for the number of energy spots
    TSpot.push_back(theParam->meanTSpot(theMeanT));
    aSpot.push_back(theParam->meanAlphaSpot(theMeanAlpha));
    bSpot.push_back((aSpot[i]-1.)/TSpot[i]);
    //    myHistos->fill("h7000",a[i]);
    //    myHistos->fill("h7002",E[i],a[i]);
  }
//  std::cout << " PS1 : " << myGrid->ps1TotalX0()
//         << " PS2 : " << myGrid->ps2TotalX0()
//         << " ECAL : " << myGrid->ecalTotalX0()
//         << " HCAL : " << myGrid->hcalTotalX0() 
//         << " Offset : " << myGrid->x0DepthOffset()
//         << std::endl;

 globalMaximum/=totalEnergy;
 meanDepth/=totalEnergy;
 // std::cout << " Total Energy " << totalEnergy << " Global max " << globalMaximum << std::endl;
}

void EMShower::prepareSteps()
{
  //  TimeMe theT("EMShower::compute");
  
  // Determine the longitudinal intervals
  //  std::cout << " EMShower compute" << std::endl;
  double dt;
  double radlen;
  int stps;
  int first_Ecal_step=0;
  int last_Ecal_step=0;

  // The maximum is in principe 8 (with 5X0 steps in the ECAL)
  steps.reserve(24);

  /*  
  std::cout << " PS1 : " << theGrid->ps1TotalX0()
	    << " PS2 : " << theGrid->ps2TotalX0()
	    << " PS2 and ECAL : " << theGrid->ps2eeTotalX0()
	    << " ECAL : " << theGrid->ecalTotalX0()
	    << " HCAL : " << theGrid->hcalTotalX0() 
	    << " Offset : " << theGrid->x0DepthOffset()
	    << std::endl;
  */
  
  radlen = -theGrid->x0DepthOffset();
  
  // Preshower Layer 1
  radlen += theGrid->ps1TotalX0();
  if ( radlen > 0. ) {
    steps.push_back(Step(0,radlen));
    radlen = 0.;
  }
  
  // Preshower Layer 2
  radlen += theGrid->ps2TotalX0();
  if ( radlen > 0. ) {
    steps.push_back(Step(1,radlen));
    radlen = 0.;
  }
  
  // add a step between preshower and ee
  radlen += theGrid->ps2eeTotalX0();
  if ( radlen > 0.) {
    steps.push_back(Step(5,radlen));
    radlen = 0.;  
  }
  
  // ECAL
  radlen += theGrid->ecalTotalX0();

  //  std::cout << "theGrid->ecalTotalX0() = " << theGrid->ecalTotalX0() << std::endl;

  if ( radlen > 0. ) {

    if (!bFixedLength_){
      stps=(int)((radlen+2.5)/5.);
      //    stps=(int)((radlen+.5)/1.);
      if ( stps == 0 ) stps = 1;
      dt = radlen/(double)stps;
      Step step(2,dt);
      first_Ecal_step=steps.size();
      for ( int ist=0; ist<stps; ++ist )
	steps.push_back(step);
      last_Ecal_step=steps.size()-1;
      radlen = 0.;
    } else {
      dt = 1.0;
      stps = static_cast<int>(radlen); 
      if (stps == 0) stps = 1;
      Step step(2,dt);
      first_Ecal_step=steps.size();
      for ( int ist=0; ist<stps; ++ist ) steps.push_back(step);
      dt = radlen-stps;
      if (dt>0) {
	Step stepLast (2,dt);
	steps.push_back(stepLast);
      }
      last_Ecal_step=steps.size()-1;
      //      std::cout << "radlen = "  << radlen << " stps = " << stps << " dt = " << dt << std::endl;
      radlen = 0.;

    }
  } 

  // I should had a gap here ! 
 
 // HCAL 
 radlen += theGrid->hcalTotalX0();
 if ( radlen > 0. ) {
   double dtFrontHcal=theGrid->totalX0()-theGrid->hcalTotalX0();
   // One single step for the full HCAL
   if(dtFrontHcal<30.) 
     {
       dt=30.-dtFrontHcal;
       Step step(3,dt);
       steps.push_back(step);
     }
 } 

 nSteps=steps.size();
 if(nSteps==0) return;
 double ESliceTot=0.;
 double MeanDepth=0.;
 depositedEnergy.resize(nSteps);
 meanDepth.resize(nSteps);
 double t=0.;

 int offset=0;
 for(unsigned iStep=0;iStep<nSteps;++iStep)
   {
     ESliceTot=0.;
     MeanDepth=0.;
     double realTotalEnergy=0;
     dt=steps[iStep].second;
     t+=dt;
     for ( unsigned int i=0; i<nPart; ++i ) {
       depositedEnergy[iStep].push_back(deposit(t,a[i],b[i],dt));     
       ESliceTot +=depositedEnergy[iStep][i];
       MeanDepth += deposit(t,a[i]+1.,b[i],dt)/b[i]*a[i];
       realTotalEnergy+=depositedEnergy[iStep][i]*E[i];
     }

     if( ESliceTot > 0. )  // can happen for the shower tails; this depth will be skipped anyway
       MeanDepth/=ESliceTot;
     else
       MeanDepth=t-dt;

     meanDepth[iStep]=MeanDepth;
     if(realTotalEnergy<0.001)
       {
	 offset-=1;
       }
   }

 innerDepth=meanDepth[first_Ecal_step];
 if(last_Ecal_step+offset>=0)
   outerDepth=meanDepth[last_Ecal_step+offset];
 else
   outerDepth=innerDepth;

 stepsCalculated=true;
}

void
EMShower::compute() {

  double samplingWidth = theECAL->da() + theECAL->dp();
  double theECALX0 = theECAL->radLenIncm();

  //  double one_over_resoSquare = 1./(theECAL->resE()*theECAL->resE());





  double t = 0.;
  double dt = 0.;
  if(!stepsCalculated) prepareSteps();

  // Prepare the grids in EcalHitMaker
  // theGrid->setInnerAndOuterDepth(innerDepth,outerDepth);
  float pstot=0.;
  float ps2tot=0.;
  float ps1tot=0.;
  bool status=false; 
  //  double E1 = 0.;  // Energy layer 1
  //  double E2 = 0.;  // Energy layer 2
  //  double n1 = 0.;  // #mips layer 1
  //  double n2 = 0.;  // #mips layer 2
  //  double E9 = 0.;  // Energy ECAL

  // Loop over all segments for the longitudinal development
  double totECalc = 0;
  


  for (unsigned iStep=0; iStep<nSteps; ++iStep ) {
    
    // The length of the shower in this segment
    dt = steps[iStep].second;

    //    std::cout << " Detector " << steps[iStep].first << " t " << t << " " << dt << std::endl;

    // The elapsed length
    t += dt;
    
    // In what detector are we ?
    unsigned detector=steps[iStep].first;
       
    bool presh1 = detector==0;
    bool presh2 = detector==1;
    bool ecal = detector==2;
    bool hcal = detector==3;
    bool vfcal = detector==4;
    bool gap = detector==5;

    // Temporary. Will be removed 
    if ( theHCAL==NULL) hcal=false;

    // Keep only ECAL for now
    if ( vfcal ) continue;

    // Nothing to do in the gap
    if( gap ) continue;

    //    cout << " t = " << t << endl;
    // Build the grid of crystals at this ECAL depth
    // Actually, it might be useful to check if this grid is empty or not. 
    // If it is empty (because no crystal at this depth), it is of no use 
    // (and time consuming) to generate the spots
    

   // middle of the step
    double tt = t-0.5*dt; 

    double realTotalEnergy=0.;
    for ( unsigned int i=0; i<nPart; ++i ) {
      realTotalEnergy += depositedEnergy[iStep][i]*E[i];
    }

//    std::cout << " Step " << tt << std::endl;
//    std::cout << "ecal " << ecal << " hcal "  << hcal <<std::endl;

    // If the amount of energy is greater than 1 MeV, make a new grid
    // otherwise put in the previous one.    
    bool usePreviousGrid=(realTotalEnergy<0.001);   

    // If the amount of energy is greater than 1 MeV, make a new grid
    // otherwise put in the previous one.    

    // If less than 1 kEV. Just skip
    if(iStep>2&&realTotalEnergy<0.000001) continue;

    if (ecal && !usePreviousGrid) 
      {
	status=theGrid->getPads(meanDepth[iStep]);
      }
    if (hcal) 
      {
	status=theHcalHitMaker->setDepth(tt);
      }
    if((ecal || hcal) && !status) continue;
    
    bool detailedShowerTail=false;
    // check if a detailed treatment of the rear leakage should be applied
    if(ecal && !usePreviousGrid) 
      {
	detailedShowerTail=(t-dt > theGrid->getX0back());
      }
    
    // The particles of the shower are processed in parallel
    for ( unsigned int i=0; i<nPart; ++i ) {

      //      double Edepo=deposit(t,a[i],b[i],dt);

     //  integration of the shower profile between t-dt and t
      double dE = (!hcal)? depositedEnergy[iStep][i]:1.-deposit(a[i],b[i],t-dt);

      // no need to do the full machinery if there is ~nothing to distribute)
      if(dE*E[i]<0.000001) continue;


      if (ecal && !theECAL->isHom()) {
	double mean = dE*E[i];
	double sigma = theECAL->resE()*sqrt(mean);
	
	/*
	  double meanLn = log(mean);
	  double kLn = sigma/mean+1;
	  double sigmaLn = log(kLn);
	*/

	double dE0 = dE;

	//	  std::cout << "dE before shoot = " << dE << std::endl;
	dE = random->gaussShoot(mean, sigma)/E[i];
	
	//	  myGammaGenerator->setParameters(aSam,bSam,0);
	//	  dE = myGammaGenerator->shoot()/E[i];
	//	  std::cout << "dE shooted = " << dE << " E[i] = " << E[i] << std::endl; 
	if (dE*E[i] < 0.000001) continue;
	photos[i] = photos[i]*dE/dE0;
	
      }



      /*
      if (ecal && !theParam->ecalProperties()->isHom()){

	double cSquare = TMath::Power(theParam->ecalProperties()->resE(),2);
	double aSam = dE/cSquare;
	double bSam = 1./cSquare;

	//	dE = dE*gam(bSam*dE, aSam)/tgamma(aSam);
      }
      */

      totECalc +=dE;
      
      if (dbe && fabs(dt-1.)< 1e-5 && ecal) {
	dbe->cd();             
	if (!dbe->get("EMShower/LongitudinalShape")) {}//std::cout << "NOT FOUND IN Shower.cc" << std::endl;}
	else {
	  double dx = 1.;
	  // dE is aready in relative units from 0 to 1
	  dbe->get("EMShower/LongitudinalShape")->Fill(t, dE/dx);

	  double step = theECALX0/samplingWidth;
	  double binMin = abs((t-1)*step)+1;
	  double binMax = abs(t*step)+1;
	  double dBins = binMax-binMin;

	  /*
	  std::cout << "X0 = " << theECALX0 << " Sampling Width = " << samplingWidth << " t = " << t 
		    << " binMin = " << binMin << " binMax = " << binMax << " (t-1)*step = " 
		    << (t-1)*step+1 << " t*step = " << t*step+1 << std::endl;
	  */

	  if ( dBins < 1) {
	    dbe->get("EMShower/LongitudinalShapeLayers")->Fill(binMin, dE/dx);
	    //	    std::cout << "bin " << binMin << " filled" << std::endl;
	  }
	  else {


	    double w1 = (binMin + 1 - (t-1)*step - 1)/step;
	    double w2 = 1./step;
	    double w3 = (t*step+1-binMax)/step;

	    //double Esum = 0;

	    /*
	    std::cout <<" ((t-1)*step - binMin) = " << (binMin + 1 - (t-1)*step - 1) 
		      <<" w1 = " << w1 << " w2 = " << w2 << " w3 = " << w3
		      << " (t*step+1 - binMax) = " << (t*step+1 - binMax) << std::endl;

	    std::cout << "fill bin = " << binMin << std::endl;
	    */

	    dbe->get("EMShower/LongitudinalShapeLayers")->Fill(binMin, dE/dx*w1);
	    //Esum = dE/dx*w1;

	    for (int iBin = 1; iBin < dBins; iBin++){
	      //	      std::cout << "fill bin = " << binMin+iBin << std::endl;
	      dbe->get("EMShower/LongitudinalShapeLayers")->Fill(binMin+iBin, dE/dx*w2);
	      //	      Esum += dE/dx*w2;
	    }

	    //	    std::cout << "fill bin = " << binMax << std::endl;
	    dbe->get("EMShower/LongitudinalShapeLayers")->Fill(binMax, dE/dx*w3);	    
	    //	    Esum += dE/dx*w3;	   
	    //	    std::cout << "Esum = " << Esum << " dE/dx = " << dE/dx << std::endl;
	  }


	}
	//(dbe->get("TransverseShape"))->Fill(ri,log10(1./1000.*eSpot/0.2));

      }
 
      // The number of energy spots (or mips)
      double nS = 0;
      
      // ECAL case : Account for photostatistics and long'al non-uniformity
      if (ecal) {


	//	double aSam = E[i]*dE*one_over_resoSquare;
	//	double bSam = one_over_resoSquare;


	dE = random->poissonShoot(dE*photos[i])/photos[i];
	double z0 = random->gaussShoot(0.,1.);
	dE *= 1. + z0*theECAL->lightCollectionUniformity();

	// Expected spot number
	nS = ( theNumberOfSpots[i] * gam(bSpot[i]*tt,aSpot[i]) 
	                           * bSpot[i] * dt 
		                   / tgamma(aSpot[i]) );
	
      // Preshower : Expected number of mips + fluctuation
      }
      else if ( hcal ) {

	nS = ( theNumberOfSpots[i] * gam(bSpot[i]*tt,aSpot[i]) 
	       * bSpot[i] * dt 
	       / tgamma(aSpot[i]))* theHCAL->spotFraction();
	double nSo = nS ;
	nS = random->poissonShoot(nS);
	// 'Quick and dirty' fix (but this line should be better removed):
	if( nSo > 0. && nS/nSo < 10.) dE *= nS/nSo;

//	if(true)
//	  {
//	    std::cout << " theHCAL->spotFraction = " <<theHCAL->spotFraction() <<std::endl;
//	    std::cout << " nSpot Ecal : " << nSo/theHCAL->spotFraction() << " Final " << nS << std::endl;
//	  }
      }
      else if ( presh1 ) {
	
	nS = random->poissonShoot(dE*E[i]*theLayer1->mipsPerGeV());
	//	std::cout << " dE *E[i] (1)" << dE*E[i] << " " << dE*E[i]*theLayer1->mipsPerGeV() << "  "<< nS << std::endl;
	pstot+=dE*E[i];
	ps1tot+=dE*E[i];
	dE = nS/(E[i]*theLayer1->mipsPerGeV());

	//        E1 += dE*E[i]; 
	//	n1 += nS; 
	//	if (presh2) { E2 += SpotEnergy; ++n2; }
      
      } else if ( presh2 ) {
	
	nS = random->poissonShoot(dE*E[i]*theLayer2->mipsPerGeV());
	//	std::cout << " dE *E[i] (2) " << dE*E[i] << " " << dE*E[i]*theLayer2->mipsPerGeV() << "  "<< nS << std::endl;
	pstot+=dE*E[i];
	ps2tot+=dE*E[i];
        dE = nS/(E[i]*theLayer2->mipsPerGeV());

	//        E2 += dE*E[i]; 
	//	n2 += nS; 
	
      }

      if(detailedShowerTail)
	myGammaGenerator->setParameters(floor(a[i]+0.5),b[i],t-dt);
	


      //    myHistos->fill("h100",t,dE);
      
      // The lateral development parameters  
 
      // Energy of the spots
      double eSpot = (nS>0.) ? dE/nS : 0.;
      double SpotEnergy=eSpot*E[i];

      if(hasPreshower&&(presh1||presh2)) thePreshower->setSpotEnergy(0.00009);
      if(hcal) 
	{
	  SpotEnergy*=theHCAL->hOverPi();
	  theHcalHitMaker->setSpotEnergy(SpotEnergy);
	}
      // Poissonian fluctuations for the number of spots
      //    int nSpot = random->poissonShoot(nS);
      int nSpot = (int)(nS+0.5);
      
      
      // Fig. 11 (right) *** Does not match.
      //    myHistos->fill("h101",t,(double)nSpot/theNumberOfSpots);
      
      //double taui = t/T;
      double taui = tt/Ti[i];
      double proba = theParam->p(taui,E[i]);
      double theRC = theParam->rC(taui,E[i]);
      double theRT = theParam->rT(taui,E[i]);
      
      // Fig. 10
      //    myHistos->fill("h300",taui,theRC);
      //    myHistos->fill("h301",taui,theRT);
      //    myHistos->fill("h302",taui,proba);
      
      double dSpotsCore = 
	random->gaussShoot(proba*nSpot,std::sqrt(proba*(1.-proba)*nSpot));
      
      if(dSpotsCore<0) dSpotsCore=0;
      
      unsigned nSpots_core = (unsigned)(dSpotsCore+0.5);
      unsigned nSpots_tail = ((unsigned)nSpot>nSpots_core) ? nSpot-nSpots_core : 0;
      
      for(unsigned icomp=0;icomp<2;++icomp)
	{	  
	  
	  double theR=(icomp==0) ? theRC : theRT ;    
	  unsigned ncompspots=(icomp==0) ? nSpots_core : nSpots_tail;
	  
	  RadialInterval radInterval(theR,ncompspots,SpotEnergy,random);
	  if(ecal)
	    {
	      if(icomp==0)
		{
		  setIntervals(icomp,radInterval);
		}
	      else
		{
		  setIntervals(icomp,radInterval);
		}
	    }
	  else
	    {
	      radInterval.addInterval(100.,1.);// 100% of the spots
	    }
	  
	  radInterval.compute();
	   // irad = 0 : central circle; irad=1 : outside

	   unsigned nrad=radInterval.nIntervals();

	   for(unsigned irad=0;irad<nrad;++irad)
	     {
	       double spote=radInterval.getSpotEnergy(irad);
	       if(ecal) theGrid->setSpotEnergy(spote);
	       if(hcal) theHcalHitMaker->setSpotEnergy(spote);
	       unsigned nradspots=radInterval.getNumberOfSpots(irad);
	       double umin=radInterval.getUmin(irad);
	       double umax=radInterval.getUmax(irad);
	       // Go for the lateral development
	       //	       std::cout << "Couche " << iStep << " irad = " << irad << " Ene = " << E[i] << " eSpot = " << eSpot << " spote = " << spote << " nSpot = " << nS << std::endl;

	       for ( unsigned  ispot=0; ispot<nradspots; ++ispot ) 
		 {
		   double z3=random->flatShoot(umin,umax);
		   double ri=theR * std::sqrt(z3/(1.-z3)) ;



		   //Fig. 12
		   /*
		   if ( 2. < t && t < 3. ) 
		     dbe->fill("h401",ri,1./1000.*eSpot/dE/0.2);
		   if ( 6. < t && t < 7. ) 
		     dbe->fill("h402",ri,1./1000.*eSpot/dE/0.2);
		   if ( 19. < t && t < 20. ) 
		     dbe->fill("h403",ri,1./1000.*eSpot/dE/0.2);
		   */
		   // Fig. 13 (top)
		   if (dbe && fabs(dt-1.)< 1e-5 && ecal) {
		     dbe->cd();             
		     if (!dbe->get("EMShower/TransverseShape")) {}//std::cout << "NOT FOUND IN Shower.cc" << std::endl;}
		     else {
		       double drho = 0.1;
		       double dx = 1;
		       // spote is a real energy we have to normalise it by E[i] which is the energy of the particle i
		       dbe->get("EMShower/TransverseShape")->Fill(ri,1/E[i]*spote/drho);
		       dbe->get("EMShower/ShapeRhoZ")->Fill(ri, t, 1/E[i]*spote/(drho*dx));
		     }
		   } else {
		     //		     std::cout << "dt =  " << dt << " length = " << t << std::endl;  
		   }
	       

		   // Generate phi
		   double phi = 2.*M_PI*random->flatShoot();
		   
		   // Add the hit in the crystal
		   //	if( ecal ) theGrid->addHit(ri*theECAL->moliereRadius(),phi);
		   // Now the *moliereRadius is done in EcalHitMaker
		   if ( ecal )
		     {
		       if(detailedShowerTail) 
			 {
			   //			   std::cout << "About to call addHitDepth " << std::endl;
			   double depth;
			   do
			     {
			       depth=myGammaGenerator->shoot();
			     }
			   while(depth>t);
			   theGrid->addHitDepth(ri,phi,depth);
			   //			   std::cout << " Done " << std::endl;
			 }
		       else
			 theGrid->addHit(ri,phi);
		     }
		   else if (hasPreshower&&presh1) thePreshower->addHit(ri,phi,1);
		   else if (hasPreshower&&presh2) thePreshower->addHit(ri,phi,2);
		   else if (hcal) 
		     {
		       //		       std::cout << " About to add a spot in the HCAL" << status << std::endl;
		       theHcalHitMaker->addHit(ri,phi);
		       //		       std::cout << " Added a spot in the HCAL" << status << std::endl;
		     }
		   //	if (ecal) E9 += SpotEnergy;
		   //	if (presh1) { E1 += SpotEnergy; ++n1; }
		   //	if (presh2) { E2 += SpotEnergy; ++n2; }

		   Etot[i] += spote;
		 }
	     }
	}
      //      std::cout << " Done with the step " << std::endl;
      // The shower!
      //myHistos->fill("h500",theSpot.z(),theSpot.perp());
    }
    //    std::cout << " nPart " << nPart << std::endl;
  }
  //  std::cout << " Finshed the step loop " << std::endl;
  //  myHistos->fill("h500",E1+0.7*E2,E9);
  //  myHistos->fill("h501",n1+0.7*n2,E9);
  //  myHistos->fill("h400",n1);
  //  myHistos->fill("h401",n2);
  //  myHistos->fill("h402",E9+E1+0.7*E2);
  //  if(!standalone)theGrid->printGrid();
  double Etotal=0.;
  for(unsigned i=0;i<nPart;++i)
    {
      //      myHistos->fill("h10",Etot[i]);
      Etotal+=Etot[i];
    }

  //  std::cout << "Etotal = " << Etotal << " nPart = "<< nPart << std::endl; 
  //  std::cout << "totECalc = " << totECalc << std::endl;

  //  myHistos->fill("h20",Etotal);
  //  if(thePreshower)
  //    std::cout << " PS " << thePreshower->layer1Calibrated() << " " << thePreshower->layer2Calibrated() << " " << thePreshower->totalCalibrated() << " " << ps1tot << " " <<ps2tot << " " << pstot << std::endl;
}


double
EMShower::gam(double x, double a) const {
  // A stupid gamma function
  return std::pow(x,a-1.)*std::exp(-x);
}

//double 
//EMShower::deposit(double t, double a, double b, double dt) {
//
//  // The number of integration steps (about 1 / X0)
//  int numberOfSteps = (int)dt+1;
//
//  // The size if the integration step
//  double integrationstep = dt/(double)numberOfSteps;
//
//  // Half this size
//  double halfis = 0.5*integrationstep;
//
//  double dE = 0.;
//  
//  for(double tt=t-dt+halfis;tt<t;tt+=integrationstep) {
//
//    // Simpson integration over each of these steps
//    dE +=   gam(b*(tt-halfis),a) 
//       + 4.*gam(b* tt        ,a)
//       +    gam(b*(tt+halfis),a);
//
//  }
//
//  // Normalization
//  dE *= b*integrationstep/tgamma(a)/6.;
//
//  // There we go.
//  return dE;
//}

double
EMShower::deposit(double t, double a, double b, double dt) {
  myIncompleteGamma.a().setValue(a);
  double b1=b*(t-dt);
  double b2=b*t;
  double result = 0.;  
  double rb1=(b1!=0.) ? myIncompleteGamma(b1) : 0.;
  double rb2=(b2!=0.) ?  myIncompleteGamma(b2) : 0.;
  result = (rb2-rb1);
  return result;
}


void EMShower::setIntervals(unsigned icomp,  RadialInterval& rad)
{
  //  std::cout << " Got the pointer " << std::endl;
  const std::vector<double>& myValues((icomp)?theParam->getTailIntervals():theParam->getCoreIntervals());
  //  std::cout << " Got the vector " << myValues.size () << std::endl;
  unsigned nvals=myValues.size()/2;
  for(unsigned iv=0;iv<nvals;++iv)
    {
      //      std::cout << myValues[2*iv] << " " <<  myValues[2*iv+1] <<std::endl;
      rad.addInterval(myValues[2*iv],myValues[2*iv+1]);
    } 
} 

void EMShower::setPreshower(PreshowerHitMaker * const myPresh)
{
  if(myPresh!=NULL)
    {
      thePreshower = myPresh;
      hasPreshower=true;
    }
}


void EMShower::setHcal(HcalHitMaker * const myHcal)
{
  theHcalHitMaker = myHcal;
}

double
EMShower::deposit( double a, double b, double t) {
  //  std::cout << " Deposit " << std::endl;
  myIncompleteGamma.a().setValue(a);
  double b2=b*t;
  double result = 0.;
  if(fabs(b2) < 1.e-9 ) b2 = 1.e-9;
  result=myIncompleteGamma(b2);
  //  std::cout << " deposit t = " << t  << " "  << result <<std::endl;
  return result;

}
