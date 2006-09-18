//FAMOS Headers
#include "FastSimulation/ShowerDevelopment/interface/EMShower.h"
#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/PreshowerHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/HcalHitMaker.h"
#include "FastSimulation/Utilities/interface/Histos.h"
//Anaphe headers
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"

#include <math.h>

using std::vector;

EMShower::EMShower(EMECALShowerParametrization* const myParam, 
		   vector<const RawParticle*>* const myPart, 
		   EcalHitMaker * const myGrid,
		   PreshowerHitMaker * const myPresh)
			 
  : theParam(myParam), 
    thePart(myPart), 
    theGrid(myGrid),
    thePreshower(myPresh)    
{ 

  // Get the Famos Histos pointer
  myHistos = Histos::instance();
  myGammaGenerator = GammaFunctionGenerator::instance();
  
  //  std::cout << " Hello EMShower " << std::endl;
  
  hasPreshower = myPresh!=NULL;
  theECAL = myParam->ecalProperties();
  theHCAL = myParam->hcalProperties();
  theLayer1 = myParam->layer1Properties();
  theLayer2 = myParam->layer2Properties();

  
  double fotos = theECAL->photoStatistics() 
               * theECAL->lightCollectionEfficiency();

  nPart = thePart->size();
  totalEnergy = 0.;
  globalMeanDepth = 0.;

  // Initialize the shower parameters for each particle
  for ( unsigned int i=0; i<nPart; ++i ) {
    
    // The particle and the shower energy
    Etot.push_back(0.);
    E.push_back(((*thePart)[i])->e());
    totalEnergy+=E[i];
    double lny = log ( E[i] / theECAL->criticalEnergy() );

    // Average and Sigma for T and alpha
    double theMeanT        = myParam->meanT(lny);
    double theMeanAlpha    = myParam->meanAlpha(lny);
    double theMeanLnT      = myParam->meanLnT(lny);
    double theMeanLnAlpha  = myParam->meanLnAlpha(lny);
    double theSigmaLnT     = myParam->sigmaLnT(lny);
    double theSigmaLnAlpha = myParam->sigmaLnAlpha(lny);

    // The correlation matrix
    double theCorrelation = myParam->correlationAlphaT(lny);
    double rhop = sqrt( (1.+theCorrelation)/2. );
    double rhom = sqrt( (1.-theCorrelation)/2. );

    // The number of spots in ECAL / HCAL
    theNumberOfSpots.push_back(myParam->nSpots(E[i]));
    //    theNumberOfSpots.push_back(myParam->nSpots(E[i])*spotFraction);
    //theNumberOfSpots = RandPoissonQ::shoot(myParam->nSpots(myPart->e()));

    // Photo-statistics
    photos.push_back(E[i] * fotos);
    
    // The longitudinal shower development parameters
    // Fluctuations of alpha, T and beta
    double z1=0.;
    double z2=0.;
    double aa=0.;

    // Protect against too large fluctuations (a < 1) for small energies
    while ( aa <= 1. ) {
      z1 = RandGaussQ::shoot(0.,1.);
      z2 = RandGaussQ::shoot(0.,1.);
      aa = exp(theMeanLnAlpha + theSigmaLnAlpha * (z1*rhop-z2*rhom));
    }

    a.push_back(aa);
    T.push_back(exp(theMeanLnT + theSigmaLnT * (z1*rhop+z2*rhom)));
    b.push_back((a[i]-1.)/T[i]);
    meanDepth.push_back(a[i]/b[i]);
    globalMeanDepth += meanDepth[i]*E[i];
    Ti.push_back(
      a[i]/b[i] * (exp(theMeanLnAlpha)-1.) / exp(theMeanLnAlpha));
  
    // The parameters for the number of energy spots
    TSpot.push_back(theParam->meanTSpot(theMeanT));
    aSpot.push_back(theParam->meanAlphaSpot(theMeanAlpha));
    bSpot.push_back((aSpot[i]-1.)/TSpot[i]);
    //    myHistos->fill("h7000",a[i]);
    //    myHistos->fill("h7002",E[i],a[i]);
  }
  //  cout << " PS1 : " << myGrid->ps1TotalX0()
  //       << " PS2 : " << myGrid->ps2TotalX0()
  //       << " ECAL : " << myGrid->ecalTotalX0()
  //       << " HCAL : " << myGrid->hcalTotalX0() 
  //       << " Offset : " << myGrid->x0DepthOffset()
  //       << endl;

 globalMeanDepth/=totalEnergy;
}

void
EMShower::compute() {
  //  TimeMe theT("EMShower::compute");
  
  // Determine the longitudinal intervals
  //  std::cout << " EMShower compute" << std::endl;
  double dt;
  double radlen;
  int stps;
  
//  std::cout << " PS1 : " << theGrid->ps1TotalX0()
//	    << " PS2 : " << theGrid->ps2TotalX0()
//	    << " ECAL : " << theGrid->ecalTotalX0()
//	    << " HCAL : " << theGrid->hcalTotalX0() 
//	    << " Offset : " << theGrid->x0DepthOffset()
//	    << std::endl;
  
  
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
  
  // ECAL
  radlen += theGrid->ecalTotalX0();
  if ( radlen > 0. ) {
    stps=(int)((radlen+2.5)/5.);
    if ( stps == 0 ) stps = 1;
    dt = radlen/(double)stps;
    Step step(2,dt);
    for ( int ist=0; ist<stps; ++ist )
      steps.push_back(step);
    radlen = 0.;
  } 
 
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
 
 
 double t = 0.;
 
  //  double E1 = 0.;  // Energy layer 1
  //  double E2 = 0.;  // Energy layer 2
  //  double n1 = 0.;  // #mips layer 1
  //  double n2 = 0.;  // #mips layer 2
  //  double E9 = 0.;  // Energy ECAL
  
  // Loop over all segments for the longitudinal development
  for ( step_iterator step=steps.begin(); step<steps.end(); ++step ) {
    
    // The length of the shower in this segment
    dt = step->second;

    // The elapsed length
    t += dt;

    // In what detector are we ?
    unsigned detector=step->first;
    bool presh1 = detector==0;
    bool presh2 = detector==1;
    bool ecal = detector==2;
    bool hcal = detector==3;
    bool vfcal = detector==4;

    // Temporary. Will be removed 
    if ( theHCAL==NULL) hcal=false;

    // Keep only ECAL for now
    if ( vfcal ) continue;

    //    cout << " t = " << t << endl;
    // Build the grid of crystals at this ECAL depth
    // Actually, it might be useful to check if this grid is empty or not. 
    // If it is empty (because no crystal at this depth), it is of no use 
    // (and time consuming) to generate the spots
    
    bool status=true;

   // middle of the step
    double tt = t-0.5*dt; 

//    std::cout << " Step " << tt << std::endl;
//    std::cout << "ecal " << ecal << " hcal "  << hcal <<std::endl;
    if (ecal) status=theGrid->getQuads(tt);
    if (hcal) 
      {
	status=theHcalHitMaker->setDepth(tt);
      }
    if(!status) continue;
    
    bool detailedShowerTail=false;
    // check if a detailed treatment of the rear leakage should be applied
    if(ecal) 
      {
	detailedShowerTail=(t-dt > theGrid->getX0back());
      }
    
    // The particles of the shower are processed in parallel
    for ( unsigned int i=0; i<nPart; ++i ) {

     //  integration of the shower profile between t-dt and t
      double dE = (!hcal)? deposit(t,a[i],b[i],dt):1.-deposit(a[i],b[i],t-dt);

      if(detailedShowerTail)
	{
	  myGammaGenerator->setParameters(floor(a[i]+0.5),b[i],t-dt);
	}
      
      // The number of energy spots (or mips)
      double nS = 0;
      
      // ECAL case : Account for photostatistics and long'al non-uniformity
      if (ecal) {

	dE = RandPoissonQ::shoot(dE*photos[i])/photos[i];
	double z0 = RandGaussQ::shoot(0.,1.);
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
	
	nS = RandPoissonQ::shoot(nS);
	dE *= nS/nSo;
//	if(true)
//	  {
//	    std::cout << " theHCAL->spotFraction = " <<theHCAL->spotFraction() <<std::endl;
//	    std::cout << " nSpot Ecal : " << nSo/theHCAL->spotFraction() << " Final " << nS << std::endl;
//	  }
      }
      else if ( presh1 ) {
	
	nS = RandPoissonQ::shoot(dE*E[i]*theLayer1->mipsPerGeV());
	dE = nS/(E[i]*theLayer1->mipsPerGeV());
	//        E1 += dE*E[i]; 
	//	n1 += nS; 
	//	if (presh2) { E2 += SpotEnergy; ++n2; }
      
      } else if ( presh2 ) {
	
	nS = RandPoissonQ::shoot(dE*E[i]*theLayer2->mipsPerGeV());
        dE = nS/(E[i]*theLayer2->mipsPerGeV());
	//        E2 += dE*E[i]; 
	//	n2 += nS; 
	
      }

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
      //    int nSpot = RandPoissonQ::shoot(nS);
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
	RandGaussQ::shoot(proba*nSpot,sqrt(proba*(1.-proba)*nSpot));
      
      if(dSpotsCore<0) dSpotsCore=0;
      
      unsigned nSpots_core = (unsigned)(dSpotsCore+0.5);
      unsigned nSpots_tail = ((unsigned)nSpot>nSpots_core) ? nSpot-nSpots_core : 0;
      
      for(unsigned icomp=0;icomp<2;++icomp)
	{	  
	  
	  double theR=(icomp==0) ? theRC : theRT ;    
	  unsigned ncompspots=(icomp==0) ? nSpots_core : nSpots_tail;
	  
	  RadialInterval radInterval(theR,ncompspots,SpotEnergy);
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
	       for ( unsigned  ispot=0; ispot<nradspots; ++ispot ) 
		 {
		   double z3=RandFlat::shoot(umin,umax);
		   double ri=theR * sqrt(z3/(1.-z3)) ;

		   //Fig. 12
		   /*
		     if ( 2. < t && t < 3. ) 
		     myHistos->fill("h401",ri,1./1000.*eSpot/dE/0.2);
		     if ( 6. < t && t < 7. ) 
		     myHistos->fill("h402",ri,1./1000.*eSpot/dE/0.2);
		     if ( 19. < t && t < 20. ) 
		     myHistos->fill("h403",ri,1./1000.*eSpot/dE/0.2);
		   */
		   // Fig. 13 (top)
		   //      myHistos->fill("h400",ri,1./1000.*eSpot/0.2);
		   
		   // Generate phi
		   double phi = 2.*M_PI*RandFlat::shoot();
		   
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
      myHistos->fill("h10",Etot[i]);
      Etotal+=Etot[i];
    }
  myHistos->fill("h20",Etotal);
}


double
EMShower::gam(double x, double a) const {
  // A stupid gamma function
  return pow(x,a-1.)*exp(-x);
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
  return (myIncompleteGamma(b2)-myIncompleteGamma(b1));
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
  double result=myIncompleteGamma(b2);
  //  std::cout << " deposit t = " << t  << " "  << result <<std::endl;
  return result;
}
