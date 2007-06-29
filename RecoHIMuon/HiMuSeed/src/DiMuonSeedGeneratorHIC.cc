#include "RecoHIMuon/HiMuSeed/interface/DiMuonSeedGeneratorHIC.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/Common/interface/Handle.h"
using namespace edm;
using namespace std;

DiMuonSeedGeneratorHIC::DiMuonSeedGeneratorHIC(edm::InputTag rphirecHitsTag0,
                                               const MagneticField* magfield0, 
                                               const GeometricSearchTracker* theTracker0, 
					       int aMult = 1):
					       TTRHbuilder(0),
					       rphirecHitsTag(rphirecHitsTag0),
					       magfield(magfield0),
					       theTracker(theTracker0),
					       theLowMult(aMult)

{
  
// initialization

  theHicConst= new HICConst();
  thePropagator=new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );
   
  
}

DiMuonSeedGeneratorHIC::SeedContainer DiMuonSeedGeneratorHIC::produce(const edm::Event& e, const edm::EventSetup& iSetup, 
                                                    FreeTrajectoryState& theFtsTracker, 
						    TrajectoryStateOnSurface& newtsos,
						    FreeTrajectoryState& theFtsMuon,
						    const TransientTrackingRecHitBuilder* RecHitBuilder,
						    const MeasurementTracker* measurementTracker,
						    vector<DetLayer*>* theDetLayer ) {
						    

    theMeasurementTracker = measurementTracker;
    theLayerMeasurements = new LayerMeasurements(theMeasurementTracker);
    					    
    SeedContainer seedContainer;
    
    bl = theTracker->barrelLayers();
    fpos = theTracker->posForwardLayers();
    fneg = theTracker->negForwardLayers();

  if(TTRHbuilder == 0){
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilderHandle;
    iSetup.get<TransientRecHitRecord>().get("WithoutRefit",theBuilderHandle);
    TTRHbuilder = theBuilderHandle.product();
  }


    edm::Handle<SiStripRecHit2DCollection> rphirecHits0;
    e.getByLabel( rphirecHitsTag ,rphirecHits0);
    const SiStripRecHit2DCollection* rphirecHits = rphirecHits0.product();   
 
    cout <<" Start MuSeedGeneratorHIC ======================================= "<<endl;
    cout <<" MuSeedGeneratorHIC::Number of layers in initial container= "<<theDetLayer->size()<<endl;
    cout <<" Trajectory state on tracker surface "<<newtsos.globalPosition().perp()<<endl;
   
    int npair=0;
//
//  For each fts do a cycle on possible layers.
//
    bool itrust = true;
    int nSig =3;
    
    HICSeedMeasurementEstimator theEstimator(itrust,nSig);
    
    double phipred, zpred, phiupd, zupd;
    
    
    
    for( vector<DetLayer* >::const_iterator ml=theDetLayer->begin(); ml != theDetLayer->end(); ml++)
    {
     const BarrelDetLayer* bl = dynamic_cast<const BarrelDetLayer*>(*ml);
     const ForwardDetLayer* fl = dynamic_cast<const ForwardDetLayer*>(*ml);
      if( bl != 0 )
      { 
       // The last layer for barrel
       phipred = (double)theHicConst->phicut[12];
       phiupd = (double)theHicConst->phiro[12];
       zpred = (double)theHicConst->zcut[12];
       zupd = (double)theHicConst->tetro[12];
       cout<<" DiMuonSeedGenerator::propagate to barrel layer surface "<<bl->specificSurface().radius()<<endl;
      }
        else
         {
	   if ( fl != 0 ) 
           {
              // The last layer for endcap
              phipred = (double)theHicConst->phicutf[13];
              phiupd = (double)theHicConst->phirof[13];
              zpred = (double)theHicConst->tetcutf[13];
              zupd = (double)theHicConst->tetrof[13];
	      
              cout<<" DiMuonSeedGenerator::propagate to endcap layer surface "<<fl->position().z()<<" Mult="<<
              theLowMult<<endl;
           } // end fl
         }// end else
// ==========================================       
       theEstimator.set(phipred, zpred);
       
       std::cout<<" DiMuonSeedGenerator::estimator::set cuts "<<std::endl;

       std::vector<TrajectoryMeasurement> vTM = theLayerMeasurements->measurements((**ml),newtsos, *thePropagator, theEstimator);
       
       
       std::cout<<" Size of compatible TM "<<vTM.size()<<std::endl;
       

       for(std::vector<TrajectoryMeasurement>::iterator it=vTM.begin(); it!=vTM.end(); it++)
       {
       TrajectoryMeasurement newtm;
       if( bl != 0 )
       {
        newtm = barrelUpdateSeed(theFtsMuon,(*it));
       }
         else
         {
            newtm = forwardUpdateSeed(theFtsMuon,(*it));
         }
          seedContainer.push_back(DiMuonTrajectorySeed(newtm,theFtsMuon,theLowMult));  
       }
    }       
    return  seedContainer; 
  
}      

TrajectoryMeasurement DiMuonSeedGeneratorHIC::barrelUpdateSeed ( 
                                                                  const FreeTrajectoryState& FTSOLD, 
                                                                  const TrajectoryMeasurement& tm 
							        ) const

{
  const DetLayer* dl = tm.layer();
  const TransientTrackingRecHit::ConstRecHitPointer rh = tm.recHit(); 

  if(!(rh->isValid())) 
  {
     return tm;
  }
  FreeTrajectoryState FTS = *(tm.forwardPredictedState().freeTrajectoryState());
  
//
// Find code a DetLayer from the map and remember as a code of SeedLayer for current trajectory.
//
//  MyRingedLayer * theMyRingedLayer = (*theCorrespondantLayers.find(dl)).second;
//  int icode = theMyRingedLayer->getLayerCode();
//  theMyRingedLayer->setSeedLayerCode(icode);


//
// Define local variables.
//     
  int imin = 0;
  int imax = 0;
  int imin1 = 0;
  int imax1 = 0;
  double phi = FTSOLD.parameters().position().phi(); 
  double pt = FTS.parameters().momentum().perp();
  double aCharge = FTS.parameters().charge();
  AlgebraicSymMatrix55 e = FTS.curvilinearError().matrix();
  double dpt = 3.*e(1,1);
  
//
// Calculate a bin for lower and upper boundary of PT interval available for track.  
//  
  int imax0 = (int)((pt+dpt-theHicConst->ptboun)/theHicConst->step) + 1;
  int imin0 = (int)((pt-dpt-theHicConst->ptboun)/theHicConst->step) + 1;
  if( imin0 < 1 ) imin0 = 1;
#ifdef DIMUONGENERATOR_DB	
	cout<<" imin0,imax0 "<<imin0<<" "<<imax0<<" pt,dpt "<<pt<<" "<<dpt<<endl;
#endif	

  double dens,df,ptmax,ptmin;

  GlobalPoint realhit = (*rh).globalPosition();
  df = abs(realhit.phi() - phi);
  if(df > pi) df = twopi-df;
  if(df > 1.e-5) 
  {
      dens = 1./df;
   } //end if
     else
      {
	  dens = 100000.;
      } // end else
	
	//
	// Calculate new imin, imax, pt (works till 20GeV/c)
	// It is necessary to parametrized for different Pt value with some step (to be done)
	//
	
	ptmax = (dens-(double)(theHicConst->phias[26]))/(double)(theHicConst->phibs[26]) + theHicConst->ptbmax;
	ptmin = (dens-(double)(theHicConst->phiai[26]))/(double)(theHicConst->phibi[26]) + theHicConst->ptbmax;
	imax = (int)((ptmax-theHicConst->ptboun)/theHicConst->step)+1;
	imin = (int)((ptmin-theHicConst->ptboun)/theHicConst->step)+1;
	if(imin > imax) return TrajectoryMeasurement();
	if(imax < 1) return TrajectoryMeasurement();
	imin1 = max(imin,imin0);
	imax1 = min(imax,imax0);
	if(imin1 > imax1) {
	return TrajectoryMeasurement(); // bad rhit
	}
//
// Define new trajectory. 
//
	double ptnew = theHicConst->ptboun + theHicConst->step * (imax1 + imin1)/2. - theHicConst->step/2.;  // recalculated PT of track
	
	//
	// new theta angle of track
	//
	
	double dfmax = 1./((double)(theHicConst->phias[26])+(double)(theHicConst->phibs[26])*(ptnew-theHicConst->ptbmax));
	double dfmin = 1./((double)(theHicConst->phiai[26])+(double)(theHicConst->phibi[26])*(ptnew-theHicConst->ptbmax));
	double dfcalc = abs(dfmax+dfmin)/2.;
	double phinew = phi+aCharge*dfcalc;
	
	//    
	// Recalculate phi, and Z.     
	//
	
	double rad = 100.*ptnew/(0.3*4.);
	double alf = 2.*asin(realhit.perp()/rad);
	double alfnew = phinew - aCharge*alf;
        
	//
	// Fill GlobalPoint,GlobalVector    
        //
	
	double delx = realhit.z()-theHicConst->zvert;
	double delr = sqrt( realhit.y()*realhit.y()+realhit.x()*realhit.x() );
	double theta = atan2(delr,delx);
//	
// Each trajectory in tracker starts from real point	
//	GlobalPoint xnew0( realhit.perp()*cos(phinew), realhit.perp()*sin(phinew), realhit.z() ); 

	GlobalPoint xnew0( realhit.x(), realhit.y(), realhit.z() ); 

	GlobalVector pnew0(ptnew*cos(alfnew),ptnew*sin(alfnew),ptnew/tan(theta));
	AlgebraicSymMatrix m(5,0);
	m(1,1) = 0.5*ptnew; m(2,2) = theHicConst->phiro[12];
	m(3,3) = theHicConst->tetro[12];
	m(4,4) = theHicConst->phiro[12]; 
	m(5,5) = theHicConst->tetro[12];
	   
	TrajectoryStateOnSurface updatedTsosOnDet=TrajectoryStateOnSurface
	  ( GlobalTrajectoryParameters( xnew0, pnew0, (int)aCharge, &(*magfield) ),
					      CurvilinearTrajectoryError(m), dl->surface()  );
     TransientTrackingRecHit::RecHitPointer itt = TTRHbuilder->build(&(*rh)); 
  return TrajectoryMeasurement(tm.forwardPredictedState(), updatedTsosOnDet, itt, (float)1.,dl );
  
} 
  
TrajectoryMeasurement DiMuonSeedGeneratorHIC::forwardUpdateSeed ( 
                                                                         const FreeTrajectoryState& FTSOLD, 
                                                                         const TrajectoryMeasurement& tm
							               ) const
{

  const DetLayer* dl = tm.layer();
  const TransientTrackingRecHit::ConstRecHitPointer rh = tm.recHit();   
  if(!(rh->isValid())) 
  {
     return tm;
  }
  
  FreeTrajectoryState FTS = *(tm.forwardPredictedState().freeTrajectoryState());

//
// Find code a DetLayer from the map and remember as a code of SeedLayer for current trajectory.
//
  
//  MyRingedLayer * theMyRingedLayer = (*theCorrespondantLayers.find(theDetLayer)).second;
//  int icode = theMyRingedLayer->getLayerCode();
//  theMyRingedLayer->setSeedLayerCode(icode);

//
// Define local variables.
//     

  double phi = FTSOLD.parameters().position().phi(); 
  double aCharge = FTS.parameters().charge();
  AlgebraicSymMatrix55 e = FTS.curvilinearError().matrix();
  double pt = FTS.parameters().momentum().perp();
  double pz = FTS.parameters().momentum().z();
  double dpt = 0.6*pt;
  
        GlobalPoint realhit = rh->globalPosition();
	
	double df = abs(realhit.phi() - phi);
	
//	cout<<" DiMuonSeedGeneratorHIC::forwardUpdateSeed "<<phi<<" "<<realhit.phi()<<endl;
	
	if(df > pi) df = twopi-df;
	//
	// calculate the new Pl
	//

	double delx = realhit.z() - theHicConst->zvert;
	double delr = sqrt(realhit.y()*realhit.y()+realhit.x()*realhit.x());
	double theta = atan2( delr, delx );
	double ptmin = 0.;
        double ptmax = 0.;
	double ptnew = 0.;
	double pznew = 0.;
	
// old ok  double pznew = abs((aCharge*theHicConst->forwparam[1])/(df-theHicConst->forwparam[0]));

	if( abs(FTSOLD.parameters().momentum().eta()) > 1.9 )
	{
//	   cout<<" First parametrization "<<df<<endl;
	   pznew = abs(( df - 0.0191878 )/(-0.0015952))/3.;
	   
	   if( df > 0.1 ) pznew = 5;
	   
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	if( abs(FTSOLD.parameters().momentum().eta()) > 1.7 && abs(FTSOLD.parameters().momentum().eta()) < 1.9 )
	{
//	   cout<<" Second parametrization "<<df<<endl;
	
	   pznew = abs(( df - 0.38 )/(-0.009))/3.;
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	if( abs(FTSOLD.parameters().momentum().eta()) > 1.6 && abs(FTSOLD.parameters().momentum().eta()) < 1.7 )
	{
//	   cout<<" Third parametrization "<<df<<endl;
	
	   pznew = abs(( df - 0.9 )/(-0.02))/3.;
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	if( abs(FTSOLD.parameters().momentum().eta()) > 0.7 && abs(FTSOLD.parameters().momentum().eta()) < 1.6 )
	{
//	   cout<<" Forth parametrization "<<df<<endl;
	
	   double dfinv = 0.;
	   if( df < 0.0000001 ) {
	        dfinv = 1000000.; 
	   }
	     else
	     {
	        dfinv = 1/df;
	     }	
	   ptmin = (dfinv - 4.)/0.7 + 3.;
	   if( ptmin < 2. ) ptmin = 2.;
	   ptmax = (dfinv - 0.5)/0.3 + 3.;
	   ptnew = ( ptmin + ptmax )/2.;
	   pznew = ptnew/tan( theta );
	}
	
//	cout<<" Paramters of algorithm "<<df<<" "<<theHicConst->forwparam[1]<<" "<<theHicConst->forwparam[0]<<endl;
//	cout<<" check "<<pt<<" "<<ptnew<<" "<<dpt<<" pz "<<pznew<<" "<<pz<<endl;
        //
	// Check if it is valid
	//  	
	if( (pt - ptnew)/pt < -2 || (pt - ptnew)/pt > 1 )
	{
	   return TrajectoryMeasurement(); // bad rhit
	}
	//
	// Recalculate phi, and Z.     
	//
	double alf = theHicConst->atra * ( realhit.z() - theHicConst->zvert )/abs(pznew);
	double alfnew = realhit.phi() + aCharge*alf;
	GlobalPoint xnew0(realhit.x(), realhit.y(), realhit.z()); 
	GlobalVector pnew0( ptnew*cos(alfnew), ptnew*sin(alfnew), pznew );
	
	if( abs(FTSOLD.parameters().momentum().eta()) < 1.7 && abs(FTSOLD.parameters().momentum().eta()) > 0.8 )
	{
	    if( realhit.perp() < 80. ) {
	      return TrajectoryMeasurement();
	    }  
	}

        if( FTSOLD.parameters().momentum().perp() > 2.0){
	
	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.0 && abs(FTSOLD.parameters().momentum().eta()) >= 1.7 )
	  {
	    if( realhit.perp() > 80. || realhit.perp() < 60. ) {
	      return TrajectoryMeasurement();
	  }  
	  }
	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.4 && abs(FTSOLD.parameters().momentum().eta()) >= 2.0 )
	  {
//	    if( realhit.perp() > 60. || realhit.perp() < 40. ) {	  
	    if( realhit.perp() > 75. || realhit.perp() < 40. ) {
	      return TrajectoryMeasurement();
	    }  
	  }
	  
	}  
        else  // pt<2
	{
	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.0 && abs(FTSOLD.parameters().momentum().eta()) >= 1.7 )
	  {
	    if( realhit.perp() > 84. || realhit.perp() < 40. ) {
	      return TrajectoryMeasurement();
	    }  
	  }
 	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.4 && abs(FTSOLD.parameters().momentum().eta()) >= 2.0 )
	  {
	    if( realhit.perp() > 84. || realhit.perp() < 40. ) {
	      return TrajectoryMeasurement();
	    }  
	  }
       } // pt ><2


	
	AlgebraicSymMatrix m(5,0);        
	m(1,1) = abs(0.5*pznew); m(2,2) = theHicConst->phiro[13]; 
	m(3,3) = theHicConst->tetro[13];
	m(4,4) = theHicConst->phiro[13]; 
	m(5,5) = theHicConst->tetro[13];
	
	TrajectoryStateOnSurface updatedTsosOnDet=TrajectoryStateOnSurface
	  (GlobalTrajectoryParameters( xnew0, pnew0, (int)aCharge, &(*magfield) ),
					     CurvilinearTrajectoryError(m), dl->surface() );
          TransientTrackingRecHit::RecHitPointer itt = TTRHbuilder->build(&(*rh));
  return TrajectoryMeasurement( tm.forwardPredictedState(), updatedTsosOnDet, itt,(float)1., dl );
}

