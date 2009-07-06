#include "RecoHIMuon/HiMuSeed/interface/DiMuonSeedGeneratorHIC.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "DataFormats/Common/interface/Handle.h"


using namespace edm;
using namespace std;
//#define DEBUG

namespace cms {
DiMuonSeedGeneratorHIC::DiMuonSeedGeneratorHIC(edm::InputTag rphirecHitsTag0,
                                               const MagneticField* magfield0, 
                                               const GeometricSearchTracker* theTracker0, 
                                               const HICConst* hh,
                                               const string bb,
					       int aMult = 1)
{
  
// initialization

  theTracker = theTracker0;
  theHICConst = hh;
  theLowMult = aMult;
  magfield = magfield0;
  rphirecHitsTag = rphirecHitsTag0;
  TTRHbuilder = 0;
  builderName = bb; 
  
  thePropagator=new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );
  
}

map<DetLayer*,DiMuonSeedGeneratorHIC::SeedContainer> DiMuonSeedGeneratorHIC::produce(const edm::Event& e, const edm::EventSetup& iSetup, 
                                                    FreeTrajectoryState& theFtsTracker, 
						    TrajectoryStateOnSurface& newtsos,
						    FreeTrajectoryState& theFtsMuon,
						    const TransientTrackingRecHitBuilder* RecHitBuilder,
						    const MeasurementTracker* measurementTracker,
						    vector<DetLayer*>* theDetLayer ) {
						    

    theMeasurementTracker = measurementTracker;
    theLayerMeasurements = new LayerMeasurements(theMeasurementTracker);

  //  cout<<" Point 0 "<<endl;
					    
    std::map<DetLayer*,DiMuonSeedGeneratorHIC::SeedContainer> seedMap;
//    SeedContainer seedContainer;
    
    bl = theTracker->barrelLayers();
    fpos = theTracker->posForwardLayers();
    fneg = theTracker->negForwardLayers();

  if(TTRHbuilder == 0){
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilderHandle;
    iSetup.get<TransientRecHitRecord>().get("WithoutRefit",theBuilderHandle);
    iSetup.get<TransientRecHitRecord>().get(builderName,theBuilderHandle);
    TTRHbuilder = theBuilderHandle.product();
  }
//  cout<<" Point 1 "<<endl;   
//    int npair=0;
//
//  For each fts do a cycle on possible layers.
//
    int nSig =3;
    bool itrust=true; 
    HICSeedMeasurementEstimator theEstimator(itrust,nSig);
   
   // cout<<" Point 2 "<<endl;
 
    double phipred=0.; double zpred=0.; double phiupd=0.;double zupd=0.;
    
    for( vector<DetLayer* >::const_iterator ml=theDetLayer->begin(); ml != theDetLayer->end(); ml++)
    {
//
// For each layer, we have its own seed container
//
     SeedContainer seedContainer;

     const BarrelDetLayer* bl = dynamic_cast<const BarrelDetLayer*>(*ml);
     const ForwardDetLayer* fl = dynamic_cast<const ForwardDetLayer*>(*ml);
      if( bl != 0 )
      { 
       // The last layer for barrel
       phipred = (double)theHICConst->phicut[12];
       phiupd = (double)theHICConst->phiro[12];
       zpred = (double)theHICConst->zcut[12];
       zupd = (double)theHICConst->tetro[12];
     //  cout<<" DiMuonSeedGenerator::propagate to barrel layer surface "<<bl->specificSurface().radius()<<endl;
      }
        else
         {
	   if ( fl != 0 ) 
           {

              // The last layer for endcap
              phipred = (double)theHICConst->phicutf[13];
              phiupd = (double)theHICConst->phirof[13];
              zpred = (double)theHICConst->tetcutf[13];
              zupd = (double)theHICConst->tetrof[13];
	      
       //       cout<<" DiMuonSeedGenerator::propagate to endcap layer surface "<<fl->position().z()<<" Mult="<<
//              theLowMult<<endl;
           } // end fl
         }// end else
// ==========================================       
       theEstimator.set(phipred, zpred);
#ifdef DEBUG       
       std::cout<<" DiMuonSeedGenerator::estimator::set cuts "<<std::endl;
#endif
       std::vector<TrajectoryMeasurement> vTM = theLayerMeasurements->measurements((**ml),newtsos, *thePropagator, theEstimator);
#ifdef DEBUG
       std::cout<<" DiMuonSeedGenerator::Size of compatible TM found by measurements "<<vTM.size()<<std::endl;
#endif
       
//        int measnum = 0;

       for(std::vector<TrajectoryMeasurement>::iterator it=vTM.begin(); it!=vTM.end(); it++)
       {
       
#ifdef DEBUG
      const TransientTrackingRecHit::ConstRecHitPointer rh = (*it).recHit();
      if(!(rh->isValid())) {
        cout<<" DiMuonSeedGenerator::rechit not valid "<<endl; 
      } else {
	GlobalPoint realhit = (*rh).globalPosition();
       cout<<" DiMuonSeedGenerator::Compatible TM: "<<realhit.perp()<<" "
       <<realhit.phi()<<" "<<realhit.z()<<endl;
      }
#endif       
       
       pair<TrajectoryMeasurement,bool> newtmr;

       if( bl != 0 )
       {
#ifdef DEBUG
        cout<<" DiMuonSeedGenerator::Barrel seed "<<endl;
#endif
        newtmr = barrelUpdateSeed(theFtsMuon,(*it));

       }
         else
         {
#ifdef DEBUG
            cout<<" DiMuonSeedGenerator::Endcap seed "<<endl;
#endif
            newtmr = forwardUpdateSeed(theFtsMuon,(*it));
         }
#ifdef DEBUG
           cout<<" DiMuonSeedGenerator::Estimate seed "<<newtmr.first.estimate()<<" True or false  "<<newtmr.second<<endl;
#endif
          if(newtmr.second) seedContainer.push_back(DiMuonTrajectorySeed(newtmr.first,theFtsMuon,theLowMult));
       }
       seedMap[*ml] = seedContainer;
    }       
    delete theLayerMeasurements;
    return  seedMap; 
  
}      

pair<TrajectoryMeasurement,bool> DiMuonSeedGeneratorHIC::barrelUpdateSeed ( 
                                                                  const FreeTrajectoryState& FTSOLD, 
                                                                  const TrajectoryMeasurement& tm 
							        ) const
{

  bool good=false;
#ifdef DEBUG
  std::cout<<" DiMuonSeedGeneratorHIC::barrelUpdateSeed::BarrelSeed "<<std::endl;
#endif
  const DetLayer* dl = tm.layer();
 // std::cout<<" BarrelSeed 0"<<std::endl;  
  const TransientTrackingRecHit::ConstRecHitPointer rh = tm.recHit(); 
 // std::cout<<" BarrelSeed 1"<<std::endl;
  if(!(rh->isValid())) 
  {
#ifdef DEBUG
     std::cout<<" DiMuonSeedGeneratorHIC::barrelUpdateSeed::hit is not valid "<<std::endl;
#endif
     return pair<TrajectoryMeasurement,bool>(tm,good);
  }
#ifdef DEBUG
     std::cout<<" DiMuonSeedGeneratorHIC::barrelUpdateSeed::hit is  valid "<<std::endl;
#endif
//  std::cout<<" BarrelSeed 2"<<std::endl;
  FreeTrajectoryState FTS = *(tm.forwardPredictedState().freeTrajectoryState());

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

  double dptup = 0.35*pt;
  double dptdown = 0.7*pt;
  double ptshift = 0.22*pt;
  
//  std::cout<<" BarrelSeed 3 "<<std::endl;  
//
// Calculate a bin for lower and upper boundary of PT interval available for track.  
//  
  int imax0 = (int)((pt+ptshift+dptup-theHICConst->ptboun)/theHICConst->step) + 1;
  int imin0 = (int)((pt+ptshift-dptdown-theHICConst->ptboun)/theHICConst->step) + 1;
  if( imin0 < 1 ) imin0 = 1;
#ifdef DEBUG	
	std::cout<<" DiMuonSeedGeneratorHIC::barrelUpdateSeed::imin0,imax0 "<<imin0<<" "<<imax0<<" pt,dpt "<<pt+ptshift<<" "<<dptup<<" "<<dptdown<<std::endl;
#endif	

  double dens,df,ptmax,ptmin;

  GlobalPoint realhit = (*rh).globalPosition();
  df = fabs(realhit.phi() - phi);

  double pi=4.*atan(1.);
  double twopi=8.*atan(1.);

  if(df > pi) df = twopi-df;
  if(df > 1.e-5) 
  {
      dens = 1./df;
   } //end if
     else
      {
	  dens = 100000.;
      } // end else
  //   std::cout<<" Phi rh "<<realhit.phi()<<" phumu "<<phi<<" df "<<df<<" dens "<<dens<<std::endl;	
	//
	// Calculate new imin, imax, pt (works till 20GeV/c)
	// It is necessary to parametrized for different Pt value with some step (to be done)
	//
	
	ptmax = (dens-(double)(theHICConst->phias[26]))/(double)(theHICConst->phibs[26]) + theHICConst->ptbmax;
	ptmin = (dens-(double)(theHICConst->phiai[26]))/(double)(theHICConst->phibi[26]) + theHICConst->ptbmax;
#ifdef DEBUG
     std::cout<<" Phias,phibs,phiai,phibi "<<theHICConst->phias[26]<<" "<<theHICConst->phibs[26]<<" "<<
     theHICConst->phiai[26]<<" "<<theHICConst->phibi[26]<<" "<<theHICConst->ptbmax<<std::endl;
     std::cout<<" ptmin= "<<ptmin<<" ptmax "<<ptmax<<std::endl;
     std::cout<<" ptboun "<<theHICConst->ptboun<<" "<<theHICConst->step<<std::endl;
#endif 
	imax = (int)((ptmax-theHICConst->ptboun)/theHICConst->step)+1;
	imin = (int)((ptmin-theHICConst->ptboun)/theHICConst->step)+1;
	if(imin > imax) {
#ifdef DEBUG	
                std::cout<<" imin>imax "<<imin<<" "<<imax<<std::endl; 
#endif		
           return pair<TrajectoryMeasurement,bool>(tm,good);}
	if(imax < 1) { 
#ifdef DEBUG	
              std::cout<<"imax < 1 "<<imax<<std::endl; 
#endif	      
              return pair<TrajectoryMeasurement,bool>(tm,good);}

	imin1 = max(imin,imin0);
	imax1 = min(imax,imax0);
	if(imin1 > imax1) {
#ifdef DEBUG	
         std::cout<<" imin,imax "<<imin<<" "<<imax<<std::endl; 
         std::cout<<" imin,imax "<<imin0<<" "<<imax0<<std::endl;
         std::cout<<" imin1>imax1 "<<imin1<<" "<<imax1<<std::endl;
#endif	
         return pair<TrajectoryMeasurement,bool>(tm,good);
	}

//
// Define new trajectory. 
//
	double ptnew = theHICConst->ptboun + theHICConst->step * (imax1 + imin1)/2. - theHICConst->step/2.;  // recalculated PT of track
	
	//
	// new theta angle of track
	//
	
	double dfmax = 1./((double)(theHICConst->phias[26])+(double)(theHICConst->phibs[26])*(ptnew-theHICConst->ptbmax));
	double dfmin = 1./((double)(theHICConst->phiai[26])+(double)(theHICConst->phibi[26])*(ptnew-theHICConst->ptbmax));
	double dfcalc = fabs(dfmax+dfmin)/2.;
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
	
	double delx = realhit.z()-theHICConst->zvert;
	double delr = sqrt( realhit.y()*realhit.y()+realhit.x()*realhit.x() );
	double theta = atan2(delr,delx);	
	
//	std::cout<<" Point 3 "<<std::endl;
//	
// Each trajectory in tracker starts from real point	
//	GlobalPoint xnew0( realhit.perp()*cos(phinew), realhit.perp()*sin(phinew), realhit.z() ); 

	GlobalPoint xnew0( realhit.x(), realhit.y(), realhit.z() ); 

	GlobalVector pnew0(ptnew*cos(alfnew),ptnew*sin(alfnew),ptnew/tan(theta));

	AlgebraicSymMatrix55 m;
	m(0,0) = 0.5*ptnew; m(1,1) = theHICConst->phiro[12];
	m(2,2) = theHICConst->tetro[12];
	m(3,3) = theHICConst->phiro[12]; 
	m(4,4) = theHICConst->tetro[12];
	   
	TrajectoryStateOnSurface updatedTsosOnDet=TrajectoryStateOnSurface
	  ( GlobalTrajectoryParameters( xnew0, pnew0, (int)aCharge, &(*magfield) ),
					      CurvilinearTrajectoryError(m), dl->surface()  );
  
     float estimate = 1.;
     TrajectoryMeasurement newtm(tm.forwardPredictedState(), updatedTsosOnDet, rh, estimate, dl );
     good=true;
     pair<TrajectoryMeasurement,bool> newtmr(newtm,good);
    // std::cout<<" Barrel newtm estimate= "<<newtmr.first.estimate()<<" "<<newtmr.second<<std::endl; 
  return newtmr;
} 
  
pair<TrajectoryMeasurement,bool> DiMuonSeedGeneratorHIC::forwardUpdateSeed ( 
                                                                         const FreeTrajectoryState& FTSOLD, 
                                                                         const TrajectoryMeasurement& tm
							               ) const
{
  bool good=false;
#ifdef DEBUG
  std::cout<<" DiMuonSeedGeneratorHIC::forwardUpdateSeed::EndcapSeed::start "<<std::endl;
#endif
  const DetLayer* dl = tm.layer();
//  std::cout<<" EndcapSeed 0"<<std::endl;
  const TransientTrackingRecHit::ConstRecHitPointer rh = tm.recHit();
//  std::cout<<" EndcapSeed 1"<<std::endl;   
  if(!(rh->isValid())) 
  {
#ifdef DEBUG
  std::cout<<" DiMuonSeedGeneratorHIC::forwardUpdateSeed::EndcapSeed::EndcapSeed::hit is not valid "<<std::endl;
#endif

     return pair<TrajectoryMeasurement,bool>(tm,good);
  }

#ifdef DEBUG
  std::cout<<" DiMuonSeedGeneratorHIC::forwardUpdateSeed::EndcapSeed::EndcapSeed::valid "<<std::endl;
#endif 

  FreeTrajectoryState FTS = *(tm.forwardPredictedState().freeTrajectoryState());

//
// Define local variables.
//     

  double phi = FTSOLD.parameters().position().phi(); 
  double aCharge = FTS.parameters().charge();
  AlgebraicSymMatrix55 e = FTS.curvilinearError().matrix();
  double pt = FTS.parameters().momentum().perp();
 // double pz = FTS.parameters().momentum().z();
 // double dpt = 0.6*pt;
  
//  std::cout<<" Point 0 "<<std::endl;
  double pi=4.*atan(1.);
  double twopi=8.*atan(1.);
 
 
        GlobalPoint realhit = rh->globalPosition();

//  std::cout<<" Point 1 "<<std::endl;
	
	double df = fabs(realhit.phi() - phi);

	if(df > pi) df = twopi-df;

#ifdef DEBUG
        cout<<" DiMuonSeedGeneratorHIC::forwardUpdateSeed::phipred::phihit::df "<<phi<<" "<<realhit.phi()<<" "<<df<<endl;
#endif

	//
	// calculate the new Pl
	//

	double delx = realhit.z() - theHICConst->zvert;
	double delr = sqrt(realhit.y()*realhit.y()+realhit.x()*realhit.x());
	double theta = atan2( delr, delx );
	double ptmin = 0.;
        double ptmax = 0.;
	double ptnew = 0.;
	double pznew = 0.;
	
// old ok  double pznew = abs((aCharge*theHicConst->forwparam[1])/(df-theHicConst->forwparam[0]));

	if( fabs(FTSOLD.parameters().momentum().eta()) > 1.9 )
	{
#ifdef DEBUG	
	   cout<<" First parametrization "<<df<<endl;
#endif	   
	   pznew = fabs(( df - 0.0191878 )/(-0.0015952))/3.;
	   
	   if( df > 0.1 ) pznew = 5.;
	   if( fabs(pznew)<3.) pznew = 3.;
	   
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	if( fabs(FTSOLD.parameters().momentum().eta()) > 1.7 && fabs(FTSOLD.parameters().momentum().eta()) < 1.9 )
	{
#ifdef DEBUG	
	   cout<<" Second parametrization "<<df<<endl;
#endif	
	   pznew = fabs(( df - 0.38 )/(-0.009))/3.;
	   if( fabs(pznew)<2.) pznew = 2.;
	   
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	if( fabs(FTSOLD.parameters().momentum().eta()) > 1.6 && fabs(FTSOLD.parameters().momentum().eta()) < 1.7 )
	{
#ifdef DEBUG	
	   cout<<" Third parametrization "<<df<<endl;
#endif	
	   pznew = fabs(( df - 0.9 )/(-0.02))/3.;
	   if( fabs(pznew)<1.) pznew = 1.;
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	if( fabs(FTSOLD.parameters().momentum().eta()) > 0.7 && fabs(FTSOLD.parameters().momentum().eta()) < 1.6 )
	{
#ifdef DEBUG
	   cout<<" Forth parametrization "<<df<<endl;
#endif	
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
	
// std::cout<<" Point 6 "<<std::endl;
#ifdef DEBUG	
	std::cout<<" Paramters of algorithm "<<df<<" "<<theHICConst->forwparam[1]<<" "<<theHICConst->forwparam[0]<<std::endl;
        std::cout<<" dfinv  "<<dfinv<<" ptmax "<<ptmax<<" ptmin "<<ptmin<<std::endl;
//	std::cout<<" check "<<pt<<" "<<ptnew<<" "<<dpt<<" pz "<<pznew<<" "<<pz<<std::endl;
#endif
        }
        //
	// Check if it is valid
	//  	
	if( (pt - ptnew)/pt < -2 || (pt - ptnew)/pt > 1 )
	{
#ifdef DEBUG
            cout<<" Return fake 0 pt::ptnew "<<pt<<" "<<ptnew<<endl;
#endif
	   return pair<TrajectoryMeasurement,bool>(tm,good); // bad rhit
	}
    //        cout<<" Start recalculation 0 "<<endl;
	//
	// Recalculate phi, and Z.     
	//
	double alf = theHICConst->atra * ( realhit.z() - theHICConst->zvert )/fabs(pznew);
	double alfnew = realhit.phi() + aCharge*alf;
	GlobalPoint xnew0(realhit.x(), realhit.y(), realhit.z()); 
	GlobalVector pnew0( ptnew*cos(alfnew), ptnew*sin(alfnew), pznew );
#ifdef DEBUG	
        cout<<" Start recalculation 1 FTSOLD eta, r hit, pt "<<FTSOLD.parameters().momentum().eta()<<" "<<realhit.perp()<<
                                                                   " "<<FTSOLD.parameters().momentum().perp()<<endl;	
#endif	
	if( fabs(FTSOLD.parameters().momentum().eta()) < 1.7 && fabs(FTSOLD.parameters().momentum().eta()) > 0.8 )
	{
	    if( realhit.perp() < 80. ) {
//	    if( realhit.perp() < 72. ) {
#ifdef DEBUG
              cout<<" Return fake 1 "<<realhit.perp()<<endl;
#endif	      
	      return pair<TrajectoryMeasurement,bool>(tm,good);
	    }  
	}
// std::cout<<" Point 9 "<<std::endl;

        if( FTSOLD.parameters().momentum().perp() > 2.0){
	  if( fabs(FTSOLD.parameters().momentum().eta()) < 2.0 && fabs(FTSOLD.parameters().momentum().eta()) >= 1.7 )
	  {
	    if( realhit.perp() > 100. || realhit.perp() < 60. ) {
#ifdef DEBUG	    
              cout<<" Return fake 2 "<<endl;
#endif	      
	      return pair<TrajectoryMeasurement,bool>(tm,good);
	  }  
	  }
	  if( fabs(FTSOLD.parameters().momentum().eta()) < 2.4 && fabs(FTSOLD.parameters().momentum().eta()) >= 2.0 )
	  {
	    if( realhit.perp() > 75. || realhit.perp() < 40. ) {
//	    if( realhit.perp() > 82. || realhit.perp() < 40. ) {
#ifdef DEBUG	    
              cout<<" Return fake 3 "<<endl;
#endif	      
	      return pair<TrajectoryMeasurement,bool>(tm,good);
	    }  
	  }
	  
	}  
        else  // pt<2
	{
	  if( fabs(FTSOLD.parameters().momentum().eta()) < 2.0 && fabs(FTSOLD.parameters().momentum().eta()) >= 1.7 )
	  {	  
	    if( realhit.perp() > 84. || realhit.perp() < 40. ) {
#ifdef DEBUG	    
              cout<<" Return fake 4 "<<endl;
#endif	      
	      return pair<TrajectoryMeasurement,bool>(tm,good);
	    }  
	  }
 	  if( fabs(FTSOLD.parameters().momentum().eta()) < 2.4 && fabs(FTSOLD.parameters().momentum().eta()) >= 2.0 )
	  {
	    if( realhit.perp() > 84. || realhit.perp() < 40. ) {
#ifdef DEBUG	    
              cout<<" Return fake 5 "<<endl;
#endif	      
	      return pair<TrajectoryMeasurement,bool>(tm,good);
	    }  
	  }
       } // pt ><2
#ifdef DEBUG
          cout<<" Create new TM "<<endl;
#endif	
	AlgebraicSymMatrix55 m;        
	m(0,0) = fabs(0.5*pznew); 
	m(1,1) = theHICConst->phiro[13]; 
	m(2,2) = theHICConst->tetro[13];
	m(3,3) = theHICConst->phiro[13]; 
	m(4,4) = theHICConst->tetro[13];
	
	TrajectoryStateOnSurface updatedTsosOnDet=TrajectoryStateOnSurface
	  (GlobalTrajectoryParameters( xnew0, pnew0, (int)aCharge, &(*magfield) ),
					     CurvilinearTrajectoryError(m), dl->surface() );

       float estimate=1.;
     TrajectoryMeasurement newtm(tm.forwardPredictedState(), updatedTsosOnDet, rh,estimate, dl);
     good=true;
      pair<TrajectoryMeasurement,bool> newtmr(newtm,good);
#ifdef DEBUG      
     std::cout<<" Endcap newtm estimate= "<<newtmr.first.estimate()<<" "<<newtmr.second<<" pt "<<pnew0.perp()<<" pz "<<pnew0.z()<<std::endl;
#endif
  return newtmr;
}
}
