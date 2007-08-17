#include "RecoHIMuon/HiMuSeed/interface/DiMuonSeedGeneratorHIC.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "DataFormats/Common/interface/Handle.h"
//#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
//#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
//#include "RecoTracker/TkSeedingLayers/src/HitExtractorSTRP.h"

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
    					    
    SeedContainer seedContainer;
    
    bl = theTracker->barrelLayers();
    fpos = theTracker->posForwardLayers();
    fneg = theTracker->negForwardLayers();

  if(TTRHbuilder == 0){
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilderHandle;
    iSetup.get<TransientRecHitRecord>().get("WithoutRefit",theBuilderHandle);
    TTRHbuilder = theBuilderHandle.product();
  }


//    edm::Handle<SiStripRecHit2DCollection> rphirecHits0;
//    e.getByLabel( rphirecHitsTag ,rphirecHits0);
//    const SiStripRecHit2DCollection* rphirecHits = rphirecHits0.product();   
 
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
//     std::string layername;
//     SeedingLayer::Side side=SeedingLayer::Barrel;

     const BarrelDetLayer* bl = dynamic_cast<const BarrelDetLayer*>(*ml);
     const ForwardDetLayer* fl = dynamic_cast<const ForwardDetLayer*>(*ml);
      if( bl != 0 )
      { 
//       layername = "TOB6";
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
//              layername = "TEC9";
//              side = SeedingLayer::PosEndcap;
//              if(fl->position().z() < 0.) side = SeedingLayer::NegEndcap; 

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
  
        std::vector<DetLayer::DetWithState>  compatible =(*ml)->compatibleDets( newtsos, *thePropagator, theEstimator);
       
       
       std::cout<<" Size of compatible dets "<<compatible.size()<<std::endl;
       
        int measnum = 0;
//        int idLayer = 13;

//   HitExtractorSTRP extSTRP((*ml),side,idLayer);
//   extSTRP.useRPhiHits(rphirecHits);
//   HitExtractor * extractor = extSTRP.clone();
//   Look to TkSeedingLayers/src/SeedingLayerSetsBuilder.cc
//   SeedLayer = seedlayer(layername,(*ml), TTRHbuilder, hitExtractor, false, 0.,0.);
	
  for (std::vector< DetLayer::DetWithState > ::iterator dws = compatible.begin();dws != compatible.end();dws++)
    {
      const DetId presentdetid = dws->first->geographicalId();//get the det Id
      TrajectoryStateOnSurface restep = dws->second;


        std::cout<<" Stereo or not "<<dws->first->components ().size()
                             <<"(presentdetid.rawId()) "<<presentdetid.rawId()<<std::endl;

      //get the rechits on this module
      TransientTrackingRecHit::ConstRecHitContainer  thoseHits = theMeasurementTracker->idToDet(presentdetid)->recHits(restep);
      
      if( thoseHits.size() == 0 ) continue;
      
      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator it = thoseHits.begin();it != thoseHits.end(); it++)
        {
       pair<bool, TrajectoryStateOnSurface> newtm;
       const TrackingRecHit * rechit = (*it)->hit();
	
	if (!(*it)->isValid())
	{
	     std::cout<<" Invalid measurement "<< measnum <<endl;
	     measnum++;
	     continue;
	}
       if( bl != 0 )
       {
        std::cout<<" Measurement with valid rechit in barrel "<<measnum<<std::endl;
	newtm = barrelUpdateSeed(theFtsMuon,restep,(*it));
	
       }
         else
	 {
            std::cout<<" Measurement with valid rechit in endcap "<<measnum<<std::endl;
	    newtm = forwardUpdateSeed(theFtsMuon,restep,(*it));
	 }
        	edm::OwnVector<const TrackingRecHit> theRecHits;
	     if( newtm.first ) {
//	        TrajectoryStateOnSurface updateTsos = newtm.second;
                 TrajectoryStateOnSurface updateTsos = newtsos;
		int detid = presentdetid.rawId();
		std::cout<<" Mydetid "<<detid<<std::endl;
	        const TrackingRecHit* myhit = (*it)->hit();	
//		theRecHits.push_back(myhit->clone());
		
		//std::cout<<"Size of RecHits "<<theRecHits.size()<<std::endl;
		
		int theLowMult = 1;
//		DiMuonTrajectorySeed dmts(updateTsos,theFtsMuon,theRecHits,theLowMult,detid);
                DiMuonTrajectorySeed dmts(restep,theFtsMuon,myhit,theLowMult,detid);		
		std::cout<<"DiMuonTrajectorySeed created "<<std::endl; 
		
		
	        seedContainer.push_back(dmts.TrajSeed()); 
	     }	
	     measnum++; 
        } 
	
	std::cout<<" Next detector "<<std::endl;      
    }  // compatible detectors

	std::cout<<" Next Layer "<<std::endl;      
    }
    std::cout<<" Size of seedContainer "<<seedContainer.size()<<std::endl;
    return  seedContainer; 
  
}      

pair<bool, TrajectoryStateOnSurface> DiMuonSeedGeneratorHIC::barrelUpdateSeed ( 
                                                 FreeTrajectoryState& FTSOLD,
						 TrajectoryStateOnSurface& tsos, 
                                                 const TransientTrackingRecHit::ConstRecHitPointer rh 
					       ) const

{

  std::cout<<" Start barrelUpdateSeed "<<std::endl;
  FreeTrajectoryState FTS = *(tsos.freeTrajectoryState());
  
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
  
  std::cout<<" Point 0 "<<std::endl;  
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
  
  std::cout<<" Point 1 "<<std::endl; 
  
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
	
  std::cout<<" Point 1.1 "<<std::endl;	
	
	if(imin > imax) return std::make_pair(false, tsos);
	if(imax < 1) return std::make_pair(false, tsos);
	imin1 = max(imin,imin0);
	imax1 = min(imax,imax0);
	if(imin1 > imax1) {
        std::cout<<" Bad rechit "<<std::endl;
	return std::make_pair(false, tsos); // bad rhit
	}  
	
	std::cout<<" Point 2 "<<std::endl;
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
	
	std::cout<<" Point 3 "<<std::endl;
//	
// Each trajectory in tracker starts from real point	
//	GlobalPoint xnew0( realhit.perp()*cos(phinew), realhit.perp()*sin(phinew), realhit.z() ); 

	GlobalPoint xnew0( realhit.x(), realhit.y(), realhit.z() ); 

	GlobalVector pnew0(ptnew*cos(alfnew),ptnew*sin(alfnew),ptnew/tan(theta));
//	AlgebraicSymMatrix m(5,0);
//	m(1,1) = 0.5*ptnew; 
//        m(2,2) = theHicConst->phiro[12];
//	m(3,3) = theHicConst->tetro[12];
//	m(4,4) = theHicConst->phiro[12]; 
//	m(5,5) = theHicConst->tetro[12];
//	
//	std::cout<<" Point 4 "<<std::endl;
//		   
//	TrajectoryStateOnSurface updatedTsosOnDet=TrajectoryStateOnSurface
//	  ( GlobalTrajectoryParameters( xnew0, pnew0, (int)aCharge, &(*magfield) ),
//					      CurvilinearTrajectoryError(m), dl->surface()  );
					           
     cout<<" Updated TM barrel "<<endl;
     
//  return std::make_pair(true,updatedTsosOnDet);
     return std::make_pair(true, tsos); 
} 
  
pair<bool, TrajectoryStateOnSurface> DiMuonSeedGeneratorHIC::forwardUpdateSeed ( 
                                                                         FreeTrajectoryState& FTSOLD, 
									 TrajectoryStateOnSurface& tsos,
                                                                         const TransientTrackingRecHit::ConstRecHitPointer rh
							               ) const
{
  
  std::cout<<" Start endcapUpdateSeed "<<std::endl; 
  FreeTrajectoryState FTS = *(tsos.freeTrajectoryState());   

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
  
  std::cout<<" Point 0 "<<std::endl;
  
        GlobalPoint realhit = rh->globalPosition();

  std::cout<<" Point 1 "<<std::endl;
	
	double df = abs(realhit.phi() - phi);
	
	cout<<" DiMuonSeedGeneratorHIC::forwardUpdateSeed "<<phi<<" "<<realhit.phi()<<endl;
	
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
	
  std::cout<<" Point 2 "<<std::endl;
	
// old ok  double pznew = abs((aCharge*theHicConst->forwparam[1])/(df-theHicConst->forwparam[0]));

	if( abs(FTSOLD.parameters().momentum().eta()) > 1.9 )
	{
//	   cout<<" First parametrization "<<df<<endl;
	   pznew = abs(( df - 0.0191878 )/(-0.0015952))/3.;
	   
	   if( df > 0.1 ) pznew = 5;
	   
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	
 std::cout<<" Point 3 "<<std::endl;
	
	if( abs(FTSOLD.parameters().momentum().eta()) > 1.7 && abs(FTSOLD.parameters().momentum().eta()) < 1.9 )
	{
//	   cout<<" Second parametrization "<<df<<endl;
	
	   pznew = abs(( df - 0.38 )/(-0.009))/3.;
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}

 std::cout<<" Point 4 "<<std::endl;

	if( abs(FTSOLD.parameters().momentum().eta()) > 1.6 && abs(FTSOLD.parameters().momentum().eta()) < 1.7 )
	{
//	   cout<<" Third parametrization "<<df<<endl;
	
	   pznew = abs(( df - 0.9 )/(-0.02))/3.;
           if( FTSOLD.parameters().position().z() < 0. ) pznew = (-1)*pznew;
	   ptnew = pznew * tan( theta );
	}
	
 std::cout<<" Point 5 "<<std::endl;
	
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
 std::cout<<" Point 6 "<<std::endl;
	
//	cout<<" Paramters of algorithm "<<df<<" "<<theHicConst->forwparam[1]<<" "<<theHicConst->forwparam[0]<<endl;
//	cout<<" check "<<pt<<" "<<ptnew<<" "<<dpt<<" pz "<<pznew<<" "<<pz<<endl;
        //
	// Check if it is valid
	//  	
	if( (pt - ptnew)/pt < -2 || (pt - ptnew)/pt > 1 )
	{
	   return std::make_pair(false, tsos);; // bad rhit
	}
	//
	// Recalculate phi, and Z.     
	//
	
 std::cout<<" Point 7 "<<std::endl;
	
	double alf = theHicConst->atra * ( realhit.z() - theHicConst->zvert )/abs(pznew);
	double alfnew = realhit.phi() + aCharge*alf;
	GlobalPoint xnew0(realhit.x(), realhit.y(), realhit.z()); 
	GlobalVector pnew0( ptnew*cos(alfnew), ptnew*sin(alfnew), pznew );
	
 std::cout<<" Point 8 "<<std::endl;
	
	if( abs(FTSOLD.parameters().momentum().eta()) < 1.7 && abs(FTSOLD.parameters().momentum().eta()) > 0.8 )
	{
	    if( realhit.perp() < 80. ) {
	      return std::make_pair(false, tsos);
	    }  
	}
 std::cout<<" Point 9 "<<std::endl;

        if( FTSOLD.parameters().momentum().perp() > 2.0){
 std::cout<<" Point 9.1 "<<std::endl;	
	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.0 && abs(FTSOLD.parameters().momentum().eta()) >= 1.7 )
	  {
 std::cout<<" Point 9.2 "<<std::endl;	    
	    if( realhit.perp() > 80. || realhit.perp() < 60. ) {
 std::cout<<" Point 9.3 "<<std::endl;	    
	      return std::make_pair(false, tsos);
	  }  
	  }
	  
	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.4 && abs(FTSOLD.parameters().momentum().eta()) >= 2.0 )
	  {
 std::cout<<" Point 9.4 "<<std::endl;	  
	    if( realhit.perp() > 75. || realhit.perp() < 40. ) {
	    
 std::cout<<" Point 9.5 "<<std::endl;
 	      return std::make_pair(false, tsos);
	    }  
	  }
	  
	}  
        else  // pt<2
	{
 std::cout<<" Point 9.6 "<<std::endl;	
	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.0 && abs(FTSOLD.parameters().momentum().eta()) >= 1.7 )
	  {
 std::cout<<" Point 9.7 "<<std::endl;	
	  
	    if( realhit.perp() > 84. || realhit.perp() < 40. ) {
 std::cout<<" Point 9.8 "<<std::endl;	
	    
	      return std::make_pair(false, tsos);
	    }  
	  }
 	  if( abs(FTSOLD.parameters().momentum().eta()) < 2.4 && abs(FTSOLD.parameters().momentum().eta()) >= 2.0 )
	  {
 std::cout<<" Point 9.9 "<<std::endl;	
	  
	    if( realhit.perp() > 84. || realhit.perp() < 40. ) {
 std::cout<<" Point 9.10 "<<std::endl;	
	    
	      return std::make_pair(false, tsos);
	    }  
	  }
       } // pt ><2


 std::cout<<" Point 10 "<<std::endl;
	
//	AlgebraicSymMatrix m(5,0);        
//	m(1,1) = abs(0.5*pznew); m(2,2) = theHicConst->phiro[13]; 
//	m(3,3) = theHicConst->tetro[13];
//	m(4,4) = theHicConst->phiro[13]; 
//	m(5,5) = theHicConst->tetro[13];
//	
  // std::cout<<" Point 11 "<<std::endl;
   //	
//	TrajectoryStateOnSurface updatedTsosOnDet=TrajectoryStateOnSurface
//	  (GlobalTrajectoryParameters( xnew0, pnew0, (int)aCharge, &(*magfield) ),
//					     CurvilinearTrajectoryError(m), dl->surface() );
//          TransientTrackingRecHit::RecHitPointer itt = TTRHbuilder->build(&(*rh));
	  
	  cout<<" Updated TM endcap "<<endl;
	  
//  return std::make_pair(true, updatedTsosOnDet);
    return std::make_pair(true, tsos);
}

