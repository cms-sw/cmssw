#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoHIMuon/HiMuTracking/interface/HICMeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
using namespace edm;
using namespace std;

std::pair<bool,double> 
HICMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TransientTrackingRecHit& aRecHit) const {
    switch (aRecHit.dimension()) {
        case 1: return estimate<1>(tsos,aRecHit);
        case 2: return estimate<2>(tsos,aRecHit);
        case 3: return estimate<3>(tsos,aRecHit);
        case 4: return estimate<4>(tsos,aRecHit);
        case 5: return estimate<5>(tsos,aRecHit);
    }
    throw cms::Exception("RecHit of invalid size (not 1,2,3,4,5)");
}

template <unsigned int D> std::pair<bool,double> 
HICMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TransientTrackingRecHit& aRecHit) const {
  typedef typename AlgebraicROOTObject<D>::Vector Vec;
  typedef typename AlgebraicROOTObject<D>::SymMatrix Mat;
  double est = 0.;
// If RecHit is not valid   
  if(!(aRecHit.isValid())) return HitReturnType(false,est); 
// Check windows

  double dphi = abs(tsos.freeTrajectoryState()->parameters().position().phi() - aRecHit.globalPosition().phi() - thePhiBoundMean);
  double dz = abs( tsos.freeTrajectoryState()->parameters().position().z() - aRecHit.globalPosition().z() - theZBoundMean );
  double dr = abs( tsos.freeTrajectoryState()->parameters().position().perp() - aRecHit.globalPosition().perp() - theZBoundMean );

  if( dphi > thePhiBound ) {
    return HitReturnType(false,est);
  }
  if( dz > theZBound ) {
    return HitReturnType(false,est);
  }
  if( dr > theZBound ) {
    return HitReturnType(false,est);
  }

    
  MeasurementExtractor me(tsos);
  Vec r = asSVector<D>(aRecHit.parameters()) - me.measuredParameters<D>(aRecHit);
  Mat R = asSMatrix<D>(aRecHit.parametersError()) + me.measuredError<D>(aRecHit);
  //int ierr = ! R.Invert(); // if (ierr != 0) throw exception; // 
  R.Invert();
  est = ROOT::Math::Similarity(r, R);
  
    if( est > theChi2Cut )
    {
      return HitReturnType(false,est);
    }
  
  return HitReturnType(true,est);
}

bool HICMeasurementEstimator::estimate( const TrajectoryStateOnSurface& ts, 
					const BoundPlane& plane) const
{

//  cout<<" start estimate plane "<<endl;
  double pi = 4.*atan(1.);
  double twopi = 2.*pi;
  float theZError = plane.bounds().length() + 4.;
  float thePhiError = 2.*plane.bounds().width()/plane.position().perp();

#ifdef HICMEASUREMENTESTIMATOR_DEBUG  
//  if( theTrue == 1 )
//  {				      

  cout<<" ======================================================================================== ";
  cout<<" Estimate detector::   tsos      :     detector   :   Error "<<endl;
  cout<<" R                 "<<ts.globalPosition().perp()<<" "<<plane.position().perp()<<" "<<theZError<<endl;
  cout<<" Phi               "<<ts.globalPosition().phi()<<" "<<plane.position().phi()<<" "<<thePhiError<<endl;
  cout<<" Z                 "<<ts.globalPosition().z()<<" "<<plane.position().z()<<" "<<theZError<<endl;
//  }
#endif

  bool flag = false;
  if(abs(ts.globalPosition().perp()-plane.position().perp())<theZError){
   if(abs(ts.globalPosition().z()-plane.position().z())<theZError){
   float phi1 = ts.globalPosition().phi();
   float phi2 = plane.position().phi();
   if(phi1<0.) phi1 = twopi+phi1;
   if(phi2<0.) phi2 = twopi+phi2;
   float dfi = abs(phi1-phi2);
   if(dfi>pi) dfi = twopi-dfi;
      if(dfi<thePhiError) flag = true;
   }
  }
#ifdef HICMEASUREMENTESTIMATOR_DEBUG
//  if( theTrue == 1 )
//  {				      
    cout<<" Estimate = "<<flag<<endl;
//  }
#endif

  return flag;
  
}

vector<double> HICMeasurementEstimator::setCuts(Trajectory& traj, const DetLayer* b)
{
     vector<double> theCuts;
     const DetLayer* a = traj.data().back().layer();
     const DetLayer* first = traj.data().front().layer();
     const DetLayer* last = traj.data().front().layer();
     thePhiWinMean = 0.;
     theZWinMean = 0.;
     thePhiWin = 0.;
     theZWin = 0.;
     theNewCut = 5.;
     theNewCutB = 5.;
     
     thePhiWinMeanB = 0.002;
     theZWinMeanB = 0.;
     thePhiWinB = 0.008;
     theZWinB = 12.;
          
     theZCutMean = 0.;
     thePhiCutMean = 0.;
     thePhiCut = 0.;
     theZCut = 0.;
     
     theLayer = b;
     theLastLayer = a;

     
     theTrajectorySize = traj.data().size();
     
     
     if( theBarrel.size() == 0 || theForward.size() == 0 )
     {
       cout<<" HICMeasurementEstimator::setCuts:: no datector map "<<endl;
        return theCuts;
     }
     
     if( a->location() == GeomDetEnumerators::barrel )
     {
       if( first->location() == GeomDetEnumerators::barrel ) 
       {
        thePhiWin = theHicConst.phiwinbar[(*theBarrel.find(first)).second][(*theBarrel.find(a)).second][(*theBarrel.find(b)).second];
        theZWin = theHicConst.zwinbar[(*theBarrel.find(first)).second][(*theBarrel.find(a)).second][(*theBarrel.find(b)).second];
        thePhiCut = theHicConst.phicutbar[(*theBarrel.find(first)).second][(*theBarrel.find(theLastLayer)).second][(*theBarrel.find(b)).second];
        theZCut = theHicConst.zcutbar[(*theBarrel.find(first)).second][(*theBarrel.find(theLastLayer)).second][(*theBarrel.find(b)).second];

       }
         else
	 {
          thePhiWin = theHicConst.phiwinfbb[(*theForward.find(first)).second][(*theBarrel.find(a)).second][(*theBarrel.find(b)).second];
          theZWin = theHicConst.zwinfbb[(*theForward.find(first)).second][(*theBarrel.find(a)).second][(*theBarrel.find(b)).second];
          thePhiCut = theHicConst.phicutfbb[(*theForward.find(first)).second][(*theBarrel.find(theLastLayer)).second][(*theBarrel.find(b)).second];
          theZCut = theHicConst.zcutfbb[(*theForward.find(first)).second][(*theBarrel.find(theLastLayer)).second][(*theBarrel.find(b)).second];
//#ifdef LOWMULT
           if(theLowMult == 1)
	   {
//          if( b->module() == pixel ) theNewCut = 20.;
           }
//#endif


	 }


	theCuts.push_back(thePhiWin); theCuts.push_back(theZWin);
	theCuts.push_back(thePhiCut); theCuts.push_back(theZCut);

        return theCuts;
     }
     if( a->location() == GeomDetEnumerators::endcap && b->location() == GeomDetEnumerators::endcap)
     {
        if( a->surface().position().z() > 0. )
	{
        thePhiWin = theHicConst.phiwinfrw[(*theForward.find(first)).second][(*theForward.find(a)).second][(*theForward.find(b)).second];
        theZWin = theHicConst.zwinfrw[(*theForward.find(first)).second][(*theForward.find(a)).second][(*theForward.find(b)).second];
        thePhiCut = theHicConst.phicutfrw[(*theForward.find(first)).second][(*theForward.find(theLastLayer)).second][(*theForward.find(b)).second];
        theZCut = theHicConst.zcutfrw[(*theForward.find(first)).second][(*theForward.find(theLastLayer)).second][(*theForward.find(b)).second];
        }
	  else
	  {
           thePhiWin = theHicConst.phiwinfrw[(*theBackward.find(first)).second][(*theBackward.find(a)).second][(*theBackward.find(b)).second];
           theZWin = theHicConst.zwinfrw[(*theBackward.find(first)).second][(*theBackward.find(a)).second][(*theBackward.find(b)).second];
           thePhiCut = theHicConst.phicutfrw[(*theBackward.find(first)).second][(*theBackward.find(theLastLayer)).second][(*theBackward.find(b)).second];
           theZCut = theHicConst.zcutfrw[(*theBackward.find(first)).second][(*theBackward.find(theLastLayer)).second][(*theBackward.find(b)).second];
	  }
	
        if( theLowMult == 1 )
	{
        if( b->subDetector() == GeomDetEnumerators::PixelEndcap ) theNewCut = 20.;
	if( traj.measurements().size() == 1 ) theNewCut = 20.; 
	theNewCutB = 30.;
	}
	
        thePhiWinMeanB = 0.004;
        thePhiWinB = 0.04;


	theCuts.push_back(thePhiWin); theCuts.push_back(theZWin);
	theCuts.push_back(thePhiCut); theCuts.push_back(theZCut);
	
        return theCuts;
     }
     if( a->location() == GeomDetEnumerators::endcap && b->location() == GeomDetEnumerators::barrel )
     {
       
       if( a->surface().position().z() > 0. )
       {
        thePhiWin = theHicConst.phiwinbfrw[(*theForward.find(first)).second][(*theForward.find(a)).second][(*theBarrel.find(b)).second];
        theZWin = theHicConst.zwinbfrw[(*theForward.find(first)).second][(*theForward.find(a)).second][(*theBarrel.find(b)).second];
        thePhiCut = theHicConst.phicutbfrw[(*theForward.find(first)).second][(*theForward.find(theLastLayer)).second][(*theBarrel.find(b)).second];
        theZCut = theHicConst.zcutbfrw[(*theForward.find(first)).second][(*theForward.find(theLastLayer)).second][(*theBarrel.find(b)).second];
       }
          else
	  {
            thePhiWin = theHicConst.phiwinbfrw[(*theBackward.find(first)).second][(*theBackward.find(a)).second][(*theBarrel.find(b)).second];
            theZWin = theHicConst.zwinbfrw[(*theBackward.find(first)).second][(*theBackward.find(a)).second][(*theBarrel.find(b)).second];
            thePhiCut = theHicConst.phicutbfrw[(*theBackward.find(first)).second][(*theBackward.find(theLastLayer)).second][(*theBarrel.find(b)).second];
            theZCut = theHicConst.zcutbfrw[(*theBackward.find(first)).second][(*theBackward.find(theLastLayer)).second][(*theBarrel.find(b)).second];
	  }	
	
        if( b->subDetector() ==  GeomDetEnumerators::PixelBarrel) theNewCut = 20.;

        thePhiWinMeanB = 0.004;
        thePhiWinB = 0.015;

	theCuts.push_back(thePhiWin); theCuts.push_back(theZWin);
	theCuts.push_back(thePhiCut); theCuts.push_back(theZCut);
	
        return theCuts;
     }
     cout<<" HICMeasurementEstimator::setCuts::Error: unknown detector layer "<<endl;
     return theCuts;
}

void HICMeasurementEstimator::chooseCuts(int& i)
{

      theChi2Cut = theNewCut;
//      cout<<" Choose Chi2Cut "<<theChi2Cut<<endl;
      
   if( i == 1 )
   {
      thePhiBound = thePhiWin;
      theZBound = theZWin;
      thePhiBoundMean = thePhiWinMean;
      theZBoundMean = theZWinMean;
//      cout<<" HICMeasurementEstimator::chooseCuts  "<<i<<" "<<thePhiBound<<" "<<theZBound<<endl;
   } 
      if( i == 2 )
   {
      thePhiBound = thePhiWinB;
      theZBound = theZWinB;
      thePhiBoundMean = thePhiWinMean;
      theZBoundMean = theZWinMean;
//      cout<<" HICMeasurementEstimator::chooseCuts  "<<i<<" "<<thePhiBound<<" "<<theZBound<<endl;
   }  
      if( i == 3 )
   {
      thePhiBound = thePhiCut;
      theZBound = theZCut;
      thePhiBoundMean = thePhiCutMean;
      theZBoundMean = theZCutMean;
//      cout<<" HICMeasurementEstimator::chooseCuts  "<<i<<" "<<thePhiBound<<" "<<theZBound<<endl;
   }
     
   theCutType = i;
}

int HICMeasurementEstimator::getDetectorCode(const DetLayer* a)
{
     int layer = 0;
     if( a->location() == GeomDetEnumerators::barrel ) 
     {
           layer = (*theBarrel.find(a)).second;
     }	   
     if( a->location() == GeomDetEnumerators::endcap ) 
     {
	   if( a->surface().position().z() > 0. ) { layer = 100+(*theForward.find(a)).second;}
	   if( a->surface().position().z() < 0. ) {layer = -100-(*theBackward.find(a)).second;}	   
     }
     return layer;
}

void HICMeasurementEstimator::setHICDetMap()
{

  std::cout<<" Set Detector Map... "<<std::endl;
  
  int ila=0;
  for ( std::vector<BarrelDetLayer*>::const_iterator ilayer = bl.begin(); ilayer != bl.end(); ilayer++)
  {
     theBarrel[(*ilayer)]=ila;
     ila++;
  }
//
// The same for forward part.
//

   int ilf1 = 0; 
   int ilf2 = 0;
   for ( vector<ForwardDetLayer*>::const_iterator ilayer = fpos.begin(); 
                                                    ilayer != fpos.end(); ilayer++)
   {
                theForward[(*ilayer)] = ilf1;
                ilf1++;
   }
   for ( vector<ForwardDetLayer*>::const_iterator ilayer = fneg.begin(); 
                                                    ilayer != fneg.end(); ilayer++)
   {
                   theBackward[(*ilayer)] = ilf2;
                   ilf2++;
   }

}
