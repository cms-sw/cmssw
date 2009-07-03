#include "RecoHIMuon/HiMuPropagator/interface/HICTkOuterStartingLayerFinder.h"

#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//#define PROPAGATOR_DB
using namespace std;
using namespace edm;

//#define DEBUG

namespace cms{
HICTkOuterStartingLayerFinder::HICTkOuterStartingLayerFinder(int& numberOfSigmas, const MagneticField * mf, 
                                 const GeometricSearchTracker* th, const HICConst* hh):
				 NumberOfSigm(numberOfSigmas),
				 magfield(mf),
				 theTracker(th),
                                 theHICConst(hh)      
{

// Get tracking geometry         
    
    theBarrelLayers = theTracker->barrelLayers();
    forwardPosLayers = theTracker->posForwardLayers();
    forwardNegLayers = theTracker->negForwardLayers();
    //cout<<" HICTkOuterStartingLayerFinder::zvert "<<theHICConst->zvert<<endl;
}

HICTkOuterStartingLayerFinder::LayerContainer HICTkOuterStartingLayerFinder::startingLayers(FreeTrajectoryState& fts)
{
  vector<DetLayer*> seedlayers;
  
  
  BoundSurface* surc = (BoundSurface*)&((theBarrelLayers.back())->specificSurface());
  
  length=surc->bounds().length()/2.;
  
  double maxcoor=fabs(fts.parameters().position().z())+NumberOfSigm*fts.curvilinearError().matrix()(4,4);
  
  //
  //  barrel part (muon and tracker)
  //
  
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::startingLayers::maxcoor "<<fabs(fts.parameters().position().z())<<" "<<
  NumberOfSigm<<" "<<fts.curvilinearError().matrix()(4,4)<<" maxcoor "<<maxcoor<<" length "<<length<<std::endl;
#endif
  
  if(maxcoor<length) {
    seedlayers.push_back( theBarrelLayers.back());
    seedlayers.push_back( *(theBarrelLayers.end()-2));
    return seedlayers;
  }

  bool checkBarrel;
  
  if(fts.parameters().position().z() < 0.){
    checkBarrel = findForwardLayers( fts, forwardNegLayers, seedlayers);
  } else {
    checkBarrel = findForwardLayers( fts, forwardPosLayers, seedlayers);
  }
  
  if (!checkBarrel) return seedlayers;
//
// One should attach barrel layers
//  
  if(fts.parameters().position().z() < 0.){
    return findBarrelLayers( fts, forwardNegLayers, seedlayers);
  }else{
    return findBarrelLayers( fts, forwardPosLayers, seedlayers);
  }

  //  return seedlayers;
  
}

bool HICTkOuterStartingLayerFinder::findForwardLayers( const FreeTrajectoryState& fts,
						    std::vector<ForwardDetLayer*>& fls, 
						    HICTkOuterStartingLayerFinder::LayerContainer& seedlayers){
	      
  bool barrel = false;

#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::start "<<NumberOfSigm<<std::endl;
#endif  

  double outrad, zseed, rseed, theta, atrack, btrack;
  double dz, dr, a1, zdet, newzmin, newzmax;  
  std::vector<ForwardDetLayer*>::const_iterator flayer; 
  double mincoor=fabs(fts.parameters().position().z())-
                                                      NumberOfSigm*fts.curvilinearError().matrix()(4,4);
						      
//  double zdetlast=(fls.front())->surface().position().z();
  double zdetlast=length;
  outrad=(fls.back())->specificSurface().outerRadius();
  
  zseed=fts.parameters().position().z();  
  rseed=fts.parameters().position().perp();
  dz = 3.*NumberOfSigm*fts.curvilinearError().matrix()(4,4); // ok
  dr = NumberOfSigm*fts.curvilinearError().matrix()(4,4);
  
  theta=fts.parameters().momentum().theta();
  atrack=tan(theta);
  btrack=rseed-atrack*zseed;
//  zvert=-btrack/atrack;

#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers "<<rseed<<" dr "<<dr<<" outrad "<<outrad<<std::endl;
#endif   

  if(rseed+dr<outrad){
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::add last forward layer "<<rseed<<" dr "<<dr<<" outrad "<<outrad<<std::endl;
#endif
    seedlayers.push_back(fls.back());
    zdetlast = fabs((fls.back())->surface().position().z());

  }else{
    if(rseed>outrad) {
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::1 zseed "<<zseed<<" dz "<<dz<<std::endl;
#endif

    newzmin=abs(zseed)-3.*dz; // ok 8*dz now 3*dz
//    newzmin = fabs(zseed)-30.; // ok 16.06.08
    newzmax=fabs(zseed)+dz/(2.*NumberOfSigm); // ok dz
    } else {
    a1=(rseed+dr)/(fabs(zseed)-fabs(theHICConst->zvert));
    if(zseed<0.) a1=-1.*a1;

#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::2 zseed "<<zseed<<" dz "<<dz<<" "<<(fls.back())->surface().position().z()<<std::endl;
#endif

    newzmin=abs(outrad/a1)-3.*dz; //ok 6*dz now 3*dz 
//    newzmin = fabs(zseed)-30.; // ok 16.06.08
    newzmax=fabs((fls.back())->surface().position().z())+dz;
    }
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::newzmin,newzmax "<<newzmin<<" "<<newzmax<<std::endl;
#endif
    
    for(flayer=fls.end()-1;flayer!=fls.begin()-1;flayer--){
     
      zdet=(**flayer).surface().position().z();
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::zdet "<<zdet<<" thickness "<<(**flayer).surface().bounds().thickness()<<std::endl;
#endif
           
//      if(abs(newzmin)<=abs(zdet+(**flayer).surface().bounds().thickness())
//                 && abs(zdet-(**flayer).surface().bounds().thickness())<=abs(newzmax)){ 

      if(fabs(zdet)<length) break;

      if(fabs(newzmin)<=fabs(zdet)+(**flayer).surface().bounds().thickness()
                 && fabs(zdet)-(**flayer).surface().bounds().thickness()<=fabs(newzmax)){

#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findForwardLayers::add layer "<<zdet<<std::endl;
#endif
         
        seedlayers.push_back(&(**flayer));
	zdetlast=zdet;

      } //zmin

    } //flayer
  }
//  if(mincoor<abs(zdetlast) ){
#ifdef DEBUG

   std::cout<<"HICTkOuterStartingLayerFinder::zdetlast,mincoor "<<zdetlast<<" "<<mincoor<<std::endl;

#endif
  if(fabs(mincoor)<fabs(zdetlast)||fabs(zdetlast)<140.){
#ifdef DEBUG
   std::cout<<"HICTkOuterStartingLayerFinder::add barrel layers to forward "<<std::endl;
#endif
    barrel=true;
  }
  return barrel;
}
HICTkOuterStartingLayerFinder::LayerContainer HICTkOuterStartingLayerFinder::findBarrelLayers( const FreeTrajectoryState& fts,
									 std::vector<ForwardDetLayer*>& fls, LayerContainer& seedlayers)
{	      
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findBarrelLayers::start::zdetlast "<<length<<std::endl;
#endif
  std::vector<BarrelDetLayer*>::const_iterator blayer; 
 //  
 // double zdetlast=(fls.front())->surface().position().z();

  double zdetlast = length+10.;
  //double zseed=fts.parameters().position().z();  
  //double rseed=fts.parameters().position().perp();
  //double dz = NumberOfSigm*fts.curvilinearError().matrix()(5,5);
  //double atrack=tan(fts.parameters().momentum().theta());
  //double btrack=rseed-atrack*zseed;
//  double zvert=-btrack/atrack;
  double r,rmin,rmax;   
  
  BoundSurface* surc = (BoundSurface*)&((theBarrelLayers.back())->surface());
  double zbarrel=surc->bounds().length()/2.;
  BoundCylinder* bc = dynamic_cast<BoundCylinder*>(surc);
  double barrelradius=bc->radius();

  double a1=barrelradius/(fabs(zdetlast)-fabs(theHICConst->zvert));


  rmin=a1*zbarrel-(theBarrelLayers.back())->surface().bounds().thickness();
  rmax=barrelradius+(theBarrelLayers.back())->surface().bounds().thickness();
     
//  if(fabs(zseed)-dz<zbarrel){
//    rmax=barrelradius+(theBarrelLayers.back())->surface().bounds().thickness();
//  } else{
//    a2=barrelradius/(fabs(zseed-theHICConst->zvert)-dz);
//    if(zseed<0.) a2=-1.*a2;
//    rmax=a2*zbarrel+(theBarrelLayers.back())->surface().bounds().thickness();
//  cout<<" Check a2,rmax "<<a2<<" "<<rmax<<endl;
//  }
     
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findBarrelLayers::rmin,rmax "<<rmin<<" "<<rmax<<std::endl;
#endif 

  for(blayer=theBarrelLayers.end()-1;blayer!=theBarrelLayers.begin()-1;blayer--){
  
  BoundSurface* sc = (BoundSurface*)&((*blayer)->surface());
  r=(dynamic_cast<BoundCylinder*>(sc))->radius();

            
    if(r>rmin&&r<=rmax){
      seedlayers.push_back(&(**blayer));
#ifdef DEBUG
  std::cout<<"HICTkOuterStartingLayerFinder::findBarrelLayers::add "<<r<<std::endl;
#endif
    }
  }//blayer barrel  
  return seedlayers;
}
}
