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

namespace cms{
HICTkOuterStartingLayerFinder::HICTkOuterStartingLayerFinder(int& numberOfSigmas, const MagneticField * mf, 
                                 const GeometricSearchTracker* th):
				 NumberOfSigm(numberOfSigmas),
				 magfield(mf),
				 theTracker(th)     
{

// Get tracking geometry         
    
    theBarrelLayers = theTracker->barrelLayers();
    forwardPosLayers = theTracker->posForwardLayers();
    forwardNegLayers = theTracker->negForwardLayers();
  
  
  HICConst hicconst;
  zvert=(double)hicconst.zvert;
  cout<<" HICTkOuterStartingLayerFinder::zvert "<<zvert<<endl;
}

HICTkOuterStartingLayerFinder::LayerContainer HICTkOuterStartingLayerFinder::startingLayers(FreeTrajectoryState& fts)
{
  vector<DetLayer*> seedlayers;
  
  
  BoundSurface* surc = (BoundSurface*)&((theBarrelLayers.back())->specificSurface());
  
  double length=surc->bounds().length()/2.;
  
  double maxcoor=abs(fts.parameters().position().z())+NumberOfSigm*fts.curvilinearError().matrix()(5,5);
  
  //
  //  barrel part (muon and tracker)
  //
  
  
  if(maxcoor<length) {
    seedlayers.push_back( theBarrelLayers.back());
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
  
  double outrad, zseed, rseed, theta, atrack, btrack;
  double dz, dr, a1, zdet, newzmin, newzmax;  
  std::vector<ForwardDetLayer*>::const_iterator flayer; 
  double mincoor=abs(fts.parameters().position().z())-
                                                      NumberOfSigm*fts.curvilinearError().matrix()(5,5);
						      
  double zdetlast=(fls.front())->surface().position().z();
  outrad=(fls.back())->specificSurface().outerRadius();
  
  zseed=fts.parameters().position().z();  
  rseed=fts.parameters().position().perp();
  dz = 3.*NumberOfSigm*fts.curvilinearError().matrix()(5,5); // ok
  dr = NumberOfSigm*fts.curvilinearError().matrix()(5,5);
  
  theta=fts.parameters().momentum().theta();
  atrack=tan(theta);
  btrack=rseed-atrack*zseed;
//  zvert=-btrack/atrack;
   
  if(rseed+dr<outrad){
    seedlayers.push_back(fls.back());
  }else{
    if(rseed>outrad) {
    newzmin=abs(zseed)-dz; // ok 8*dz
    newzmax=abs(zseed)+dz/(2.*NumberOfSigm); // ok dz
    } else {
    a1=(rseed+dr)/(abs(zseed)-abs(zvert));
    if(zseed<0.) a1=-1.*a1;
    newzmin=abs(outrad/a1)-dz; //ok 6*dz
    newzmax=abs((fls.back())->surface().position().z())+dz;
    }
    
    for(flayer=fls.end()-1;flayer!=fls.begin()-1;flayer--){
     
      zdet=(**flayer).surface().position().z();
           
      if(abs(newzmin)<=abs(zdet+(**flayer).surface().bounds().thickness())
                 && abs(zdet-(**flayer).surface().bounds().thickness())<=abs(newzmax)){ 
         
        seedlayers.push_back(&(**flayer));
	zdetlast=zdet;

      } //zmin
    } //flayer
  }
  if(mincoor<abs(zdetlast)){
    barrel=true;
  }
  return barrel;
}
HICTkOuterStartingLayerFinder::LayerContainer HICTkOuterStartingLayerFinder::findBarrelLayers( const FreeTrajectoryState& fts,
									 std::vector<ForwardDetLayer*>& fls, LayerContainer& seedlayers)
{	      
  std::vector<BarrelDetLayer*>::const_iterator blayer; 
 //  
  double zdetlast=(fls.front())->surface().position().z();
  double zseed=fts.parameters().position().z();  
  double rseed=fts.parameters().position().perp();
  double dz = NumberOfSigm*fts.curvilinearError().matrix()(5,5);
  double atrack=tan(fts.parameters().momentum().theta());
  double btrack=rseed-atrack*zseed;
//  double zvert=-btrack/atrack;
  double r,rmin,rmax,a2;   
  
  BoundSurface* surc = (BoundSurface*)&((theBarrelLayers.back())->surface());
  double zbarrel=surc->bounds().length()/2.;
  BoundCylinder* bc = dynamic_cast<BoundCylinder*>(surc);
  double barrelradius=bc->radius();

  double a1=barrelradius/(abs(zdetlast)-abs(zvert));


  rmin=a1*zbarrel-(theBarrelLayers.back())->surface().bounds().thickness();
     
  if(abs(zseed)-dz<zbarrel){
    rmax=barrelradius+(theBarrelLayers.back())->surface().bounds().thickness();
  } else{
    a2=barrelradius/(abs(zseed-zvert)-dz);
    if(zseed<0.) a2=-1.*a2;
    rmax=a2*zbarrel+(theBarrelLayers.back())->surface().bounds().thickness();
    cout<<" Check a2,rmax "<<a2<<" "<<rmax<<endl;
  }
     

  for(blayer=theBarrelLayers.end()-1;blayer!=theBarrelLayers.begin()-1;blayer--){
  
  BoundSurface* sc = (BoundSurface*)&((*blayer)->surface());
  r=(dynamic_cast<BoundCylinder*>(sc))->radius();

            
    if(r>rmin&&r<=rmax){
      seedlayers.push_back(&(**blayer));
    }
  }//blayer barrel  
  return seedlayers;
}
}
