#include "RecoTracker/ConversionSeedGenerators/interface/SeedForPhotonConversionFromQuadruplets.h"

#include <TVector3.h>
#include "RecoTracker/ConversionSeedGenerators/interface/Conv4HitsReco.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"

//#define mydebug_sguazz
//#define quadDispLine
template <class T> T sqr( T t) {return t*t;}

const TrajectorySeed * SeedForPhotonConversionFromQuadruplets::trajectorySeed(
    TrajectorySeedCollection & seedCollection,
    const SeedingHitSet & phits,
    const SeedingHitSet & mhits,
    const TrackingRegion & region,
    const edm::EventSetup& es,
    std::stringstream& ss,
    std::vector<Quad>& quadV)
{

  //  return 0; //FIXME, remove this line to make the code working. 

  pss = &ss;
 
  if ( phits.size() < 2) return 0;
  if ( mhits.size() < 2) return 0;

  //PUT HERE THE QUADRUPLET ALGORITHM, AND IN CASE USE THE METHODS ALREADY DEVELOPED, ADAPTING THEM

  //
  // Rozzo ma efficace (per ora)
  //

#ifdef mydebug_sguazz
  std::cout << " --------------------------------------------------------------------------" << "\n";   
  std::cout << "  Starting a hit quad fast reco " << "\n";   
  std::cout << " --------------------------------------------------------------------------" << "\n";   
#endif

  //
  // Let's build the 4 hits
  TransientTrackingRecHit::ConstRecHitPointer ptth1 = phits[0];
  TransientTrackingRecHit::ConstRecHitPointer ptth2 = phits[1];
  TransientTrackingRecHit::ConstRecHitPointer mtth1 = mhits[0];
  TransientTrackingRecHit::ConstRecHitPointer mtth2 = mhits[1];

  GlobalPoint vHit[4];
  vHit[0]=ptth2->globalPosition();
  vHit[1]=ptth1->globalPosition();
  vHit[2]=mtth1->globalPosition();
  vHit[3]=mtth2->globalPosition();
  //double zErr2[4];
  //zErr2[0]=ptth2->globalPositionError().czz();
  //zErr2[1]=ptth1->globalPositionError().czz();
  //zErr2[2]=mtth1->globalPositionError().czz();
  //zErr2[3]=mtth2->globalPositionError().czz();
  //double perpErr2[4];
  //perpErr2[0]=ptth2->globalPositionError().rerr(ptth2->globalPosition());
  //perpErr2[1]=ptth1->globalPositionError().rerr(ptth1->globalPosition());
  //perpErr2[2]=mtth1->globalPositionError().rerr(mtth1->globalPosition());
  //perpErr2[3]=mtth2->globalPositionError().rerr(mtth2->globalPosition());

  //Photon source vertex primary vertex
  GlobalPoint vgPhotVertex=region.origin();
  TVector3 vPhotVertex(vgPhotVertex.x(), vgPhotVertex.y(), vgPhotVertex.z());

  TVector3 h1(vHit[0].x(),vHit[0].y(),vHit[0].z());
  TVector3 h2(vHit[1].x(),vHit[1].y(),vHit[1].z());
  TVector3 h3(vHit[2].x(),vHit[2].y(),vHit[2].z());
  TVector3 h4(vHit[3].x(),vHit[3].y(),vHit[3].z());
  
  Conv4HitsReco quad(vPhotVertex, h1, h2, h3, h4);
  quad.SetMaxNumberOfIterations(100);
#ifdef mydebug_sguazz
  quad.Dump();
#endif
  TVector3 candVtx;
  double candPtPlus, candPtMinus;
  //  double truePtPlus, truePtMinus;
  double rPlus, rMinus;
  int nite = quad.ConversionCandidate(candVtx, candPtPlus, candPtMinus); 

  if ( ! (nite && abs(nite) < 25 && nite != -1000 && nite != -2000) ) return 0;

  TVector3 plusCenter = quad.GetPlusCenter(rPlus);
  TVector3 minusCenter = quad.GetMinusCenter(rMinus);
 
#ifdef mydebug_sguazz
    std::cout << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << "\n";   
    std::cout << " >>>>>>>>>>> Conv Cand: " << " Vertex X: " << candVtx.X() << " [cm] Y: " << candVtx.Y() << " [cm] pt+: " << candPtPlus<< " [GeV] pt-: " << candPtMinus << " [GeV]; #its: " << nite << "\n";   
#endif



    //Do a very simple fit to estimate the slope
    double quadPhotCotTheta = 0.;
    double quadZ0 = 0.;
    simpleGetSlope(ptth2, ptth1, mtth1, mtth2, region, quadPhotCotTheta, quadZ0);


    double quadPhotPhi = (candVtx-vPhotVertex).Phi();

    TVector3 fittedPrimaryVertex(vgPhotVertex.x(), vgPhotVertex.y(),quadZ0);

    candVtx.SetZ(candVtx.Perp()*quadPhotCotTheta+quadZ0);
    GlobalPoint convVtxGlobalPoint(candVtx.X(),candVtx.Y(),candVtx.Z());

    //
    // Comparing new quad with old quad
    //
    //Arbitration
    Quad thisQuad;
    thisQuad.x = candVtx.X();
    thisQuad.y = candVtx.Y();
    thisQuad.z = candVtx.Z();
    thisQuad.ptPlus = candPtPlus;
    thisQuad.ptMinus = candPtMinus;
    thisQuad.cot = quadPhotCotTheta;
    if ( similarQuadExist(thisQuad, quadV) ) return 0;
    
    // not able to get the mag field... doing the dirty way
    //
    // Plus
    FastHelix helixPlus(ptth2->globalPosition(), ptth1->globalPosition(), convVtxGlobalPoint, es, convVtxGlobalPoint);
    GlobalTrajectoryParameters kinePlus = helixPlus.stateAtVertex().parameters();
    kinePlus = GlobalTrajectoryParameters(convVtxGlobalPoint,
					  GlobalVector(candPtPlus*cos(quadPhotPhi),candPtPlus*sin(quadPhotPhi),candPtPlus*quadPhotCotTheta),
					  1,
					  & kinePlus.magneticField()
					  );

    //
    // Minus
    FastHelix helixMinus(mtth2->globalPosition(), mtth1->globalPosition(), convVtxGlobalPoint, es, convVtxGlobalPoint);
    GlobalTrajectoryParameters kineMinus = helixMinus.stateAtVertex().parameters();
    kineMinus = GlobalTrajectoryParameters(convVtxGlobalPoint,
					  GlobalVector(candPtMinus*cos(quadPhotPhi),candPtMinus*sin(quadPhotPhi),candPtMinus*quadPhotCotTheta),
					  -1,
					  & kineMinus.magneticField()
					  );

    float sinThetaPlus = sin(kinePlus.momentum().theta());
    float sinThetaMinus = sin(kineMinus.momentum().theta());
    float ptmin = region.ptMin();
    //vertexBounds da region
    GlobalVector vertexBounds(region.originRBound(),region.originRBound(),region.originZBound());

    CurvilinearTrajectoryError errorPlus = initialError(vertexBounds, ptmin,  sinThetaPlus);
    CurvilinearTrajectoryError errorMinus = initialError(vertexBounds, ptmin,  sinThetaMinus);
    FreeTrajectoryState ftsPlus(kinePlus, errorPlus);
    FreeTrajectoryState ftsMinus(kineMinus, errorMinus);
    
    //FIXME: here probably you want to go in parallel with phits and mhits
    //NB: the seedCollection is filled (the important thing) the return of the last TrajectorySeed is not used, but is needed
    //to maintain the inheritance
    
#ifdef quadDispLine
    double vError = region.originZBound();
    if ( vError > 15. ) vError = 1.;
    std::cout << "QuadDispLine " 
	      << vgPhotVertex.x() << " " << vgPhotVertex.y() << " " << vgPhotVertex.z() << " " << vError << " "
	      << vHit[0].x() << " " << vHit[0].y() << " " << vHit[0].z() << " " << sqrt(getSqrEffectiveErrorOnZ(ptth2, region)) << " "
	      << vHit[1].x() << " " << vHit[1].y() << " " << vHit[1].z() << " " << sqrt(getSqrEffectiveErrorOnZ(ptth1, region)) << " "
	      << vHit[2].x() << " " << vHit[2].y() << " " << vHit[2].z() << " " << sqrt(getSqrEffectiveErrorOnZ(mtth1, region)) << " "
	      << vHit[3].x() << " " << vHit[3].y() << " " << vHit[3].z() << " " << sqrt(getSqrEffectiveErrorOnZ(mtth2, region)) << " "
	      << candVtx.X() << " " << candVtx.Y() << " " << candVtx.Z() << " "
	      << fittedPrimaryVertex.X() << " " << fittedPrimaryVertex.Y() << " " << fittedPrimaryVertex.Z() << " "
	      << candPtPlus  << " " << rPlus << " " << plusCenter.X() << " " << plusCenter.Y() << " " 
	      << candPtMinus << " " << rMinus << " " << minusCenter.X() << " " << minusCenter.Y() << " " 
	      << nite << " " << chi2 << "\n";   
#endif
#ifdef mydebug_sguazz
    std::cout << " >>>>> Hit quad fast reco done >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << "\n";   
    uint32_t detid;
    std::cout << "[SeedForPhotonConversionFromQuadruplets]\n ptth1 " ;
    detid=ptth1->geographicalId().rawId();
    //    po.print(std::cout , detid );
    std::cout << " \t " << detid << " " << ptth1->localPosition()  << " " << ptth1->globalPosition()    ;
    detid=ptth2->geographicalId().rawId();
    std::cout << " \n\t ptth2 ";
    //    po.print(std::cout , detid );
    std::cout << " \t " << detid << " " << ptth2->localPosition()  << " " << ptth2->globalPosition()  
	      << "\nhelix momentum " << kinePlus.momentum() << " pt " << kinePlus.momentum().perp() << " radius " << 1/kinePlus.transverseCurvature() << " q " << kinePlus.charge(); 
    std::cout << " \n\t mtth1 ";
    detid=mtth1->geographicalId().rawId();
    std::cout << " \t " << detid << " " << mtth1->localPosition()  << " " << mtth1->globalPosition()    ;
    std::cout << " \n\t mtth2 ";
    detid=mtth2->geographicalId().rawId();
    //    po.print(std::cout , detid );
    std::cout << " \t " << detid << " " << mtth2->localPosition()  << " " << mtth2->globalPosition()  
	      << "\nhelix momentum " << kineMinus.momentum() << " pt " << kineMinus.momentum().perp() << " radius " << 1/kineMinus.transverseCurvature() << " q " << kineMinus.charge(); 
    std::cout << "\n <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";   
#endif

    buildSeed(seedCollection,phits,ftsPlus,es); 
    return buildSeed(seedCollection,mhits,ftsMinus,es); 

}


GlobalTrajectoryParameters SeedForPhotonConversionFromQuadruplets::initialKinematic(
      const SeedingHitSet & hits, 
      const GlobalPoint & vertexPos, 
      const edm::EventSetup& es,
      const float cotTheta) const
{
  GlobalTrajectoryParameters kine;

  TransientTrackingRecHit::ConstRecHitPointer tth1 = hits[0];
  TransientTrackingRecHit::ConstRecHitPointer tth2 = hits[1];


  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, es, vertexPos);
  kine = helix.stateAtVertex().parameters();

  //force the pz/pt equal to the measured one
  if(fabs(cotTheta)<cotTheta_Max)
    kine = GlobalTrajectoryParameters(kine.position(),
				      GlobalVector(kine.momentum().x(),kine.momentum().y(),kine.momentum().perp()*cotTheta),
				      kine.charge(),
				      & kine.magneticField()
				      );
  else
    kine = GlobalTrajectoryParameters(kine.position(),
				      GlobalVector(kine.momentum().x(),kine.momentum().y(),kine.momentum().perp()*cotTheta_Max),
				      kine.charge(),
				      & kine.magneticField()
				      );

#ifdef mydebug_seed
  uint32_t detid;
  (*pss) << "[SeedForPhotonConversionFromQuadruplets] initialKinematic tth1 " ;
  detid=tth1->geographicalId().rawId();
  po.print(*pss, detid );
  (*pss) << " \t " << detid << " " << tth1->localPosition()  << " " << tth1->globalPosition()    ;
  detid= tth2->geographicalId().rawId();
  (*pss) << " \n\t tth2 ";
  po.print(*pss, detid );
  (*pss) << " \t " << detid << " " << tth2->localPosition()  << " " << tth2->globalPosition()  
	 << "\nhelix momentum " << kine.momentum() << " pt " << kine.momentum().perp() << " radius " << 1/kine.transverseCurvature(); 
#endif

  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);
  bool isBOFF = ( std::abs(bfield->inTesla(GlobalPoint(0,0,0)).z()) < 1e-3 );
  if (isBOFF && (theBOFFMomentum > 0)) {
    kine = GlobalTrajectoryParameters(kine.position(),
                              kine.momentum().unit() * theBOFFMomentum,
                              kine.charge(),
                              &*bfield);
  }
  return kine;
}



CurvilinearTrajectoryError SeedForPhotonConversionFromQuadruplets::
initialError( 
	     const GlobalVector& vertexBounds, 
	     float ptMin,  
	     float sinTheta) const
{
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit
  // information.
  GlobalError vertexErr( sqr(vertexBounds.x()), 0, 
			 sqr(vertexBounds.y()), 0, 0,
			 sqr(vertexBounds.z())
			 );
  
 
  AlgebraicSymMatrix55 C = ROOT::Math::SMatrixIdentity();

// FIXME: minC00. Prevent apriori uncertainty in 1/P from being too small, 
// to avoid instabilities.
// N.B. This parameter needs optimising ...
  float sin2th = sqr(sinTheta);
  float minC00 = 1.0;
  C[0][0] = std::max(sin2th/sqr(ptMin), minC00);
  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy
  C[3][3] = transverseErr;
  C[4][4] = zErr*sin2th + transverseErr*(1-sin2th);

  return CurvilinearTrajectoryError(C);
}

const TrajectorySeed * SeedForPhotonConversionFromQuadruplets::buildSeed(
    TrajectorySeedCollection & seedCollection,
    const SeedingHitSet & hits,
    const FreeTrajectoryState & fts,
    const edm::EventSetup& es) const
{
  // get tracker
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get propagator
  edm::ESHandle<Propagator>  propagatorHandle;
  es.get<TrackingComponentsRecord>().get(thePropagatorLabel, propagatorHandle);
  const Propagator*  propagator = &(*propagatorHandle);
  
  // get updator
  KFUpdator  updator;
  
  // Now update initial state track using information from seed hits.
  
  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;
  
  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size() && iHit<1; iHit++) {
    hit = hits[iHit]->hit();
    TrajectoryStateOnSurface state = (iHit==0) ? 
      propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return 0;
    
    TransientTrackingRecHit::ConstRecHitPointer tth = hits[iHit]; 
    
    TransientTrackingRecHit::RecHitPointer newtth =  refitHit( tth, state);

    
    if (!checkHit(state,newtth,es)) return 0;

    updatedState =  updator.update(state, *newtth);
    if (!updatedState.isValid()) return 0;
    
    seedHits.push_back(newtth->hit()->clone());
#ifdef mydebug_seed
    uint32_t detid = hit->geographicalId().rawId();
    (*pss) << "\n[SeedForPhotonConversionFromQuadruplets] hit " << iHit;
    po.print(*pss, detid);
    (*pss) << " "  << detid << "\t lp " << hit->localPosition()
	   << " tth " << tth->localPosition() << " newtth " << newtth->localPosition() << " state " << state.globalMomentum().perp();
#endif
  } 
  
  
  PTrajectoryStateOnDet const &  PTraj = 
      trajectoryStateTransform::persistentState(updatedState, hit->geographicalId().rawId());
  
  seedCollection.push_back( TrajectorySeed(PTraj,seedHits,alongMomentum));
  return &seedCollection.back();
}

TransientTrackingRecHit::RecHitPointer SeedForPhotonConversionFromQuadruplets::refitHit(
      const TransientTrackingRecHit::ConstRecHitPointer &hit, 
      const TrajectoryStateOnSurface &state) const
{
  //const TransientTrackingRecHit* a= hit.get();
  //return const_cast<TransientTrackingRecHit*> (a);
  //This was modified otherwise the rechit will have just the local x component and local y=0
  // To understand how to modify for pixels

  //const TSiStripRecHit2DLocalPos* b = dynamic_cast<const TSiStripRecHit2DLocalPos*>(a);
  //return const_cast<TSiStripRecHit2DLocalPos*>(b);
  return hit->clone(state);
}

//
// Below: stupid utils method by sguazz
//
//
void SeedForPhotonConversionFromQuadruplets::
stupidPrint(std::string s,float* d){
  (*pss) << "\n" << s << "\t";
  for(size_t i=0;i<2;++i)
      (*pss) << std::setw (60)  << d[i] << std::setw(1) << " | ";
}    

void SeedForPhotonConversionFromQuadruplets::
stupidPrint(std::string s,double* d){
  (*pss) << "\n" << s << "\t";
  for(size_t i=0;i<2;++i)
      (*pss) << std::setw (60) << d[i] << std::setw(1) << " | ";
}    

void SeedForPhotonConversionFromQuadruplets::
stupidPrint(const char* s,GlobalPoint* d){
  (*pss) << "\n" << s << "\t";
  for(size_t i=0;i<2;++i)
    (*pss) << std::setw(20) << d[i] << " r " << d[i].perp() << " phi " << d[i].phi() << " | ";
}    

void SeedForPhotonConversionFromQuadruplets::
stupidPrint(const char* s, GlobalPoint* d, int n){
  (*pss) << "\n" << s << "\n";
  for(int i=0;i<n;++i)
    (*pss) << std::setw(20) << d[i] << " r " << d[i].perp() << " phi " << d[i].phi() << "\n";
}    

#include "DataFormats/Math/interface/deltaPhi.h"

void SeedForPhotonConversionFromQuadruplets::
bubbleSortVsPhi(GlobalPoint arr[], int n, GlobalPoint vtx) {
  bool swapped = true;
  int j = 0;
  GlobalPoint tmp;
  while (swapped) {
    swapped = false;
    j++;
    for (int i = 0; i < n - j; i++) {
      if ( reco::deltaPhi( (arr[i]-vtx).phi(), (arr[i + 1]-vtx).phi() ) > 0. ) {
	tmp = arr[i];
	arr[i] = arr[i + 1];
	arr[i + 1] = tmp;
	swapped = true;
      }
    }
  }
}

void SeedForPhotonConversionFromQuadruplets::
bubbleReverseSortVsPhi(GlobalPoint arr[], int n, GlobalPoint vtx) {
  bool swapped = true;
  int j = 0;
  GlobalPoint tmp;
  while (swapped) {
    swapped = false;
    j++;
    for (int i = 0; i < n - j; i++) {
      if ( reco::deltaPhi( (arr[i]-vtx).phi(), (arr[i + 1]-vtx).phi() ) < 0. ) {
	tmp = arr[i];
	arr[i] = arr[i + 1];
	arr[i + 1] = tmp;
	swapped = true;
      }
    }
  }
}


double SeedForPhotonConversionFromQuadruplets::
simpleGetSlope(const TransientTrackingRecHit::ConstRecHitPointer &ohit, const TransientTrackingRecHit::ConstRecHitPointer &nohit, const TransientTrackingRecHit::ConstRecHitPointer &ihit, const TransientTrackingRecHit::ConstRecHitPointer &nihit, const TrackingRegion & region, double & cotTheta, double & z0){

  double x[5], y[5], e2y[5];

  //The fit is done filling x with r values, y with z values of the four hits and the vertex
  //
  //Hits
  x[0] = ohit->globalPosition().perp();
  y[0] = ohit->globalPosition().z();
  e2y[0] = getSqrEffectiveErrorOnZ(ohit, region);
  //
  x[1] = nohit->globalPosition().perp();
  y[1] = nohit->globalPosition().z();
  e2y[1] = getSqrEffectiveErrorOnZ(nohit, region);
  //
  x[2] = nihit->globalPosition().perp();
  y[2] = nihit->globalPosition().z();
  e2y[2] = getSqrEffectiveErrorOnZ(nihit, region);
  //
  x[3] = ihit->globalPosition().perp();
  y[3] = ihit->globalPosition().z();
  e2y[3] = getSqrEffectiveErrorOnZ(ihit, region);
  //
  //Vertex
  x[4] = region.origin().perp();
  y[4] = region.origin().z();
  double vError = region.originZBound();
  if ( vError > 15. ) vError = 1.;
  e2y[4] = sqr(vError);

  double e2z0;
  double chi2 = verySimpleFit(5, x, y, e2y, z0, e2z0, cotTheta);

  return chi2;

}

double SeedForPhotonConversionFromQuadruplets::verySimpleFit(int size, double* ax, double* ay, double* e2y, double& p0, double& e2p0, double& p1){

  //#include "RecoTracker/ConversionSeedGenerators/interface/verySimpleFit.icc"
  return 0;
}

double SeedForPhotonConversionFromQuadruplets::getSqrEffectiveErrorOnZ(const TransientTrackingRecHit::ConstRecHitPointer &hit, const TrackingRegion & region){

  //
  //Fit-wise the effective error on Z is the sum in quadrature of the error on Z 
  //and the error on R correctly projected by using hit-vertex direction

  double sqrProjFactor = sqr((hit->globalPosition().z()-region.origin().z())/(hit->globalPosition().perp()-region.origin().perp()));
  return (hit->globalPositionError().czz()+sqrProjFactor*hit->globalPositionError().rerr(hit->globalPosition()));

}

bool SeedForPhotonConversionFromQuadruplets::similarQuadExist(Quad & thisQuad, std::vector<Quad>& quadV){
  
  BOOST_FOREACH( Quad quad, quadV )
    {
      double dx = thisQuad.x-quad.x;
      double dy = thisQuad.y-quad.y;
      double dz = abs(thisQuad.z-quad.z);
      if ( sqrt(dx*dx+dy*dy)<1. && 
	   dz<3. && 
	   abs(thisQuad.ptPlus-quad.ptPlus)<0.5*quad.ptPlus &&
	   abs(thisQuad.ptMinus-quad.ptMinus)<0.5*quad.ptMinus &&
	   abs(thisQuad.cot-quad.cot)<0.3*quad.cot
	   ) return true;
    }
  
  quadV.push_back(thisQuad);
  return false;

}
