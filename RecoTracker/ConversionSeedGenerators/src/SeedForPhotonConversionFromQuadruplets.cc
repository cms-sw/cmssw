#include "RecoTracker/ConversionSeedGenerators/interface/SeedForPhotonConversionFromQuadruplets.h"

#include "RecoTracker/ConversionSeedGenerators/interface/Conv4HitsReco2.h"

#include "TRandom3.h"

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
//ClusterShapeIncludes
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//#define mydebug_knuenz
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
    std::vector<Quad>& quadV,
    edm::ParameterSet& SeedComparitorPSet,
    edm::ParameterSet& QuadCutPSet)
{


//// CUT DEFINITIONS ////


	bool rejectAllQuads=QuadCutPSet.getParameter<bool>("rejectAllQuads");
	if(rejectAllQuads) return 0;

	bool applyDeltaPhiCuts=QuadCutPSet.getParameter<bool>("apply_DeltaPhiCuts");
    bool ClusterShapeFiltering=QuadCutPSet.getParameter<bool>("apply_ClusterShapeFilter");
    bool applyArbitration=QuadCutPSet.getParameter<bool>("apply_Arbitration");
    bool applydzCAcut=QuadCutPSet.getParameter<bool>("apply_zCACut");
    double CleaningmaxRadialDistance=QuadCutPSet.getParameter<double>("Cut_DeltaRho");//cm
    double BeamPipeRadiusCut=QuadCutPSet.getParameter<double>("Cut_BeamPipeRadius");//cm
    double CleaningMinLegPt = QuadCutPSet.getParameter<double>("Cut_minLegPt"); //GeV
    double maxLegPt = QuadCutPSet.getParameter<double>("Cut_maxLegPt"); //GeV
    double dzcut=QuadCutPSet.getParameter<double>("Cut_zCA");//cm

    double toleranceFactorOnDeltaPhiCuts=0.1;

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

  //Photon source vertex primary vertex
  GlobalPoint vgPhotVertex=region.origin();
  math::XYZVector vPhotVertex(vgPhotVertex.x(), vgPhotVertex.y(), vgPhotVertex.z());

  math::XYZVector h1(vHit[0].x(),vHit[0].y(),vHit[0].z());
  math::XYZVector h2(vHit[1].x(),vHit[1].y(),vHit[1].z());
  math::XYZVector h3(vHit[2].x(),vHit[2].y(),vHit[2].z());
  math::XYZVector h4(vHit[3].x(),vHit[3].y(),vHit[3].z());

// At this point implement cleaning cuts before building the seed:::

/*
  Notes:

h1, h2: positron
h3, h4: electron

P1, P2: positron, ordered with radius
M1, M2: electron, ordered with radius

Evan's notation:
V1=P1
V2=M1
V3=P2
V4=M2

*/

  math::XYZVector P1;
  math::XYZVector P2;
  math::XYZVector M1;
  math::XYZVector M2;

  if(h1.x()*h1.x()+h1.y()*h1.y() < h2.x()*h2.x()+h2.y()*h2.y()){
	  P1=h1;
	  P2=h2;
  }
  else{
	  P1=h2;
	  P2=h1;
  }

  if(h3.x()*h3.x()+h3.y()*h3.y() < h4.x()*h4.x()+h4.y()*h4.y()){
	  M1=h3;
	  M2=h4;
  }
  else{
	  M1=h4;
	  M2=h3;
  }

////////////////////////
// Intersection-point:::
////////////////////////
/*
Calculate the intersection point of the lines P2-P1 and M2-M1.
If this point is in the beam pipe, or if the distance of this
point to the layer of the most inner hit of the seed is less
than CleaningmaxRadialDistance cm, the combination is rejected.
*/


  math::XYZVector IP(0,0,0);

  //Line1:
  double kP=(P1.y()-P2.y())/(P1.x()-P2.x());
  double dP=P1.y()-kP*P1.x();
  //Line2:
  double kM=(M1.y()-M2.y())/(M1.x()-M2.x());
  double dM=M1.y()-kM*M1.x();
  //Intersection:
  double IPx=(dM-dP)/(kP-kM);
  double IPy=kP*IPx+dP;

  IP.SetXYZ(IPx,IPy,0);

  double IPrho=TMath::Sqrt(IP.x()*IP.x()+IP.y()*IP.y());
  double P1rho2=P1.x()*P1.x()+P1.y()*P1.y();
  double M1rho2=M1.x()*M1.x()+M1.y()*M1.y();
  double maxIPrho2=IPrho+CleaningmaxRadialDistance; maxIPrho2*=maxIPrho2;

  if( IPrho<BeamPipeRadiusCut || P1rho2>maxIPrho2 || M1rho2>maxIPrho2){
	  return 0;
  }

  if(applyDeltaPhiCuts) {

  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);
  math::XYZVector QuadMean(0,0,0);
  QuadMean.SetXYZ((M1.x()+M2.x()+P1.x()+P2.x())/4.,(M1.y()+M2.y()+P1.y()+P2.y())/4.,(M1.z()+M2.z()+P1.z()+P2.z())/4.);

  double fBField = bfield->inTesla(GlobalPoint(QuadMean.x(),QuadMean.y(),QuadMean.z())).z();

  double rMax=CleaningMinLegPt/(0.01*0.3*fBField);
  double rMax_squared=rMax*rMax;
  double Mx=M1.x();
  double My=M1.y();

  math::XYZVector B(0,0,0);
  math::XYZVector C(0,0,0);

  if(rMax_squared*4. > Mx*Mx+My*My){

////////////////////////
// Cleaning P1 points:::
////////////////////////

	  //Cx, Cy = Coordinates of circle center
	  //C_=line that contains the circle center

  //C_=k*x+d
  double k=-Mx/My;
  double d=My/2.-k*Mx/2.;

#ifdef mydebug_knuenz
std::cout << "k" << k << std::endl;
std::cout << "d" << d << std::endl;
#endif

  //Cx1,2 and Cy1,2 are the two points that have a distance of rMax to 0,0
  double CsolutionPart1=-2*k*d;
  double CsolutionPart2=TMath::Sqrt(4*k*k*d*d-4*(1+k*k)*(d*d-rMax_squared));
  double CsolutionPart3=2*(1+k*k);
  double Cx1=(CsolutionPart1+CsolutionPart2)/CsolutionPart3;
  double Cx2=(CsolutionPart1-CsolutionPart2)/CsolutionPart3;
  double Cy1=k*Cx1+d;
  double Cy2=k*Cx2+d;


  // Decide between solutions: phi(C) > phi(P)
  double Cx,Cy;
  math::XYZVector C1(Cx1,Cy1,0);
  if(C1.x()*M1.y()-C1.y()*M1.x()<0){
	  Cx=Cx1;
	  Cy=Cy1;
  }
  else{
	  Cx=Cx2;
	  Cy=Cy2;
  }
  C.SetXYZ(Cx,Cy,0);

#ifdef mydebug_knuenz
	std::cout << "Cx1" << Cx1 << std::endl;
	std::cout << "Cx2" << Cx2 << std::endl;
	std::cout << "Cy1" << Cy1 << std::endl;
	std::cout << "Cy2" << Cy2 << std::endl;
	std::cout << "Cx" << Cx << std::endl;
	std::cout << "Cy" << Cy << std::endl;
#endif

// Find Tangent at 0,0 to minPtCircle and point (Bx,By) on the first layer which bisects the allowed angle
  k=-Cx/Cy;
  d=0;
  double Bx1=TMath::Sqrt(Mx*Mx+My*My/(1+k*k));
  double Bx2=-Bx1;
  double By1=k*Bx1+d;
  double By2=k*Bx2+d;

#ifdef mydebug_knuenz
	std::cout << "k" << k << std::endl;
	std::cout << "d" << d << std::endl;
#endif

// Decide between solutions: phi(B) < phi(P)
  double Bx,By;
  math::XYZVector B1(Bx1,By1,0);
  if(M1.x()*B1.y()-M1.y()*B1.x()<0){
	  Bx=Bx1;
	  By=By1;
  }
  else{
	  Bx=Bx2;
	  By=By2;
  }
  B.SetXYZ(Bx,By,0);

#ifdef mydebug_knuenz
	std::cout << "Bx1" << Bx1 << std::endl;
	std::cout << "Bx2" << Bx2 << std::endl;
	std::cout << "By1" << By1 << std::endl;
	std::cout << "By2" << By2 << std::endl;
	std::cout << "Bx" << Bx << std::endl;
	std::cout << "By" << By << std::endl;
#endif

  double DeltaPhiMaxM1P1=DeltaPhiManual(M1,B)*2;

#ifdef mydebug_knuenz
    std::cout << "DeltaPhiMaxM1P1 " << DeltaPhiMaxM1P1 << std::endl;
	std::cout << "M1.DeltaPhi(P1) " << DeltaPhiManual(M1,P1) << std::endl;
    std::cout << "rho P1: " << TMath::Sqrt(P1.x()*P1.x()+P1.y()*P1.y()) <<  "phi P1: " << P1.Phi() << std::endl;
    std::cout << "rho M1: " << TMath::Sqrt(M1.x()*M1.x()+M1.y()*M1.y()) <<  "phi M1: " << M1.Phi() << std::endl;
#endif

//Finally Cut on DeltaPhi of P1 and M1

    double tol_DeltaPhiMaxM1P1=DeltaPhiMaxM1P1*toleranceFactorOnDeltaPhiCuts;
    double DeltaPhiManualM1P1=DeltaPhiManual(M1,P1);

if(DeltaPhiManualM1P1>DeltaPhiMaxM1P1+tol_DeltaPhiMaxM1P1 || DeltaPhiManualM1P1<0-tol_DeltaPhiMaxM1P1){
	return 0;
}

  }//if rMax > rLayerM1


////////////////////////
// Cleaning M2 points:::
////////////////////////

//  if(B.DeltaPhi(P1)>0){//normal algo (with minPt circle)

	  double rM2_squared=M2.x()*M2.x()+M2.y()*M2.y();
	  if(rMax_squared*4. > rM2_squared){//if minPt circle is smaller than 2*M2-layer radius, algo makes no sense

		  //Chordales equation (line containing the two intersection points of the two circles)
		  double k=-C.x()/C.y();
		  double d=(rM2_squared-rMax_squared+C.x()*C.x()+C.y()*C.y())/(2*C.y());

		  double M2solutionPart1=-2*k*d;
		  double M2solutionPart2=TMath::Sqrt(4*k*k*d*d-4*(1+k*k)*(d*d-rM2_squared));
		  double M2solutionPart3=2+2*k*k;
		  double M2xMax1=(M2solutionPart1+M2solutionPart2)/M2solutionPart3;
		  double M2xMax2=(M2solutionPart1-M2solutionPart2)/M2solutionPart3;
		  double M2yMax1=k*M2xMax1+d;
		  double M2yMax2=k*M2xMax2+d;

		  //double M2xMax,M2yMax;
		  math::XYZVector M2MaxVec1(M2xMax1,M2yMax1,0);
		  math::XYZVector M2MaxVec2(M2xMax2,M2yMax2,0);
		  math::XYZVector M2MaxVec(0,0,0);
		  if(M2MaxVec1.x()*M2MaxVec2.y()-M2MaxVec1.y()*M2MaxVec2.x()<0){
			  M2MaxVec.SetXYZ(M2xMax2,M2yMax2,0);
		  }
		  else{
			  M2MaxVec.SetXYZ(M2xMax1,M2yMax1,0);
		  }

		  double DeltaPhiMaxM2=DeltaPhiManual(M2MaxVec,M1);

#ifdef mydebug_knuenz
		  	std::cout << "C.x() " << C.x() << std::endl;
		  	std::cout << "C.y() " << C.y() << std::endl;
		  	std::cout << "M1.x() " << M1.x() << std::endl;
		  	std::cout << "M1.y() " << M1.y() << std::endl;
		  	std::cout << "M2.x() " << M2.x() << std::endl;
		  	std::cout << "M2.y() " << M2.y() << std::endl;
		  	std::cout << "k " << k << std::endl;
		  	std::cout << "d " << d << std::endl;
		  	std::cout << "M2xMax1 " << M2xMax1 << std::endl;
		  	std::cout << "M2xMax2 " << M2xMax2 << std::endl;
		  	std::cout << "M2yMax1 " << M2yMax1 << std::endl;
		  	std::cout << "M2yMax2 " << M2yMax2 << std::endl;
		  	std::cout << "M2xMax " << M2MaxVec.x() << std::endl;
		  	std::cout << "M2yMax " << M2MaxVec.y() << std::endl;
		  	std::cout << "rM2_squared " << rM2_squared << std::endl;
		  	std::cout << "rMax " << rMax << std::endl;
		  	std::cout << "DeltaPhiMaxM2 " << DeltaPhiMaxM2 << std::endl;
#endif


		    double tol_DeltaPhiMaxM2=DeltaPhiMaxM2*toleranceFactorOnDeltaPhiCuts;
		    double DeltaPhiManualM2M1=DeltaPhiManual(M2,M1);

		  if(DeltaPhiManualM2M1>DeltaPhiMaxM2+tol_DeltaPhiMaxM2 || DeltaPhiManualM2M1<0-tol_DeltaPhiMaxM2){
			  return 0;
		  }

		  //Using the lazy solution for P2: DeltaPhiMaxP2=DeltaPhiMaxM2
		  double DeltaPhiManualP1P2=DeltaPhiManual(P1,P2);
		  if(DeltaPhiManualP1P2>DeltaPhiMaxM2+tol_DeltaPhiMaxM2 || DeltaPhiManualP1P2<0-tol_DeltaPhiMaxM2){
		  	return 0;
		  }

	  }

}

//  }

//  if(B.DeltaPhi(P1)<0){//different algo (without minPt circle)

//  }

// End of pre-seed cleaning

	  
	  
  Conv4HitsReco2 quad(vPhotVertex, h1, h2, h3, h4);
  quad.SetMaxNumberOfIterations(100);
#ifdef mydebug_sguazz
  quad.Dump();
#endif
  math::XYZVector candVtx;
  double candPtPlus, candPtMinus;
  //double rPlus, rMinus;
  int nite = quad.ConversionCandidate(candVtx, candPtPlus, candPtMinus);

  if ( ! (nite && abs(nite) < 25 && nite != -1000 && nite != -2000) ) return 0;

//  math::XYZVector plusCenter = quad.GetPlusCenter(rPlus);
//  math::XYZVector minusCenter = quad.GetMinusCenter(rMinus);

#ifdef mydebug_sguazz
    std::cout << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << "\n";
    std::cout << " >>>>>>>>>>> Conv Cand: " << " Vertex X: " << candVtx.X() << " [cm] Y: " << candVtx.Y() << " [cm] pt+: " << candPtPlus<< " [GeV] pt-: " << candPtMinus << " [GeV]; #its: " << nite << "\n";
#endif

    //Add here cuts
    double minLegPt = CleaningMinLegPt;
    double maxRadialDistance = CleaningmaxRadialDistance;

    //
    // Cut on leg's transverse momenta
    if ( candPtPlus < minLegPt ) return 0;
    if ( candPtMinus < minLegPt ) return 0;
    //
    if ( candPtPlus > maxLegPt ) return 0;
    if ( candPtMinus > maxLegPt ) return 0;
    //
    // Cut on radial distance between estimated conversion vertex and inner hits
    double cr = TMath::Sqrt(candVtx.Perp2());
    double maxr2 = (maxRadialDistance + cr); maxr2*=maxr2;
    if (h2.Perp2() > maxr2) return 0;
    if (h3.Perp2() > maxr2) return 0;


// At this point implement cleaning cuts after building the seed

    //ClusterShapeFilter_knuenz:::
    std::string comparitorName = SeedComparitorPSet.getParameter<std::string>("ComponentName");
    SeedComparitor * theComparitor = (comparitorName == "none") ? 0 :  SeedComparitorFactory::get()->create( comparitorName, SeedComparitorPSet);

    if(ClusterShapeFiltering){
  	  if (theComparitor) theComparitor->init(es);

		  GlobalTrajectoryParameters pkine;
		  GlobalTrajectoryParameters mkine;

		  TransientTrackingRecHit::ConstRecHitPointer ptth1 = phits[0];
		  TransientTrackingRecHit::ConstRecHitPointer ptth2 = phits[1];
		  TransientTrackingRecHit::ConstRecHitPointer mtth1 = mhits[0];
		  TransientTrackingRecHit::ConstRecHitPointer mtth2 = mhits[1];

		  GlobalPoint vertexPos(candVtx.x(),candVtx.y(),candVtx.z());

		  float ptMinReg=0.1;
		  GlobalTrackingRegion region(ptMinReg,vertexPos,0,0,true);

		  FastHelix phelix(ptth2->globalPosition(), mtth1->globalPosition(), vertexPos, es, vertexPos);
		  pkine = phelix.stateAtVertex().parameters();
		  FastHelix mhelix(mtth2->globalPosition(), mtth1->globalPosition(), vertexPos, es, vertexPos);
		  mkine = mhelix.stateAtVertex().parameters();

		  if(theComparitor&&!theComparitor->compatible(phits, pkine, phelix, region)) { return 0; }
		  if(theComparitor&&!theComparitor->compatible(mhits, mkine, mhelix, region)) { return 0; }
    }


    //Do a very simple fit to estimate the slope
    double quadPhotCotTheta = 0.;
    double quadZ0 = 0.;
    simpleGetSlope(ptth2, ptth1, mtth1, mtth2, region, quadPhotCotTheta, quadZ0);

    double quadPhotPhi = (candVtx-vPhotVertex).Phi();

    math::XYZVector fittedPrimaryVertex(vgPhotVertex.x(), vgPhotVertex.y(),quadZ0);

    candVtx.SetZ(TMath::Sqrt(candVtx.Perp2())*quadPhotCotTheta+quadZ0);
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
        if ( similarQuadExist(thisQuad, quadV) && applyArbitration ) return 0;

    // not able to get the mag field... doing the dirty way
    //
    // Plus
    FastHelix helixPlus(ptth2->globalPosition(), ptth1->globalPosition(), convVtxGlobalPoint, es, convVtxGlobalPoint);
    GlobalTrajectoryParameters kinePlus = helixPlus.stateAtVertex().parameters();
    kinePlus = GlobalTrajectoryParameters(convVtxGlobalPoint,
					  GlobalVector(candPtPlus*cos(quadPhotPhi),candPtPlus*sin(quadPhotPhi),candPtPlus*quadPhotCotTheta),
					  1,//1
					  & kinePlus.magneticField()
					  );

    //
    // Minus
    FastHelix helixMinus(mtth2->globalPosition(), mtth1->globalPosition(), convVtxGlobalPoint, es, convVtxGlobalPoint);
    GlobalTrajectoryParameters kineMinus = helixMinus.stateAtVertex().parameters();
    kineMinus = GlobalTrajectoryParameters(convVtxGlobalPoint,
					  GlobalVector(candPtMinus*cos(quadPhotPhi),candPtMinus*sin(quadPhotPhi),candPtMinus*quadPhotCotTheta),
					  -1,//-1
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
//	      << candPtPlus  << " " << rPlus << " " << plusCenter.X() << " " << plusCenter.Y() << " "
//	      << candPtMinus << " " << rMinus << " " << minusCenter.X() << " " << minusCenter.Y() << " "
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

    

    bool buildSeedBoolPos = buildSeedBool(seedCollection,phits,ftsPlus,es,applydzCAcut,region, dzcut);
    bool buildSeedBoolNeg = buildSeedBool(seedCollection,mhits,ftsMinus,es,applydzCAcut,region, dzcut);

    
    
    if( buildSeedBoolPos && buildSeedBoolNeg ){
        buildSeed(seedCollection,phits,ftsPlus,es,false,region);
        buildSeed(seedCollection,mhits,ftsMinus,es,false,region);
	}

    return 0;

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
    const edm::EventSetup& es,
    bool apply_dzCut,
    const TrackingRegion & region) const
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
  for ( unsigned int iHit = 0; iHit < hits.size() && iHit<2; iHit++) {
    hit = hits[iHit]->hit();
    TrajectoryStateOnSurface state = (iHit==0) ?
      propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());

    TransientTrackingRecHit::ConstRecHitPointer tth = hits[iHit];

    TransientTrackingRecHit::RecHitPointer newtth =  refitHit( tth, state);

    updatedState =  updator.update(state, *newtth);

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





bool SeedForPhotonConversionFromQuadruplets::buildSeedBool(
    TrajectorySeedCollection & seedCollection,
    const SeedingHitSet & hits,
    const FreeTrajectoryState & fts,
    const edm::EventSetup& es,
    bool apply_dzCut,
    const TrackingRegion & region,
    double dzcut) const
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
  for ( unsigned int iHit = 0; iHit < hits.size() && iHit<2; iHit++) {
    hit = hits[iHit]->hit();
    TrajectoryStateOnSurface state = (iHit==0) ?
      propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) {
    	return false;}

    TransientTrackingRecHit::ConstRecHitPointer tth = hits[iHit];

    TransientTrackingRecHit::RecHitPointer newtth =  refitHit( tth, state);


    if (!checkHit(state,newtth,es)){
		return false;
    }
    
    updatedState =  updator.update(state, *newtth);
    if (!updatedState.isValid()){
    	return false;
    }

    seedHits.push_back(newtth->hit()->clone());
  }

  if(apply_dzCut){
  /// Implement here the dz cut:::

  double zCA;

  math::XYZVector EstMomGam(updatedState.globalMomentum().x(),updatedState.globalMomentum().y(),updatedState.globalMomentum().z());
  math::XYZVector EstPosGam(updatedState.globalPosition().x(),updatedState.globalPosition().y(),updatedState.globalPosition().z());

  double EstMomGamLength=TMath::Sqrt(EstMomGam.x()*EstMomGam.x()+EstMomGam.y()*EstMomGam.y()+EstMomGam.z()*EstMomGam.z());
  math::XYZVector EstMomGamNorm(EstMomGam.x()/EstMomGamLength,EstMomGam.y()/EstMomGamLength,EstMomGam.z()/EstMomGamLength);

  //Calculate dz of point of closest approach of the two lines (WA approach) -> cut on dz
  
  	  
  const GlobalPoint EstPosGamGlobalPoint(updatedState.globalPosition().x(),updatedState.globalPosition().y(),updatedState.globalPosition().z());
  const GlobalVector EstMomGamGlobalVector(updatedState.globalMomentum().x(),updatedState.globalMomentum().y(),updatedState.globalMomentum().z());


  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);
  const MagneticField* magField = bfield.product();
  TrackCharge qCharge = 0;
  
  const GlobalTrajectoryParameters myGlobalTrajectoryParameter(EstPosGamGlobalPoint, EstMomGamGlobalVector, qCharge, magField);
  
  AlgebraicSymMatrix66 aCovarianceMatrix;
  
  for (int i =0;i<6;++i)
     for (int j =0;j<6;++j)
    	 aCovarianceMatrix(i, j) = 1e-4;
 
  CartesianTrajectoryError myCartesianError (aCovarianceMatrix);
 
  const FreeTrajectoryState stateForProjectionToBeamLine(myGlobalTrajectoryParameter,myCartesianError);

  const GlobalPoint BeamSpotGlobalPoint(0,0,0);

  const reco::BeamSpot::Point BeamSpotPoint(region.origin().x(),region.origin().y(),region.origin().z());

  TSCBLBuilderNoMaterial tscblBuilder;
  
  double CovMatEntry=0.;
  reco::BeamSpot::CovarianceMatrix cov;
  for (int i=0;i<3;++i) {
	  cov(i,i) = CovMatEntry;
  }
   reco::BeamSpot::BeamType BeamType_=reco::BeamSpot::Unknown;

  reco::BeamSpot myBeamSpot(BeamSpotPoint, 0.,0.,0.,0., cov,BeamType_);
       
  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(stateForProjectionToBeamLine,myBeamSpot);
  if (tscbl.isValid()==false) {
	  zCA=0;
  }
  else{
	  GlobalPoint v = tscbl.trackStateAtPCA().position(); // Position of closest approach to BS
	  zCA=v.z();
  }
  
/*  //Calculate dz of point of closest approach of the two lines -> cut on dz

  double newX,newY,newR;
  double Rbuff=TMath::Sqrt(EstPosGam.x()*EstPosGam.x()+EstPosGam.y()*EstPosGam.y());
  double deltas,s,sbuff;
  double rMin=1e9;

  double InitXp=EstPosGam.x()+1*EstMomGamNorm.x();
  double InitXm=EstPosGam.x()-1*EstMomGamNorm.x();
  double InitYp=EstPosGam.y()+1*EstMomGamNorm.y();
  double InitYm=EstPosGam.y()-1*EstMomGamNorm.y();

  if(InitXp*InitXp+InitYp*InitYp < InitXm*InitXm+InitYm*InitYm)  {s=5; deltas=5;}
  else {s=-5; deltas=-5;}

  int nTurns=0;
  int nIterZ=0;
  for (int i_dz=0;i_dz<1000;i_dz++){
  newX=EstPosGam.x()+s*EstMomGamNorm.x();
  newY=EstPosGam.y()+s*EstMomGamNorm.y();
  newR=TMath::Sqrt(newX*newX+newY*newY);
  if(newR>Rbuff) {deltas=-1*deltas/10;nTurns++;}
  else {Rbuff=newR;}
  if(newR<rMin) {rMin=newR; sbuff=s;}
  s=s+deltas;
  nIterZ++;
  if(nTurns>1) break;
  }

  zCA=EstPosGam.z()+sbuff*EstMomGamNorm.z();
*/
  
#ifdef mydebug_knuenz
  std::cout<< "zCA: " << zCA <<std::endl;
#endif

  if(TMath::Abs(zCA)>dzcut) return false;


  }


  return true;
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
      //double dz = abs(thisQuad.z-quad.z);
      //std::cout<<"thisQuad.x="<<thisQuad.x<<"  "<<"quad.x="<<quad.x<<"  "<<"thisQuad.y="<<thisQuad.y<<"  "<<"quad.y="<<quad.y<<"  "<<"thisQuad.z"<<thisQuad.z<<"  "<<"quad.z="<<quad.z<<"  "<<"thisQuad.ptPlus"<<thisQuad.ptPlus<<"  "<<"quad.ptPlus="<<quad.ptPlus<<"  "<<"thisQuad.ptMinus="<<thisQuad.ptMinus<<"  "<<"quad.ptMinus="<<quad.ptMinus<<"  "<<"thisQuad.cot="<<thisQuad.cot<<"  "<<"quad.cot="<<quad.cot<<std::endl; //ValDebug
      //std::cout<<"x1-x2="<<dx<<"y1-y2="<<dy<<"dx*dx+dy*dy="<<dx*dx+dy*dy<<std::endl; //ValDebug
      //std::cout<<"thisQuad.ptPlus-quad.ptPlus="<<thisQuad.ptPlus-quad.ptPlus<<"abs(thisQuad.ptPlus-quad.ptPlus)="<<abs(thisQuad.ptPlus-quad.ptPlus)<<std::endl; //ValDebug
      //std::cout <<sqrt(dx*dx+dy*dy)<<" <1? "<<dz<<" <3? "<<abs(thisQuad.ptPlus-quad.ptPlus)<<" <0.5? "<<abs(thisQuad.ptMinus-quad.ptMinus)<<" <0.5? "<<abs(thisQuad.cot-quad.cot)<<" <0.3? "<<std::endl; //ValDebug
      //if ( sqrt(dx*dx+dy*dy)<1.)
	  //	  std::cout <<sqrt(dx*dx+dy*dy)<<" <1? "<<dz<<" <3? "<<abs(thisQuad.ptPlus-quad.ptPlus)<<" <"<<0.5*quad.ptPlus<<"? "<<abs(thisQuad.ptMinus-quad.ptMinus)<<" <"<<0.5*quad.ptMinus<<"? "<<abs(thisQuad.cot-quad.cot)<<" <"<<0.3*quad.cot<<"? "<<std::endl; //ValDebug
    // ( sqrt(dx*dx+dy*dy)<1. &&
	//z<3. &&
	//bs(thisQuad.ptPlus-quad.ptPlus)<0.5*quad.ptPlus &&
	//bs(thisQuad.ptMinus-quad.ptMinus)<0.5*quad.ptMinus &&
	//bs(thisQuad.cot-quad.cot)<0.3*quad.cot
	//
    //  {
    //  //std::cout<<"Seed rejected due to arbitration"<<std::endl;
    //  return true;
    //  }
      if ( sqrt(dx*dx+dy*dy)<1. &&
	   fabs(thisQuad.ptPlus-quad.ptPlus)<0.5*quad.ptPlus &&
	   fabs(thisQuad.ptMinus-quad.ptMinus)<0.5*quad.ptMinus
	   )
    	  {
    	  //std::cout<<"Seed rejected due to arbitration"<<std::endl;
    	  return true;
    	  }
    }
  quadV.push_back(thisQuad);
  return false;

}

double SeedForPhotonConversionFromQuadruplets::DeltaPhiManual(math::XYZVector v1, math::XYZVector v2){


	double  kPI = TMath::Pi();
	double  kTWOPI     = 2.*kPI;
	double DeltaPhiMan=v1.Phi()-v2.Phi();
	while (DeltaPhiMan >= kPI) DeltaPhiMan -= kTWOPI;
	while (DeltaPhiMan < -kPI) DeltaPhiMan += kTWOPI;

	return DeltaPhiMan;

}
