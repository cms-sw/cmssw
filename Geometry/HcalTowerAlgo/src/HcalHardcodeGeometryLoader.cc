#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "HcalHardcodeGeometryData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

typedef CaloCellGeometry::CCGFloat CCGFloat ;

HcalHardcodeGeometryLoader::HcalHardcodeGeometryLoader(const HcalTopology& ht) : 
   theTopology ( 0   ) ,
   extTopology ( &ht )
{
   init();
}

void 
HcalHardcodeGeometryLoader::init() 
{
  theBarrelRadius = 190.;
  theHBThickness = 93.6; // just from drawings.  All thicknesses needs to be done right

  theHB15aThickness = theHBThickness * 12.0/17.0; // relative weight from layer count!
  theHB15bThickness = theHBThickness * 4.0/17.0;  // relative weight from layer count!
  theHB16aThickness = theHBThickness * 1.0/17.0;  // relative weight from layer count!
  theHB16bThickness = theHBThickness * 7.0/17.0;  // relative weight from layer count!

  theOuterRadius  = 406;
  theHOThickness = 1.;

  theHEZPos[0] = 388.0;
  theHEZPos[1] = 397.0;
  theHEZPos[2] = 450.0;
  theHEZPos[3] = 568.0;

  theHFZPos[0] = 1100.0;
  theHFZPos[1] = 1120.0;
  theHFThickness = 165;

}


HcalHardcodeGeometryLoader::ReturnType
HcalHardcodeGeometryLoader::load( DetId::Detector det, 
				  int             subdet ) 
{
   HcalSubdetector hsub=static_cast<HcalSubdetector>( subdet );
   ReturnType hg( new HcalGeometry( *extTopology) );

   if( hg->cornersMgr() == 0 ) hg->allocateCorners( extTopology->ncells() );
   if( hg->parMgr()     == 0 ) hg->allocatePar( 
       HcalGeometry::k_NumberOfParametersPerShape*hg->numberOfShapes(),
      HcalGeometry::k_NumberOfParametersPerShape ) ;

   switch (hsub) 
   {
      case (HcalBarrel) :  fill(hsub, extTopology->firstHBRing(), extTopology->lastHBRing(), hg ); break;
      case (HcalEndcap) :  fill(hsub, extTopology->firstHERing(), extTopology->lastHERing(), hg ); break;
      case (HcalForward) : fill(hsub, extTopology->firstHFRing(), extTopology->lastHFRing(), hg ); break;
      case (HcalOuter) :   fill(hsub, extTopology->firstHORing(), extTopology->lastHORing(), hg ); break;
      default:  break;
   }
   return hg;
}

HcalHardcodeGeometryLoader::ReturnType
HcalHardcodeGeometryLoader::load() 
{
   ReturnType hg( new HcalGeometry( *extTopology ) ) ;

   if( hg->cornersMgr() == 0 ) hg->allocateCorners(  extTopology->ncells() );
   if( hg->parMgr()     == 0 ) hg->allocatePar( 500, 5 ) ;

   fill(HcalBarrel,  extTopology->firstHBRing(), extTopology->lastHBRing(), hg); 
   fill(HcalEndcap,  extTopology->firstHERing(), extTopology->lastHERing(), hg); 
   fill(HcalForward, extTopology->firstHFRing(), extTopology->lastHFRing(), hg); 
   fill(HcalOuter,   extTopology->firstHORing(), extTopology->lastHORing(), hg);
   return hg;
}

void 
HcalHardcodeGeometryLoader::fill( HcalSubdetector subdet, 
				  int             firstEtaRing, 
				  int             lastEtaRing, 
				  ReturnType      geom           ) 
{
   // start by making the new HcalDetIds
   std::vector<HcalDetId> hcalIds;
   int nDepthSegments, startingDepthSegment;
   for(int etaRing = firstEtaRing; etaRing <= lastEtaRing; ++etaRing) {
      extTopology->depthBinInformation(subdet, etaRing, nDepthSegments, startingDepthSegment);
      unsigned int nPhiBins = extTopology->nPhiBins(etaRing);
      unsigned int phiInc=72/std::max(36u,nPhiBins);
      for(int idepth = 0; idepth < nDepthSegments; ++idepth) {
	 int depthBin = startingDepthSegment + idepth;
	 
	 for(unsigned iphi = 1; iphi <= 72; iphi+=phiInc) {
	    for(int zsign = -1; zsign <= 1; zsign += 2) {
	       HcalDetId id( subdet, zsign * etaRing, iphi, depthBin);
	       if (extTopology->valid(id)) hcalIds.push_back(id);
	    }
	 } 
      }
   }

   edm::LogInfo("HcalHardcodeGeometry") << "Number of HCAL DetIds made: " 
					<< subdet 
					<< " " << hcalIds.size();
   // for each new HcalDetId, make a CaloCellGeometry
   for(std::vector<HcalDetId>::const_iterator hcalIdItr = hcalIds.begin();
       hcalIdItr != hcalIds.end(); ++hcalIdItr)
   {
      makeCell(*hcalIdItr,geom) ;
   }
}
     

inline double theta_from_eta(double eta){return (2.0*atan(exp(-eta)));}

void
HcalHardcodeGeometryLoader::makeCell( const HcalDetId& detId ,
				      ReturnType       geom    ) const 
{
   // the two eta boundaries of the cell
   double eta1, eta2;
   HcalSubdetector subdet = detId.subdet();
   int etaRing = detId.ietaAbs();
   if(subdet == HcalForward) 
   {
      eta1 = theHFEtaBounds[etaRing-extTopology->firstHFRing()];
      eta2 = theHFEtaBounds[etaRing-extTopology->firstHFRing()+1];
   } 
   else if (etaRing==28 && detId.depth()==3) 
   {
      // double-width in eta at depth 3 in ieta=28
      eta1 = theHBHEEtaBounds[etaRing-1];
      eta2 = theHBHEEtaBounds[etaRing+1];
   } 
   else 
   {
      eta1 = theHBHEEtaBounds[etaRing-1];
      eta2 = theHBHEEtaBounds[etaRing];
   }
   double eta = 0.5*(eta1+eta2) * detId.zside();
   double deta = 0.5*(eta2-eta1);
   double theta = theta_from_eta(eta);

   int nDepthBins, startingBin;
   extTopology->depthBinInformation(subdet,etaRing,nDepthBins,startingBin);

   // in radians
   double dphi_nominal = 2.0*M_PI / extTopology->nPhiBins(1); // always the same
   double dphi_half = M_PI / extTopology->nPhiBins(etaRing); // half-width
  
   double phi_low = dphi_nominal*(detId.iphi()-1); // low-edge boundaries are constant...
   double phi = phi_low+dphi_half;

   bool isBarrel = (subdet == HcalBarrel || subdet == HcalOuter);

   double x,y,z,r;
   double thickness;

   if(isBarrel) 
   {
      if(subdet == HcalBarrel) 
      {
	 if (detId.ietaAbs()==15) 
	 {
	    if (detId.depth()==1) 
	    {
	       r = theBarrelRadius;
	       thickness = theHB15aThickness;
	    } 
	    else 
	    {
	       r = theBarrelRadius+theHB15aThickness;
	       thickness = theHB15bThickness;
	    }
	 } 
	 else if (detId.ietaAbs()==16) 
	 {
	    if (detId.depth()==1) 
	    {
	       r = theBarrelRadius;
	       thickness = theHB16aThickness;
	    } 
	    else 
	    {
	       r = theBarrelRadius+theHB16aThickness;
	       thickness = theHB16bThickness;
	    }
	 } 
	 else 
	 {
	    r = theBarrelRadius;
	    thickness = theHBThickness;
	 }
      } 
      else 
      { // HO
	 r = theOuterRadius;
	 thickness = theHOThickness;
      } 
      z = r * sinh(eta); // sinh(eta) == 1/tan(theta)
      thickness *= cosh(eta); // cosh(eta) == 1/sin(theta)
   } 
   else 
   {
      int depth = detId.depth();
      if(subdet == HcalEndcap) 
      { // Sadly, Z must be made a function of ieta.
	 if (nDepthBins==1) 
	 {
	    z = theHEZPos[depth - 1];
	    thickness = theHEZPos[3] - z;
	 } 
	 else if (nDepthBins==2 && depth==2) 
	 {
	    z = theHEZPos[depth - 1];
	    if (etaRing==29) // special case for tower 29
	       thickness = theHEZPos[depth] - z;
	    else
	       thickness = theHEZPos[3] - z;
	 } 
	 else 
	 {
	    z = theHEZPos[depth - 1];
	    thickness = theHEZPos[depth] - z;
	 }
	 thickness /= fabs(tanh(eta));
      } 
      else 
      {
	 z = theHFZPos[depth - 1];
	 thickness = theHFThickness-(theHFZPos[depth-1]-theHFZPos[0]);
      }
      z*=detId.zside(); // get the sign right.
      r = z * tan(theta);
      assert(r>0.);
   }

   x = r * cos(phi);
   y = r * sin(phi);
   GlobalPoint point(x,y,z);

   std::vector<CCGFloat> hp ;
   hp.reserve(5) ;

   if (subdet==HcalForward) 
   {
      hp.push_back( deta ) ;
      hp.push_back( dphi_half ) ;
      hp.push_back( thickness/2 ) ;
      hp.push_back( fabs( point.eta() ) ) ;
      hp.push_back( fabs( point.z() ) ) ;
   } 
   else 
   { 
      const double mysign ( isBarrel ? 1 : -1 ) ;
      hp.push_back( deta ) ;
      hp.push_back( dphi_half ) ;
      hp.push_back( mysign*thickness/2. ) ;
      hp.push_back( fabs( point.eta() ) ) ;
      hp.push_back( fabs( point.z() ) ) ;
   }
   geom->newCell( point, point, point,
		  CaloCellGeometry::getParmPtr( hp, 
						geom->parMgr(), 
						geom->parVecVec() ),
		  detId );
}


