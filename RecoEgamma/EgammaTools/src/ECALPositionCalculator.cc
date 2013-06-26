
#include "RecoEgamma/EgammaTools/interface/ECALPositionCalculator.h"

// Framework includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h"

static const float R_ECAL           = 136.5;
static const float Z_Endcap         = 328.0;
static const float etaBarrelEndcap  = 1.479;

double ECALPositionCalculator::ecalPhi(const MagneticField *magField,const math::XYZVector &momentum, const math::XYZPoint &vertex, int charge)
{

   // Get kinematic variables

   float ptParticle = momentum.Rho();
   float etaParticle = momentum.eta();
   float phiParticle = momentum.phi();
   float vRho = vertex.Rho();

   // Magnetic field

   const float RBARM = 1.357 ;  	// was 1.31 , updated on 16122003
   const float ZENDM = 3.186 ;  	// was 3.15 , updated on 16122003

   float rbend = RBARM-(vRho/100.0); 	//Assumed vRho in cm
   float bend  = 0.3 * magField->inTesla(GlobalPoint(0.,0.,0.)).z() * rbend / 2.0; 
   float phi = 0.0;

   if( fabs(etaParticle) <=  etaBarrelEndcap)
   {
      if (fabs(bend/ptParticle) <= 1.) 
      {
         phi = phiParticle - asin(bend/ptParticle)*charge;
	 if(phi >  Geom::pi()) phi = phi - Geom::twoPi();
	 if(phi < -Geom::pi()) phi = phi + Geom::twoPi();
      } else {
         edm::LogWarning("") << "[EcalPositionFromTrack::phiTransformation] Warning: "
    			     << "Too low Pt, giving up";
	 return phiParticle;
      }

   } // end if in the barrel
  
   if(fabs(etaParticle) > etaBarrelEndcap)
   {
      float rHit = 0.0;
      rHit = ZENDM / sinh(fabs(etaParticle));
      if (fabs(((rHit-(vRho/100.0))/rbend)*bend/ptParticle) <= 1.0)
      {
         phi = phiParticle - asin(((rHit-(vRho/100.0)) / rbend)*bend/ptParticle)*charge;
	 if(phi >  Geom::pi()) phi = phi - Geom::twoPi();
	 if(phi < -Geom::pi()) phi = phi + Geom::twoPi();
      } else {
         edm::LogWarning("") << "[EcalPositionFromTrack::phiTransformation] Warning: "
		             << "Too low Pt, giving up";
	 return phiParticle;
      }  

    } // end if in the endcap
  
   return phi;

}

double ECALPositionCalculator::ecalEta(const math::XYZVector &momentum, const math::XYZPoint &vertex)
{

   // Get kinematic variables

   float etaParticle = momentum.eta();
   float vZ = vertex.z();
   float vRho = vertex.Rho();

   if (etaParticle != 0.0)
   {
      float theta = 0.0;
      float zEcal = (R_ECAL-vRho)*sinh(etaParticle)+vZ;
      
      if(zEcal != 0.0) theta = atan(R_ECAL/zEcal);
      if(theta < 0.0) theta = theta + Geom::pi() ;

      float ETA = - log(tan(0.5*theta));
      
      if( fabs(ETA) > etaBarrelEndcap )
      {
         float Zend = Z_Endcap ;
	 if(etaParticle < 0.0 )  Zend = - Zend;
	 float Zlen = Zend - vZ ;
	 float RR = Zlen/sinh(etaParticle);
	 theta = atan((RR+vRho)/Zend);
	 if(theta < 0.0) theta = theta+Geom::pi();
	 ETA = -log(tan(0.5*theta));
      }
      return ETA;

    } else {
       edm::LogWarning("")  << "[EcalPositionFromTrack::etaTransformation] Warning: "
			    << "Eta equals to zero, not correcting";
       return etaParticle;
    }

}

