#include "RecoHI/HiEgammaAlgos/interface/HICaloUtil.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace edm;
using namespace reco;

#define PI 3.141592653589793238462643383279502884197169399375105820974945


const double HICaloUtil::kEEtaBarrelEndcap = 1.479;
const double HICaloUtil::kER_ECAL          = 136.5;
const double HICaloUtil::kEZ_Endcap        = 328.0;
const double HICaloUtil::kERBARM           = 1.357;
const double HICaloUtil::kEZENDM           = 3.186;


//________________________________________________________________________
double HICaloUtil::EcalEta(const Candidate &p)
{  
   // Calculate the eta of the particle wrt the center of the detector.

   double rhovert = sqrt(p.vertex().X()*p.vertex().X()+p.vertex().Y()*p.vertex().Y());
   double zvert   = p.vertex().Z();
   return EcalEta(p.eta(), zvert, rhovert);
}


//________________________________________________________________________
double HICaloUtil::EcalEta(double EtaParticle, double Zvertex, double plane_Radius)
{  
   // Calculate the eta of the particle wrt center of detector.

   if(EtaParticle != 0.) {
      double Theta = 0.0  ;
      double ZEcal = (kER_ECAL-plane_Radius)*sinh(EtaParticle)+Zvertex;
 
      if(ZEcal != 0.0) Theta = atan(kER_ECAL/ZEcal);
      if(Theta<0.0) Theta = Theta+PI ;
      
      double ETA = - log(tan(0.5*Theta));
 
      if( fabs(ETA) > kEEtaBarrelEndcap )
      {
         double Zend = kEZ_Endcap ;
         if(EtaParticle<0.0 )  Zend = -Zend ;
         double Zlen = Zend - Zvertex ;
         double RR = Zlen/sinh(EtaParticle);
         Theta = atan((RR+plane_Radius)/Zend);
         if(Theta<0.0) Theta = Theta+PI ;
         ETA = - log(tan(0.5*Theta));
      }
                 
      return ETA;
   } 

   return EtaParticle;
}


//________________________________________________________________________
double HICaloUtil::EcalPhi(const Candidate &p)
{  
   // Calculate the phi of the particle transported to the Ecal.

   int charge     = p.charge();
   double phi     = p.phi();
   double rhovert =  sqrt(p.vertex().X()*p.vertex().X()+p.vertex().Y()*p.vertex().Y());
   if(charge) phi   = EcalPhi(p.pt(), p.eta(), phi, charge, rhovert);
   return phi;
}


//________________________________________________________________________
double HICaloUtil::EcalPhi(double PtParticle, double EtaParticle, 
                              double PhiParticle, int ChargeParticle, double Rstart)
{
   // Calculate the phi of the particle transported to the Ecal.

   double Rbend = kERBARM-(Rstart/100.); //assumed rstart in cm
   double Bend  = 0.3 * 4. * Rbend/ 2.0 ;
 
   //PHI correction
   double PHI = 0.0 ;
   if( fabs(EtaParticle) <=  kEEtaBarrelEndcap) {
      if (fabs(Bend/PtParticle)<=1.) {
         PHI = PhiParticle - asin(Bend/PtParticle)*ChargeParticle;
         if(PHI >  PI) {PHI = PHI - 2*PI;}
         if(PHI < -PI) {PHI = PHI + 2*PI;}
         return PHI;
      } 
   } else {
      double Rhit = 0.0 ;
      Rhit = kEZENDM / sinh(fabs(EtaParticle));
      if (fabs(((Rhit-(Rstart/100.))/Rbend)*Bend/PtParticle)<=1.) {
         PHI = PhiParticle - asin(((Rhit-(Rstart/100.))/Rbend)*Bend/PtParticle)*ChargeParticle;
         if(PHI >  PI) {PHI = PHI - 2*PI;}
         if(PHI < -PI) {PHI = PHI + 2*PI;}
         return PHI;
      } else {
         return PhiParticle;
      }
   }

   return PhiParticle;
}
