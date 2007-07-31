#include "DataFormats/EgammaCandidates/interface/PixelMatchElectron.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

  const double pi = M_PI, pi2 = 2 * M_PI;
 
using namespace reco;


PixelMatchElectron::PixelMatchElectron(const SuperClusterRef scl, const TrackRef t,
					     const GlobalPoint tssuperPos, const GlobalVector tssuperMom, const GlobalPoint tsseedPos, const GlobalVector tsseedMom, const double HoE) :
               hadOverEm_(HoE), superCluster_(scl), track_(t)   
{
  //
  // electron particle quantities
  //

  //Initialise to E from cluster + direction from track
  //  const math::XYZVector trackMom = gsfTrack_->momentum();  //FIXME: later on: impactModeMomentum, GsfUtil
  const math::XYZVector trackMom = track_->momentum();  //FIXME: later on: impactModeMomentum, GsfUtil
  double scale = superCluster_->energy()/trackMom.R();    
  math::XYZTLorentzVectorD momentum= math::XYZTLorentzVector(trackMom.x()*scale,
                          trackMom.y()*scale,
                          trackMom.z()*scale,
			  superCluster_->energy());
  setCharge(track_->charge());
  setP4(momentum);
  setVertex(Point(track_->vertex()));
  setPdgId( -11 * charge() );

  math::XYZPoint trackPos= track_->vertex();
  trackPositionAtVtx_=math::XYZVector(trackPos.x(),trackPos.y(),trackPos.z());
  trackPositionAtCalo_=math::XYZVector(tssuperPos.x(),
                                       tssuperPos.y(),
                                       tssuperPos.z());
  trackMomentumAtCalo_=math::XYZVector(tssuperMom.x(),
                                       tssuperMom.y(),
                                       tssuperMom.z());
  //
  // supercluster - track at impact match parameters
  //
  eSuperClusterOverP_=-1;
  if (trackMom.R()!=0) eSuperClusterOverP_= superCluster_->energy()/trackMom.R();
  float trackEta = ecalEta(
						  track_->innerMomentum().eta(),
                                                  track_->innerPosition().z(),
                                                  track_->innerPosition().Rho());

  float trackPhi = ecalPhi(
						  track_->innerMomentum().Rho(),
                                                  track_->innerMomentum().eta(),
                                                  track_->innerMomentum().phi(),
						  track_->charge(),
                                                  track_->innerPosition().Rho());

  deltaEtaSuperClusterAtVtx_=superCluster_->position().eta() - trackEta;
  float dphi                =superCluster_->position().phi() - trackPhi;
  if (fabs(dphi)>pi)
      dphi = dphi < 0? pi2 + dphi : dphi - pi2;
  deltaPhiSuperClusterAtVtx_ = dphi;

  // 
  // seed cluster - track at calo match quantities
  //

  const BasicClusterRef seedClus = superCluster_->seed();
  eSeedClusterOverPout_ = -1;
  //  if (tsseed.globalMomentum().mag() > 0.)
  //    eSeedClusterOverPout_ = seedClus->energy()/tsseed.globalMomentum().mag();
  if (tsseedMom.mag() > 0.)
    eSeedClusterOverPout_ = seedClus->energy()/tsseedMom.mag();

  //  deltaEtaSeedClusterAtCalo_ = seedClus->eta() - tsseed.globalPosition().eta();
  //  dphi                       = seedClus->phi() - tsseed.globalPosition().phi();
  deltaEtaSeedClusterAtCalo_ = seedClus->eta() - tsseedPos.eta();
  dphi                       = seedClus->phi() - tsseedPos.phi();
  if (fabs(dphi)>pi)
    dphi = dphi < 0? pi2 + dphi : dphi - pi2;
  deltaPhiSeedClusterAtCalo_ = dphi;

  //
  // other quantities
  //

  //temporary
  momentumFromEpCombination_=false;
  trackMomentumError_=0;
  
  //FIXME  hadOverEm_ = superCluster_->seed()->getHoe();
  //FIXME  hadOverEm_ *= seedClus->energy()/superCluster_->energy();

}
 
//FIXME!!
static const float R_ECAL           = 136.5;
static const float Z_Endcap         = 328.0;
static const float etaBarrelEndcap  = 1.479; 



float PixelMatchElectron::ecalEta(float EtaParticle , float Zvertex, float plane_Radius)
{
  if (EtaParticle!= 0.)
    {
      float Theta = 0.0  ;
      float ZEcal = (R_ECAL-plane_Radius)*sinh(EtaParticle)+Zvertex;
      
      if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
      if(Theta<0.0) Theta = Theta+Geom::pi() ;

      float ETA = - log(tan(0.5*Theta));
      
      if( fabs(ETA) > etaBarrelEndcap )
	{
	  float Zend = Z_Endcap ;
	  if(EtaParticle<0.0 )  Zend = -Zend ;
	  float Zlen = Zend - Zvertex ;
	  float RR = Zlen/sinh(EtaParticle);
	  Theta = atan((RR+plane_Radius)/Zend);
	  if(Theta<0.0) Theta = Theta+Geom::pi() ;
	  ETA = - log(tan(0.5*Theta));
	}
      return ETA;
    }
  else
    {
      edm::LogWarning("")  << "[EcalPositionFromTrack::etaTransformation] Warning: Eta equals to zero, not correcting" ;
      return EtaParticle;
    }
}

float PixelMatchElectron::ecalPhi(float PtParticle, float EtaParticle, float PhiParticle, int ChargeParticle, float Rstart)
{
  //Magnetic field
  const float RBARM = 1.357 ;  // was 1.31 , updated on 16122003
  const float ZENDM = 3.186 ;  // was 3.15 , updated on 16122003
  float Rbend = RBARM-(Rstart/100.); //Assumed Rstart in cm
  float Bend  = 0.3 * 4. * Rbend/ 2.0 ;

  //---PHI correction
  float PHI = 0.0 ;
  if( fabs(EtaParticle) <=  etaBarrelEndcap)
    {
      if (fabs(Bend/PtParticle)<=1.)
	{
	  PHI = PhiParticle - asin(Bend/PtParticle)*ChargeParticle;
	  if(PHI >  Geom::pi()) {PHI = PHI - Geom::twoPi();}
	  if(PHI < -Geom::pi()) {PHI = PHI + Geom::twoPi();}
	}
      else
	{
	  edm::LogWarning("") << "[EcalPositionFromTrack::phiTransformation] Warning:Too low Pt, giving up ";
	  return PhiParticle;
	}
    }
  
  if( fabs(EtaParticle) >  etaBarrelEndcap )
    {
      float Rhit = 0.0 ;
      Rhit = ZENDM / sinh(fabs(EtaParticle));
      if (fabs(((Rhit-(Rstart/100.))/Rbend)*Bend/PtParticle)<=1.)
	{
	  PHI = PhiParticle - asin(((Rhit-(Rstart/100.))/Rbend)*Bend/PtParticle)*ChargeParticle;
	  if(PHI >  Geom::pi()) {PHI = PHI - Geom::twoPi();}
	  if(PHI < -Geom::pi()) {PHI = PHI + Geom::twoPi();}
	}
      else
	{
	  edm::LogWarning("") <<"[EcalPositionFromTrack::phiTransformation] Warning:Too low Pt, giving up ";
	  return PhiParticle;
	}
      
    }
  
  //---Return the result
  return PHI;
}

bool PixelMatchElectron::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( track(), o->track() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
  return false;
}

PixelMatchElectron * PixelMatchElectron::clone() const { 
  return new PixelMatchElectron( * this ); 
}
