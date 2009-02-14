#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 

#include <cmath>

const double pi = M_PI, pi2 = 2 * M_PI;
 
using namespace reco;

GsfElectron::GsfElectron()
 : scSigmaEtaEta_(std::numeric_limits<float>::infinity()),
   scSigmaIEtaIEta_(std::numeric_limits<float>::infinity()),
   scE1x5_(0.), scE2x5Max_(0.), scE5x5_(0.),
   shFracInnerHits_(0.)
{ }

GsfElectron::GsfElectron
 ( const LorentzVector & p4,
   const SuperClusterRef scl,
   const GsfTrackRef gsfTrack,
   const GlobalPoint & tssuperPos, const GlobalVector & tssuperMom,
   const GlobalPoint & tsseedPos, const GlobalVector & tsseedMom,
   const GlobalPoint & innPos, const GlobalVector & innMom,
   const GlobalPoint & vtxPos, const GlobalVector & vtxMom,
   const GlobalPoint & outPos, const GlobalVector & outMom,
   double hadOverEm1,double hadOverEm2,
   float scSigmaEtaEta, float scSigmaIEtaIEta,
   float scE1x5, float scE2x5Max, float scE5x5,
   const TrackRef ctfTrack, const float shFracInnerHits,
   const BasicClusterRef electronCluster,
   const GlobalPoint & tselePos, const GlobalVector & tseleMom
 )
 : hadOverEm1_(hadOverEm1), hadOverEm2_(hadOverEm2), superCluster_(scl), track_(gsfTrack),
   scSigmaEtaEta_(scSigmaEtaEta), scSigmaIEtaIEta_(scSigmaIEtaIEta),
   scE1x5_(scE1x5), scE2x5Max_(scE2x5Max), scE5x5_(scE5x5),
   ctfTrack_(ctfTrack), shFracInnerHits_(shFracInnerHits),
   electronCluster_(electronCluster)
 {
  setCharge(track_->charge()) ;
  setP4(p4) ;
  setVertex(Point(vtxPos)) ;
  setPdgId( -11 * charge()) ;

  trackPositionAtVtx_=math::XYZPoint(vtxPos.x(),vtxPos.y(),vtxPos.z());
  trackPositionAtCalo_=math::XYZPoint(tssuperPos.x(),
                                       tssuperPos.y(),
                                       tssuperPos.z());
  trackMomentumAtCalo_=math::XYZVector(tssuperMom.x(),
                                       tssuperMom.y(),
                                       tssuperMom.z());
  trackMomentumAtVtx_=math::XYZVector(vtxMom.x(),
                                      vtxMom.y(),
                                      vtxMom.z());
  trackMomentumOut_=math::XYZVector(tsseedMom.x(),
                                        tsseedMom.y(),
                                        tsseedMom.z());
  trackMomentumAtEleClus_=math::XYZVector(tseleMom.x(),
                                        tseleMom.y(),
                                        tseleMom.z());
  //
  // supercluster - track at impact match parameters
  //
  superClusterEnergy_=superCluster_->energy();
  eSuperClusterOverP_=-1;
  if (vtxMom.mag() > 0) eSuperClusterOverP_= superCluster_->energy()/vtxMom.mag();
  
  deltaEtaSuperClusterAtVtx_ = superCluster_->eta() - tssuperPos.eta();
  float dphi                       = superCluster_->phi() - tssuperPos.phi();
  if (fabs(dphi)>pi)
    dphi = dphi < 0? pi2 + dphi : dphi - pi2;
  deltaPhiSuperClusterAtVtx_ = dphi;

  // 
  // seed cluster - track at calo match quantities
  //
  const BasicClusterRef seedClus = superCluster_->seed();
  eSeedClusterOverPout_ = -1;
  if (tsseedMom.mag() > 0.)
    eSeedClusterOverPout_ = seedClus->energy()/tsseedMom.mag();

  deltaEtaSeedClusterAtCalo_ = seedClus->eta() - tsseedPos.eta();
  dphi                       = seedClus->phi() - tsseedPos.phi();
  if (fabs(dphi)>pi)
    dphi = dphi < 0? pi2 + dphi : dphi - pi2;
  deltaPhiSeedClusterAtCalo_ = dphi;

  // 
  // ele cluster - track at calo match quantities
  //
  eEleClusterOverPout_ = -1;
  if (tseleMom.mag() > 0.)
    eEleClusterOverPout_ = electronCluster->energy()/tseleMom.mag();

  deltaEtaEleClusterAtCalo_ = electronCluster->eta() - tselePos.eta();
  dphi                       = electronCluster->phi() - tselePos.phi();
  if (fabs(dphi)>pi)
    dphi = dphi < 0? pi2 + dphi : dphi - pi2;
  deltaPhiEleClusterAtCalo_ = dphi;

  eSeedClusterOverP_ = -1;
  if (vtxMom.mag() > 0) 
    eSeedClusterOverP_= seedClus->energy()/vtxMom.mag();

  //
  // other quantities
  //
  fbrem_ = 1.e30;
  if (outMom.mag() > 0.) fbrem_ = (innMom.mag() - outMom.mag()) / outMom.mag();

  energyScaleCorrected_=false;
  momentumFromEpCombination_=false;
  trackMomentumError_=0;
  
  isEB_=false;
  isEE_=false;
  isEBEEGap_=false;
  isEBEtaGap_=false;
  isEBPhiGap_=false;
  isEEDeeGap_=false;
  isEERingGap_=false;
  
}

void GsfElectron::correctElectronEnergyScale(const float newEnergy) {
  
  math::XYZTLorentzVectorD momentum=p4();
  momentum*=newEnergy/momentum.e();
  setP4(momentum);
  hadOverEm1_ *=superClusterEnergy_/newEnergy; 
  hadOverEm2_ *=superClusterEnergy_/newEnergy; 
  eSuperClusterOverP_*=newEnergy/superClusterEnergy_;
  superClusterEnergy_=newEnergy;
 
  energyScaleCorrected_=true;    
}
 
void GsfElectron::correctElectronFourMomentum(const math::XYZTLorentzVectorD & momentum,float & enErr, float & tmErr) {
 
  setP4(momentum);
  energyError_ = enErr;
  trackMomentumError_ = tmErr;
  momentumFromEpCombination_=true;
}
 
void GsfElectron::classifyElectron(const int myclass)
{
  electronClass_ = myclass;
}

bool GsfElectron::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 && 
	   ( checkOverlap( gsfTrack(), o->gsfTrack() ) ||
	     checkOverlap( superCluster(), o->superCluster() ) ) 
	   );
  return false;
}

GsfElectron * GsfElectron::clone() const { 
  return new GsfElectron( * this ); 
}

bool GsfElectron::isElectron() const {
  return true;
}
