#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h" 

#include <cmath>

const double pi = M_PI, pi2 = 2 * M_PI;
 
using namespace reco;

GsfElectron::GsfElectron(const LorentzVector & p4,
					     const SuperClusterRef scl,
					     const GsfTrackRef gsft,
					     const GlobalPoint tssuperPos, const GlobalVector tssuperMom, 
                                             const GlobalPoint tsseedPos, const GlobalVector tsseedMom, 
					     const GlobalPoint innPos, const GlobalVector innMom, 
					     const GlobalPoint vtxPos, const GlobalVector vtxMom, 
					     const GlobalPoint outPos, const GlobalVector outMom, 
					     const double HoE) :
  hadOverEm_(HoE), superCluster_(scl), track_(gsft)
 {
  //
  // electron particle quantities
  //

  //Initialise to E from cluster + direction from track
//   double scale = superCluster_->energy()/vtxMom.mag();    
//   math::XYZTLorentzVectorD momentum= math::XYZTLorentzVector(vtxMom.x()*scale,
//                           vtxMom.y()*scale,
//                           vtxMom.z()*scale,
// 			  superCluster_->energy());
   setCharge(track_->charge());
   setP4(p4);
   setVertex(Point(vtxPos));
   setPdgId( -11 * charge());

  trackPositionAtVtx_=math::XYZVector(vtxPos.x(),vtxPos.y(),vtxPos.z());
  trackPositionAtCalo_=math::XYZVector(tssuperPos.x(),
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
  //
  // supercluster - track at impact match parameters
  //
  superClusterEnergy_=superCluster_->energy();
  eSuperClusterOverP_=-1;
  //  if (innMom.R()!=0) eSuperClusterOverP_= superCluster_->energy()/innMom.R();
  if (innMom.mag()!=0) eSuperClusterOverP_= superCluster_->energy()/innMom.mag();
//   float trackEta = ecalEta(
// 						  track_->innerMomentum().eta(),
//                                                   track_->innerPosition().z(),
//                                                   track_->innerPosition().Rho());

//   float trackPhi = ecalPhi(
// 						  track_->innerMomentum().Rho(),
//                                                   track_->innerMomentum().eta(),
//                                                   track_->innerMomentum().phi(),
// 						  track_->charge(),
//                                                   track_->innerPosition().Rho());

//   deltaEtaSuperClusterAtVtx_=superCluster_->position().eta() - trackEta;
//   float dphi                =superCluster_->position().phi() - trackPhi;
//   if (fabs(dphi)>pi)
//       dphi = dphi < 0? pi2 + dphi : dphi - pi2;
//   deltaPhiSuperClusterAtVtx_ = dphi;

  // 
  // seed cluster - track at calo match quantities
  //

  const BasicClusterRef seedClus = superCluster_->seed();
  eSeedClusterOverPout_ = -1;
  //  if (tsseed.globalMomentum().mag() > 0.)
  //    eSeedClusterOverPout_ = seedClus->energy()/tsseed.globalMomentum().mag();
  //  GlobalPoint tsseedPos=seedTsos.globalPosition();
  //  GlobalVector tsseedMom=seedTsos.globalMomentum();
  if (tsseedMom.mag() > 0.)
    eSeedClusterOverPout_ = seedClus->energy()/tsseedMom.mag();

  deltaEtaSeedClusterAtCalo_ = seedClus->eta() - tsseedPos.eta();
  float dphi                       = seedClus->phi() - tsseedPos.phi();
  if (fabs(dphi)>pi)
    dphi = dphi < 0? pi2 + dphi : dphi - pi2;
  deltaPhiSeedClusterAtCalo_ = dphi;

  //
  // other quantities
  //
  energyScaleCorrected_=false;
  momentumFromEpCombination_=false;
  trackMomentumError_=0;
  
  //FIXME  hadOverEm_ = superCluster_->seed()->getHoe();
  //FIXME  hadOverEm_ *= seedClus->energy()/superCluster_->energy();

}

void GsfElectron::correctElectronEnergyScale(const float newEnergy) {
  
  //   float newEnergy = thecorr->getCorrectedEnergy();
 
  //momentum_*=newEnergy/momentum_.e();
  math::XYZTLorentzVectorD momentum=p4();
  momentum*=newEnergy/momentum.e();
  setP4(momentum);
  hadOverEm_ *=superClusterEnergy_/newEnergy; 
  eSuperClusterOverP_*=newEnergy/superClusterEnergy_;
  superClusterEnergy_=newEnergy;
 
  energyScaleCorrected_=true;    
}
 
//void GsfElectron::correctElectronFourMomentum(const PElectronMomentumCorrector *thecorr) {
void GsfElectron::correctElectronFourMomentum(const math::XYZTLorentzVectorD & momentum,float & enErr, float & tmErr) {
 
  //   momentum_ = thecorr->getBestMomentum();
  setP4(momentum);
  //   energyError_ = thecorr->getSCEnergyError();
  energyError_ = enErr;
  //   trackMomentumError_ = thecorr->getTrackMomentumError();
  trackMomentumError_ = tmErr;

  momentumFromEpCombination_=true;
}
 
//void GsfElectron::classifyElectron(const PElectronClassification *theclassifier) {
//   electronClass_ = theclassifier->getClass();
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
