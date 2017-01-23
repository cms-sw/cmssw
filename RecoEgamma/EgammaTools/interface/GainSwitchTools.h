#ifndef RecoEgamma_EGammaTools_GainSwitchTools_h
#define RecoEgamma_EGammaTools_GainSwitchTools_h

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <vector>

class DetId;
namespace reco{
  class SuperCluster;
}
class CaloTopology;
class CaloGeometry;

class GainSwitchTools {

public:

  enum class ShowerShapeType{
    Full5x5=0,Fractions //Full 5x5 would be better known as "NoFractions", its not a full 5x5, fractions is the standard showershape
  };

  //this should really live in EcalClusterTools
  static int nrCrysWithFlagsIn5x5(const DetId& id,const std::vector<int>& flags,const EcalRecHitCollection* recHits,const CaloTopology *topology);
  
  //note, right now the weights are showing the GS flags so the collections here have to be pure multifit
  static bool hasEBGainSwitch(const reco::SuperCluster& superClus,const EcalRecHitCollection* recHits);
  static bool hasEBGainSwitchIn5x5(const reco::SuperCluster& superClus,const EcalRecHitCollection* recHits,const CaloTopology *topology);
  static bool hasEBGainSwitch(const EcalRecHitCollection* recHits);
  
  static const std::vector<int> gainSwitchFlags(){return gainSwitchFlags_;}
  static float newRawEnergyNoFracs(const reco::SuperCluster& superClus,const std::vector<DetId> gainSwitchedHitIds,const EcalRecHitCollection* oldRecHits,const EcalRecHitCollection* newRecHits);

  //needs to be multifit rec-hits currently as weights dont have gs flags set
  static std::vector<DetId> gainSwitchedIdsIn5x5(const DetId& id,const EcalRecHitCollection* recHits,const CaloTopology* topology);
  
  
  static reco::SuperClusterRef matchSCBySeedCrys(const reco::SuperCluster& sc,edm::Handle<reco::SuperClusterCollection> scColl );
  static reco::SuperClusterRef matchSCBySeedCrys(const reco::SuperCluster& sc,edm::Handle<reco::SuperClusterCollection> scColl,int maxDEta,int maxDPhi);

  template<bool noZS>
  static reco::GsfElectron::ShowerShape 
  redoEcalShowerShape(reco::GsfElectron::ShowerShape showerShape,const reco::SuperClusterRef& superClus, const EcalRecHitCollection* recHits,const CaloTopology* topology,const CaloGeometry* geometry);
  template<bool noZS>
  static reco::Photon::ShowerShape 
  redoEcalShowerShape(reco::Photon::ShowerShape showerShape,const reco::SuperClusterRef& superClus, const EcalRecHitCollection* recHits,const CaloTopology* topology,const CaloGeometry* geometry);

  
  //so the no fractions showershape for electrons had hcalDepth1/2 corrected by the regression energy, hence we need to know the type
  static void 
  correctHadem(reco::GsfElectron::ShowerShape& showerShape,float eNewOverEOld,
	       const GainSwitchTools::ShowerShapeType ssType);
  
  static void
  correctHadem(reco::Photon::ShowerShape& showerShape,float eNewOverEOld);
  

private:
  static int calDIEta(int lhs,int rhs);
  static int calDIPhi(int lhs,int rhs);
private:
  static const std::vector<int> gainSwitchFlags_;
 
  
};



template<bool noZS>
reco::GsfElectron::ShowerShape 
GainSwitchTools::redoEcalShowerShape(reco::GsfElectron::ShowerShape showerShape,const reco::SuperClusterRef& superClus, const EcalRecHitCollection* recHits,const CaloTopology* topology,const CaloGeometry* geometry)
{
  const reco::CaloCluster & seedClus = *(superClus->seed());
  
  std::vector<float> covariances = EcalClusterToolsT<noZS>::covariances(seedClus,recHits,topology,geometry);
  std::vector<float> localCovariances = EcalClusterToolsT<noZS>::localCovariances(seedClus,recHits,topology);
  showerShape.sigmaEtaEta = sqrt(covariances[0]);
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]);
  showerShape.sigmaIphiIphi =!edm::isNotFinite(localCovariances[2]) ? sqrt(localCovariances[2]) : 0;
  showerShape.e1x5 = EcalClusterToolsT<noZS>::e1x5(seedClus,recHits,topology);
  showerShape.e2x5Max = EcalClusterToolsT<noZS>::e2x5Max(seedClus,recHits,topology);
  showerShape.e5x5 = EcalClusterToolsT<noZS>::e5x5(seedClus,recHits,topology);
  showerShape.r9 = EcalClusterToolsT<noZS>::e3x3(seedClus,recHits,topology)/superClus->rawEnergy();  
  const float see_by_spp = showerShape.sigmaIetaIeta*showerShape.sigmaIphiIphi;
  if(  see_by_spp > 0 ) {
    showerShape.sigmaIetaIphi = localCovariances[1] / see_by_spp;
  } else if ( localCovariances[1] > 0 ) {
    showerShape.sigmaIetaIphi = 1.f;
  } else {
    showerShape.sigmaIetaIphi = -1.f;
  }
  showerShape.eMax          = EcalClusterTools::eMax(seedClus,recHits);
  showerShape.e2nd          = EcalClusterTools::e2nd(seedClus,recHits);
  showerShape.eTop          = EcalClusterTools::eTop(seedClus,recHits,topology);
  showerShape.eLeft         = EcalClusterTools::eLeft(seedClus,recHits,topology);
  showerShape.eRight        = EcalClusterTools::eRight(seedClus,recHits,topology);
  showerShape.eBottom       = EcalClusterTools::eBottom(seedClus,recHits,topology);
  return showerShape;
}
  
template<bool noZS>
reco::Photon::ShowerShape 
GainSwitchTools::redoEcalShowerShape(reco::Photon::ShowerShape showerShape,const reco::SuperClusterRef& superClus, const EcalRecHitCollection* recHits,const CaloTopology* topology,const CaloGeometry* geometry)
{
  const reco::CaloCluster & seedClus = *(superClus->seed());
  
  std::vector<float> covariances = EcalClusterToolsT<noZS>::covariances(seedClus,recHits,topology,geometry);
  std::vector<float> localCovariances = EcalClusterToolsT<noZS>::localCovariances(seedClus,recHits,topology);
  showerShape.sigmaEtaEta = sqrt(covariances[0]);
  showerShape.e1x5 = EcalClusterToolsT<noZS>::e1x5(seedClus,recHits,topology); 
  showerShape.e2x5 = EcalClusterToolsT<noZS>::e2x5Max(seedClus,recHits,topology);
  showerShape.e3x3 = EcalClusterToolsT<noZS>::e3x3(seedClus,recHits,topology);
  showerShape.e5x5 = EcalClusterToolsT<noZS>::e5x5(seedClus,recHits,topology);
  showerShape.maxEnergyXtal =  EcalClusterToolsT<noZS>::eMax(seedClus,recHits);
  //showerShape.effSigmaRR fine as its preshower, this only does ECAL shapes
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]);
  showerShape.sigmaIphiIphi =!edm::isNotFinite(localCovariances[2]) ? sqrt(localCovariances[2]) : 0;
  showerShape.e2nd =  EcalClusterToolsT<noZS>::e2nd(seedClus,recHits);
  showerShape.eTop =  EcalClusterToolsT<noZS>::eTop(seedClus,recHits,topology);
  showerShape.eLeft =  EcalClusterToolsT<noZS>::eLeft(seedClus,recHits,topology);
  showerShape.eRight =  EcalClusterToolsT<noZS>::eRight(seedClus,recHits,topology);
  showerShape.eBottom =  EcalClusterToolsT<noZS>::eBottom(seedClus,recHits,topology);
  showerShape.e1x3 = EcalClusterToolsT<noZS>::e1x3(seedClus,recHits,topology);
  showerShape.e2x2 = EcalClusterToolsT<noZS>::e2x2(seedClus,recHits,topology);
  showerShape.e2x5Max = EcalClusterToolsT<noZS>::e2x5Max(seedClus,recHits,topology);
  showerShape.e2x5Left = EcalClusterToolsT<noZS>::e2x5Left(seedClus,recHits,topology);
  showerShape.e2x5Right = EcalClusterToolsT<noZS>::e2x5Right(seedClus,recHits,topology);
  showerShape.e2x5Top = EcalClusterToolsT<noZS>::e2x5Top(seedClus,recHits,topology);
  showerShape.e2x5Bottom = EcalClusterToolsT<noZS>::e2x5Bottom(seedClus,recHits,topology);

  return showerShape;
}



#endif
