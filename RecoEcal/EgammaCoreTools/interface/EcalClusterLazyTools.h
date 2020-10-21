#ifndef RecoEcal_EgammaCoreTools_EcalClusterLazyTools_h
#define RecoEcal_EgammaCoreTools_EcalClusterLazyTools_h

/** \class EcalClusterLazyTools
 *
 * various cluster tools (e.g. cluster shapes)
 *
 * \author Federico Ferri
 *
 * \version $Id:
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <optional>

class CaloTopology;
class CaloGeometry;
class CaloSubdetectorTopology;

class EcalClusterLazyToolsBase {
public:
  struct ESData {
    CaloGeometry const &caloGeometry;
    CaloTopology const &caloTopology;
    EcalIntercalibConstants const &ecalIntercalibConstants;
    EcalADCToGeVConstant const &ecalADCToGeV;
    EcalLaserDbService const &ecalLaserDbService;
  };

  class ESGetTokens {
  public:
    ESGetTokens(edm::ConsumesCollector &&cc)
        : caloGeometryToken_{cc.esConsumes()},
          caloTopologyToken_{cc.esConsumes()},
          ecalIntercalibConstantsToken_{cc.esConsumes()},
          ecalADCToGeVConstantToken_{cc.esConsumes()},
          ecalLaserDbServiceToken_{cc.esConsumes()} {}

    ESData get(edm::EventSetup const &eventSetup) const {
      return {.caloGeometry = eventSetup.getData(caloGeometryToken_),
              .caloTopology = eventSetup.getData(caloTopologyToken_),
              .ecalIntercalibConstants = eventSetup.getData(ecalIntercalibConstantsToken_),
              .ecalADCToGeV = eventSetup.getData(ecalADCToGeVConstantToken_),
              .ecalLaserDbService = eventSetup.getData(ecalLaserDbServiceToken_)};
    }

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
    edm::ESGetToken<CaloTopology, CaloTopologyRecord> caloTopologyToken_;
    edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> ecalIntercalibConstantsToken_;
    edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> ecalADCToGeVConstantToken_;
    edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> ecalLaserDbServiceToken_;
  };

  EcalClusterLazyToolsBase(const edm::Event &ev,
                           ESData const &esData,
                           edm::EDGetTokenT<EcalRecHitCollection> token1,
                           edm::EDGetTokenT<EcalRecHitCollection> token2,
                           std::optional<edm::EDGetTokenT<EcalRecHitCollection>> token3);

  // get time of basic cluster seed crystal
  float BasicClusterSeedTime(const reco::BasicCluster &cluster);
  // error-weighted average of time from constituents of basic cluster
  float BasicClusterTime(const reco::BasicCluster &cluster, const edm::Event &ev);
  // get BasicClusterSeedTime of the seed basic cluser of the supercluster
  float SuperClusterSeedTime(const reco::SuperCluster &cluster);
  // get BasicClusterTime of the seed basic cluser of the supercluster
  inline float SuperClusterTime(const reco::SuperCluster &cluster, const edm::Event &ev) {
    return BasicClusterTime((*cluster.seed()), ev);
  }

  // mapping for preshower rechits
  std::map<DetId, EcalRecHit> rechits_map_;
  // get Preshower hit array
  std::vector<float> getESHits(double X,
                               double Y,
                               double Z,
                               const std::map<DetId, EcalRecHit> &rechits_map,
                               const CaloGeometry *geometry,
                               CaloSubdetectorTopology const *topology_p,
                               int row = 0,
                               int plane = 1);
  // get Preshower hit shape
  float getESShape(const std::vector<float> &ESHits0);
  // get Preshower effective sigmaRR
  float eseffsirir(const reco::SuperCluster &cluster);
  float eseffsixix(const reco::SuperCluster &cluster);
  float eseffsiyiy(const reco::SuperCluster &cluster);

protected:
  EcalRecHitCollection const *getEcalRecHitCollection(const reco::BasicCluster &cluster) const;

  const CaloGeometry *geometry_;
  const CaloTopology *topology_;
  const EcalRecHitCollection *ebRecHits_;
  const EcalRecHitCollection *eeRecHits_;
  const EcalRecHitCollection *esRecHits_;

  std::unique_ptr<CaloSubdetectorTopology const> ecalPS_topology_ = nullptr;

  EcalIntercalibConstants const *ical = nullptr;
  EcalIntercalibConstantMap const *icalMap = nullptr;
  EcalADCToGeVConstant const *agc = nullptr;
  EcalLaserDbService const *laser = nullptr;

private:
  void getESRecHits(const edm::Event &ev, edm::EDGetTokenT<EcalRecHitCollection> const &esRecHitsToken);

};  // class EcalClusterLazyToolsBase

template <class ClusterTools>
class EcalClusterLazyToolsT : public EcalClusterLazyToolsBase {
public:
  EcalClusterLazyToolsT(const edm::Event &ev,
                        ESData const &esData,
                        edm::EDGetTokenT<EcalRecHitCollection> token1,
                        edm::EDGetTokenT<EcalRecHitCollection> token2)
      : EcalClusterLazyToolsBase(ev, esData, token1, token2, std::nullopt) {}

  EcalClusterLazyToolsT(const edm::Event &ev,
                        ESData const &esData,
                        edm::EDGetTokenT<EcalRecHitCollection> token1,
                        edm::EDGetTokenT<EcalRecHitCollection> token2,
                        edm::EDGetTokenT<EcalRecHitCollection> token3)
      : EcalClusterLazyToolsBase(ev, esData, token1, token2, token3) {}

  // Get the rec hit energies in a rectangle matrix around the seed.
  std::vector<float> energyMatrix(reco::BasicCluster const &cluster, int size) const {
    auto recHits = getEcalRecHitCollection(cluster);
    DetId maxId = ClusterTools::getMaximum(cluster, recHits).first;

    std::vector<float> energies;
    for (auto const &detId : CaloRectangleRange(size, maxId, *topology_)) {
      energies.push_back(ClusterTools::recHitEnergy(detId, recHits));
    }

    return energies;
  }

  // various energies in the matrix nxn surrounding the maximum energy crystal of the input cluster
  //
  // NOTE (29/10/08): we now use an eta/phi coordinate system rather than
  //                  phi/eta to minmise possible screwups, for now e5x1 isnt
  //                  defined all the majority of people who call it actually
  //                  want e1x5 and it is thought it is better that their code
  //                  doesnt compile rather than pick up the wrong function
  //                  therefore in this version and later e1x5 = e5x1 in the old
  //                  version so 1x5 is 1 crystal in eta and 5 crystals in phi
  //                  note e3x2 does not have a definate eta/phi geometry, it
  //                  takes the maximum 3x2 block containing the seed regardless
  //                  of whether that 3 in eta or phi
  float e1x3(const reco::BasicCluster &cluster) const {
    return ClusterTools::e1x3(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e3x1(const reco::BasicCluster &cluster) const {
    return ClusterTools::e3x1(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e1x5(const reco::BasicCluster &cluster) const {
    return ClusterTools::e1x5(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e5x1(const reco::BasicCluster &cluster) const {
    return ClusterTools::e5x1(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e2x2(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2x2(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e3x2(const reco::BasicCluster &cluster) const {
    return ClusterTools::e3x2(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e3x3(const reco::BasicCluster &cluster) const {
    return ClusterTools::e3x3(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e4x4(const reco::BasicCluster &cluster) const {
    return ClusterTools::e4x4(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float e5x5(const reco::BasicCluster &cluster) const {
    return ClusterTools::e5x5(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  int n5x5(const reco::BasicCluster &cluster) const {
    return ClusterTools::n5x5(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  // energy in the 2x5 strip right of the max crystal (does not contain max crystal)
  // 2 crystals wide in eta, 5 wide in phi.
  float e2x5Right(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2x5Right(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  // energy in the 2x5 strip left of the max crystal (does not contain max crystal)
  float e2x5Left(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2x5Left(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  // energy in the 5x2 strip above the max crystal (does not contain max crystal)
  // 5 crystals wide in eta, 2 wide in phi.
  float e2x5Top(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2x5Top(cluster, getEcalRecHitCollection(cluster), topology_);
  }

  // energy in the 5x2 strip below the max crystal (does not contain max crystal)
  float e2x5Bottom(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2x5Bottom(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  // energy in a 2x5 strip containing the seed (max) crystal.
  // 2 crystals wide in eta, 5 wide in phi.
  // it is the maximum of either (1x5left + 1x5center) or (1x5right + 1x5center)
  float e2x5Max(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2x5Max(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  // energies in the crystal left, right, top, bottom w.r.t. to the most energetic crystal
  float eLeft(const reco::BasicCluster &cluster) const {
    return ClusterTools::eLeft(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float eRight(const reco::BasicCluster &cluster) const {
    return ClusterTools::eRight(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float eTop(const reco::BasicCluster &cluster) const {
    return ClusterTools::eTop(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  float eBottom(const reco::BasicCluster &cluster) const {
    return ClusterTools::eBottom(cluster, getEcalRecHitCollection(cluster), topology_);
  }
  // the energy of the most energetic crystal in the cluster
  float eMax(const reco::BasicCluster &cluster) const {
    return ClusterTools::eMax(cluster, getEcalRecHitCollection(cluster));
  }
  // the energy of the second most energetic crystal in the cluster
  float e2nd(const reco::BasicCluster &cluster) const {
    return ClusterTools::e2nd(cluster, getEcalRecHitCollection(cluster));
  }

  // get the DetId and the energy of the maximum energy crystal of the input cluster
  std::pair<DetId, float> getMaximum(const reco::BasicCluster &cluster) const {
    return ClusterTools::getMaximum(cluster, getEcalRecHitCollection(cluster));
  }
  std::vector<float> energyBasketFractionEta(const reco::BasicCluster &cluster) const {
    return ClusterTools::energyBasketFractionEta(cluster, getEcalRecHitCollection(cluster));
  }
  std::vector<float> energyBasketFractionPhi(const reco::BasicCluster &cluster) const {
    return ClusterTools::energyBasketFractionPhi(cluster, getEcalRecHitCollection(cluster));
  }

  // return a vector v with v[0] = etaLat, v[1] = phiLat, v[2] = lat
  std::vector<float> lat(const reco::BasicCluster &cluster, bool logW = true, float w0 = 4.7) const {
    return ClusterTools::lat(cluster, getEcalRecHitCollection(cluster), geometry_, logW, w0);
  }

  // return a vector v with v[0] = covEtaEta, v[1] = covEtaPhi, v[2] = covPhiPhi
  std::vector<float> covariances(const reco::BasicCluster &cluster, float w0 = 4.7) const {
    return ClusterTools::covariances(cluster, getEcalRecHitCollection(cluster), topology_, geometry_, w0);
  }

  // return a vector v with v[0] = covIEtaIEta, v[1] = covIEtaIPhi, v[2] = covIPhiIPhi
  // this function calculates differences in eta/phi in units of crystals not
  // global eta/phi this is gives better performance in the crack regions of
  // the calorimeter but gives otherwise identical results to covariances
  // function this is only defined for the barrel, it returns covariances when
  // the cluster is in the endcap Warning: covIEtaIEta has been studied by
  // egamma, but so far covIPhiIPhi hasnt been studied extensively so there
  // could be a bug in the covIPhiIEta or covIPhiIPhi calculations. I dont
  // think there is but as it hasnt been heavily used, there might be one
  std::vector<float> localCovariances(const reco::BasicCluster &cluster, float w0 = 4.7) const {
    return ClusterTools::localCovariances(cluster, getEcalRecHitCollection(cluster), topology_, w0);
  }
  std::vector<float> scLocalCovariances(const reco::SuperCluster &cluster, float w0 = 4.7) const {
    return ClusterTools::scLocalCovariances(cluster, getEcalRecHitCollection(cluster), topology_, w0);
  }
  double zernike20(const reco::BasicCluster &cluster, double R0 = 6.6, bool logW = true, float w0 = 4.7) const {
    return ClusterTools::zernike20(cluster, getEcalRecHitCollection(cluster), geometry_, R0, logW, w0);
  }
  double zernike42(const reco::BasicCluster &cluster, double R0 = 6.6, bool logW = true, float w0 = 4.7) const {
    return ClusterTools::zernike42(cluster, getEcalRecHitCollection(cluster), geometry_, R0, logW, w0);
  }

};  // class EcalClusterLazyToolsT

namespace noZS {
  typedef EcalClusterLazyToolsT<noZS::EcalClusterTools> EcalClusterLazyTools;
}
typedef EcalClusterLazyToolsT<::EcalClusterTools> EcalClusterLazyTools;

#endif
