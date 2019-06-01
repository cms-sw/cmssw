/**

Description: Producer for EcalRecHits to be used for pi0/eta ECAL calibration. 


 Implementation:
     <Notes on implementation>
*/
//
// Original Authors:  Vladimir Litvine , Yong Yang

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

// ES stuff
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"

//
//Ecal status
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

// class declaration
//
#include <algorithm>
#include <utility>

#include "TLorentzVector.h"
#include <vector>

namespace edm {
  class ConfigurationDescriptions;
}

class HLTEcalResonanceFilter : public edm::EDFilter {
public:
  explicit HLTEcalResonanceFilter(const edm::ParameterSet &);
  ~HLTEcalResonanceFilter() override;

  bool filter(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  // ----------member data ---------------------------
  void doSelection(int detector,
                   const reco::BasicClusterCollection *clusterCollection,
                   const EcalRecHitCollection *hitCollection,
                   const EcalChannelStatus &channelStatus,
                   const CaloSubdetectorTopology *topology_p,
                   std::map<int, std::vector<EcalRecHit> > &RecHits5x5_clus,
                   std::vector<int> &indCandClus,     ///good cluster all ,  5x5 rechit done already during the loop
                   std::vector<int> &indIsoClus,      /// Iso cluster all , 5x5 rechit not yet done
                   std::vector<int> &indClusSelected  /// saved so far, all
  );

  void makeClusterES(
      float x, float y, float z, const CaloSubdetectorGeometry *iSubGeom, const CaloSubdetectorTopology *topology_p);

  void calcPaircluster(const reco::BasicCluster &bc1,
                       const reco::BasicCluster &bc2,
                       float &mpair,
                       float &ptpair,
                       float &etapair,
                       float &phipair);

  bool checkStatusOfEcalRecHit(const EcalChannelStatus &channelStatus, const EcalRecHit &rh);

  void calcShowerShape(const reco::BasicCluster &bc,
                       const EcalChannelStatus &channelStatus,
                       const EcalRecHitCollection *recHits,
                       const CaloSubdetectorTopology *topology_p,
                       bool calc5x5,
                       std::vector<EcalRecHit> &rechit5x5,
                       float res[]);

  void convxtalid(int &, int &);
  int diff_neta_s(int, int);
  int diff_nphi_s(int, int);

  static float DeltaPhi(float phi1, float phi2);
  static float GetDeltaR(float eta1, float eta2, float phi1, float phi2);

  // Input hits & clusters
  edm::InputTag barrelHits_;
  edm::InputTag endcapHits_;
  edm::InputTag barrelClusters_;
  edm::InputTag endcapClusters_;

  edm::EDGetTokenT<EBRecHitCollection> barrelHitsToken_;
  edm::EDGetTokenT<EERecHitCollection> endcapHitsToken_;
  edm::EDGetTokenT<ESRecHitCollection> preshHitsToken_;
  edm::EDGetTokenT<reco::BasicClusterCollection> barrelClustersToken_;
  edm::EDGetTokenT<reco::BasicClusterCollection> endcapClustersToken_;

  ///output hits
  std::string BarrelHits_;
  std::string EndcapHits_;
  std::string ESHits_;

  ///for pi0/eta ->gg barrel
  bool doSelBarrel_;
  double selePtGamma_;
  double selePtPair_;
  double seleMinvMaxBarrel_;
  double seleMinvMinBarrel_;
  double seleS4S9Gamma_;
  double seleS9S25Gamma_;
  double seleBeltDR_;
  double seleBeltDeta_;
  double seleIso_;
  double ptMinForIsolation_;
  bool removePi0CandidatesForEta_;
  double massLowPi0Cand_;
  double massHighPi0Cand_;
  bool store5x5RecHitEB_;

  //for pi0/eta ->gg endcap
  bool doSelEndcap_;
  double region1_EndCap_;
  double selePtGammaEndCap_region1_;
  double selePtPairEndCap_region1_;
  double region2_EndCap_;
  double selePtGammaEndCap_region2_;
  double selePtPairEndCap_region2_;
  double selePtGammaEndCap_region3_;
  double selePtPairEndCap_region3_;
  double selePtPairMaxEndCap_region3_;
  double seleMinvMaxEndCap_;
  double seleMinvMinEndCap_;
  double seleS4S9GammaEndCap_;
  double seleS9S25GammaEndCap_;
  double seleIsoEndCap_;
  double seleBeltDREndCap_;
  double seleBeltDetaEndCap_;
  double ptMinForIsolationEndCap_;
  bool store5x5RecHitEE_;

  bool useRecoFlag_;
  bool useDBStatus_;
  int flagLevelRecHitsToUse_;
  int statusLevelRecHitsToUse_;

  bool storeRecHitES_;
  edm::InputTag preshHitProducer_;  // name of module/plugin/producer producing hits
  int preshNclust_;
  float preshClustECut;
  double etThresh_;
  double calib_planeX_;
  double calib_planeY_;
  double mip_;
  double gamma_;

  PreshowerClusterAlgo *presh_algo_;  // algorithm doing the real work

  std::map<DetId, EcalRecHit> m_esrechit_map;
  std::set<DetId> m_used_strips;

  int debug_;
};
