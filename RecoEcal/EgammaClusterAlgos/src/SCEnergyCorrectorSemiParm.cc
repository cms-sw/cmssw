#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorSemiParm.h"

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vdt/vdtMath.h>

using namespace reco;

//--------------------------------------------------------------------------------------------------
SCEnergyCorrectorSemiParm::SCEnergyCorrectorSemiParm()
    : foresteb_(nullptr),
      forestee_(nullptr),
      forestsigmaeb_(nullptr),
      forestsigmaee_(nullptr),
      calotopo_(nullptr),
      calogeom_(nullptr) {}

//--------------------------------------------------------------------------------------------------
SCEnergyCorrectorSemiParm::~SCEnergyCorrectorSemiParm() {}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::setTokens(const edm::ParameterSet &iConfig, edm::ConsumesCollector &cc) {
  isHLT_ = iConfig.getParameter<bool>("isHLT");
  applySigmaIetaIphiBug_ = iConfig.getParameter<bool>("applySigmaIetaIphiBug");
  tokenEBRecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEB"));
  tokenEERecHits_ = cc.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRecHitsEE"));

  regressionKeyEB_ = iConfig.getParameter<std::string>("regressionKeyEB");
  uncertaintyKeyEB_ = iConfig.getParameter<std::string>("uncertaintyKeyEB");
  regressionKeyEE_ = iConfig.getParameter<std::string>("regressionKeyEE");
  uncertaintyKeyEE_ = iConfig.getParameter<std::string>("uncertaintyKeyEE");

  if (not isHLT_) {
    tokenVertices_ = cc.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));
  } else {
    eThreshold_ = iConfig.getParameter<double>("eRecHitThreshold");
  }
}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::setEventSetup(const edm::EventSetup &es) {
  es.get<CaloTopologyRecord>().get(calotopo_);
  es.get<CaloGeometryRecord>().get(calogeom_);

  edm::ESHandle<GBRForestD> readereb;
  edm::ESHandle<GBRForestD> readerebvar;
  edm::ESHandle<GBRForestD> readeree;
  edm::ESHandle<GBRForestD> readereevar;

  es.get<GBRDWrapperRcd>().get(regressionKeyEB_, readereb);
  es.get<GBRDWrapperRcd>().get(uncertaintyKeyEB_, readerebvar);
  es.get<GBRDWrapperRcd>().get(regressionKeyEE_, readeree);
  es.get<GBRDWrapperRcd>().get(uncertaintyKeyEE_, readereevar);

  foresteb_ = readereb.product();
  forestsigmaeb_ = readerebvar.product();
  forestee_ = readeree.product();
  forestsigmaee_ = readereevar.product();
}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::setEvent(const edm::Event &e) {
  e.getByToken(tokenEBRecHits_, rechitsEB_);
  e.getByToken(tokenEERecHits_, rechitsEE_);

  if (not isHLT_)
    e.getByToken(tokenVertices_, vertices_);
  else {
    nHitsAboveThreshold_ = 0;
    const EcalRecHitCollection *recHitsEB = (rechitsEB_.isValid() ? rechitsEB_.product() : nullptr);
    const EcalRecHitCollection *recHitsEE = (rechitsEE_.isValid() ? rechitsEE_.product() : nullptr);

    if (nullptr != recHitsEB) {
      for (EcalRecHitCollection::const_iterator it = recHitsEB->begin(); it != recHitsEB->end(); ++it) {
        if (it->energy() > eThreshold_)
          nHitsAboveThreshold_++;
      }
    }

    if (nullptr != recHitsEE) {
      for (EcalRecHitCollection::const_iterator it = recHitsEE->begin(); it != recHitsEE->end(); ++it) {
        if (it->energy() > eThreshold_)
          nHitsAboveThreshold_++;
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
std::pair<double, double> SCEnergyCorrectorSemiParm::getCorrections(const reco::SuperCluster &sc) const {
  std::pair<double, double> p;
  p.first = -1;
  p.second = -1;

  // protect against HGCal, don't mod the object
  if (EcalTools::isHGCalDet(sc.seed()->seed().det()))
    return p;

  const reco::CaloCluster &seedCluster = *(sc.seed());
  const bool iseb = seedCluster.hitsAndFractions()[0].first.subdetId() == EcalBarrel;
  const EcalRecHitCollection *recHits = iseb ? rechitsEB_.product() : rechitsEE_.product();

  const CaloTopology *topo = calotopo_.product();

  const double raw_energy = sc.rawEnergy();
  const int numberOfClusters = sc.clusters().size();

  std::vector<float> localCovariances = EcalClusterTools::localCovariances(seedCluster, recHits, topo);

  if (not isHLT_) {
    std::array<float, 30> eval;

    const float eLeft = EcalClusterTools::eLeft(seedCluster, recHits, topo);
    const float eRight = EcalClusterTools::eRight(seedCluster, recHits, topo);
    const float eTop = EcalClusterTools::eTop(seedCluster, recHits, topo);
    const float eBottom = EcalClusterTools::eBottom(seedCluster, recHits, topo);

    float sigmaIetaIeta = sqrt(localCovariances[0]);
    float sigmaIetaIphi = std::numeric_limits<float>::max();
    float sigmaIphiIphi = std::numeric_limits<float>::max();

    if (!edm::isNotFinite(localCovariances[2]))
      sigmaIphiIphi = sqrt(localCovariances[2]);

    // extra shower shapes
    const float see_by_spp =
        sigmaIetaIeta * (applySigmaIetaIphiBug_ ? std::numeric_limits<float>::max() : sigmaIphiIphi);
    if (see_by_spp > 0) {
      sigmaIetaIphi = localCovariances[1] / see_by_spp;
    } else if (localCovariances[1] > 0) {
      sigmaIetaIphi = 1.f;
    } else {
      sigmaIetaIphi = -1.f;
    }

    // calculate sub-cluster variables
    std::vector<float> clusterRawEnergy;
    clusterRawEnergy.resize(std::max(3, numberOfClusters), 0);
    std::vector<float> clusterDEtaToSeed;
    clusterDEtaToSeed.resize(std::max(3, numberOfClusters), 0);
    std::vector<float> clusterDPhiToSeed;
    clusterDPhiToSeed.resize(std::max(3, numberOfClusters), 0);
    float clusterMaxDR = 999.;
    float clusterMaxDRDPhi = 999.;
    float clusterMaxDRDEta = 999.;
    float clusterMaxDRRawEnergy = 0.;

    size_t iclus = 0;
    float maxDR = 0;
    edm::Ptr<reco::CaloCluster> pclus;
    const edm::Ptr<reco::CaloCluster> &theseed = sc.seed();
    // loop over all clusters that aren't the seed
    auto clusend = sc.clustersEnd();
    for (auto clus = sc.clustersBegin(); clus != clusend; ++clus) {
      pclus = *clus;

      if (theseed == pclus)
        continue;
      clusterRawEnergy[iclus] = pclus->energy();
      clusterDPhiToSeed[iclus] = reco::deltaPhi(pclus->phi(), theseed->phi());
      clusterDEtaToSeed[iclus] = pclus->eta() - theseed->eta();

      // find cluster with max dR
      const auto the_dr = reco::deltaR(*pclus, *theseed);
      if (the_dr > maxDR) {
        maxDR = the_dr;
        clusterMaxDR = maxDR;
        clusterMaxDRDPhi = clusterDPhiToSeed[iclus];
        clusterMaxDRDEta = clusterDEtaToSeed[iclus];
        clusterMaxDRRawEnergy = clusterRawEnergy[iclus];
      }
      ++iclus;
    }

    // SET INPUTS
    eval[0] = vertices_->size();
    eval[1] = raw_energy;
    eval[2] = sc.etaWidth();
    eval[3] = sc.phiWidth();
    eval[4] = EcalClusterTools::e3x3(seedCluster, recHits, topo) / raw_energy;
    eval[5] = seedCluster.energy() / raw_energy;
    eval[6] = EcalClusterTools::eMax(seedCluster, recHits) / raw_energy;
    eval[7] = EcalClusterTools::e2nd(seedCluster, recHits) / raw_energy;
    eval[8] = (eLeft + eRight != 0.f ? (eLeft - eRight) / (eLeft + eRight) : 0.f);
    eval[9] = (eTop + eBottom != 0.f ? (eTop - eBottom) / (eTop + eBottom) : 0.f);
    eval[10] = sigmaIetaIeta;
    eval[11] = sigmaIetaIphi;
    eval[12] = sigmaIphiIphi;
    eval[13] = std::max(0, numberOfClusters - 1);
    eval[14] = clusterMaxDR;
    eval[15] = clusterMaxDRDPhi;
    eval[16] = clusterMaxDRDEta;
    eval[17] = clusterMaxDRRawEnergy / raw_energy;
    eval[18] = clusterRawEnergy[0] / raw_energy;
    eval[19] = clusterRawEnergy[1] / raw_energy;
    eval[20] = clusterRawEnergy[2] / raw_energy;
    eval[21] = clusterDPhiToSeed[0];
    eval[22] = clusterDPhiToSeed[1];
    eval[23] = clusterDPhiToSeed[2];
    eval[24] = clusterDEtaToSeed[0];
    eval[25] = clusterDEtaToSeed[1];
    eval[26] = clusterDEtaToSeed[2];
    if (iseb) {
      EBDetId ebseedid(seedCluster.seed());
      eval[27] = ebseedid.ieta();
      eval[28] = ebseedid.iphi();
    } else {
      EEDetId eeseedid(seedCluster.seed());
      eval[27] = eeseedid.ix();
      eval[28] = eeseedid.iy();
      //seed cluster eta is only needed for the 106X Ultra Legacy regressions
      //and was not used in the 74X regression however as its just an extra varaible
      //at the end, its harmless to add for the 74X regression
      eval[29] = seedCluster.eta();
    }

    //magic numbers for MINUIT-like transformation of BDT output onto limited range
    //(These should be stored inside the conditions object in the future as well)
    constexpr double meanlimlow = 0.2;
    constexpr double meanlimhigh = 2.0;
    constexpr double meanoffset = meanlimlow + 0.5 * (meanlimhigh - meanlimlow);
    constexpr double meanscale = 0.5 * (meanlimhigh - meanlimlow);

    constexpr double sigmalimlow = 0.0002;
    constexpr double sigmalimhigh = 0.5;
    constexpr double sigmaoffset = sigmalimlow + 0.5 * (sigmalimhigh - sigmalimlow);
    constexpr double sigmascale = 0.5 * (sigmalimhigh - sigmalimlow);

    const GBRForestD *forestmean = iseb ? foresteb_ : forestee_;
    const GBRForestD *forestsigma = iseb ? forestsigmaeb_ : forestsigmaee_;

    //these are the actual BDT responses
    double rawmean = forestmean->GetResponse(eval.data());
    double rawsigma = forestsigma->GetResponse(eval.data());

    //apply transformation to limited output range (matching the training)
    double mean = meanoffset + meanscale * vdt::fast_sin(rawmean);
    double sigma = sigmaoffset + sigmascale * vdt::fast_sin(rawsigma);

    double ecor = mean * (eval[1]);
    const double sigmacor = sigma * ecor;

    p.first = ecor;
    p.second = sigmacor;

  } else {
    std::array<float, 7> eval;
    float clusterMaxDR = 999.;

    size_t iclus = 0;
    float maxDR = 0;
    edm::Ptr<reco::CaloCluster> pclus;
    const edm::Ptr<reco::CaloCluster> &theseed = sc.seed();

    // loop over all clusters that aren't the seed
    auto clusend = sc.clustersEnd();
    for (auto clus = sc.clustersBegin(); clus != clusend; ++clus) {
      pclus = *clus;

      if (theseed == pclus)
        continue;

      // find cluster with max dR
      const auto the_dr = reco::deltaR(*pclus, *theseed);
      if (the_dr > maxDR) {
        maxDR = the_dr;
        clusterMaxDR = maxDR;
      }
      ++iclus;
    }

    // SET INPUTS
    eval[0] = nHitsAboveThreshold_;
    eval[1] = sc.eta();
    eval[2] = sc.phiWidth();
    eval[3] = EcalClusterTools::e3x3(seedCluster, recHits, topo) / raw_energy;
    eval[4] = std::max(0, numberOfClusters - 1);
    eval[5] = clusterMaxDR;
    eval[6] = raw_energy;

    //magic numbers for MINUIT-like transformation of BDT output onto limited range
    //(These should be stored inside the conditions object in the future as well)
    constexpr double meanlimlow = 0.2;
    constexpr double meanlimhigh = 2.0;
    constexpr double meanoffset = meanlimlow + 0.5 * (meanlimhigh - meanlimlow);
    constexpr double meanscale = 0.5 * (meanlimhigh - meanlimlow);

    const GBRForestD *forestmean = iseb ? foresteb_ : forestee_;

    double rawmean = forestmean->GetResponse(eval.data());
    double mean = meanoffset + meanscale * vdt::fast_sin(rawmean);

    double ecor = mean * eval[6];
    if (!iseb)
      ecor = mean * eval[6] + sc.preshowerEnergy();

    p.first = ecor;
    //p.second unchanged
  }

  return p;
}

//--------------------------------------------------------------------------------------------------
void SCEnergyCorrectorSemiParm::modifyObject(reco::SuperCluster &sc) {
  std::pair<double, double> cor = getCorrections(sc);
  if (cor.first < 0)
    return;
  sc.setEnergy(cor.first);
  sc.setCorrectedEnergy(cor.first);
  if (!isHLT_ && cor.second >= 0.)
    sc.setCorrectedEnergyUncertainty(cor.second);
}
