#include "RecoHI/HiEgammaAlgos/plugins/HiEgammaSCCorrectionMaker.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include <string>

HiEgammaSCCorrectionMaker::HiEgammaSCCorrectionMaker(const edm::ParameterSet& ps) {
  // The verbosity level
  std::string debugString = ps.getParameter<std::string>("VerbosityLevel");
  if (debugString == "DEBUG")
    verbosity_ = HiEgammaSCEnergyCorrectionAlgo::pDEBUG;
  else if (debugString == "INFO")
    verbosity_ = HiEgammaSCEnergyCorrectionAlgo::pINFO;
  else
    verbosity_ = HiEgammaSCEnergyCorrectionAlgo::pERROR;

  // the input producers
  rHInputProducerTag_ = ps.getParameter<edm::InputTag>("recHitProducer");
  sCInputProducerTag_ = ps.getParameter<edm::InputTag>("rawSuperClusterProducer");
  rHInputProducer_ = consumes<EcalRecHitCollection>(rHInputProducerTag_);
  sCInputProducer_ = consumes<reco::SuperClusterCollection>(sCInputProducerTag_);
  std::string sCAlgo_str = ps.getParameter<std::string>("superClusterAlgo");

  // determine which BasicCluster algo we are correcting for
  //And obtain forrection parameters form cfg file
  edm::ParameterSet fCorrPset;
  if (sCAlgo_str == "Hybrid") {
    sCAlgo_ = reco::CaloCluster::hybrid;
    fCorrPset = ps.getParameter<edm::ParameterSet>("hyb_fCorrPset");
  } else if (sCAlgo_str == "Island") {
    sCAlgo_ = reco::CaloCluster::island;
    fCorrPset = ps.getParameter<edm::ParameterSet>("isl_fCorrPset");
  } else if (sCAlgo_str == "DynamicHybrid") {
    sCAlgo_ = reco::CaloCluster::dynamicHybrid;
    fCorrPset = ps.getParameter<edm::ParameterSet>("dyn_fCorrPset");
  } else if (sCAlgo_str == "Multi5x5") {
    sCAlgo_ = reco::CaloCluster::multi5x5;
    fCorrPset = ps.getParameter<edm::ParameterSet>("fix_fCorrPset");
  } else {
    edm::LogError("HiEgammaSCCorrectionMakerError")
        << "Error! SuperClusterAlgo in config file must be Hybrid or Island: " << sCAlgo_str
        << "  Using Hybrid by default";
    sCAlgo_ = reco::CaloCluster::hybrid;
  }

  // set correction algo parameters
  applyEnergyCorrection_ = ps.getParameter<bool>("applyEnergyCorrection");
  sigmaElectronicNoise_ = ps.getParameter<double>("sigmaElectronicNoise");

  etThresh_ = ps.getParameter<double>("etThresh");

  // set the producer parameters
  outputCollection_ = ps.getParameter<std::string>("corectedSuperClusterCollection");
  produces<reco::SuperClusterCollection>(outputCollection_);

  // instanciate the correction algo object
  energyCorrector_ = std::make_unique<HiEgammaSCEnergyCorrectionAlgo>(sigmaElectronicNoise_, fCorrPset, verbosity_);
}

HiEgammaSCCorrectionMaker::~HiEgammaSCCorrectionMaker() = default;

void HiEgammaSCCorrectionMaker::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry* geometry_p;

  edm::ESHandle<CaloTopology> pTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopo_);
  const CaloTopology& topology = *theCaloTopo_;

  std::string rHInputCollection = rHInputProducerTag_.instance();
  if (rHInputCollection == "EcalRecHitsEB") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  } else if (rHInputCollection == "EcalRecHitsEE") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  } else if (rHInputCollection == "EcalRecHitsPS") {
    geometry_p = geometry.getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  } else {
    std::string str =
        "\n\nSCCorrectionMaker encountered invalied ecalhitcollection type: " + rHInputCollection + ".\n\n";
    throw(std::runtime_error(str.c_str()));
  }

  // Get raw SuperClusters from the event
  Handle<reco::SuperClusterCollection> pRawSuperClusters;
  try {
    evt.getByToken(sCInputProducer_, pRawSuperClusters);
  } catch (cms::Exception& ex) {
    edm::LogError("HiEgammaSCCorrectionMakerError")
        << "Error! can't get the rawSuperClusters " << sCInputProducerTag_.label();
  }

  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHits;
  try {
    evt.getByToken(rHInputProducer_, pRecHits);
  } catch (cms::Exception& ex) {
    edm::LogError("HiEgammaSCCorrectionMakerError") << "Error! can't get the RecHits " << rHInputProducerTag_.label();
  }

  // Create a pointer to the RecHits and raw SuperClusters
  const EcalRecHitCollection* hitCollection = pRecHits.product();
  const reco::SuperClusterCollection* rawClusters = pRawSuperClusters.product();

  // Define a collection of corrected SuperClusters to put back into the event
  auto corrClusters = std::make_unique<reco::SuperClusterCollection>();

  //  Loop over raw clusters and make corrected ones
  reco::SuperClusterCollection::const_iterator aClus;
  for (aClus = rawClusters->begin(); aClus != rawClusters->end(); aClus++) {
    reco::SuperCluster newClus;
    if (applyEnergyCorrection_)
      newClus = energyCorrector_->applyCorrection(*aClus, *hitCollection, sCAlgo_, *geometry_p, topology);
    else
      newClus = *aClus;

    if (newClus.energy() * sin(newClus.position().theta()) > etThresh_) {
      corrClusters->push_back(newClus);
    }
  }
  // Put collection of corrected SuperClusters into the event
  evt.put(std::move(corrClusters), outputCollection_);
}

DEFINE_FWK_MODULE(HiEgammaSCCorrectionMaker);
