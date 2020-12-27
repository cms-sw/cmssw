#include "RecoEcal/EgammaClusterProducers/interface/EgammaSCCorrectionMaker.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

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

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"

#include <string>

EgammaSCCorrectionMaker::EgammaSCCorrectionMaker(const edm::ParameterSet& ps) {
  // the input producers
  rHTag_ = ps.getParameter<edm::InputTag>("recHitProducer");
  rHInputProducer_ = consumes<EcalRecHitCollection>(rHTag_);
  sCInputProducer_ = consumes<reco::SuperClusterCollection>(ps.getParameter<edm::InputTag>("rawSuperClusterProducer"));
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
    edm::LogError("EgammaSCCorrectionMakerError")
        << "Error! SuperClusterAlgo in config file must be Hybrid or Island: " << sCAlgo_str
        << "  Using Hybrid by default";
    sCAlgo_ = reco::CaloCluster::hybrid;
  }

  // set correction algo parameters
  applyEnergyCorrection_ = ps.getParameter<bool>("applyEnergyCorrection");
  applyCrackCorrection_ = ps.getParameter<bool>("applyCrackCorrection");
  applyLocalContCorrection_ =
      ps.existsAs<bool>("applyLocalContCorrection") ? ps.getParameter<bool>("applyLocalContCorrection") : false;

  energyCorrectorName_ = ps.getParameter<std::string>("energyCorrectorName");
  crackCorrectorName_ = ps.existsAs<std::string>("crackCorrectorName")
                            ? ps.getParameter<std::string>("crackCorrectorName")
                            : std::string("EcalClusterCrackCorrection");

  modeEB_ = ps.getParameter<int>("modeEB");
  modeEE_ = ps.getParameter<int>("modeEE");

  sigmaElectronicNoise_ = ps.getParameter<double>("sigmaElectronicNoise");

  etThresh_ = ps.getParameter<double>("etThresh");

  // set the producer parameters
  outputCollection_ = ps.getParameter<std::string>("corectedSuperClusterCollection");
  produces<reco::SuperClusterCollection>(outputCollection_);

  // instanciate the correction algo object
  energyCorrector_ = std::make_unique<EgammaSCEnergyCorrectionAlgo>(sigmaElectronicNoise_);

  // energy correction class
  if (applyEnergyCorrection_)
    energyCorrectionFunction_ = EcalClusterFunctionFactory::get()->create(energyCorrectorName_, ps);
  //energyCorrectionFunction_ = EcalClusterFunctionFactory::get()->create("EcalClusterEnergyCorrection", ps);

  if (applyCrackCorrection_)
    crackCorrectionFunction_ = EcalClusterFunctionFactory::get()->create(crackCorrectorName_, ps);

  if (applyLocalContCorrection_)
    localContCorrectionFunction_ = std::make_unique<EcalBasicClusterLocalContCorrection>(consumesCollector());
}

void EgammaSCCorrectionMaker::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;

  // initialize energy correction class
  if (applyEnergyCorrection_)
    energyCorrectionFunction_->init(es);

  // initialize energy correction class
  if (applyCrackCorrection_)
    crackCorrectionFunction_->init(es);

  // initialize containemnt correction class
  if (applyLocalContCorrection_)
    localContCorrectionFunction_->init(es);

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry* geometry_p;

  std::string rHInputCollection = rHTag_.instance();
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
  evt.getByToken(sCInputProducer_, pRawSuperClusters);

  // Get the RecHits from the event
  Handle<EcalRecHitCollection> pRecHits;
  evt.getByToken(rHInputProducer_, pRecHits);

  // Create a pointer to the RecHits and raw SuperClusters
  const EcalRecHitCollection* hitCollection = pRecHits.product();
  const reco::SuperClusterCollection* rawClusters = pRawSuperClusters.product();

  // Define a collection of corrected SuperClusters to put back into the event
  auto corrClusters = std::make_unique<reco::SuperClusterCollection>();

  //  Loop over raw clusters and make corrected ones
  reco::SuperClusterCollection::const_iterator aClus;
  int i = 0;
  for (aClus = rawClusters->begin(); aClus != rawClusters->end(); aClus++) {
    reco::SuperCluster enecorrClus, crackcorrClus, localContCorrClus;

    i++;

    if (applyEnergyCorrection_)
      enecorrClus = energyCorrector_->applyCorrection(*aClus,
                                                      *hitCollection,
                                                      sCAlgo_,
                                                      geometry_p,
                                                      energyCorrectionFunction_.get(),
                                                      energyCorrectorName_,
                                                      modeEB_,
                                                      modeEE_);
    else
      enecorrClus = *aClus;

    if (applyCrackCorrection_)
      crackcorrClus = EgammaSCEnergyCorrectionAlgo::applyCrackCorrection(enecorrClus, crackCorrectionFunction_.get());
    else
      crackcorrClus = enecorrClus;

    if (applyLocalContCorrection_)
      localContCorrClus =
          EgammaSCEnergyCorrectionAlgo::applyLocalContCorrection(crackcorrClus, *localContCorrectionFunction_);
    else
      localContCorrClus = crackcorrClus;

    if (localContCorrClus.energy() * sin(localContCorrClus.position().theta()) > etThresh_) {
      corrClusters->push_back(localContCorrClus);
    }
  }

  // Put collection of corrected SuperClusters into the event
  evt.put(std::move(corrClusters), outputCollection_);
}
