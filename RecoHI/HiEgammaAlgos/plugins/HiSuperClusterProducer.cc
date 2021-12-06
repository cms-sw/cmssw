#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoHI/HiEgammaAlgos/interface/HiBremRecoveryClusterAlgo.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

class HiSuperClusterProducer : public edm::stream::EDProducer<> {
public:
  HiSuperClusterProducer(const edm::ParameterSet& ps);

  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

private:
  const edm::EDPutTokenT<reco::SuperClusterCollection> endcapSuperclusterPut_;
  const edm::EDPutTokenT<reco::SuperClusterCollection> barrelSuperclusterPut_;

  const edm::EDGetTokenT<reco::BasicClusterCollection> eeClustersToken_;
  const edm::EDGetTokenT<reco::BasicClusterCollection> ebClustersToken_;

  const float barrelEtaSearchRoad_;
  const float barrelPhiSearchRoad_;
  const float endcapEtaSearchRoad_;
  const float endcapPhiSearchRoad_;
  const float seedTransverseEnergyThreshold_;
  const float barrelBCEnergyThreshold_;
  const float endcapBCEnergyThreshold_;

  const bool doBarrel_;
  const bool doEndcaps_;

  std::unique_ptr<HiBremRecoveryClusterAlgo> bremAlgo_p;

  double totalE = 0.0;
  int noSuperClusters = 0;

  void getClusterPtrVector(edm::Event& evt,
                           const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
                           reco::CaloClusterPtrVector*);

  void produceSuperclustersForECALPart(edm::Event& evt,
                                       const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
                                       edm::EDPutTokenT<reco::SuperClusterCollection> const& putToken);

  void outputValidationInfo(reco::SuperClusterCollection& superclusterCollection);
};

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HiSuperClusterProducer);

HiSuperClusterProducer::HiSuperClusterProducer(const edm::ParameterSet& ps)
    : endcapSuperclusterPut_{produces<reco::SuperClusterCollection>(
          ps.getParameter<std::string>("endcapSuperclusterCollection"))},
      barrelSuperclusterPut_{
          produces<reco::SuperClusterCollection>(ps.getParameter<std::string>("barrelSuperclusterCollection"))},
      eeClustersToken_{consumes<reco::BasicClusterCollection>(
          edm::InputTag(ps.getParameter<std::string>("endcapClusterProducer"),
                        ps.getParameter<std::string>("endcapClusterCollection")))},
      ebClustersToken_{consumes<reco::BasicClusterCollection>(
          edm::InputTag(ps.getParameter<std::string>("barrelClusterProducer"),
                        ps.getParameter<std::string>("barrelClusterCollection")))},
      barrelEtaSearchRoad_{float(ps.getParameter<double>("barrelEtaSearchRoad"))},
      barrelPhiSearchRoad_{float(ps.getParameter<double>("barrelPhiSearchRoad"))},
      endcapEtaSearchRoad_{float(ps.getParameter<double>("endcapEtaSearchRoad"))},
      endcapPhiSearchRoad_{float(ps.getParameter<double>("endcapPhiSearchRoad"))},
      seedTransverseEnergyThreshold_{float(ps.getParameter<double>("seedTransverseEnergyThreshold"))},
      barrelBCEnergyThreshold_{float(ps.getParameter<double>("barrelBCEnergyThreshold"))},
      endcapBCEnergyThreshold_{float(ps.getParameter<double>("endcapBCEnergyThreshold"))},
      doBarrel_{ps.getParameter<bool>("doBarrel")},
      doEndcaps_{ps.getParameter<bool>("doEndcaps")}

{
  // The verbosity level
  HiBremRecoveryClusterAlgo::VerbosityLevel verbosity;
  std::string verbosityString = ps.getParameter<std::string>("VerbosityLevel");
  if (verbosityString == "DEBUG")
    verbosity = HiBremRecoveryClusterAlgo::pDEBUG;
  else if (verbosityString == "WARNING")
    verbosity = HiBremRecoveryClusterAlgo::pWARNING;
  else if (verbosityString == "INFO")
    verbosity = HiBremRecoveryClusterAlgo::pINFO;
  else
    verbosity = HiBremRecoveryClusterAlgo::pERROR;

  if (verbosityString == "INFO") {
    std::cout << "Barrel BC Energy threshold = " << barrelBCEnergyThreshold_ << std::endl;
    std::cout << "Endcap BC Energy threshold = " << endcapBCEnergyThreshold_ << std::endl;
  }

  bremAlgo_p = std::make_unique<HiBremRecoveryClusterAlgo>(barrelEtaSearchRoad_,
                                                           barrelPhiSearchRoad_,
                                                           endcapEtaSearchRoad_,
                                                           endcapPhiSearchRoad_,
                                                           seedTransverseEnergyThreshold_,
                                                           barrelBCEnergyThreshold_,
                                                           endcapBCEnergyThreshold_,
                                                           verbosity);
}

void HiSuperClusterProducer::endJob() {
  double averEnergy = 0.;
  std::ostringstream str;
  str << "HiSuperClusterProducer::endJob()\n"
      << "  total # reconstructed super clusters: " << noSuperClusters << "\n"
      << "  total energy of all clusters: " << totalE << "\n";
  if (noSuperClusters > 0) {
    averEnergy = totalE / noSuperClusters;
    str << "  average SuperCluster energy = " << averEnergy << "\n";
  }
  edm::LogInfo("HiSuperClusterProducerInfo") << str.str() << "\n";
}

void HiSuperClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  if (doEndcaps_)
    produceSuperclustersForECALPart(evt, eeClustersToken_, endcapSuperclusterPut_);

  if (doBarrel_)
    produceSuperclustersForECALPart(evt, ebClustersToken_, barrelSuperclusterPut_);
}

void HiSuperClusterProducer::produceSuperclustersForECALPart(
    edm::Event& evt,
    const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
    edm::EDPutTokenT<reco::SuperClusterCollection> const& putToken) {
  // get the cluster collection out and turn it to a BasicClusterRefVector:
  reco::CaloClusterPtrVector clusterPtrVector_p{};
  getClusterPtrVector(evt, clustersToken, &clusterPtrVector_p);

  // run the brem recovery and get the SC collection
  reco::SuperClusterCollection superclusters_ap = bremAlgo_p->makeSuperClusters(clusterPtrVector_p);

  // count the total energy and the number of superclusters
  for (auto const& cluster : superclusters_ap) {
    totalE += cluster.energy();
    noSuperClusters++;
  }

  // put the SC collection in the event
  evt.emplace(putToken, std::move(superclusters_ap));
}

void HiSuperClusterProducer::getClusterPtrVector(edm::Event& evt,
                                                 const edm::EDGetTokenT<reco::BasicClusterCollection>& clustersToken,
                                                 reco::CaloClusterPtrVector* clusterPtrVector_p) {
  edm::Handle<reco::BasicClusterCollection> bccHandle = evt.getHandle(clustersToken);

  if (!(bccHandle.isValid())) {
    edm::LogError("HiSuperClusterProducerError") << "could not get a handle on the BasicCluster Collection!";
    clusterPtrVector_p = nullptr;
  }

  const reco::BasicClusterCollection* clusterCollection_p = bccHandle.product();
  for (unsigned int i = 0; i < clusterCollection_p->size(); i++) {
    clusterPtrVector_p->push_back(reco::CaloClusterPtr(bccHandle, i));
  }
}
