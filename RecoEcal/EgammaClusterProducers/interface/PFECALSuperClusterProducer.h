#ifndef RecoEcal_EgammaClusterProducers_PFECALSuperClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_PFECALSuperClusterProducer_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgo.h"

/**\class PFECALSuperClusterProducer 

\author Nicolas Chanon
Additional authors for Mustache: Y. Gershtein, R. Patel, L. Gray
\date   July 2012
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;
class GBRForest;
class GBRWrapperRcd;


class PFECALSuperClusterProducer : public edm::EDProducer {
 public:  
  explicit PFECALSuperClusterProducer(const edm::ParameterSet&);
  ~PFECALSuperClusterProducer();

  virtual void beginRun(const edm::Run& iR, const edm::EventSetup& iE);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  

 private:  
  // ----------member data ---------------------------

  /// clustering algorithm 
  PFECALSuperClusterAlgo                  superClusterAlgo_;
  PFECALSuperClusterAlgo::clustering_type _theclusteringtype;
  PFECALSuperClusterAlgo::energy_weight   _theenergyweight;

  std::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_;

  /// verbose ?
  bool   verbose_;
  // regression
  bool use_regression;
  float rinputs[33];
  std::string eb_reg_key, ee_reg_key;
  const GBRWrapperRcd* gbr_record;
  edm::ESHandle<GBRForest> eb_reg, ee_reg;
  double calculateRegressedEnergy(const reco::SuperCluster&);
  
  edm::EDGetTokenT<edm::View<reco::PFCluster> >   inputTagPFClusters_;
  edm::EDGetTokenT<edm::View<reco::PFCluster> >   inputTagPFClustersES_;

  std::string PFBasicClusterCollectionBarrel_;
  std::string PFSuperClusterCollectionBarrel_;
  std::string PFBasicClusterCollectionEndcap_;
  std::string PFSuperClusterCollectionEndcap_;
  std::string PFBasicClusterCollectionPreshower_;
  std::string PFSuperClusterCollectionEndcapWithPreshower_;

};

#endif
