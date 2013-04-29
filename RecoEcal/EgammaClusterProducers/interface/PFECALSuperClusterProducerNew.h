#ifndef RecoEcal_EgammaClusterProducers_PFECALSuperClusterProducerNew_h_
#define RecoEcal_EgammaClusterProducers_PFECALSuperClusterProducerNew_h_

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoEcal/EgammaClusterAlgos/interface/PFECALSuperClusterAlgoNew.h"

/**\class PFECALSuperClusterProducerNew 

\author Nicolas Chanon
Additional authors for Mustache: Y. Gershtein, R. Patel, L. Gray
\date   July 2012
*/

class CaloSubdetectorTopology;
class CaloSubdetectorGeometry;
class DetId;


class PFECALSuperClusterProducerNew : public edm::EDProducer {
 public:
  explicit PFECALSuperClusterProducerNew(const edm::ParameterSet&);
  ~PFECALSuperClusterProducerNew();

  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  

 private:

  // ----------member data ---------------------------

  /// clustering algorithm 
  PFECALSuperClusterAlgoNew                  superClusterAlgo_;
  PFECALSuperClusterAlgoNew::clustering_type _theclusteringtype;

  std::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_;

  /// verbose ?
  bool   verbose_;
  
  edm::InputTag    inputTagPFClusters_;
  edm::InputTag    inputTagPFClustersES_;

  std::string PFBasicClusterCollectionBarrel_;
  std::string PFSuperClusterCollectionBarrel_;
  std::string PFBasicClusterCollectionEndcap_;
  std::string PFSuperClusterCollectionEndcap_;
  std::string PFBasicClusterCollectionPreshower_;
  std::string PFSuperClusterCollectionEndcapWithPreshower_;

};

#endif
