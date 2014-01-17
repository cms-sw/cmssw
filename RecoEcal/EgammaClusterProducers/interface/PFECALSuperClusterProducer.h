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
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h" 

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

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
class EcalClusterTools;


class PFECALSuperClusterProducer : public edm::EDProducer {
 public:  
  explicit PFECALSuperClusterProducer(const edm::ParameterSet&);
  ~PFECALSuperClusterProducer();

  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
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

  std::string PFBasicClusterCollectionBarrel_;
  std::string PFSuperClusterCollectionBarrel_;
  std::string PFBasicClusterCollectionEndcap_;
  std::string PFSuperClusterCollectionEndcap_;
  std::string PFBasicClusterCollectionPreshower_;
  std::string PFSuperClusterCollectionEndcapWithPreshower_;
  std::string PFClusterAssociationEBEE_;
  std::string PFClusterAssociationES_;

};

#endif
