#ifndef RecoEcal_EgammaClusterProducers_PreshowerClusterProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerClusterProducer_h
/** \class PreshowerClusterProducer
 **   example of producer for BasicCluster from recHits
 **
 **  $Id: PreshowerClusterProducer.h,v 1.1 2006/04/13 14:40:05 rahatlou Exp $
 **  $Date: 2006/04/13 14:40:05 $
 **  $Revision: 1.1 $
 **  \author Shahram Rahatlou, University of Rome & INFN, April 2006
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

// PreshowerClusterProducer inherits from EDProducer, so it can be a module:
class PreshowerClusterProducer : public edm::EDProducer {

 public:

  PreshowerClusterProducer (const edm::ParameterSet& ps);
  ~PreshowerClusterProducer();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es);
  const ESDetId getClosestCellInPlane( const reco::Point&, const int&) const;

 private:

  //typedef math::XYZPoint Point;
  PreshowerClusterAlgo* presh_algo_; // algorithm doing the real work

  std::string hitProducer_;   // name of module/plugin/producer producing hits
  std::string hitCollection_; // secondary name given to collection of hits by hitProducer
  std::string clusterCollection1_;  // secondary name to be given to collection of cluster produced in this module
  std::string clusterCollection2_;  

  int PreshNclust_;

  double calib_plane1_;
  double calib_plane2_;
  double miptogev_;

  std::string SClusterCollection_;    // name of super cluster collection

};
#endif

