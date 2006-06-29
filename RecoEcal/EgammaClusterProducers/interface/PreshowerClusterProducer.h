#ifndef RecoEcal_EgammaClusterProducers_PreshowerClusterProducer_h
#define RecoEcal_EgammaClusterProducers_PreshowerClusterProducer_h

#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "RecoEcal/EgammaClusterAlgos/interface/PreshowerClusterAlgo.h"

#include "TH1.h"
class TFile;


class PreshowerClusterProducer : public edm::EDProducer {

 public:

  typedef math::XYZPoint Point;

  explicit PreshowerClusterProducer (const edm::ParameterSet& ps);

  ~PreshowerClusterProducer();

  virtual void produce( edm::Event& evt, const edm::EventSetup& es);
  const ESDetId getClosestCellInPlane_(const GlobalPoint &point, const int plane) const;

 private:

  int nEvt_;         // internal counter of events

  //clustering parameters:
  std::string preshHitProducer_;   // name of module/plugin/producer producing hits
  std::string preshHitCollection_; // secondary name given to collection of hits by hitProducer
  std::string preshClusterCollectionX_;  // secondary name to be given to collection of cluster produced in this module
  std::string preshClusterCollectionY_;  
  std::string endcapSClusterCollection_;
  std::string endcapSClusterProducer_;

  int preshNclust_;

  // association parameters:
  std::string assocSClusterCollection_;    // name of super cluster output collection

  double calib_planeX_;
  double calib_planeY_;
  double miptogev_;

  PreshowerClusterAlgo * presh_algo; // algorithm doing the real work

  DebugLevel debugL;

  virtual void beginJob(edm::EventSetup const&);
  virtual void endJob();
  TH1F* h1_esE_x;
  TH1F* h1_esE_y;
  TH1F* h1_esEta_x;
  TH1F* h1_esEta_y;
  TH1F* h1_esPhi_x;
  TH1F* h1_esPhi_y;
  TH1F* h1_esNhits_x;
  TH1F* h1_esNhits_y;
  TH1F* h1_esDeltaE;
  std::string outputFile_; // output file
  TFile*  rootFile_;

};
#endif

