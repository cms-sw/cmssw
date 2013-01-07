#ifndef TrackerAlignment_AlignmentPrescaler_H
#define TrackerAlignment_AlignmentPrescaler_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TH1F.h"

class TrackerTopology;

class AlignmentPrescaler : public edm::EDProducer{

 public:
  AlignmentPrescaler(const edm::ParameterSet &iConfig);
  ~AlignmentPrescaler();
  void beginJob();
  void endJob();
  virtual void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) ;

 private:
  edm::InputTag src_;//tracks in input
  edm::InputTag srcQualityMap_;//Hit-quality association map

  std::string prescfilename_;//name of the file containing the TTree with the prescaling factors
  std::string presctreename_;//name of the  TTree with the prescaling factors

  TFile *fpresc_;
  TTree *tpresc_;
  TRandom3 *myrand_;
 

  int layerFromId (const DetId& id, const TrackerTopology* tTopo) const;

  unsigned int detid_;
  float hitPrescFactor_, overlapPrescFactor_;
  int totnhitspxl_;
};
#endif
