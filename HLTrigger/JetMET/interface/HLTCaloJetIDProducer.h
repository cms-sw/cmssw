#ifndef HLTCaloJetIDProducer_h
#define HLTCaloJetIDProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTCaloJetIDProducer : public edm::EDProducer {
 public:
  explicit HLTCaloJetIDProducer(const edm::ParameterSet&);
  ~HLTCaloJetIDProducer();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void beginJob() ; 
  virtual void produce(edm::Event &, const edm::EventSetup&);

 private:
  edm::EDGetTokenT<reco::CaloJetCollection> m_theCaloJetToken;

  edm::InputTag jetsInput_;
  double min_EMF_;         // minimum EMF
  double max_EMF_;         // maximum EMF
  int min_N90_;            // mininum N90
  int min_N90hits_;        // mininum Nhit90

  reco::helper::JetIDHelper jetID_;

};

#endif
