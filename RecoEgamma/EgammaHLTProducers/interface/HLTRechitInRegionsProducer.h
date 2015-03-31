#ifndef RecoEgamma_EgammayHLTProducers_HLTRechitInRegionsProducer_h_
#define RecoEgamma_EgammayHLTProducers_HLTRechitInRegionsProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

// Reco candidates
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

// Geometry and topology
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"


template<typename T1>
class HLTRechitInRegionsProducer : public edm::stream::EDProducer<> {
 typedef std::vector<T1> T1Collection;
 typedef typename T1::const_iterator T1iterator;
  
 public:
  
  HLTRechitInRegionsProducer(const edm::ParameterSet& ps);
  ~HLTRechitInRegionsProducer();

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  void getEtaPhiRegions(std::vector<EcalEtaPhiRegion> *, T1Collection, const L1CaloGeometry&, bool);
    
  const bool useUncalib_;

  const bool doIsolated_;

  const edm::EDGetTokenT<T1Collection> l1TokenIsolated_;
  const edm::EDGetTokenT<T1Collection> l1TokenNonIsolated_;
  const double l1LowerThr_;
  const double l1UpperThr_;
  const double l1LowerThrIgnoreIsolation_;

  const double regionEtaMargin_;
  const double regionPhiMargin_;

  const std::vector<edm::InputTag> hitLabels;
  const std::vector<std::string> productLabels;

  std::vector<edm::EDGetTokenT<EcalRecHitCollection>> hitTokens;
  std::vector<edm::EDGetTokenT<EcalUncalibratedRecHitCollection>> uncalibHitTokens;
};


#endif


