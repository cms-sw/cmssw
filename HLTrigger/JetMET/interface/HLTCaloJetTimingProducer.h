#ifndef  HLTCaloJetTimingProducer_h_
#define  HLTCaloJetTimingProducer_h_

/** \class HLTCaloJetTimingProducer
 *
 *  \brief  This produces timing and associated ecal cell information for calo jets 
 *  \author Matthew Citron
 *
 *
 */


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class declaration
//
class HLTCaloJetTimingProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTCaloJetTimingProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  // Input collections
  edm::InputTag jetLabel_;
  edm::InputTag ecalEBLabel_;
  edm::InputTag ecalEELabel_;
  // Include endcap jets or only barrel
  bool barrelOnly_;

  edm::EDGetTokenT<reco::CaloJetCollection> jetInputToken;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEBToken;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEEToken;
};
#endif  // HLTCaloJetTimingProducer_h_
