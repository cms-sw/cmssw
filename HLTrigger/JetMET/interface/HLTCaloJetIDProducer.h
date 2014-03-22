#ifndef HLTCaloJetIDProducer_h_
#define HLTCaloJetIDProducer_h_

/** \class HLTCaloJetIDProducer
 *
 *  \brief  This applies CaloJet ID and produces a jet collection with jets that pass the ID.
 *  \author a Jet/MET person
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  This receives a CaloJet collection, selects jets that pass CaloJet ID,
 *  and makes an output CaloJet collection with only jets that pass.
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"


namespace edm {
    class ConfigurationDescriptions;
}

namespace reco {
    namespace helper {
        class JetIDHelper;
    }
}

// Class declaration
class HLTCaloJetIDProducer : public edm::EDProducer {
  public:
    explicit HLTCaloJetIDProducer(const edm::ParameterSet & iConfig);
    ~HLTCaloJetIDProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:
    int min_N90_;                     ///< mininum N90
    int min_N90hits_;                 ///< mininum N90hits
    double min_EMF_;                  ///< minimum EMF
    double max_EMF_;                  ///< maximum EMF
    edm::InputTag inputTag_;          ///< input CaloJet collection
    edm::ParameterSet jetIDParams_;   ///< CaloJet ID parameters

    /// A helper to calculates calo jet ID variables.
    reco::helper::JetIDHelper jetIDHelper_;

    edm::EDGetTokenT<reco::CaloJetCollection> m_theCaloJetToken;
};

#endif  // HLTCaloJetIDProducer_h_
