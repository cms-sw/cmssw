#ifndef HLTPFJetIDProducer_h_
#define HLTPFJetIDProducer_h_

/** \class HLTPFJetIDProducer
 *
 *  \brief  This applies PFJet ID and produces a jet collection with jets that pass the ID.
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *
 *  This receives a PFJet collection, selects jets that pass PFJet ID,
 *  and makes an output PFJet collection with only jets that pass.
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"


namespace edm {
   class ConfigurationDescriptions;
}

// Class declaration
class HLTPFJetIDProducer : public edm::stream::EDProducer<> {
  public:
    explicit HLTPFJetIDProducer(const edm::ParameterSet & iConfig);
    ~HLTPFJetIDProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:
    double minPt_;
    double maxEta_;
    double CHF_;              ///< charged hadron fraction
    double NHF_;              ///< neutral hadron fraction
    double CEF_;              ///< charged EM fraction
    double NEF_;              ///< neutral EM fraction
    double maxCF_;            ///< total charged energy fraction
    int NCH_;                 ///< number of charged constituents
    int NTOT_;                ///< number of constituents
    edm::InputTag inputTag_;  ///< input PFJet collection

    edm::EDGetTokenT<reco::PFJetCollection> m_thePFJetToken;
};

#endif  // HLTPFJetIDProducer_h_
