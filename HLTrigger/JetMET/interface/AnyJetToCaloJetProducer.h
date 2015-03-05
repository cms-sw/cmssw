#ifndef AnyJetToCaloJetProducer_H
#define AnyJetToCaloJetProducer_H

// Author: S. Lowette

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
   class ConfigurationDescriptions;
}

class AnyJetToCaloJetProducer: public edm::stream::EDProducer<> {

  public:

    explicit AnyJetToCaloJetProducer(const edm::ParameterSet&);
    ~AnyJetToCaloJetProducer();

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions); 
    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:

    edm::EDGetTokenT<edm::View<reco::Jet>> m_theGenericJetToken;
    edm::InputTag jetSrc_;

};

#endif
