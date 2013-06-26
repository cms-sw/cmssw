#ifndef AnyJetToCaloJetProducer_H
#define AnyJetToCaloJetProducer_H

// Author: S. Lowette

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


class AnyJetToCaloJetProducer: public edm::EDProducer {

  public:

    explicit AnyJetToCaloJetProducer(const edm::ParameterSet&);
    ~AnyJetToCaloJetProducer();

    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:

    edm::InputTag jetSrc_;

};

#endif
