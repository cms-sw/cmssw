// -*- C++ -*-
//
// Package:    METProducers
// Class:      ElseMETProducer
//
/**\class ElseMETProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef ElseMETProducer_h
#define ElseMETProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//____________________________________________________________________________||
namespace cms
{
  class ElseMETProducer: public edm::stream::EDProducer<>
    {
    public:
      explicit ElseMETProducer(const edm::ParameterSet&);
      ~ElseMETProducer() override { }
      void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      double globalThreshold_;

    };
}

//____________________________________________________________________________||
#endif // ElseMETProducer_h
