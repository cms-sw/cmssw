// -*- C++ -*-
//
// Package:    METProducers
// Class:      PFClusterMETProducer
//
/**\class PFClusterMETProducer

 Description: An stream::EDProducer<> for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef PFClusterMETProducer_h
#define PFClusterMETProducer_h

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
  class PFClusterMETProducer: public edm::stream::EDProducer<>
    {
    public:
      explicit PFClusterMETProducer(const edm::ParameterSet&);
      virtual ~PFClusterMETProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      double globalThreshold_;

    };
}

//____________________________________________________________________________||
#endif // PFClusterMETProducer_h
