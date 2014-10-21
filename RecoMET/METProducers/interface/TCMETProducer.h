// -*- C++ -*-
//
// Package:    METProducers
// Class:      TCMETProducer
//
/**\class TCMETProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef TCMETProducer_h
#define TCMETProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMET/METAlgorithms/interface/TCMETAlgo.h"

//____________________________________________________________________________||
namespace cms
{
  class TCMETProducer: public edm::stream::EDProducer<>
    {
    public:
      explicit TCMETProducer(const edm::ParameterSet&);
      virtual ~TCMETProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      TCMETAlgo tcMetAlgo_;

    };
}

//____________________________________________________________________________||
#endif // TCMETProducer_h
