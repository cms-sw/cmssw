// -*- C++ -*-
//
// Package:    METProducers
// Class:      CaloMETProducer
//
/**\class CaloMETProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef CaloMETProducer_h
#define CaloMETProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//____________________________________________________________________________||
namespace metsig
{
  class SignAlgoResolutions;
}

//____________________________________________________________________________||
namespace cms
{
  class CaloMETProducer: public edm::stream::EDProducer<>
    {
    public:
      explicit CaloMETProducer(const edm::ParameterSet&);
      virtual ~CaloMETProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      bool calculateSignificance_;
      metsig::SignAlgoResolutions *resolutions_;

      bool noHF_;

      double globalThreshold_;

    };
}

//____________________________________________________________________________||
#endif // CaloMETProducer_h
