// -*- C++ -*-
//
// Package:    METProducers
// Class:      GenMETProducer
//
/**\class GenMETProducer

 Description: An EDProducer for GenMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef GenMETProducer_h
#define GenMETProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//____________________________________________________________________________||
namespace cms
{
  class GenMETProducer: public edm::EDProducer
    {
    public:
      explicit GenMETProducer(const edm::ParameterSet&);
      virtual ~GenMETProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      double globalThreshold_;

      bool onlyFiducial_;

      bool applyFiducialThresholdForFractions_;

      bool usePt_;

    };
}

//____________________________________________________________________________||
#endif // GenMETProducer_h
