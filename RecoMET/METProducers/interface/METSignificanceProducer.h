// -*- C++ -*-
//
// Package:    METProducers
// Class:      METSignificanceProducer
//
/**\class METSignificanceProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef METSignificanceProducer_h
#define METSignificanceProducer_h

//____________________________________________________________________________||
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/JetReco/interface/PFJet.h"

//____________________________________________________________________________||
namespace metsig
{
    class SignAlgoResolutions;
}

//____________________________________________________________________________||
namespace cms
{
  class METSignificanceProducer: public edm::stream::EDProducer<>
    {
    public:
      explicit METSignificanceProducer(const edm::ParameterSet&);
      virtual ~METSignificanceProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      // ----------member data ---------------------------

      edm::InputTag pfjetsTag_;
      edm::InputTag metTag_;
      edm::InputTag pfcandidatesTag_;
      std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > srcLeptons_;
      Int_t metsSize_;

      double jetThreshold_;
      std::vector<double> jetEtas_;
      std::vector<double> jetParams_;
      std::vector<double> pjetParams_;

      std::string resAlg;
      std::string resEra;

    };
}

//____________________________________________________________________________||
#endif // METSignificanceProducer_h
