// -*- C++ -*-
//
// Package:    METProducers
// Class:      PFMETProducer
//
/**\class PFMETProducer

 Description: An EDProducer for CaloMET

 Implementation:

*/
//
//
//

//____________________________________________________________________________||
#ifndef PFMETProducer_h
#define PFMETProducer_h

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
  class PFMETProducer: public edm::stream::EDProducer<>
    {
    public:
      explicit PFMETProducer(const edm::ParameterSet&);
      virtual ~PFMETProducer() { }
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

    private:

      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      bool calculateSignificance_;
      //metsig::SignAlgoResolutions *resolutions_;

      double globalThreshold_;
      double jetThreshold_;

      edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
      std::vector< edm::EDGetTokenT<edm::View<reco::Candidate> > > lepTokens_;
      std::string resAlg_;
      std::string resEra_;

      std::vector<double> jetEtas_;
      std::vector<double> jetParams_;
      std::vector<double> pjetParams_;

    };
}

//____________________________________________________________________________||
#endif // PFMETProducer_h
