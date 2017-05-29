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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/METSignificance.h"

#include "JetMETCorrections/Modules/interface/JetResolution.h"

#include "TVector.h"

#include <string>


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

      reco::METCovMatrix getMETCovMatrix(const edm::Event& event, const edm::EventSetup&, 
					 const edm::Handle<edm::View<reco::Candidate> >& input) const;


      edm::EDGetTokenT<edm::View<reco::Candidate> > inputToken_;

      bool calculateSignificance_;
      metsig::METSignificance* metSigAlgo_;

      double globalThreshold_;
      double jetThreshold_;

      edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
      std::vector< edm::EDGetTokenT<edm::View<reco::Candidate> > > lepTokens_;
      std::string jetSFType_;
      std::string jetResPtType_;
      std::string jetResPhiType_;
      edm::EDGetTokenT<double> rhoToken_;
  };
}

//____________________________________________________________________________||
#endif // PFMETProducer_h
