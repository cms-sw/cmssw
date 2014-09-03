// -*- C++ -*-
//
// Package:    METProducers
// Class:      GenMETProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/GenMETProducer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"

#include <string.h>

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  GenMETProducer::GenMETProducer(const edm::ParameterSet& iConfig)
    : inputToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src")))
    , globalThreshold_(iConfig.getParameter<double>("globalThreshold"))
    , onlyFiducial_(iConfig.getParameter<bool>("onlyFiducialParticles"))
    , applyFiducialThresholdForFractions_(iConfig.getParameter<bool>("applyFiducialThresholdForFractions"))
    , usePt_(iConfig.getParameter<bool>("usePt"))
  {
    std::string alias = iConfig.exists("alias") ? iConfig.getParameter<std::string>("alias") : "";
    produces<reco::GenMETCollection>().setBranchAlias(alias);
  }


//____________________________________________________________________________||
  void GenMETProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByToken(inputToken_, input);

    CommonMETData commonMETdata;

    GenSpecificAlgo gen;
    std::auto_ptr<reco::GenMETCollection> genmetcoll;
    genmetcoll.reset(new reco::GenMETCollection);
    genmetcoll->push_back(gen.addInfo(input, &commonMETdata, globalThreshold_, onlyFiducial_, applyFiducialThresholdForFractions_, usePt_));
    event.put(genmetcoll);
  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(GenMETProducer);
}

//____________________________________________________________________________||
