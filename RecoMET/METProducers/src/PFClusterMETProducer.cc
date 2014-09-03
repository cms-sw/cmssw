// -*- C++ -*-
//
// Package:    METProducers
// Class:      PFClusterMETProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/PFClusterMETProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFClusterMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/PFClusterSpecificAlgo.h"

#include <string.h>

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  PFClusterMETProducer::PFClusterMETProducer(const edm::ParameterSet& iConfig)
    : inputToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src")))
    , globalThreshold_(iConfig.getParameter<double>("globalThreshold"))
  {
    std::string alias = iConfig.exists("alias") ? iConfig.getParameter<std::string>("alias") : "";
    produces<reco::PFClusterMETCollection>().setBranchAlias(alias);
  }


//____________________________________________________________________________||
  void PFClusterMETProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByToken(inputToken_, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(*input.product(), globalThreshold_);

    PFClusterSpecificAlgo pfcluster;
    std::auto_ptr<reco::PFClusterMETCollection> pfclustermetcoll;
    pfclustermetcoll.reset (new reco::PFClusterMETCollection);

    pfclustermetcoll->push_back(pfcluster.addInfo(input, commonMETdata));
    event.put(pfclustermetcoll);
  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(PFClusterMETProducer);
}

//____________________________________________________________________________||
