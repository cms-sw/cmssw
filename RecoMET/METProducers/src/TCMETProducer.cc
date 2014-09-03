// -*- C++ -*-
//
// Package:    METProducers
// Class:      TCMETProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/TCMETProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/METReco/interface/METFwd.h"

#include <string.h>

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  TCMETProducer::TCMETProducer(const edm::ParameterSet& iConfig)
  {
    std::string alias = iConfig.exists("alias") ? iConfig.getParameter<std::string>("alias") : "";

    produces<reco::METCollection>().setBranchAlias(alias);

    tcMetAlgo_.configure(iConfig, consumesCollector());
  }

//____________________________________________________________________________||
  void TCMETProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
    std::auto_ptr<reco::METCollection> tcmetcoll;
    tcmetcoll.reset(new reco::METCollection);
    tcmetcoll->push_back(tcMetAlgo_.CalculateTCMET(event, setup));
    event.put(tcmetcoll);
  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(TCMETProducer);
}

//____________________________________________________________________________||
