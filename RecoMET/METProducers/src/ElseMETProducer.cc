// -*- C++ -*-
//
// Package:    METProducers
// Class:      ElseMETProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/ElseMETProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"

#include <string.h>

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  ElseMETProducer::ElseMETProducer(const edm::ParameterSet& iConfig)
    : inputToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src")))
    , globalThreshold_(iConfig.getParameter<double>("globalThreshold"))
  {
    std::string alias = iConfig.exists("alias") ? iConfig.getParameter<std::string>("alias") : "";

    produces<reco::METCollection>().setBranchAlias(alias);
  }


//____________________________________________________________________________||
  void ElseMETProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByToken(inputToken_, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(*input.product(), globalThreshold_);

    math::XYZTLorentzVector p4(commonMETdata.mex, commonMETdata.mey, 0.0, commonMETdata.met);
    math::XYZPoint vtx(0,0,0);
    reco::MET met(commonMETdata.sumet, p4, vtx);
    std::auto_ptr<reco::METCollection> metcoll;
    metcoll.reset(new reco::METCollection);
    metcoll->push_back(met);
    event.put(metcoll);
  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(ElseMETProducer);
}

//____________________________________________________________________________||
