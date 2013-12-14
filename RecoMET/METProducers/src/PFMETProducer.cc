// -*- C++ -*-
//
// Package:    METProducers
// Class:      PFMETProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/PFMETProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/interface/METAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/SignPFSpecificAlgo.h"

#include <string.h>

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  PFMETProducer::PFMETProducer(const edm::ParameterSet& iConfig)
    : inputToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src")))
    , calculateSignificance_(iConfig.getParameter<bool>("calculateSignificance"))
    , resolutions_(0)
    , globalThreshold_(iConfig.getParameter<double>("globalThreshold"))
  {
    if(calculateSignificance_)
      {
	jetsLabel_ = iConfig.getParameter<edm::InputTag>("jets");
	jetToken_ = consumes<edm::View<reco::PFJet> >(iConfig.getParameter<edm::InputTag>("jets"));
      }

    if (calculateSignificance_)
      {
	resolutions_ = new metsig::SignAlgoResolutions(iConfig);
      }

    std::string alias(iConfig.getParameter<std::string>("alias"));
    produces<reco::PFMETCollection>().setBranchAlias(alias.c_str());
  }


//____________________________________________________________________________||
  void PFMETProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByToken(inputToken_, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(input, globalThreshold_);

    PFSpecificAlgo pf;

    std::auto_ptr<reco::PFMETCollection> pfmetcoll;
    pfmetcoll.reset(new reco::PFMETCollection);
    reco::PFMET pfmet = pf.addInfo(input, commonMETdata);

    if(calculateSignificance_)
      {
	metsig::SignPFSpecificAlgo pfsignalgo;
	pfsignalgo.setResolutions(resolutions_);

	edm::Handle<edm::View<reco::PFJet> > jets;
	event.getByToken(jetToken_, jets);
	pfsignalgo.addPFJets(jets.product());
	pfmet.setSignificanceMatrix(pfsignalgo.mkSignifMatrix(input));
      }

    pfmetcoll->push_back(pfmet);
    event.put(pfmetcoll);
  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(PFMETProducer);
}

//____________________________________________________________________________||
