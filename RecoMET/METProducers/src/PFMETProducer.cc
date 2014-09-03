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
#include "DataFormats/METReco/interface/PFMET.h"
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
	jetToken_ = consumes<edm::View<reco::PFJet> >(iConfig.getParameter<edm::InputTag>("jets"));
	resolutions_ = new metsig::SignAlgoResolutions(iConfig);
      }

    std::string alias = iConfig.exists("alias") ? iConfig.getParameter<std::string>("alias") : "";

    produces<reco::PFMETCollection>().setBranchAlias(alias);
  }

//____________________________________________________________________________||
  void PFMETProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
    edm::Handle<edm::View<reco::Candidate> > input;
    event.getByToken(inputToken_, input);

    METAlgo algo;
    CommonMETData commonMETdata = algo.run(*input.product(), globalThreshold_);

    const math::XYZTLorentzVector p4(commonMETdata.mex, commonMETdata.mey, 0.0, commonMETdata.met);
    const math::XYZPoint vtx(0.0, 0.0, 0.0);

    PFSpecificAlgo pf;
    SpecificPFMETData specific = pf.run(*input.product());

    reco::PFMET pfmet(specific, commonMETdata.sumet, p4, vtx);

    if(calculateSignificance_)
      {
	metsig::SignPFSpecificAlgo pfsignalgo;
	pfsignalgo.setResolutions(resolutions_);

	edm::Handle<edm::View<reco::PFJet> > jets;
	event.getByToken(jetToken_, jets);
	pfsignalgo.addPFJets(jets.product());
	pfmet.setSignificanceMatrix(pfsignalgo.mkSignifMatrix(input));
      }

    std::auto_ptr<reco::PFMETCollection> pfmetcoll;
    pfmetcoll.reset(new reco::PFMETCollection);

    pfmetcoll->push_back(pfmet);
    event.put(pfmetcoll);
  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(PFMETProducer);
}

//____________________________________________________________________________||
