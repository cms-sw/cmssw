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
#include "RecoMET/METAlgorithms/interface/METSignificance.h"

#include "TVector.h"

#include <string.h>

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  PFMETProducer::PFMETProducer(const edm::ParameterSet& iConfig)
    : inputToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src")))
    , calculateSignificance_(iConfig.getParameter<bool>("calculateSignificance"))
    , globalThreshold_(iConfig.getParameter<double>("globalThreshold"))
  {
    if(calculateSignificance_)
      {
	jetToken_ = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jets"));
   std::vector<edm::InputTag> srcLeptonsTags = iConfig.getParameter< std::vector<edm::InputTag> >("leptons");
   for(std::vector<edm::InputTag>::const_iterator it=srcLeptonsTags.begin();it!=srcLeptonsTags.end();it++) {
      lepTokens_.push_back( consumes<edm::View<reco::Candidate> >( *it ) );
   }
   resAlg_ = iConfig.getParameter<std::string>("jetResAlgo");
   resEra_ = iConfig.getParameter<std::string>("jetResEra");
   jetThreshold_ = iConfig.getParameter<double>("jetThreshold");
   jetEtas_ = iConfig.getParameter<std::vector<double>>("jeta");
   jetParams_ = iConfig.getParameter<std::vector<double>>("jpar");
   pjetParams_ = iConfig.getParameter<std::vector<double>>("pjpar");
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

    // candidates
    std::vector<reco::Candidate::LorentzVector> candidates;
    for(edm::View<reco::Candidate>::const_iterator cand = input->begin();
          cand != input->end(); ++cand) {
       candidates.push_back( cand->p4() );
    }

    // leptons
    std::vector<reco::Candidate::LorentzVector> leptons;
    for ( std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > >::const_iterator srcLeptons_i = lepTokens_.begin();
          srcLeptons_i != lepTokens_.end(); ++srcLeptons_i ) {
       edm::Handle<reco::CandidateView> leptons_i;
       event.getByToken(*srcLeptons_i, leptons_i);
       for ( reco::CandidateView::const_iterator lepton = leptons_i->begin();
             lepton != leptons_i->end(); ++lepton ) {
          leptons.push_back(lepton->p4());
       }
    }

    // jets
    edm::Handle<edm::View<reco::Jet>> inputJets;
    event.getByToken( jetToken_, inputJets );
    std::vector<reco::Jet> jets;
    for(edm::View<reco::Jet>::const_iterator jet = inputJets->begin(); jet != inputJets->end(); ++jet) {
       jets.push_back( *jet );
    }

    // resolutions
    std::string path = "CondFormats/JetMETObjects/data";
    std::string ptFileName  = path + "/" + resEra_ + "_PtResolution_" +resAlg_+".txt";
    std::string phiFileName = path + "/" + resEra_ + "_PhiResolution_"+resAlg_+".txt";

    // calculate significance matrix
    metsig::METSignificance metsig;
    metsig.addJets( jets );
    metsig.addLeptons( leptons );
    metsig.addCandidates( candidates );

    metsig.setThreshold( jetThreshold_ );
    metsig.setJetEtaBins( jetEtas_ );
    metsig.setJetParams( jetParams_ );
    metsig.setPJetParams( pjetParams_ );

    metsig.setResFiles( ptFileName, phiFileName );

    TMatrixD cov = metsig.getCovariance();
    reco::METCovMatrix sigcov;
    sigcov(0,0) = cov(0,0);
    sigcov(1,0) = cov(1,0);
    sigcov(0,1) = cov(0,1);
    sigcov(1,1) = cov(1,1);
    pfmet.setSignificanceMatrix(sigcov);
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
