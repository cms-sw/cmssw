// -*- C++ -*-
//
// Package:    METProducers
// Class:      METSignificanceProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/METSignificanceProducer.h"


//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  METSignificanceProducer::METSignificanceProducer(const edm::ParameterSet& iConfig)
  {
    
    std::vector<edm::InputTag> srcLeptonsTags = iConfig.getParameter< std::vector<edm::InputTag> >("srcLeptons");
    for(std::vector<edm::InputTag>::const_iterator it=srcLeptonsTags.begin();it!=srcLeptonsTags.end();it++) {
      lepTokens_.push_back( consumes<edm::View<reco::Candidate> >( *it ) );
    }
    
    pfjetsToken_    = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("srcPfJets"));

    metToken_ = consumes<edm::View<reco::MET> >(iConfig.getParameter<edm::InputTag>("srcMet"));
    pfCandidatesToken_ = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("srcPFCandidates"));

    metSigAlgo_ = new metsig::METSignificance(iConfig);

   produces<double>("METSignificance");
   produces<math::Error<2>::type>("METCovariance");
   
  }

//____________________________________________________________________________||
  void METSignificanceProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
   //
   // met
   //
    edm::Handle<edm::View<reco::MET> > metHandle;
    event.getByToken(metToken_, metHandle);
    const reco::MET& met = (*metHandle)[0];
    
    //
    // candidates
    //
    edm::Handle<reco::CandidateView> pfCandidates;
    event.getByToken( pfCandidatesToken_, pfCandidates );
    
    //
    // leptons
    //
   std::vector< edm::Handle<reco::CandidateView> > leptons;
   for ( std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > >::const_iterator srcLeptons_i = lepTokens_.begin();
         srcLeptons_i != lepTokens_.end(); ++srcLeptons_i ) {

      edm::Handle<reco::CandidateView> leptons_i;
      event.getByToken(*srcLeptons_i, leptons_i);
      leptons.push_back( leptons_i );

   }

   //
   // jets
   //
   edm::Handle<edm::View<reco::Jet> > jets;
   event.getByToken( pfjetsToken_, jets );
   
   //
   // compute the significance
   //
   const reco::METCovMatrix cov = metSigAlgo_->getCovariance( *jets, leptons, *pfCandidates);
   double sig  = metSigAlgo_->getSignificance(cov, met);

   std::auto_ptr<double> significance (new double);
   (*significance) = sig;
   
   std::auto_ptr<math::Error<2>::type> covPtr(new math::Error<2>::type());
   (*covPtr)(0,0) = cov(0,0);
   (*covPtr)(1,0) = cov(1,0);
   (*covPtr)(1,1) = cov(1,1);

   event.put( covPtr, "METCovariance" );
   event.put( significance, "METSignificance" );

  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(METSignificanceProducer);
}

//____________________________________________________________________________||
