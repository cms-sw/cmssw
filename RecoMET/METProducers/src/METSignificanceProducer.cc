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
   produces<double>("CovarianceMatrix00");
   produces<double>("CovarianceMatrix01");
   produces<double>("CovarianceMatrix10");
   produces<double>("CovarianceMatrix11");

  }

//____________________________________________________________________________||
  void METSignificanceProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
   //
   // met
   //
    edm::Handle<edm::View<reco::MET> > metHandle;
    event.getByToken(metToken_, metHandle);
    reco::MET met = (*metHandle)[0];
    
    //
    // candidates
    //
    edm::Handle<reco::CandidateView> pfCandidates;
    event.getByToken( pfCandidatesToken_, pfCandidates );
    
    //
    // leptons
    //
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

   //
   // jets
   //
   edm::Handle<edm::View<reco::Jet> > jets;
   event.getByToken( pfjetsToken_, jets );
   
   //
   // compute the significance
   //
   reco::METCovMatrix cov = metSigAlgo_->getCovariance( *jets, leptons, *pfCandidates);
   double sig  = metSigAlgo_->getSignificance(cov, met);

   std::auto_ptr<double> significance (new double);
   (*significance) = sig;

   std::auto_ptr<double> sigmatrix_00 (new double);
   (*sigmatrix_00) = cov(0,0);
   std::auto_ptr<double> sigmatrix_01 (new double);
   (*sigmatrix_01) = cov(1,0);
   std::auto_ptr<double> sigmatrix_10 (new double);
   (*sigmatrix_10) = cov(0,1);
   std::auto_ptr<double> sigmatrix_11 (new double);
   (*sigmatrix_11) = cov(1,1);

   event.put( significance, "METSignificance" );
   event.put( sigmatrix_00, "CovarianceMatrix00" );
   event.put( sigmatrix_01, "CovarianceMatrix01" );
   event.put( sigmatrix_10, "CovarianceMatrix10" );
   event.put( sigmatrix_11, "CovarianceMatrix11" );


  }

//____________________________________________________________________________||
  DEFINE_FWK_MODULE(METSignificanceProducer);
}

//____________________________________________________________________________||
