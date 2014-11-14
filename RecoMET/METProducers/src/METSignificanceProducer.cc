// -*- C++ -*-
//
// Package:    METProducers
// Class:      METSignificanceProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/METSignificanceProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"

#include "RecoMET/METAlgorithms/src/METSignificance.cc"

#include <string.h>

#include "TMatrixD.h"

//____________________________________________________________________________||
namespace cms
{

//____________________________________________________________________________||
  METSignificanceProducer::METSignificanceProducer(const edm::ParameterSet& iConfig)
  {

   std::vector<edm::InputTag> srcLeptonsTags = iConfig.getParameter< std::vector<edm::InputTag> >("srcLeptons");
   for(std::vector<edm::InputTag>::const_iterator it=srcLeptonsTags.begin();it!=srcLeptonsTags.end();it++) {
      srcLeptons_.push_back( consumes<edm::View<reco::Candidate> >( *it ) );
   }

   pfjetsTag_    = iConfig.getUntrackedParameter<edm::InputTag>("pfjetsTag");

   metTag_ = iConfig.getUntrackedParameter<edm::InputTag>("metTag");
   pfcandidatesTag_ = iConfig.getUntrackedParameter<edm::InputTag>("pfcandidatesTag");

   jetThreshold_ = iConfig.getParameter<double>("jetThreshold");
   jetEtas_ = iConfig.getParameter<std::vector<double>>("jeta");
   jetParams_ = iConfig.getParameter<std::vector<double>>("jpar");
   pjetParams_ = iConfig.getParameter<std::vector<double>>("pjpar");

   resAlg  = iConfig.getParameter<std::string>("jetResAlgo");
   resEra  = iConfig.getParameter<std::string>("jetResEra");

   produces<double>("METSignificance");
   produces<double>("CovarianceMatrix00");
   produces<double>("CovarianceMatrix01");
   produces<double>("CovarianceMatrix10");
   produces<double>("CovarianceMatrix11");

  }

//____________________________________________________________________________||
  void METSignificanceProducer::produce(edm::Event& event, const edm::EventSetup& setup)
  {
   using namespace edm;

   //
   // met
   //

   Handle<View<reco::MET> > metHandle;
   event.getByLabel(metTag_, metHandle);
   reco::MET met = (*metHandle)[0];

   //
   // candidates
   //

   Handle<reco::CandidateView> inputCandidates;
   event.getByLabel( pfcandidatesTag_, inputCandidates );
   std::vector<reco::Candidate::LorentzVector> candidates;
   for(View<reco::Candidate>::const_iterator cand = inputCandidates->begin();
         cand != inputCandidates->end(); ++cand) {
      candidates.push_back( cand->p4() );
   }

   //
   // leptons
   //

   std::vector<reco::Candidate::LorentzVector> leptons;
   for ( std::vector<EDGetTokenT<View<reco::Candidate> > >::const_iterator srcLeptons_i = srcLeptons_.begin();
         srcLeptons_i != srcLeptons_.end(); ++srcLeptons_i ) {

      Handle<reco::CandidateView> leptons_i;
      event.getByToken(*srcLeptons_i, leptons_i);
      for ( reco::CandidateView::const_iterator lepton = leptons_i->begin();
            lepton != leptons_i->end(); ++lepton ) {
         leptons.push_back(lepton->p4());
      }
   }

   //
   // jets
   //

   Handle<View<reco::Jet>> inputJets;
   event.getByLabel( pfjetsTag_, inputJets );
   std::vector<reco::Jet> jets;
   for(View<reco::Jet>::const_iterator jet = inputJets->begin(); jet != inputJets->end(); ++jet) {
      jets.push_back( *jet );
   }

   //
   // resolutions
   //

   std::string path = "CondFormats/JetMETObjects/data";
   std::string ptFileName  = path + "/" + resEra + "_PtResolution_" +resAlg+".txt";
   std::string phiFileName = path + "/" + resEra + "_PhiResolution_"+resAlg+".txt";

   //
   // compute the significance
   //

   metsig::METSignificance metsig;

   metsig.addMET( met );
   metsig.addJets( jets );
   metsig.addLeptons( leptons );
   metsig.addCandidates( candidates );

   metsig.setThreshold( jetThreshold_ );
   metsig.setJetEtaBins( jetEtas_ );
   metsig.setJetParams( jetParams_ );
   metsig.setPJetParams( pjetParams_ );

   metsig.setResFiles( ptFileName, phiFileName );

   TMatrixD cov = metsig.getCovariance();
   double sig  = metsig.getSignificance(cov);

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
