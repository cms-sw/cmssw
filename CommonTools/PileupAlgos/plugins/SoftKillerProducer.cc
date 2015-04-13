// -*- C++ -*-
//
// Package:    CommonTools/PileupAlgos
// Class:      SoftKillerProducer
// 
/**\class SoftKillerProducer SoftKillerProducer.cc CommonTools/PileupAlgos/plugins/SoftKillerProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Salvatore Rappoccio
//         Created:  Wed, 22 Oct 2014 15:14:20 GMT
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "fastjet/contrib/SoftKiller.hh"



//
// class declaration
//

class SoftKillerProducer : public edm::stream::EDProducer<> {
public:
  typedef math::XYZTLorentzVector LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;
  typedef std::vector< reco::PFCandidate >   PFOutputCollection;

  explicit SoftKillerProducer(const edm::ParameterSet&);
  ~SoftKillerProducer();

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT< reco::CandidateView > tokenPFCandidates_;


  double Rho_EtaMax_;
  double rParam_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
SoftKillerProducer::SoftKillerProducer(const edm::ParameterSet& iConfig) : 
  Rho_EtaMax_( iConfig.getParameter<double>("Rho_EtaMax") ),
  rParam_ ( iConfig.getParameter<double>("rParam") )
{

  tokenPFCandidates_
    = consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("PFCandidates"));

  produces<edm::ValueMap<LorentzVector> > ("SoftKillerP4s");
  produces< PFOutputCollection >();

}


SoftKillerProducer::~SoftKillerProducer()
{

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SoftKillerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::auto_ptr< PFOutputCollection > pOutput( new PFOutputCollection );

  // get PF Candidates
  edm::Handle<reco::CandidateView> pfCandidates;
  iEvent.getByToken( tokenPFCandidates_, pfCandidates);

  std::vector<fastjet::PseudoJet> fjInputs;
  for ( auto i = pfCandidates->begin(), 
	  ibegin = pfCandidates->begin(),
	  iend = pfCandidates->end(); i != iend; ++i ) {
    fjInputs.push_back( fastjet::PseudoJet( i->px(), i->py(), i->pz(), i->energy() ) );
    fjInputs.back().set_user_index( i - ibegin );
  }

  // soft killer:
  fastjet::contrib::SoftKiller soft_killer(Rho_EtaMax_, rParam_);

  double pt_threshold = 0.;
  std::vector<fastjet::PseudoJet> soft_killed_event;
  soft_killer.apply(fjInputs, soft_killed_event, pt_threshold);

  std::auto_ptr<edm::ValueMap<LorentzVector> > p4SKOut(new edm::ValueMap<LorentzVector>());
  LorentzVectorCollection skP4s;

  static const reco::PFCandidate dummySinceTranslateIsNotStatic;

  // To satisfy the value map, the size of the "killed" collection needs to be the
  // same size as the input collection, so if the constituent is killed, just set E = 0
  for ( auto j = fjInputs.begin(), 
	  jend = fjInputs.end(); j != jend; ++j ) {
    const reco::Candidate& cand = pfCandidates->at(j->user_index());
    auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(cand.pdgId());
    const reco::PFCandidate *pPF = dynamic_cast<const reco::PFCandidate*>(&cand);
    reco::PFCandidate pCand( pPF ? *pPF : reco::PFCandidate(cand.charge(), cand.p4(), id) );
    auto val = j->user_index();
    auto skmatch = find_if( soft_killed_event.begin(), soft_killed_event.end(), [&val](fastjet::PseudoJet const & i){return i.user_index() == val;} );
    LorentzVector pVec;
    if ( skmatch != soft_killed_event.end() ) {
      pVec.SetPxPyPzE(skmatch->px(),skmatch->py(),skmatch->pz(),skmatch->E());      
    } else {
      pVec.SetPxPyPzE( 0., 0., 0., 0.);
    }
    pCand.setP4(pVec);
    skP4s.push_back( pVec );
    pOutput->push_back(pCand);
  }

  //Compute the modified p4s
  edm::ValueMap<LorentzVector>::Filler  p4SKFiller(*p4SKOut);
  p4SKFiller.insert(pfCandidates,skP4s.begin(), skP4s.end() );
  p4SKFiller.fill();

  iEvent.put(p4SKOut,"SoftKillerP4s");
  iEvent.put( pOutput );

}

//define this as a plug-in
DEFINE_FWK_MODULE(SoftKillerProducer);
