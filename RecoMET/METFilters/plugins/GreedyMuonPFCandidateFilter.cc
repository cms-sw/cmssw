


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

//
// class declaration
//

class GreedyMuonPFCandidateFilter : public edm::EDFilter {
public:
  explicit GreedyMuonPFCandidateFilter(const edm::ParameterSet&);
  ~GreedyMuonPFCandidateFilter();

private:
  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  const edm::EDGetTokenT<reco::PFCandidateCollection>  tokenPFCandidates_;
      // ----------member data ---------------------------

  const double eOverPMax_;

  const bool debug_;

  const bool taggingMode_;
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
GreedyMuonPFCandidateFilter::GreedyMuonPFCandidateFilter(const edm::ParameterSet& iConfig)
   //now do what ever initialization is needed
  : tokenPFCandidates_ (consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates") ) )
  , eOverPMax_ (iConfig.getParameter<double>("eOverPMax") )
  , debug_ ( iConfig.getParameter<bool>("debug") )
  , taggingMode_ (iConfig.getParameter<bool>("taggingMode") )
{
  produces<bool>();
  produces<reco::PFCandidateCollection>("muons");
}


GreedyMuonPFCandidateFilter::~GreedyMuonPFCandidateFilter()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
GreedyMuonPFCandidateFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  bool foundMuon = false;

  auto_ptr< reco::PFCandidateCollection >
    pOutputCandidateCollection( new reco::PFCandidateCollection );

  for( unsigned i=0; i<pfCandidates->size(); i++ ) {

    const reco::PFCandidate & cand = (*pfCandidates)[i];

//    if( cand.particleId() != 3 ) // not a muon
    if( cand.particleId() != reco::PFCandidate::mu ) // not a muon
      continue;

    if(!PFMuonAlgo::isIsolatedMuon( cand.muonRef() ) ) // muon is not isolated
      continue;

    double totalCaloEnergy = cand.rawEcalEnergy() +  cand.rawHcalEnergy();
    double eOverP = totalCaloEnergy/cand.p();

    if( eOverP < eOverPMax_ )
      continue;

    foundMuon = true;

    pOutputCandidateCollection->push_back( cand );

    if( debug_ ) {
      cout<<cand<<" HCAL E="<<endl;
      cout<<"\t"<<"ECAL energy "<<cand.rawEcalEnergy()<<endl;
      cout<<"\t"<<"HCAL energy "<<cand.rawHcalEnergy()<<endl;
      cout<<"\t"<<"E/p "<<eOverP<<endl;
    }
  }

  iEvent.put( pOutputCandidateCollection, "muons" );

  bool pass = !foundMuon;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;

}

// ------------ method called once each job just before starting event loop  ------------
void
GreedyMuonPFCandidateFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
GreedyMuonPFCandidateFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(GreedyMuonPFCandidateFilter);
