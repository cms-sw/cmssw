#include "__subsys__/__plgname__/plugins/__class__Producer.h"

#include <iostream>

using namespace pat;
using namespace std;

__class__Producer::__class__Producer(const edm::ParameterSet & iConfig) : 
  public edm::EDProducer(iConfig)
{

  // Here we get the list of common includes
  muonSrc_      = iConfig.getParameter<edm::InputTag>( "muonSource"     );
  electronSrc_  = iConfig.getParameter<edm::InputTag>( "electronSource" );
  tauSrc_       = iConfig.getParameter<edm::InputTag>( "tauSource"      );
  photonSrc_    = iConfig.getParameter<edm::InputTag>( "photonSource"   );
  jetSrc_       = iConfig.getParameter<edm::InputTag>( "jetSource"      );
  metSrc_       = iConfig.getParameter<edm::InputTag>( "metSource"      );

  // Here we get the output tag name
  outputName_   = iConfig.getParameter<edm::OutputTag>("outputName");
  
  // This declares the output to the event stream
  string alias;
  produces<std::vector<__class__> >(alias = outputName_).setBranchAddress(alias);
}

__class__Producer::~__class__Producer()
{
}

__class__Producer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup )
{

  // Here is the vector where you should put your hypotheses
  std::vector<__class__ hyps;
  

  // ------------------------------------------------------------------------
  // Here is where you get the objects you need. The perl script will
  // uncomment any that you need by reading the event hypothesis text file
  // ------------------------------------------------------------------------

  edm::Handle<std::vector<pat::Muon> >     muons;
  iEvent.getByLabel(muonSrc_,              muons);

  edm::Handle<std::vector<pat::Electron> > electrons;
  iEvent.getByLabel(electronSrc_,          electrons);

  edm::Handle<std::vector<pat::Tau> >      taus;
  iEvent.getByLabel(tauSrc_,               taus);

  edm::Handle<std::vector<pat::Photon> >   photons;
  iEvent.getByLabel(photonSrc_,            photons);

  edm::Handle<std::vector<pat::Jet> >      jets;
  iEvent.getByLabel(jetSrc_,               jets);

  edm::Handle<std::vector<pat::Met> >      mets;
  iEvent.getByLabel(metSrc_,               mets);

  
  // ------------------------------------------------------------------------
  // ****** Here is where you put your event hypothesis code ******
  // ------------------------------------------------------------------------
  // A: Define a combinatorics loop.
  // Replace for ( ...;...;...) with your appropriate loop over objects, such as
  // for ( vector<Muon>::iterator imuon = muons->begin(); imuon != muons->end(); imuon++ ) 
  //    for ( vector<Muon>::iterator jmuon = imuon + 1; jmuon != muons->end(); jmuon++ )
  //   
  for ( ...; ...; ... ) {
    __class__ hyp;

    // B: Fill "hyp" with your hypothesis information and push it back to the
    // vector containing them.
    // For instance, 
    // hyp.muon1() = *imuon;
    // hyp.muon2() = *jmuon;
    // hyp.jets() = *jets;

    hyps.push_back( hyp );
  }  


  // Here is where we write the hypotheses to the event stream
  std::auto_ptr<std::vector<__class__> > ap_hyps( hyps );
  iEvent.put( ap_hyps, outputName_);

}
