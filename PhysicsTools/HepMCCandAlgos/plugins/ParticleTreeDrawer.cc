/* class ParticleTreeDrawer
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class ParticleTreeDrawer : public edm::EDAnalyzer {
public:
  ParticleTreeDrawer( const edm::ParameterSet & );
private:
  void analyze( const edm::Event &, const edm::EventSetup & );
  edm::InputTag src_;
  void printDecay( const reco::Candidate &, const std::string & pre ) const;
  edm::ESHandle<DefaultConfig::ParticleDataTable> pdt_;
  /// print parameters
  bool printP4_, printPtEtaPhi_, printVertex_, printStatus_;
  /// print 4 momenta
  void printP4( const reco::Candidate & ) const;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
#include <sstream>
using namespace std;
using namespace edm;
using namespace reco;
using namespace HepMC;

ParticleTreeDrawer::ParticleTreeDrawer( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  printP4_( cfg.getUntrackedParameter<bool>( "printP4", false ) ),
  printPtEtaPhi_( cfg.getUntrackedParameter<bool>( "printPtEtaPhi", false ) ),
  printVertex_( cfg.getUntrackedParameter<bool>( "printVertex", false ) ),
  printStatus_( cfg.getUntrackedParameter<bool>( "printStatus", false ) ) {
}

void ParticleTreeDrawer::analyze( const Event & event, const EventSetup & es ) {  
  es.getData( pdt_ );
  Handle<CandidateCollection> particles;
  event.getByLabel( src_, particles );
  for( CandidateCollection::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    if ( p->mother() == 0 ) {
      cout << "-- decay: --" << endl;
      printDecay( * p, "" );
    }
  }
}

void ParticleTreeDrawer::printP4( const reco::Candidate & c ) const {
  if ( printP4_ ) cout << " (" << c.px() << ", " << c.py() << ", " << c.pz() << "; " << c.energy() << ")"; 
  if ( printPtEtaPhi_ ) cout << " [" << c.pt() << ": " << c.eta() << ", " << c.phi() << "]";
  if ( printVertex_ ) cout << " {" << c.vx() << ", " << c.vy() << ", " << c.vz() << "}";
  if ( printStatus_ ) cout << "{status: " << status( c ) << "}";
}

void ParticleTreeDrawer::printDecay( const reco::Candidate & c, const std::string & pre ) const {
  int id = c.pdgId();
  unsigned int ndau = c.numberOfDaughters();
  const DefaultConfig::ParticleData * pd = pdt_->particle( id );  
  assert( pd != 0 );

  cout << pd->name(); 
  printP4( c );
  cout << endl;

  if ( ndau == 0 ) return;

  bool lastLevel = true;
  for( size_t i = 0; i < ndau; ++ i )
    if ( c.daughter( i )->numberOfDaughters() != 0 ) {
      lastLevel = false;
      break;
    }      

  if ( lastLevel ) {
    cout << pre << "+-> ";
    for( size_t i = 0; i < ndau; ++ i ) {
      const GenParticleCandidate * d = 
	dynamic_cast<const GenParticleCandidate *>( c.daughter( i ) );
      assert( d != 0 );
      const DefaultConfig::ParticleData * pd = pdt_->particle( d->pdgId() );  
      assert( pd != 0 );
      cout << pd->name();
      printP4( * d );
      if ( i != ndau - 1 )
	cout << " ";
    }
    cout << endl;
    return;
  }

  for( size_t i = 0; i < ndau; ++i ) {
    const GenParticleCandidate * d =
      dynamic_cast<const GenParticleCandidate *>( c.daughter( i ) );
    cout << pre << "+-> ";
    string prepre( pre );
    if ( i == ndau - 1 ) prepre += "    ";
    else prepre += "|   ";
    printDecay( * d, prepre );
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ParticleTreeDrawer );

