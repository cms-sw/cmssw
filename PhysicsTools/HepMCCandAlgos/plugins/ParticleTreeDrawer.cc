/* class ParticleTreeDrawer
 *
 * \author Luca Lista, INFN
 */
#include <sstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class ParticleTreeDrawer : public edm::EDAnalyzer {
public:
  ParticleTreeDrawer( const edm::ParameterSet & );
private:
  std::string getParticleName( int id ) const;
  void analyze( const edm::Event &, const edm::EventSetup&) override;
  edm::InputTag src_;
  void printDecay( const reco::Candidate &, const std::string & pre ) const;
  edm::ESHandle<ParticleDataTable> pdt_;
  /// print parameters
  bool printP4_, printPtEtaPhi_, printVertex_, printStatus_, printIndex_;
  /// accepted status codes
  typedef std::vector<int> vint;
  vint status_;
  /// print 4 momenta
  void printInfo( const reco::Candidate & ) const;
  /// accept candidate
  bool accept( const reco::Candidate & ) const;
  /// has valid daughters in the chain
  bool hasValidDaughters( const reco::Candidate & ) const;
  /// pointer to collection
  std::vector<const reco::Candidate *> cands_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
#include <algorithm>
using namespace std;
using namespace edm;
using namespace reco;

ParticleTreeDrawer::ParticleTreeDrawer( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  printP4_( cfg.getUntrackedParameter<bool>( "printP4", false ) ),
  printPtEtaPhi_( cfg.getUntrackedParameter<bool>( "printPtEtaPhi", false ) ),
  printVertex_( cfg.getUntrackedParameter<bool>( "printVertex", false ) ),
  printStatus_( cfg.getUntrackedParameter<bool>( "printStatus", false ) ),
  printIndex_( cfg.getUntrackedParameter<bool>( "printIndex", false ) ),
  status_( cfg.getUntrackedParameter<vint>( "status", vint() ) ) {
}

bool ParticleTreeDrawer::accept( const reco::Candidate & c ) const {
  if ( status_.size() == 0 ) return true;
  return find( status_.begin(), status_.end(), c.status() ) != status_.end();
}

bool ParticleTreeDrawer::hasValidDaughters( const reco::Candidate & c ) const {
  size_t ndau = c.numberOfDaughters();
  for( size_t i = 0; i < ndau; ++ i )
    if ( accept( * c.daughter( i ) ) )
      return true;
  return false;
}

std::string ParticleTreeDrawer::getParticleName(int id) const
{
  const ParticleData * pd = pdt_->particle( id );
  if (!pd) {
    std::ostringstream ss;
    ss << "P" << id;
    return ss.str();
  } else
    return pd->name();
}

void ParticleTreeDrawer::analyze( const Event & event, const EventSetup & es ) {  
  es.getData( pdt_ );
  Handle<View<Candidate> > particles;
  event.getByLabel( src_, particles );
  cands_.clear();
  for( View<Candidate>::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    cands_.push_back( & * p );
  }
  for( View<Candidate>::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    if ( accept( * p ) ) {
      if ( p->mother() == 0 ) {
	cout << "-- decay tree: --" << endl;
	printDecay( * p, "" );
      }
    }
  }
}

void ParticleTreeDrawer::printInfo( const Candidate & c ) const {
  if ( printP4_ ) cout << " (" << c.px() << ", " << c.py() << ", " << c.pz() << "; " << c.energy() << ")"; 
  if ( printPtEtaPhi_ ) cout << " [" << c.pt() << ": " << c.eta() << ", " << c.phi() << "]";
  if ( printVertex_ ) cout << " {" << c.vx() << ", " << c.vy() << ", " << c.vz() << "}";
  if ( printStatus_ ) cout << "{status: " << c.status() << "}";
  if ( printIndex_ ) {
    int idx = -1;
    vector<const Candidate *>::const_iterator found = find( cands_.begin(), cands_.end(), & c );
    if ( found != cands_.end() ) {
      idx = found - cands_.begin();
      cout << " <idx: " << idx << ">";
    }
  }
}

void ParticleTreeDrawer::printDecay( const Candidate & c, const string & pre ) const {
  cout << getParticleName( c.pdgId() );
  printInfo( c );
  cout << endl;

  size_t ndau = c.numberOfDaughters(), validDau = 0;
  for( size_t i = 0; i < ndau; ++ i )
    if ( accept( * c.daughter( i ) ) )
      ++ validDau;
  if ( validDau == 0 ) return;
  
  bool lastLevel = true;
  for( size_t i = 0; i < ndau; ++ i ) {
    if ( hasValidDaughters( * c.daughter( i ) ) ) {
      lastLevel = false;
      break;
    }      
  }
  
  if ( lastLevel ) {
    cout << pre << "+-> ";
    size_t vd = 0;
    for( size_t i = 0; i < ndau; ++ i ) {
      const Candidate * d = c.daughter( i );
      if ( accept( * d ) ) {
	cout << getParticleName( d->pdgId() );
	printInfo( * d );
	if ( vd != validDau - 1 )
	  cout << " ";
	vd ++;
      }
    }
    cout << endl;
    return;
  }

  for( size_t i = 0; i < ndau; ++i ) {
    const Candidate * d = c.daughter( i );
    assert( d != 0 );
    if ( accept( * d ) ) {
      cout << pre << "+-> ";
      string prepre( pre );
      if ( i == ndau - 1 ) prepre += "    ";
      else prepre += "|   ";
      printDecay( * d, prepre );
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ParticleTreeDrawer );


