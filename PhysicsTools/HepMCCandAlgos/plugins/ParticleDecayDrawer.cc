/* class ParticleDecayDrawer
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class ParticleDecayDrawer : public edm::EDAnalyzer {
public:
  ParticleDecayDrawer( const edm::ParameterSet & );
private:
  void analyze( const edm::Event &, const edm::EventSetup&) override;
  edm::InputTag src_;
  std::string decay( const reco::Candidate &, std::list<const reco::Candidate *> & ) const;
  edm::ESHandle<ParticleDataTable> pdt_;
  /// print parameters
  bool printP4_, printPtEtaPhi_, printVertex_;
  /// print 4 momenta
  std::string printP4( const reco::Candidate & ) const;
  /// accept candidate
  bool accept( const reco::Candidate &, const std::list<const reco::Candidate *> & ) const;
  /// select candidate
  bool select( const reco::Candidate & ) const;
  /// has valid daughters in the chain
  bool hasValidDaughters( const reco::Candidate & ) const;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
#include <sstream> 
#include <algorithm>
using namespace std;
using namespace edm;
using namespace reco;

ParticleDecayDrawer::ParticleDecayDrawer( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  printP4_( cfg.getUntrackedParameter<bool>( "printP4", false ) ),
  printPtEtaPhi_( cfg.getUntrackedParameter<bool>( "printPtEtaPhi", false ) ),
  printVertex_( cfg.getUntrackedParameter<bool>( "printVertex", false ) ) {
}

bool ParticleDecayDrawer::accept( const reco::Candidate & c, const list<const Candidate *> & skip ) const {
  if( find( skip.begin(), skip.end(), & c ) != skip.end() ) return false;
  return select( c );
}

bool ParticleDecayDrawer::select( const reco::Candidate & c ) const {
  return c.status() == 3;
}

bool ParticleDecayDrawer::hasValidDaughters( const reco::Candidate & c ) const {
  size_t ndau = c.numberOfDaughters();
  for( size_t i = 0; i < ndau; ++ i )
    if ( select( * c.daughter( i ) ) )
      return true;
  return false;
}

void ParticleDecayDrawer::analyze( const Event & event, const EventSetup & es ) {  
  es.getData( pdt_ );
  Handle<View<Candidate> > particles;
  event.getByLabel( src_, particles );
  list<const Candidate *> skip;
  vector<const Candidate *> nodes, moms;
  for( View<Candidate>::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    if( p->numberOfMothers() > 1 ) {
      if ( select( * p ) ) {
	skip.push_back( & * p );
	nodes.push_back( & * p );
	for( size_t j = 0; j < p->numberOfMothers(); ++ j ) {
	  const Candidate * mom = p->mother( j );
	  const Candidate * grandMom;
	  while ( ( grandMom = mom->mother() ) != 0 )
	    mom = grandMom;
	  if ( select( * mom ) ) {
	    moms.push_back( mom );
	  }
	}
      }
    }
  }
  cout << "-- decay: --" << endl;
  if( moms.size() > 0 ) {
    if ( moms.size() > 1 )
      for( size_t m = 0; m < moms.size(); ++ m ) {
	string dec = decay( * moms[ m ], skip );
	if ( ! dec.empty() )
	  cout << "{ " << dec << " } ";
      }
    else 
      cout << decay( * moms[ 0 ], skip );
  }
  if ( nodes.size() > 0 ) {
    cout << "-> ";
    if ( nodes.size() > 1 ) {
      for( size_t n = 0; n < nodes.size(); ++ n ) {    
	skip.remove( nodes[ n ] );
	string dec = decay( * nodes[ n ], skip );
	if ( ! dec.empty() ) {
	  if ( dec.find( "->", 0 ) != string::npos )
	    cout << " ( " << dec << " )";
	  else 
	    cout << " " << dec;
        }
      }
    } else {
      skip.remove( nodes[ 0 ] );
      cout << decay( * nodes[ 0 ], skip );
    }
  }
  cout << endl;
}

string ParticleDecayDrawer::printP4( const Candidate & c ) const {
  ostringstream cout;
  if ( printP4_ ) cout << " (" << c.px() << ", " << c.py() << ", " << c.pz() << "; " << c.energy() << ")"; 
  if ( printPtEtaPhi_ ) cout << " [" << c.pt() << ": " << c.eta() << ", " << c.phi() << "]";
  if ( printVertex_ ) cout << " {" << c.vx() << ", " << c.vy() << ", " << c.vz() << "}";
  return cout.str();
}

string ParticleDecayDrawer::decay( const Candidate & c, 
				   list<const Candidate *> & skip ) const {
  string out;
  if ( find( skip.begin(), skip.end(), & c ) != skip.end() ) 
    return out;
  skip.push_back( & c );

  
  int id = c.pdgId();
  const ParticleData * pd = pdt_->particle( id );  
  assert( pd != 0 );
  out += ( pd->name() + printP4( c ) );
  
  size_t validDau = 0, ndau = c.numberOfDaughters();
  for( size_t i = 0; i < ndau; ++ i )
    if ( accept( * c.daughter( i ), skip ) )
      ++ validDau;
  if ( validDau == 0 ) return out;

  out += " ->";
  
  for( size_t i = 0; i < ndau; ++ i ) {
    const Candidate * d = c.daughter( i );
    if ( accept( * d, skip ) ) {
      string dec = decay( * d, skip );
      if ( dec.find( "->", 0 ) != string::npos )
	out += ( " ( " + dec + " )" );
      else
	out += ( " " + dec );
    }
  }
  return out;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ParticleDecayDrawer );


