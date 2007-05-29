/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateProducer.cc,v 1.15 2007/05/17 15:30:53 loizides Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include <vector>
#include <map>
#include <set>

static const double mmToCm = 0.1;

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class GenParticleCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  GenParticleCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~GenParticleCandidateProducer();

 private:
  /// vector of strings
  typedef std::vector<PdtEntry> vpdt;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  edm::InputTag src_;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly_;
  /// exclude list
  vpdt excludeList_;
  /// set of excluded particle id's
  std::set<int> excludedIds_;
  /// minimum thresholds
  double ptMinNeutral_, ptMinCharged_, ptMinGluon_;
  /// keep initial protons
  bool keepInitialProtons_;
  /// suppress unfragmented partons (status=3) clones
  bool excludeUnfragmentedClones_;
  /// internal functional decomposition
  void fillIndices( const HepMC::GenEvent *, 
	     std::vector<const HepMC::GenParticle *> &, 
	     std::vector<int> &, std::vector<int> &, std::vector<std::vector<int> > & ) const;
  /// internal functional decomposition
  void fillVector( const HepMC::GenEvent *,
		   std::vector<const HepMC::GenParticle *> &
		   ) const;
  /// internal functional decomposition
  void fillMothers( const std::vector<const HepMC::GenParticle *> &,
		    std::vector<int> &, std::vector<int> & ) const;
  /// internal functional decomposition
  void fillDaughters( const std::vector<int> &, const std::vector<int> &, std::vector<std::vector<int> > & ) const;
  /// internal functional decomposition
  size_t fillSkip( const std::vector<const HepMC::GenParticle *> &, 
		   const std::vector<int> &, const std::vector<int> &, std::vector<bool> & ) const;
  /// internal functional decomposition
  void fix( const std::vector<const HepMC::GenParticle *> &,
	    const std::vector<int> &, const std::vector<int> &, 
	    const std::vector<std::vector<int> > &,
	    std::vector<bool> & ) const;
  /// internal functional decomposition
  void fillOutput( const std::vector<const HepMC::GenParticle *> &,
	     const std::vector<bool> &,
	     reco::CandidateCollection &,
	     std::vector<std::pair<reco::GenParticleCandidate *, size_t> > &,
	     std::vector<size_t> & ) const;
  /// internal functional decomposition
  void fillRefs( const std::vector<int> &, const std::vector<int> &, 
		 const reco::CandidateRefProd,
		 const std::vector<size_t> &,
		 const std::vector<std::pair<reco::GenParticleCandidate *, size_t> > &,
		 reco::CandidateCollection & ) const;
  /// charge indices
  std::vector<int> chargeP_, chargeM_;
  std::map<int, int> chargeMap_;
  int chargeTimesThree( int ) const;
};

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>
#include <algorithm>
using namespace edm;
using namespace reco;
using namespace std;
using namespace HepMC;

static const int protonId = 2212;
static const int gluonId = 21;
static const int uId = 1;
static const int tId = 6;
static const int stringId = 92;
static const int clusterId = 92;
static const int PDGCacheMax = 32768;

GenParticleCandidateProducer::GenParticleCandidateProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vpdt>( "excludeList" ) ),
  ptMinNeutral_( p.getParameter<double>( "ptMinNeutral" ) ),
  ptMinCharged_( p.getParameter<double>( "ptMinCharged" ) ),
  ptMinGluon_( p.getParameter<double>( "ptMinGluon" ) ),
  keepInitialProtons_( p.getParameter<bool>( "keepInitialProtons" ) ),
  excludeUnfragmentedClones_( p.getParameter<bool>( "excludeUnfragmentedClones" ) ),
  chargeP_( PDGCacheMax, 0 ), chargeM_( PDGCacheMax, 0 ) {
  produces<CandidateCollection>();
}

GenParticleCandidateProducer::~GenParticleCandidateProducer() { 
}

int GenParticleCandidateProducer::chargeTimesThree( int id ) const {
  if( abs( id ) < PDGCacheMax ) 
    return id > 0 ? chargeP_[ id ] : chargeM_[ - id ];
  map<int, int>::const_iterator f = chargeMap_.find( id );
  if ( f == chargeMap_.end() )
    throw edm::Exception( edm::errors::LogicError ) 
      << "invalid PDG id: " << id << endl;
  return f->second;
}

void GenParticleCandidateProducer::beginJob( const EventSetup & es ) {
  for( vpdt::iterator e = excludeList_.begin(); 
       e != excludeList_.end(); ++ e ) {
    e->setup( es );
    excludedIds_.insert( abs( e->pdgId() ) );
  }

  ESHandle<ParticleDataTable> pdt;
  es.getData( pdt );

  for( ParticleDataTable::const_iterator p = pdt->begin(); p != pdt->end(); ++ p ) {
    const HepPDT::ParticleID & id = p->first;
    int pdgId = id.pid(), apdgId = abs( pdgId );
    int q3 = id.threeCharge();
    if ( apdgId < PDGCacheMax ){
      chargeP_[ apdgId ] = q3;
      chargeM_[ apdgId ] = -q3;
    }else{
      chargeMap_[ pdgId ] = q3;
      chargeMap_[ -pdgId ] = -q3;
    } 
  }
}

void GenParticleCandidateProducer::produce( Event& evt, const EventSetup& es ) {
  ESHandle<ParticleDataTable> pdt;
  es.getData( pdt );

  Handle<HepMCProduct> mcp;
  evt.getByLabel( src_, mcp );
  const GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  const size_t size = mc->particles_size();

  vector<const GenParticle *> particles( size );
  vector<int> mothers( size ), mothers2( size );
  // need daughters vector since pointers in HepMC 
  // may be broken in some HepMC version
  vector<vector<int> > daughters( size );
  vector<bool> skip( size );
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  const CandidateRefProd ref = evt.getRefBeforePut<CandidateCollection>();
  vector<size_t> indices;
  vector<pair<GenParticleCandidate *, size_t> > candidates( size );

  /// fill indices
  fillIndices( mc, particles, mothers, mothers2, daughters );
  // fill skip vector
  if( fillSkip( particles, mothers, mothers2, skip ) > 0 )
    fix( particles, mothers, mothers2, daughters, skip );
  // fill output collection and save association
  fillOutput( particles, skip, * cands, candidates, indices );
  // fill references to daughters
  fillRefs( mothers, mothers2, ref, indices, candidates, * cands );

  evt.put( cands );
}

void GenParticleCandidateProducer::fillIndices( const GenEvent * mc,
						vector<const GenParticle *> & particles, 
						vector<int> & mothers, vector<int> & mothers2, 
						vector<vector<int> > & daughters ) const {
  // copy particle pointers
  fillVector( mc, particles );
  // fill mother indices
  fillMothers( particles, mothers, mothers2 );
  // fill daughters indices
  fillDaughters( mothers, mothers2, daughters );
}

void GenParticleCandidateProducer::fillVector( const GenEvent * mc,
					       vector<const GenParticle *> & particles) const {
  size_t idx = 0;
  for( GenEvent::particle_const_iterator p = mc->particles_begin(); 
       p != mc->particles_end(); ++ p ) {
    const GenParticle * particle = * p;
    size_t i = particle->barcode() - 1;
    if( i != idx ++ )
      throw cms::Exception( "WrongReference" )
	<< "barcodes is not properly ordered; got: " << i << " expected: " << idx ;
    particles[ i ] = particle;
  }
}

void GenParticleCandidateProducer::fillMothers( const vector<const HepMC::GenParticle *> & particles, 
						vector<int> & mothers,
						vector<int> & mothers2 ) const {
  const size_t size = particles.size();
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    const GenVertex * productionVertex = part-> production_vertex();
    if ( productionVertex == 0 ) {
      throw edm::Exception( edm::errors::InvalidReference ) 
	<< "particle has no production vertex. PDG id: " << part->pdg_id() << endl;
    } else {
      size_t numberOfMothers = productionVertex->particles_in_size();
      if ( numberOfMothers > 0 ) {
	GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
	const GenParticle * mother = * motherIt;
	mothers[ i ] = mother->barcode() - 1;
	if ( numberOfMothers > 1 ) {
	  ++ motherIt;
	  const GenParticle * mother2 = * motherIt;
	  mothers2[ i ] = mother2->barcode() - 1;
	} else {
          mothers2[ i ] = -1;
        }
      } else {
	mothers[ i ] = -1;
	mothers2[ i ] = -1;
      }
    }
  }
}

void  GenParticleCandidateProducer::fillDaughters( const vector<int> & mothers, 
						   const vector<int> & mothers2, 
						   vector<vector<int> > & daughters ) const {
  for( size_t i = 0; i < mothers.size(); ++ i ) {
    int mother = mothers[ i ];
    if ( mother != -1 )
      daughters[ mother ].push_back( i );
    int mother2 = mothers2[ i ];
    if ( mother2 != -1 )
      daughters[ mother2 ].push_back( i );
  } 
}

size_t GenParticleCandidateProducer::fillSkip( const vector<const GenParticle *> & particles, 
					       const vector<int> & mothers, 
					       const vector<int> & mothers2, 
					       vector<bool> & skip ) const {
  size_t tot = 0;
  const size_t size = particles.size();
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    const int pdgId = part->pdg_id();
    const int status = part->status();
    bool skipped = false;
    bool pass = false;
    /// keep initial protons anyway, if keepInitialProtons_ set
    if ( keepInitialProtons_ ) {
      bool initialProton = ( mothers[ i ] == -1 && pdgId == protonId );
      if ( initialProton ) pass = true;
    }
    if ( ! pass ) {
      /// skip unstable particles if stableOnly_ set
      if ( stableOnly_ && ! status == 1 ) skipped = true;
      /// skip particles with excluded id's
      else if ( excludedIds_.find( abs( pdgId ) ) != excludedIds_.end() ) skipped = true;
      /// apply minimun pt cuts on final state neutrals and charged
      else if ( status == 1 ) {
	if ( ptMinNeutral_ > 0 || ptMinCharged_ > 0 ) {
	  // --> this is slow! fix it using
	  if ( chargeTimesThree( pdgId ) == 0 ) {
	    if ( part->momentum().perp() < ptMinNeutral_ ) skipped = true;	  
	  }
	  else {
	    if ( part->momentum().perp() < ptMinCharged_ ) skipped = true;
	  }
	}
	/// apply minimum pt cut on gluons
      } else if ( ptMinGluon_ > 0 && pdgId == gluonId ) {
	if ( part->momentum().perp() < ptMinGluon_ ) skipped = true;
      }
    }
    skip[ i ] = skipped;
    if ( skipped ) ++ tot;
  }
  return tot;
}

void GenParticleCandidateProducer::fix( const vector<const GenParticle *> & particles,
					const vector<int> & mothers, const vector<int> & mothers2,
					const vector<vector<int> > & daughters,
					vector<bool> & skip ) const {
  const size_t size = particles.size();
  for( int i = size - 1; i >= 0; -- i ) {
    const GenParticle * part = particles[ i ];
    const int pdgId = part->pdg_id();
    const int status = part->status();
    if ( ! skip[ i ] ) {
      if ( status == 2 ) {
	/// skip unfragmented clones (status = 3 if excludeUnfragmentedClones_ set)
	if ( excludeUnfragmentedClones_ ) {
	  int m = mothers[ i ];
	  if( m != -1 ) {
	    const GenParticle * mother = particles[ m ];
	    if ( mother->status() == 3 && mother->pdg_id() == pdgId )
	      skip[ m ] = true;
	  }
	  int m2 = mothers2[ i ];
	  if( m2 != -1 ) {
	    const GenParticle * mother2 = particles[ m2 ];
	    if ( mother2->status() == 3 && mother2->pdg_id() == pdgId )
	      skip[ m2 ] = true;
	  }
	}
	/// drop mothers if all daughters dropped, but keep complete decays
	bool allDaughtersSkipped = true;
	const vector<int> & ds = daughters[ i ];
	for( vector<int>::const_iterator j = ds.begin(); j != ds.end(); ++ j ) {
	  if ( ! skip[ * j ] ) {
	    allDaughtersSkipped = false;
	    break;
	  }
	}
	bool canDropDaughters =
	  ( pdgId >= uId && pdgId <= tId ) || 
	  pdgId == stringId || pdgId == clusterId;
	if ( allDaughtersSkipped ) 
	  skip[ i ] = true;
	else if ( ! canDropDaughters ){
	  for( vector<int>::const_iterator j = ds.begin(); j != ds.end(); ++ j ) {
	    const GenParticle * dau = particles[ * j ];
	    if ( dau->status() == 1 )
	      skip[ * j ] = false;
	  }
	}	    
      }
    }
  }
}

void GenParticleCandidateProducer::fillOutput( const vector<const GenParticle *> & particles,
					       const vector<bool> & skip,
					       CandidateCollection & cands,
					       vector<pair<GenParticleCandidate *, size_t> > & candidates,
					       vector<size_t> & indices ) const {
  const size_t size = particles.size();
  cands.reserve( size );
  indices.reserve( size );
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    GenParticleCandidate * cand = 0;
    size_t index = 0;
    if ( ! skip[ i ] ) {
      Candidate::LorentzVector momentum( part->momentum() );
      Candidate::Point vertex( 0, 0, 0 );
      const HepMC::GenVertex * v = part->production_vertex();
      if ( v != 0 ) {
 	ThreeVector vtx = v->point3d();
	vertex.SetXYZ( vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm );
      }
      int pdgId = part->pdg_id();
      cand = new GenParticleCandidate( chargeTimesThree( pdgId ), momentum, vertex, 
				       pdgId, part->status(), false );
      auto_ptr<Candidate> ptr( cand );
      index = indices.size();
      cands.push_back( ptr );
      indices.push_back( i );
    }
    candidates[ i ] = make_pair( cand, index );
  }
  assert( candidates.size() == size );
  assert( cands.size() == indices.size() );
}

void GenParticleCandidateProducer::fillRefs( const vector<int> & mothers, const vector<int> & mothers2,
					     const CandidateRefProd ref,
					     const vector<size_t> & indices,
					     const vector<pair<GenParticleCandidate *, size_t> > & candidates,
					     reco::CandidateCollection & cands ) const {
  GenParticleCandidate * null = 0;
  for( size_t i = 0; i < cands.size(); ++ i ) {
    int m = mothers[ indices[ i ] ];
    pair<GenParticleCandidate *, size_t> mother = make_pair( null, 0 );
    while ( mother.first == 0 && m != -1 )
      if ( ( mother = candidates[ m ] ).first == 0 )
	m = ( m != -1 ) ? mothers[ m ] : -1;
    if ( mother.first != 0 ) 
      mother.first->addDaughter( CandidateRef( ref, i ) );

    int m2 = mothers2[ indices[ i ] ];
    pair<GenParticleCandidate *, size_t> mother2 = make_pair( null, 0 );
    while ( mother2.first == 0 && m2 != -1 )
      if ( ( mother2 = candidates[ m2 ] ).first == 0 )
	m2 = ( m2 != -1 ) ? mothers2[ m2 ] : -1;
    if ( mother2.first != 0 ) 
      mother2.first->addDaughter( CandidateRef( ref, i ) );
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleCandidateProducer );
