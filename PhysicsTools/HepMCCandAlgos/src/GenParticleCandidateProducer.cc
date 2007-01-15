// $Id: GenParticleCandidateProducer.cc,v 1.14 2006/12/07 18:35:51 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateProducer.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>
#include <algorithm>
#include <iostream>
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
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vstring>( "excludeList" ) ),
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
  if( id < PDGCacheMax ) 
    return id > 0 ? chargeP_[ id ] : chargeM_[ - id ];
  map<int, int>::const_iterator f = chargeMap_.find( id );
  if ( f == chargeMap_.end() )
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "invalid PDG id: " << id << endl;
  return f->second;
}

void GenParticleCandidateProducer::beginJob( const EventSetup & es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  
  for( vstring::const_iterator e = excludeList_.begin(); 
       e != excludeList_.end(); ++ e ) {
    const DefaultConfig::ParticleData * p = pdt->particle( * e );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError" )
	<< "can't find particle: " << * e;
    excludedIds_.insert( abs( p->pid() ) );
  }

  for( DefaultConfig::ParticleDataTable::const_iterator p = pdt->begin(); p != pdt->end(); ++ p ) {
    const HepPDT::ParticleID & id = p->first;
    int pdgId = id.pid(), apdgId = abs( pdgId );
    int q3 = id.threeCharge();
    if ( apdgId < PDGCacheMax )
      if ( pdgId > 0 )
	chargeP_[ apdgId ] = q3;
      else
	chargeM_[ apdgId ] = q3;
    else
      chargeMap_[ pdgId ] = q3;
  }
}

void GenParticleCandidateProducer::produce( Event& evt, const EventSetup& es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );

  Handle<HepMCProduct> mcp;
  evt.getByLabel( src_, mcp );
  const GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  const size_t size = mc->particles_size();

  vector<const GenParticle *> particles( size );
  vector<int> mothers( size );
  // need daughters vector since pointers in HepMC 
  // may be broken in some HepMC version
  vector<vector<int> > daughters( size );
  vector<bool> skip( size );
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  const CandidateRefProd ref = evt.getRefBeforePut<CandidateCollection>();
  vector<size_t> indices;
  vector<pair<GenParticleCandidate *, size_t> > candidates( size );

  /// fill indices
  fillIndices( mc, particles, mothers, daughters );
  // fill skip vector
  fillSkip( particles, mothers, skip );
  // reverse particle order to avoir recursive calls
  fix( particles, mothers, daughters, skip );
  // fill output collection and save association
  fillOutput( particles, skip, * cands, candidates, indices );
  // fill references to daughters
  fillRefs( mothers, ref, indices, candidates, * cands );

  evt.put( cands );
}

void GenParticleCandidateProducer::fillIndices( const GenEvent * mc,
						vector<const GenParticle *> & particles, 
						vector<int> & mothers, 
						vector<vector<int> > & daughters ) const {
  // copy particle pointers
  fillVector( mc, particles );
  // fill mother indices
  fillMothers( particles, mothers );
  // fill daughters indices
  fillDaughters( mothers, daughters );
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

void GenParticleCandidateProducer::fillMothers( const std::vector<const HepMC::GenParticle *> & particles, 
						std::vector<int> & mothers ) const {
  const size_t size = particles.size();
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    if ( part->hasParents() ) {
      const GenParticle * mother = part->mother();
      mothers[ i ] = mother->barcode() - 1;
    } else {
      mothers[ i ] = -1;
    }
  }
}

void  GenParticleCandidateProducer::fillDaughters( const std::vector<int> & mothers, 
					  std::vector<std::vector<int> > & daughters ) const {
  for( size_t i = 0; i < mothers.size(); ++ i ) {
    int mother = mothers[ i ];
    if ( mother != -1 )
      daughters[ mother ].push_back( i );
  } 
}

void GenParticleCandidateProducer::fillSkip( const vector<const GenParticle *> & particles, 
					     const vector<int> & mothers, 
					     vector<bool> & skip ) const {
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
      } else if ( pdgId == gluonId ) {
	if ( part->momentum().perp() < ptMinGluon_ ) skipped = true;
      }
    }
    skip[ i ] = skipped;
  }
}

void GenParticleCandidateProducer::fix( const vector<const GenParticle *> & particles,
					const vector<int> & mothers,
					const vector<vector<int> > & daughters,
					vector<bool> & skip ) const {
  if ( skip.size() == 0 ) return;
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

void GenParticleCandidateProducer::fillOutput( const std::vector<const GenParticle *> & particles,
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
      CLHEP::HepLorentzVector p4 =part->momentum();
      Candidate::LorentzVector momentum( p4.x(), p4.y(), p4.z(), p4.t() );
      Candidate::Point vertex( 0, 0, 0 );
      const HepMC::GenVertex * v = part->production_vertex();
      if ( v != 0 ) {
	HepGeom::Point3D<double> vtx = v->point3d();
	vertex.SetXYZ( vtx.x() / 10. , vtx.y() / 10. , vtx.z() / 10. );
      }
      int pdgId = part->pdg_id(), status = part->status();
      int q = chargeTimesThree( pdgId ) / 3;
      GenParticleCandidate * c = new GenParticleCandidate( q, momentum, vertex, pdgId, status );
      cand = c;
      index = indices.size();
      cands.push_back( c );
      indices.push_back( i );
    }
    candidates[ i ] = make_pair( cand, index );
  }
  assert( candidates.size() == size );
  assert( cands.size() == indices.size() );
}

void GenParticleCandidateProducer::fillRefs( const std::vector<int> & mothers,
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
    if ( mother.first != 0 ) {
      CandidateRef candRef( ref, i );
      mother.first->addDaughter( candRef );
      GenParticleCandidate & c = dynamic_cast<GenParticleCandidate &>( cands[ i ] );
      c.setMotherRef( CandidateRef( ref, mother.second ) );
    }
  }
}
