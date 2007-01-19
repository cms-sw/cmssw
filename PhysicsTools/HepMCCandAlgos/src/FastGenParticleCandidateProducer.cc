// $Id: FastGenParticleCandidateProducer.cc,v 1.2 2007/01/16 11:23:52 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/FastGenParticleCandidateProducer.h"
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

FastGenParticleCandidateProducer::FastGenParticleCandidateProducer( const ParameterSet & p ) :
  src_( p.getParameter<string>( "src" ) ),
  chargeP_( PDGCacheMax, 0 ), chargeM_( PDGCacheMax, 0 ) {
  produces<CandidateCollection>();
}

FastGenParticleCandidateProducer::~FastGenParticleCandidateProducer() { 
}

int FastGenParticleCandidateProducer::chargeTimesThree( int id ) const {
  if( id < PDGCacheMax ) 
    return id > 0 ? chargeP_[ id ] : chargeM_[ - id ];
  map<int, int>::const_iterator f = chargeMap_.find( id );
  if ( f == chargeMap_.end() )
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "invalid PDG id: " << id << endl;
  return f->second;
}

void FastGenParticleCandidateProducer::beginJob( const EventSetup & es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
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

void FastGenParticleCandidateProducer::produce( Event& evt, const EventSetup& es ) {
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
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  const CandidateRefProd ref = evt.getRefBeforePut<CandidateCollection>();

  vector<GenParticleCandidate *> candVector( size );
  /// fill indices
  fillIndices( mc, particles );
  // fill output collection and save association
  fillOutput( particles, * cands, candVector );
  // fill references to daughters
  fillRefs( particles, ref, candVector );

  evt.put( cands );
}

void FastGenParticleCandidateProducer::fillIndices( const GenEvent * mc,
						    vector<const GenParticle *> & particles ) const {
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

void FastGenParticleCandidateProducer::fillOutput( const std::vector<const GenParticle *> & particles,
						   CandidateCollection & cands, 
						   vector<GenParticleCandidate *> & candVector ) const {
  const size_t size = particles.size();
  cands.reserve( size );
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
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
    candVector[ i ] = c;
    cands.push_back( c );
  }
}

void FastGenParticleCandidateProducer::fillRefs( const std::vector<const GenParticle *> & particles,
						 const CandidateRefProd ref,
						 const vector<GenParticleCandidate *> & candVector ) const {
  for( size_t d = 0; d < candVector.size(); ++ d ) {
    const GenParticle * part = particles[ d ];
    if ( part->hasParents() ) {
      size_t m = part->mother()->barcode() - 1;
      candVector[ m ]->addDaughter( CandidateRef( ref, d ) );
    }
  }
}
