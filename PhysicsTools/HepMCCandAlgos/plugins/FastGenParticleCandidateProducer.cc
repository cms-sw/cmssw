/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: FastGenParticleCandidateProducer.cc,v 1.5.2.1.2.2 2007/05/04 13:52:52 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"
#include <vector>
#include <map>
#include <set>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class FastGenParticleCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  FastGenParticleCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~FastGenParticleCandidateProducer();

 private:
  /// vector of strings
  typedef std::vector<std::string> vstring;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string src_;
  /// unknown code treatment flag
  bool abortOnUnknownPDGCode_;
  /// internal functional decomposition
  void fillIndices( const HepMC::GenEvent *, 
		    std::vector<const HepMC::GenParticle *> &,
		    std::map<int, size_t> & ) const;
  /// internal functional decomposition
  void fillOutput( const std::vector<const HepMC::GenParticle *> &,
		   reco::CandidateCollection &, 
		   std::vector<reco::GenParticleCandidate *> & ) const;
  /// internal functional decomposition
  void fillRefs( const std::vector<const HepMC::GenParticle *> &,
		 const std::map<int, size_t> &,
		 const reco::CandidateRefProd,
		 const std::vector<reco::GenParticleCandidate *> & ) const;
  /// charge indices
  std::vector<int> chargeP_, chargeM_;
  std::map<int, int> chargeMap_;
  int chargeTimesThree( int ) const;
};

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"
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
  abortOnUnknownPDGCode_( p.getUntrackedParameter<bool>( "abortOnUnknownPDGCode", true ) ),
  chargeP_( PDGCacheMax, 0 ), chargeM_( PDGCacheMax, 0 ) {
  produces<CandidateCollection>();
}

FastGenParticleCandidateProducer::~FastGenParticleCandidateProducer() { 
}

int FastGenParticleCandidateProducer::chargeTimesThree( int id ) const {
  if( abs( id ) < PDGCacheMax ) 
    return id > 0 ? chargeP_[ id ] : chargeM_[ - id ];
  map<int, int>::const_iterator f = chargeMap_.find( id );
  if ( f == chargeMap_.end() ) {
    throw edm::Exception( edm::errors::LogicError ) 
      << "invalid PDG id: " << id << endl;
  } else {
    return HepPDT::ParticleID(id).threeCharge();
  }
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
  Handle<HepMCProduct> mcp;
  evt.getByLabel( src_, mcp );
  const GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  const size_t size = mc->particles_size();

  vector<const GenParticle *> particles( size );
  map<int, size_t> barcodes;
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  const CandidateRefProd ref = evt.getRefBeforePut<CandidateCollection>();

  vector<GenParticleCandidate *> candVector( size );
  /// fill indices
  fillIndices( mc, particles, barcodes );
  // fill output collection and save association
  fillOutput( particles, * cands, candVector );
  // fill references to daughters
  fillRefs( particles, barcodes, ref, candVector );

  evt.put( cands );
}

void FastGenParticleCandidateProducer::fillIndices( const GenEvent * mc,
						    vector<const GenParticle *> & particles,
						    map<int, size_t> & barcodes ) const {
  GenEvent::particle_const_iterator begin = mc->particles_begin(), end = mc->particles_end();
  size_t idx = 0;
  for( GenEvent::particle_const_iterator p = begin; p != end; ++ p ) {
    const GenParticle * particle = * p;
    size_t i = particle->barcode();
    if( barcodes.find(i) != barcodes.end() ) {
      throw cms::Exception( "WrongReference" )
	<< "barcodes are duplicated! " << endl;
    }
    particles[ idx ] = particle;
    barcodes.insert( make_pair( i, idx ++) );
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
    int pdgId = part->pdg_id();
    GenParticleCandidate * c = 
      new GenParticleCandidate( chargeTimesThree( pdgId ) / 3, momentum, vertex, 
				pdgId, part->status() );
    auto_ptr<Candidate> ptr( c );
    candVector[ i ] = c;
    cands.push_back( ptr );
  }
}

void FastGenParticleCandidateProducer::fillRefs( const std::vector<const GenParticle *> & particles,
						 const std::map<int, size_t> & barcodes,
						 const CandidateRefProd ref,
						 const vector<GenParticleCandidate *> & candVector ) const {
  for( size_t d = 0; d < candVector.size(); ++ d ) {
    const GenParticle * part = particles[ d ];
    if ( part->hasParents() ) {
      const GenParticle * mother = part->mother();
      size_t m = barcodes.find( mother->barcode() )->second;
      candVector[ m ]->addDaughter( CandidateRef( ref, d ) );
      const GenParticle * mother2 = part->secondMother();
      if ( mother2 != 0 && mother2 != mother ) {
	size_t m = barcodes.find( mother2->barcode() )->second;
	candVector[ m ]->addDaughter( CandidateRef( ref, d ) );
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( FastGenParticleCandidateProducer );
