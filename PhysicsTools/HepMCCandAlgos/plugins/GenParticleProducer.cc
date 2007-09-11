/* \class GenParticleProducer
 *
 * \author Luca Lista, INFN
 *
 * Convert HepMC GenEvent format into a collection of type
 * CandidateCollection containing objects of type GenParticle
 *
 * \version $Id: GenParticleProducer.cc,v 1.22 2007/06/19 17:52:55 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <vector>
#include <map>
#include <set>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }

class GenParticleProducer : public edm::EDProducer {
 public:
  /// constructor
  GenParticleProducer( const edm::ParameterSet & );
  /// destructor
  ~GenParticleProducer();

 private:
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  edm::InputTag src_;
  /// unknown code treatment flag
  bool abortOnUnknownPDGCode_;
  /// internal functional decomposition
  void fillIndices( const HepMC::GenEvent *, 
		    std::vector<const HepMC::GenParticle *>  &,
		    std::map<int, size_t> &) const;
  /// internal functional decomposition
  void fillOutput( const std::vector<const HepMC::GenParticle *> &,
		   reco::GenParticleCollection &, 
		   std::vector<reco::GenParticle *> & ) const;
  /// internal functional decomposition
  void fillRefs( const std::vector<const HepMC::GenParticle *> &,
		 const std::map<int, size_t> &,
		 const reco::GenParticleRefProd,
		 const std::vector<reco::GenParticle *> & ) const;
  /// charge indices
  std::vector<int> chargeP_, chargeM_;
  std::map<int, int> chargeMap_;
  int chargeTimesThree( int ) const;
};

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Handle.h"
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
static const double mmToCm = 0.1;

GenParticleProducer::GenParticleProducer( const ParameterSet & p ) :
  src_( p.getParameter<InputTag>( "src" ) ),
  abortOnUnknownPDGCode_( p.getUntrackedParameter<bool>( "abortOnUnknownPDGCode", true ) ),
  chargeP_( PDGCacheMax, 0 ), chargeM_( PDGCacheMax, 0 ) {
  produces<GenParticleCollection>();
}

GenParticleProducer::~GenParticleProducer() { 
}

int GenParticleProducer::chargeTimesThree( int id ) const {
  if( abs( id ) < PDGCacheMax ) 
    return id > 0 ? chargeP_[ id ] : chargeM_[ - id ];
  map<int, int>::const_iterator f = chargeMap_.find( id );
  if ( f == chargeMap_.end() ) 
    if ( abortOnUnknownPDGCode_ )
      throw edm::Exception( edm::errors::LogicError ) 
	<< "invalid PDG id: " << id << endl;
    else {
      return HepPDT::ParticleID(id).threeCharge();
    }
  return f->second;
}

void GenParticleProducer::beginJob( const EventSetup & es ) {
  ESHandle<HepPDT::ParticleDataTable> pdt;
  es.getData( pdt );
  for( HepPDT::ParticleDataTable::const_iterator p = pdt->begin(); p != pdt->end(); ++ p ) {
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

void GenParticleProducer::produce( Event& evt, const EventSetup& es ) {
  Handle<HepMCProduct> mcp;
  evt.getByLabel( src_, mcp );
  const GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  const size_t size = mc->particles_size();
  
  vector<const HepMC::GenParticle *> particles( size );
  map<int, size_t> barcodes;
  auto_ptr<GenParticleCollection> cands( new GenParticleCollection );
  const GenParticleRefProd ref = evt.getRefBeforePut<GenParticleCollection>();

  vector<reco::GenParticle *> candVector( size );
  /// fill indices
  fillIndices( mc, particles, barcodes );
  // fill output collection and save association
  fillOutput( particles, * cands, candVector );
  // fill references to daughters
  fillRefs( particles, barcodes, ref, candVector );

  evt.put( cands );
}

void GenParticleProducer::fillIndices( const GenEvent * mc,
				       vector<const HepMC::GenParticle *> & particles,
				       map<int, size_t> & barcodes ) const {
  GenEvent::particle_const_iterator begin = mc->particles_begin(), end = mc->particles_end();
  size_t idx = 0;
  for( GenEvent::particle_const_iterator p = begin; p != end; ++ p ) {
    const HepMC::GenParticle * particle = * p;
    size_t i = particle->barcode();
    if( barcodes.find(i) != barcodes.end() ) {
      throw cms::Exception( "WrongReference" )
	<< "barcodes are duplicated! " << endl;
    }
    particles[ idx ] = particle;
    barcodes.insert( make_pair( i, idx ++) );
  }
}

void GenParticleProducer::fillOutput( const vector<const HepMC::GenParticle *> & particles,
				      GenParticleCollection & cands, 
				      vector<reco::GenParticle *> & candVector ) const {
  const size_t size = particles.size();
  cands.reserve( size );
  for( size_t i = 0; i < size; ++ i ) {
    const HepMC::GenParticle * part = particles[ i ];
    Candidate::LorentzVector momentum( part->momentum() );
    Candidate::Point vertex( 0, 0, 0 );
    const GenVertex * v = part->production_vertex();
    if ( v != 0 ) {
      ThreeVector vtx = v->point3d();
      vertex.SetXYZ( vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm );
    }
    int pdgId = part->pdg_id();
    // this allocation can be optimized...
    reco::GenParticle c( chargeTimesThree( pdgId ), momentum, vertex, 
		   pdgId, part->status(), false );
    cands.push_back( c );
    candVector[ i ] = & cands.back();
  }
}

void GenParticleProducer::fillRefs( const std::vector<const HepMC::GenParticle *> & particles,
				    const std::map<int, size_t> & barcodes,
				    const GenParticleRefProd ref,
				    const vector<reco::GenParticle *> & candVector ) const {
  for( size_t d = 0; d < candVector.size(); ++ d ) {
    const HepMC::GenParticle * part = particles[ d ];
    const GenVertex * productionVertex = part->production_vertex();
    if ( productionVertex != 0 ) {
      size_t numberOfMothers = productionVertex->particles_in_size();
      if ( numberOfMothers > 0 ) {
        GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
        for( ; motherIt != productionVertex->particles_in_const_end(); motherIt++) {
          const HepMC::GenParticle * mother = * motherIt;
	  size_t m = barcodes.find( mother->barcode() )->second;
          candVector[ m ]->addDaughter( GenParticleRef( ref, d ) );  
        }
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleProducer );

