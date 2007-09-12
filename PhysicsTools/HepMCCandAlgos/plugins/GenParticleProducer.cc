/* \class GenParticleProducer
 *
 * \author Luca Lista, INFN
 *
 * Convert HepMC GenEvent format into a collection of type
 * CandidateCollection containing objects of type GenParticle
 *
 * \version $Id: GenParticleProducer.cc,v 1.1 2007/09/11 16:17:42 llista Exp $
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
  auto_ptr<GenParticleCollection> candsPtr( new GenParticleCollection( size ) );
  const GenParticleRefProd ref = evt.getRefBeforePut<GenParticleCollection>();
  GenParticleCollection & cands = * candsPtr;

  /// fill indices
  GenEvent::particle_const_iterator begin = mc->particles_begin(), end = mc->particles_end();
  size_t idx = 0;
  for( GenEvent::particle_const_iterator p = begin; p != end; ++ p ) {
    const HepMC::GenParticle * particle = * p;
    size_t i = particle->barcode();
    if( barcodes.find(i) != barcodes.end() )
      throw cms::Exception( "WrongReference" )
	<< "barcodes are duplicated! " << endl;
    particles[ idx ] = particle;
    barcodes.insert( make_pair(i, idx ++) );
  }

  // fill output collection and save association
  for( size_t i = 0; i < particles.size(); ++ i ) {
    const HepMC::GenParticle * part = particles[ i ];
    Candidate::LorentzVector p4( part->momentum() );
    Candidate::Point vertex( 0, 0, 0 );
    const GenVertex * v = part->production_vertex();
    if ( v != 0 ) {
      ThreeVector vtx = v->point3d();
      vertex.SetXYZ( vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm );
    }
    int pdgId = part->pdg_id();
    reco::GenParticle & cand = cands[ i ];
    cand.setP4( p4 );
    cand.setVertex( vertex );
    cand.setPdgId( pdgId );
    cand.setStatus( part->status() );
    cand.setThreeCharge( chargeTimesThree( pdgId ) );
  }

  // fill references to daughters
  for( size_t d = 0; d < cands.size(); ++ d ) {
    const HepMC::GenParticle * part = particles[ d ];
    const GenVertex * productionVertex = part->production_vertex();
    if ( productionVertex != 0 ) {
      size_t numberOfMothers = productionVertex->particles_in_size();
      if ( numberOfMothers > 0 ) {
        GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
        for( ; motherIt != productionVertex->particles_in_const_end(); motherIt++) {
          const HepMC::GenParticle * mother = * motherIt;
	  size_t m = barcodes.find( mother->barcode() )->second;
          cands[ m ].addDaughter( GenParticleRef( ref, d ) );  
        }
      }
    }
  }

  evt.put( candsPtr );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleProducer );

