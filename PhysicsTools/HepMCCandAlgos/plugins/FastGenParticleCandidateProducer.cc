/* \class FastGenParticleCandidateProducer
 *
 * \author Luca Lista, INFN
 *
 * Convert HepMC GenEvent format into a collection of type
 * CandidateCollection containing objects of type GenParticleCandidate
 *
 * \version $Id: FastGenParticleCandidateProducer.cc,v 1.24 2007/10/19 15:17:18 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include <vector>
#include <map>
#include <set>

namespace edm { class ParameterSet; }
namespace HepMC { class GenParticle; class GenEvent; }
namespace reco { class GenParticleCandidate; }

class FastGenParticleCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  FastGenParticleCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~FastGenParticleCandidateProducer();

 private:
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  edm::InputTag src_;
  /// unknown code treatment flag
  bool abortOnUnknownPDGCode_;
  /// save bar-codes
  bool saveBarCodes_;
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
using namespace edm;
using namespace reco;
using namespace std;
using namespace HepMC;

static const int PDGCacheMax = 32768;
static const double mmToCm = 0.1;

FastGenParticleCandidateProducer::FastGenParticleCandidateProducer( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  abortOnUnknownPDGCode_( cfg.getUntrackedParameter<bool>( "abortOnUnknownPDGCode", true ) ),
  saveBarCodes_( cfg.getUntrackedParameter<bool>( "saveBarCodes", false ) ),
  chargeP_( PDGCacheMax, 0 ), chargeM_( PDGCacheMax, 0 ) {
  produces<CandidateCollection>();
  if( saveBarCodes_ ) {
    std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
    produces<vector<int> >().setBranchAlias( alias + "BarCodes" );
  }				  
}

FastGenParticleCandidateProducer::~FastGenParticleCandidateProducer() { 
}

int FastGenParticleCandidateProducer::chargeTimesThree( int id ) const {
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

void FastGenParticleCandidateProducer::beginJob( const EventSetup & es ) {
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
  auto_ptr<vector<int> > barCodeVector( new vector<int>( size ) );
  const CandidateRefProd ref = evt.getRefBeforePut<CandidateCollection>();

  vector<GenParticleCandidate *> candVector( size );
  /// fill indices
  GenEvent::particle_const_iterator begin = mc->particles_begin(), end = mc->particles_end();
  size_t idx = 0;
  for( GenEvent::particle_const_iterator p = begin; p != end; ++ p ) {
    const GenParticle * particle = * p;
    size_t barCode = particle->barcode();
    if( barcodes.find(barCode) != barcodes.end() )
      throw cms::Exception( "WrongReference" )
	<< "barcodes are duplicated! " << endl;
    particles[ idx ] = particle;
    (*barCodeVector)[idx] = barCode;
    barcodes.insert( make_pair( barCode, idx ++) );
  }

  // fill output collection and save association
  cands->reserve( size );
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    Candidate::LorentzVector momentum( part->momentum() );
    Candidate::Point vertex( 0, 0, 0 );
    const GenVertex * v = part->production_vertex();
    if ( v != 0 ) {
      ThreeVector vtx = v->point3d();
      vertex.SetXYZ( vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm );
    }
    int pdgId = part->pdg_id();
    GenParticleCandidate * c = 
      new GenParticleCandidate( chargeTimesThree( pdgId ), momentum, vertex, 
				pdgId, part->status(), false );
    auto_ptr<Candidate> ptr( c );
    candVector[ i ] = c;
    cands->push_back( ptr );
  }

  // fill references to daughters
  for( size_t d = 0; d < candVector.size(); ++ d ) {
    const GenParticle * part = particles[ d ];
    const GenVertex * productionVertex = part->production_vertex();
    if ( productionVertex != 0 ) {
      size_t numberOfMothers = productionVertex->particles_in_size();
      if ( numberOfMothers > 0 ) {
        GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
        for( ; motherIt != productionVertex->particles_in_const_end(); motherIt++) {
          const GenParticle * mother = * motherIt;
	  size_t m = barcodes.find( mother->barcode() )->second;
          candVector[ m ]->addDaughter( CandidateRef( ref, d ) );  
          candVector[ d ]->addMother( CandidateRef( ref, m ) );
        }
      }
    }
  }

  evt.put( cands );
  if(saveBarCodes_) evt.put( barCodeVector );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( FastGenParticleCandidateProducer );

