/* \class GenParticleProducer
 *
 * \author Luca Lista, INFN
 *
 * Convert HepMC GenEvent format into a collection of type
 * CandidateCollection containing objects of type GenParticle
 *
 * \version $Id: GenParticleProducer.cc,v 1.20 2013/02/27 23:16:51 wmtan Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <vector>
#include <string>
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

  /// process one event
  virtual void produce( edm::Event& e, const edm::EventSetup&) override;
  int chargeTimesThree( int ) const;
  bool convertParticle(reco::GenParticle& cand, const HepMC::GenParticle * part);
  bool fillDaughters(reco::GenParticleCollection& cand, const HepMC::GenParticle * part, size_t index);
  bool fillIndices(const HepMC::GenEvent * mc, std::vector<const HepMC::GenParticle*>& particles, std::vector<int>& barCodeVector, int offset);
  std::map<int, size_t> barcodes_;
  reco::GenParticleRefProd ref_;

 private:
  /// source collection name  
  edm::InputTag src_;
  std::vector<std::string> vectorSrc_;
  std::string mixLabel_;

  /// whether the first event was looked at
  bool firstEvent_; 
  /// unknown code treatment flag
  bool abortOnUnknownPDGCode_;
  /// save bar-codes
  bool saveBarCodes_;
  /// charge indices
  std::vector<int> chargeP_, chargeM_;
  std::map<int, int> chargeMap_;

  /// input & output modes
  bool doSubEvent_;
  bool useCF_;

};

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//#include "SimDataFormats/HiGenData/interface/SubEventMap.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

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

GenParticleProducer::GenParticleProducer( const ParameterSet & cfg ) :
  firstEvent_(true), 
  abortOnUnknownPDGCode_( cfg.getUntrackedParameter<bool>( "abortOnUnknownPDGCode", true ) ),
  saveBarCodes_( cfg.getUntrackedParameter<bool>( "saveBarCodes", false ) ),
  chargeP_( PDGCacheMax, 0 ), chargeM_( PDGCacheMax, 0 ),
  doSubEvent_(cfg.getUntrackedParameter<bool>( "doSubEvent", false )),
  useCF_(cfg.getUntrackedParameter<bool>( "useCrossingFrame", false ))
{
  produces<GenParticleCollection>();
  if( saveBarCodes_ ) {
    std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
    produces<vector<int> >().setBranchAlias( alias + "BarCodes" );
  }				  

  if(doSubEvent_){
     vectorSrc_ = cfg.getParameter<std::vector<std::string> >( "srcVector" );
     //     produces<SubEventMap>();
  }else if(useCF_) {
    mixLabel_ = cfg.getParameter<std::string>( "mix" );
    src_ = cfg.getUntrackedParameter<InputTag>( "src" , InputTag(mixLabel_,"generator"));
  } else src_ = cfg.getParameter<InputTag>( "src" );
}

GenParticleProducer::~GenParticleProducer() { 
}

int GenParticleProducer::chargeTimesThree( int id ) const {
  if( std::abs( id ) < PDGCacheMax ) 
    return id > 0 ? chargeP_[ id ] : chargeM_[ - id ];
  map<int, int>::const_iterator f = chargeMap_.find( id );
  if ( f == chargeMap_.end() )  {
    if ( abortOnUnknownPDGCode_ )
      throw edm::Exception( edm::errors::LogicError ) 
	<< "invalid PDG id: " << id << endl;
    else
      return HepPDT::ParticleID(id).threeCharge();
  }
  return f->second;
}

void GenParticleProducer::produce( Event& evt, const EventSetup& es ) {

  if (firstEvent_) {
     ESHandle<HepPDT::ParticleDataTable> pdt;
     es.getData( pdt );
     for( HepPDT::ParticleDataTable::const_iterator p = pdt->begin(); p != pdt->end(); ++ p ) {
       const HepPDT::ParticleID & id = p->first;
       int pdgId = id.pid(), apdgId = std::abs( pdgId );
       int q3 = id.threeCharge();
       if ( apdgId < PDGCacheMax && pdgId > 0 ) {
	 chargeP_[ apdgId ] = q3;
	 chargeM_[ apdgId ] = -q3;
       } else if ( apdgId < PDGCacheMax ) {
	 chargeP_[ apdgId ] = -q3;
	 chargeM_[ apdgId ] = q3;
       } else {
	 chargeMap_[ pdgId ] = q3;
	 chargeMap_[ -pdgId ] = -q3;
       }
     }
     firstEvent_ = false; 
   }
      
   barcodes_.clear();
   
   size_t totalSize = 0;
   const GenEvent * mc = 0;   
   std::vector<Handle<HepMCProduct> > heps;
   MixCollection<HepMCProduct>* cfhepmcprod = 0;
   size_t npiles = vectorSrc_.size();

   if(useCF_){
      Handle<CrossingFrame<HepMCProduct> > cf;
      evt.getByLabel(InputTag(mixLabel_,"generator"),cf);
      cfhepmcprod = new MixCollection<HepMCProduct>(cf.product());
      npiles = cfhepmcprod->size();
      for(unsigned int icf = 0; icf < npiles; ++icf){
	 totalSize += cfhepmcprod->getObject(icf).GetEvent()->particles_size();
      }
   }else if (doSubEvent_){
      for(size_t i = 0; i < npiles; ++i){
	//	 cout<<"Tag "<<vectorSrc_[i]<<endl;
	 Handle<HepMCProduct> handle;
	 heps.push_back(handle);
	 evt.getByLabel( vectorSrc_[i], heps[i] );
	 totalSize += heps[i]->GetEvent()->particles_size();
      }
   }else{
      Handle<HepMCProduct> mcp;
      evt.getByLabel( src_, mcp );
      mc = mcp->GetEvent();
      if( mc == 0 ) 
	 throw edm::Exception( edm::errors::InvalidReference ) 
	    << "HepMC has null pointer to GenEvent" << endl;
      totalSize  = mc->particles_size();
   }
      
   // initialise containers 
   const size_t size = totalSize;
  vector<const HepMC::GenParticle *> particles( size );
  auto_ptr<GenParticleCollection> candsPtr( new GenParticleCollection( size ) );
  //  auto_ptr<SubEventMap> subsPtr( new SubEventMap() );
  auto_ptr<vector<int> > barCodeVector( new vector<int>( size ) );
  ref_ = evt.getRefBeforePut<GenParticleCollection>();
  GenParticleCollection & cands = * candsPtr;
  //  SubEventMap & subs = *subsPtr;
  size_t offset = 0;
  size_t suboffset = 0;

  /// fill indices
  if(doSubEvent_ || useCF_){
     for(size_t i = 0; i < npiles; ++i){
	barcodes_.clear();
	if(useCF_) mc = cfhepmcprod->getObject(i).GetEvent();
	else mc = heps[i]->GetEvent();

	//Look whether heavy ion/signal event
	bool isHI = false;
	const HepMC::HeavyIon * hi = mc->heavy_ion();
	if(hi && hi->Ncoll_hard() > 1) isHI = true;
	size_t num_particles = mc->particles_size();
	fillIndices(mc, particles, *barCodeVector, offset);
	// fill output collection and save association 
	for( size_t i = offset; i < offset + num_particles; ++ i ) {

	   const HepMC::GenParticle * part = particles[ i ];
	   reco::GenParticle & cand = cands[ i ];
	   // convert HepMC::GenParticle to new reco::GenParticle
	   convertParticle(cand, part);
	   cand.resetDaughters( ref_.id() );
	}

	for( size_t d = offset; d < offset + num_particles; ++ d ) {
	   const HepMC::GenParticle * part = particles[ d ];
	   const GenVertex * productionVertex = part->production_vertex();
	   int sub_id = 0;
	   if ( productionVertex != 0 ) {
	      sub_id = productionVertex->id();
	      if(!isHI) sub_id = 0;
	      // search barcode map and attach daughters 
	      fillDaughters(cands,part,d);
	   }else{
	      const GenVertex * endVertex = part->end_vertex();
	      if(endVertex != 0) sub_id = endVertex->id();
	      else throw cms::Exception( "SubEventID" )<<"SubEvent not determined. Particle has no production and no end vertex!"<<endl;
	   }
	   if(sub_id < 0) sub_id = 0;
	   int new_id = sub_id + suboffset;
	   GenParticleRef dref( ref_, d );
	   //	   subs.insert(dref,new_id);   // For SubEventMap
	   cands[d].setCollisionId(new_id); // For new GenParticle
	   LogDebug("VertexId")<<"SubEvent offset 3 : "<<suboffset;
	}
	int nsub = -2;
	if(isHI){
	   nsub = hi->Ncoll_hard()+1;
	   suboffset += nsub;
	}else{
	   suboffset += 1;
	}
	offset += num_particles;
     }
  }else{
     fillIndices(mc, particles, *barCodeVector, 0);
     
     // fill output collection and save association
     for( size_t i = 0; i < particles.size(); ++ i ) {
	const HepMC::GenParticle * part = particles[ i ];
	reco::GenParticle & cand = cands[ i ];
	// convert HepMC::GenParticle to new reco::GenParticle
	convertParticle(cand, part);
	cand.resetDaughters( ref_.id() );
     }
     
     // fill references to daughters
     for( size_t d = 0; d < cands.size(); ++ d ) {
	const HepMC::GenParticle * part = particles[ d ];
	const GenVertex * productionVertex = part->production_vertex();
	// search barcode map and attach daughters
	if ( productionVertex != 0 ) fillDaughters(cands,part,d);
	cands[d].setCollisionId(0);
     }
  }
  
  evt.put( candsPtr );
  if(saveBarCodes_) evt.put( barCodeVector );
  //  if(doSubEvent_) evt.put(subsPtr); // For SubEventMap
  if(cfhepmcprod) delete cfhepmcprod;

}

bool GenParticleProducer::convertParticle(reco::GenParticle& cand, const HepMC::GenParticle * part){
   Candidate::LorentzVector p4( part->momentum() );
   int pdgId = part->pdg_id();
   cand.setThreeCharge( chargeTimesThree( pdgId ) );
   cand.setPdgId( pdgId );
   cand.setStatus( part->status() );
   cand.setP4( p4 );
   cand.setCollisionId(0);
   const GenVertex * v = part->production_vertex();
   if ( v != 0 ) {
      ThreeVector vtx = v->point3d();
      Candidate::Point vertex( vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm );
      cand.setVertex( vertex );
   } else {
      cand.setVertex( Candidate::Point( 0, 0, 0 ) );
   }
   return true;
}

bool GenParticleProducer::fillDaughters(reco::GenParticleCollection& cands, const HepMC::GenParticle * part, size_t index){

   const GenVertex * productionVertex = part->production_vertex();
   size_t numberOfMothers = productionVertex->particles_in_size();
   if ( numberOfMothers > 0 ) {
      GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
      for( ; motherIt != productionVertex->particles_in_const_end(); motherIt++) {
	 const HepMC::GenParticle * mother = * motherIt;
	 size_t m = barcodes_.find( mother->barcode() )->second;
	 cands[ m ].addDaughter( GenParticleRef( ref_, index ) );
	 cands[ index ].addMother( GenParticleRef( ref_, m ) );
      }
   }

   return true;
}

bool GenParticleProducer::fillIndices(const HepMC::GenEvent * mc, vector<const HepMC::GenParticle*>& particles, vector<int>& barCodeVector, int offset){
   size_t idx = offset;
   HepMC::GenEvent::particle_const_iterator begin = mc->particles_begin(), end = mc->particles_end();
   for( HepMC::GenEvent::particle_const_iterator p = begin; p != end; ++ p ) {
      const HepMC::GenParticle * particle = * p;
      size_t barCode_this_event = particle->barcode();
      size_t barCode = barCode_this_event + offset;
      if( barcodes_.find(barCode) != barcodes_.end() )
	 throw cms::Exception( "WrongReference" )
	    << "barcodes are duplicated! " << endl;
      particles[idx] = particle;
      barCodeVector[idx] = barCode;
      barcodes_.insert( make_pair(barCode_this_event, idx ++) );
   }
      return true;
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleProducer );

