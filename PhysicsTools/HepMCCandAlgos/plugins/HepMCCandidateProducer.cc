/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: HepMCCandidateProducer.h,v 1.6 2006/11/29 10:47:56 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>
#include <set>

class HepMCCandidateProducer : public edm::EDProducer {
 public:
  /// constructor
  HepMCCandidateProducer( const edm::ParameterSet & );
  /// destructor
  ~HepMCCandidateProducer();

 private:
  /// vector of strings
  typedef std::vector<std::string> vstring;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event & , const edm::EventSetup & );
  /// source collection name  
  std::string src_;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly_;
  /// exclude list
  vstring excludeList_;
  /// set of excluded particle id's
  std::set<int> excludedIds_;
  /// verbose flag
  bool verbose_;
};

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>
using namespace edm;
using namespace reco;
using namespace std;

HepMCCandidateProducer::HepMCCandidateProducer( const ParameterSet & p ) :
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vstring>( "excludeList" ) ),
  verbose_( p.getUntrackedParameter<bool>( "verbose" ) ) {
  produces<CandidateCollection>();
}

HepMCCandidateProducer::~HepMCCandidateProducer() { 
}

void HepMCCandidateProducer::beginJob( const EventSetup & es ) {
  //  const PDTRecord & rec = es.get<PDTRecord>();
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  
  if ( verbose_ && stableOnly_ )
    LogInfo ( "INFO" ) << "Excluding unstable particles";
  for( vstring::const_iterator e = excludeList_.begin(); 
       e != excludeList_.end(); ++ e ) {
    const DefaultConfig::ParticleData * p = pdt->particle( * e );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError", "can't find particle" )
	<< "can't find particle: " << * e;
    if ( verbose_ )
      LogInfo ( "INFO" ) << "Excluding particle " << *e << ", id: " << p->pid();
    excludedIds_.insert( abs( p->pid() ) );
  }
}

void HepMCCandidateProducer::produce( Event& evt, const EventSetup& ) {
  Handle<HepMCProduct> mcp;
  evt.getByLabel( src_, mcp );
  const HepMC::GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( mc->particles_size() );
  for( HepMC::GenEvent::particle_const_iterator p = mc->particles_begin(); 
       p != mc->particles_end(); ++ p ) {
    if ( ! stableOnly_ || (*p)->status() == 1 ) {
      int id = abs( (*p)->pdg_id() );
      if ( excludedIds_.find( id ) == excludedIds_.end() ) {
	if ( verbose_ )
	  LogInfo( "INFO" ) << "Adding candidate for particle with id: " 
			    << (*p)->pdg_id() << ", status: " << (*p)->status();
	HepMCCandidate::GenParticleRef ref( mcp, (*p)->barcode() );
	cands->push_back( new HepMCCandidate( ref ) );
      }
    }
  }

  evt.put( cands );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( HepMCCandidateProducer );
