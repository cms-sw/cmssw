/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateSelector.h,v 1.2 2007/01/25 15:28:55 hegner Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>
#include <vector>
#include <set>

class GenParticleCandidateSelector : public edm::EDProducer {
 public:
  /// constructor
  GenParticleCandidateSelector( const edm::ParameterSet & );
  /// destructor
  ~GenParticleCandidateSelector();

 private:
  /// vector of strings
  typedef std::vector<std::string> vstring;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string src_;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly_;
  /// name of particles in include or exclude list
  vstring pNameList_;
  /// using include list?
  bool bInclude_;
  /// output string for debug
  std::string caseString_;
  /// set of excluded particle id's
  std::set<int> pIds_;
  /// verbose flag
  bool verbose_;
};

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
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

GenParticleCandidateSelector::GenParticleCandidateSelector( const ParameterSet & p ) :
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  bInclude_(0),
  verbose_( p.getUntrackedParameter<bool>( "verbose" ) ) {

  produces<CandidateCollection>();

  //check optional parameters includeList and excludeList
  const std::string excludeString("excludeList");
  const std::string includeString("includeList");
  vstring includeList, excludeList;

  std::vector<std::string> vstringParams = 
    p.getParameterNamesForType<vstring>();
  // check for include list
  bool found = std::find( vstringParams.begin(), vstringParams.end(), 
    includeString) != vstringParams.end();
  if ( found ) includeList = p.getParameter<vstring>( includeString );
  // check for exclude list
  found = std::find( vstringParams.begin(), vstringParams.end(), 
    excludeString) != vstringParams.end();
  if ( found ) excludeList = p.getParameter<vstring>( excludeString );

  // checking configuration cases
  bool bExclude(0);
  if ( includeList.size() > 0 ) bInclude_ = 1;
  if ( excludeList.size() > 0 ) bExclude = 1;

  if ( bInclude_ && bExclude ) {
    throw cms::Exception( "ConfigError", "not allowed to use both includeList and excludeList at the same time\n");
  }
  else if ( bInclude_ ) {
    caseString_ = "Including";
    pNameList_ = includeList;
  }
  else {
    caseString_ = "Excluding";
    pNameList_ = excludeList;
  }

}

GenParticleCandidateSelector::~GenParticleCandidateSelector() { 
}

void GenParticleCandidateSelector::beginJob( const EventSetup & es ) {
  //  const PDTRecord & rec = es.get<PDTRecord>();
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  
  if ( verbose_ && stableOnly_ )
    LogInfo ( "INFO" ) << "Excluding unstable particles";
  for( vstring::const_iterator name = pNameList_.begin(); 
       name != pNameList_.end(); ++ name ) {
    const DefaultConfig::ParticleData * p = pdt->particle( * name );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError", "can't find particle" )
	<< "can't find particle: " << * name;
    if ( verbose_ )
      LogInfo ( "INFO" ) << caseString_ <<" particle " << *name << ", id: " << p->pid();
    pIds_.insert( abs( p->pid() ) );
  }
}

void GenParticleCandidateSelector::produce( Event& evt, const EventSetup& ) {
  Handle<CandidateCollection> particles;
  evt.getByLabel( src_, particles );
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( particles->size() );
  size_t idx = 0;
  for( CandidateCollection::const_iterator p = particles->begin(); 
       p != particles->end(); ++ p, ++ idx ) {
    int status = reco::status( * p );
    if ( ! stableOnly_ || status == 1 ) {
      int id = abs( reco::pdgId( * p ) );
      // id not in list + exclude= keep, in list + include = keep, otherwise drop
      // -> XOR operation: end XOR bInclude; 
      if ( pIds_.find( id ) == pIds_.end() ^ bInclude_) { 
	if ( verbose_ )
	  LogInfo( "INFO" ) << "Adding candidate for particle with id: " 
			    << id << ", status: " << status;
	CandidateBaseRef ref( CandidateRef( particles, idx ) );
	cands->push_back( new ShallowCloneCandidate( ref ) );
      }
    }
  }

  evt.put( cands );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenParticleCandidateSelector );
