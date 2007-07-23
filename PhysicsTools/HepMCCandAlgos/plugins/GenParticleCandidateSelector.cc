/** class 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: GenParticleCandidateSelector.cc,v 1.5 2007/03/27 08:43:51 llista Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
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
  typedef std::vector<PdtEntry> vpdt;
  /// module init at begin of job
  void beginJob( const edm::EventSetup & );
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// source collection name  
  std::string src_;
  // selects only stable particles (HEPEVT status = 1)
  bool stableOnly_;
  /// name of particles in include or exclude list
  vpdt pdtList_;
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
#include "DataFormats/Common/interface/Handle.h"
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
  vpdt includeList, excludeList;

  vector<string> vPdtParams = p.getParameterNamesForType<vpdt>();
  // check for include list
  bool found = std::find( vPdtParams.begin(), vPdtParams.end(), includeString) != vPdtParams.end();
  if ( found ) includeList = p.getParameter<vpdt>( includeString );
  // check for exclude list
  found = std::find( vPdtParams.begin(), vPdtParams.end(), excludeString) != vPdtParams.end();
  if ( found ) excludeList = p.getParameter<vpdt>( excludeString );

  // checking configuration cases
  bool bExclude(0);
  if ( includeList.size() > 0 ) bInclude_ = 1;
  if ( excludeList.size() > 0 ) bExclude = 1;

  if ( bInclude_ && bExclude ) {
    throw cms::Exception( "ConfigError", "not allowed to use both includeList and excludeList at the same time\n");
  }
  else if ( bInclude_ ) {
    caseString_ = "Including";
    pdtList_ = includeList;
  }
  else {
    caseString_ = "Excluding";
    pdtList_ = excludeList;
  }

}

GenParticleCandidateSelector::~GenParticleCandidateSelector() { 
}

void GenParticleCandidateSelector::beginJob( const EventSetup & es ) {
  //  const PDTRecord & rec = es.get<PDTRecord>();
  ESHandle<ParticleDataTable> pdt;
  es.getData( pdt );
  
  if ( verbose_ && stableOnly_ )
    LogInfo ( "INFO" ) << "Excluding unstable particles";
  for( vpdt::iterator p = pdtList_.begin(); 
       p != pdtList_.end(); ++ p ) {
    p->setup( es );
    LogInfo ( "INFO" ) << caseString_ <<" particle " << p->name() << ", id: " << p->pdgId();
    pIds_.insert( abs( p->pdgId() ) );
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
    int status = reco::status( *p );
    if ( ! stableOnly_ || status == 1 ) {
      int id = abs( p->pdgId() );
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
