/* \class PFCandidateMerger
 * 
 * Merges two lists of PFCandidates
 *
 * \author: L. Gray (FNAL)
 *
 * Additional functionality: S. Zenz
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include <vector>

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


// DAMN AND BLAST, WE HAVE TO DO MULTIPLE COLLECTIONS
//typedef Merger<reco::PFCandidateCollection> PFCandidateListMerger;

// so I shamelessly copy CommonTools/UtilAlgos/interface/Merger.h but make it dumber

typedef reco::PFCandidateCollection InputCollection;
typedef reco::PFCandidateCollection OutputCollection;
typedef edm::clonehelper::CloneTrait<InputCollection>::type P;

class PFCandidateListMerger : public edm::EDProducer {
public:
  /// constructor from parameter set
  explicit PFCandidateListMerger( const edm::ParameterSet& );
  /// destructor
  ~PFCandidateListMerger();

private:
  /// process an event
  virtual void produce( edm::Event&, const edm::EventSetup&) override;

  typedef std::vector<edm::InputTag> vtag;

  /// labels of the collections to be merged                                                                                                                                       
  vtag src_;

  /// more labels of more collections to be merged
  /// there's only one extra right now so I'm doing stupid cut-and-paste
  /// we'll clean it some day maybe
  vtag src1_;
  std::string label1_;
};

PFCandidateListMerger::~PFCandidateListMerger(){}

PFCandidateListMerger::PFCandidateListMerger( const edm::ParameterSet& par ) :
  src_( par.getParameter<vtag>( "src" ) ),
  src1_( par.getParameter<vtag>( "src1" ) ),
  label1_(par.getParameter<std::string>("label1"))
{
  produces<OutputCollection>();
  produces<OutputCollection>(label1_);
}

void PFCandidateListMerger::produce( edm::Event& evt, const edm::EventSetup&) {

  std::auto_ptr<OutputCollection> coll( new OutputCollection );
  for( vtag::const_iterator s = src_.begin(); s != src_.end(); ++ s ) {
    edm::Handle<InputCollection> h;
    evt.getByLabel( * s, h );
    for( typename InputCollection::const_iterator c = h->begin(); c != h->end(); ++c ) {
      coll->push_back( P::clone( * c ) );
    }
  }
  evt.put( coll );

  std::auto_ptr<OutputCollection> coll1( new OutputCollection );
  for( vtag::const_iterator s = src_.begin(); s != src_.end(); ++ s ) {
    edm::Handle<InputCollection> h;
    evt.getByLabel( * s, h );
    for( typename InputCollection::const_iterator c = h->begin(); c != h->end(); ++c ) {
      coll1->push_back( P::clone( * c ) );
    }
  }
  evt.put( coll1, label1_ );
}

DEFINE_FWK_MODULE( PFCandidateListMerger );
