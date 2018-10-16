#ifndef CandAlgos_NamedCandCombiner_h
#define CandAlgos_NamedCandCombiner_h
/** \class NamedCandCombiner
 *
 * performs all possible and selected combinations
 * of particle pairs using the NamedCandCombiner utility
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: NamedCandCombiner.h,v 1.2 2009/04/22 17:51:06 kaulmer Exp $
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CommonTools/CandUtils/interface/NamedCandCombiner.h"
#include "CommonTools/CandAlgos/interface/decayParser.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "CommonTools/Utilities/interface/cutParser.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidateFwd.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/transform.h"
#include <string>
#include <vector>
#include <algorithm>

namespace edm {
  class ParameterSet;
}

namespace reco {
  namespace modules {

    struct NamedCandCombinerBase : public edm::EDProducer {
      NamedCandCombinerBase( const edm::ParameterSet & cfg ) :
	setLongLived_(false),
	setMassConstraint_(false),
	setPdgId_(false) {
        using namespace cand::parser;
	using namespace std;
	string decay(cfg.getParameter<string>("decay"));
	if(decayParser(decay, labels_))
	  for(vector<ConjInfo>::iterator label = labels_.begin();
	       label != labels_.end(); ++label)
	if(label->mode_ == ConjInfo::kPlus)
	  dauCharge_.push_back( 1 );
	else if (label->mode_ == ConjInfo::kMinus)
	  dauCharge_.push_back(-1);
	else
	  dauCharge_.push_back(0);
	else
	  throw edm::Exception(edm::errors::Configuration,
			       "failed to parse \"" + decay + "\"");

	int lists = labels_.size();
	if(lists != 2 && lists != 3)
	  throw edm::Exception(edm::errors::LogicError,
			       "invalid number of collections");
	bool found;
	const string setLongLived("setLongLived");
	vector<string> vBoolParams = cfg.getParameterNamesForType<bool>();
	found = find(vBoolParams.begin(), vBoolParams.end(), setLongLived) != vBoolParams.end();
	if(found) setLongLived_ = cfg.getParameter<bool>("setLongLived");
	const string setMassConstraint("setMassConstraint");
	found = find(vBoolParams.begin(), vBoolParams.end(), setMassConstraint) != vBoolParams.end();
	if(found) setMassConstraint_ = cfg.getParameter<bool>("setMassConstraint");
	const string setPdgId("setPdgId");
	vector<string> vIntParams = cfg.getParameterNamesForType<int>();
	found = find(vIntParams.begin(), vIntParams.end(), setPdgId) != vIntParams.end();
	if(found) { setPdgId_ = true; pdgId_ = cfg.getParameter<int>("setPdgId"); }
	tokens_ = edm::vector_transform( labels_, [this](ConjInfo const & cI){return consumes<CandidateView>(cI.tag_);} );

	name_ = cfg.getParameter<std::string>( "name" );
	roles_= cfg.getParameter<std::vector<std::string> > ( "roles" );
      }
    protected:
      /// label vector
      std::vector<cand::parser::ConjInfo> labels_;
      std::vector<EDGetTokenT<CandidateView> > tokens_;
      /// daughter charges
      std::vector<int> dauCharge_;
      /// set long lived flag
      bool setLongLived_;
      /// set mass constraint flag
      bool setMassConstraint_;
      /// set pdgId flag
      bool setPdgId_;
      /// which pdgId to set
      int pdgId_;
      /// Name of this candidate
      std::string name_;
      // Name of the roles
      std::vector<std::string> roles_;
    };

    template<typename Selector,
             typename PairSelector = AnyPairSelector,
             typename Cloner = ::combiner::helpers::NormalClone,
             typename Setup = AddFourMomenta,
             typename Init = typename ::reco::modules::EventSetupInit<Setup>::type
            >
    class NamedCandCombiner : public NamedCandCombinerBase {
      public:
      /// constructor from parameter settypedef
      explicit NamedCandCombiner( const edm::ParameterSet & cfg ) :
        NamedCandCombinerBase( cfg ),
        combiner_( name_,
		   reco::modules::make<Selector>( cfg ),
		   reco::modules::make<PairSelector>( cfg ),
		   Setup( cfg ),
		   cfg.existsAs<bool>("checkCharge")  ? cfg.getParameter<bool>("checkCharge")  : true,
		   cfg.existsAs<bool>("checkOverlap") ? cfg.getParameter<bool>("checkOverlap") : true,
		   dauCharge_ ) {
        produces<reco::NamedCompositeCandidateCollection>();
      }
	/// destructor
      virtual ~NamedCandCombiner() { }

    private:
      /// process an event
      void produce(edm::Event& evt, const edm::EventSetup& es) {
	using namespace std;
	using namespace reco;
	Init::init(combiner_.setup(), evt, es);
	int n = labels_.size();
	std::vector<edm::Handle<CandidateView> > colls(n);
	for(int i = 0; i < n; ++i) {
	  evt.getByToken(tokens_[i], colls[i]);
	}
	std::vector<CandidatePtrVector> cv;
	for(typename std::vector<edm::Handle<CandidateView> >::const_iterator c = colls.begin();
	    c != colls.end(); ++ c) {
	  CandidatePtrVector r;
	  CandidateView::size_type n_view = (*c)->size();
	  for ( CandidateView::size_type iview = 0; iview < n_view; ++iview ) {
	    CandidatePtr ri( *c, iview );
	    r.push_back( ri );
	  }
	  cv.push_back(r);
	}
	std::unique_ptr<NamedCompositeCandidateCollection> out = combiner_.combine(cv, roles_);
	if(setLongLived_ || setMassConstraint_ || setPdgId_) {
	  typename NamedCompositeCandidateCollection::iterator i = out->begin(), e = out->end();
	  for(; i != e; ++i) {
	    i->setName( name_ );
	    i->setRoles( roles_ );
	    i->applyRoles();
	    if(setLongLived_) i->setLongLived();
	    if(setMassConstraint_) i->setMassConstraint();
	    if(setPdgId_) i->setPdgId(pdgId_);
	  }
	}
	evt.put(out);
      }
      /// combiner utility
      ::NamedCandCombiner<Selector, PairSelector, Cloner, Setup> combiner_;
    };

  }
}

#endif
