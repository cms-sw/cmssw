#ifndef PhysicsTools_PFCandProducer_TopProjector_
#define PhysicsTools_PFCandProducer_TopProjector_

// system include files
#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"  

#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**\class TopProjector 
\brief 

\author Colin Bernet
\date   february 2008
*/

#include <iostream>


/// This checks a slew of possible overlaps for FwdPtr<Candidate> and derivatives. 
template < class Top, class Bottom>
  class TopProjectorFwdPtrOverlap : public std::unary_function<edm::FwdPtr<Top>, bool > { 

  public:
    typedef edm::FwdPtr<Top> TopFwdPtr;
    typedef edm::FwdPtr<Bottom> BottomFwdPtr;


    explicit TopProjectorFwdPtrOverlap( ){bottom_ = 0;}

    explicit TopProjectorFwdPtrOverlap(edm::ParameterSet const & iConfig ){ bottom_ = 0;}

    inline void setBottom(BottomFwdPtr const & bottom ) { bottom_ = &bottom; }

    bool operator() ( TopFwdPtr const & top ) const{
      bool topFwdGood = top.ptr().isNonnull() && top.ptr().isAvailable();
      bool topBckGood = top.backPtr().isNonnull() && top.backPtr().isAvailable();
      bool bottomFwdGood = bottom_->ptr().isNonnull() && bottom_->ptr().isAvailable();
      bool bottomBckGood = bottom_->backPtr().isNonnull() && bottom_->backPtr().isAvailable();

      bool matched = 
	(topFwdGood && bottomFwdGood && top.ptr().refCore() == bottom_->ptr().refCore() && top.ptr().key() == bottom_->ptr().key()) ||
	(topFwdGood && bottomBckGood && top.ptr().refCore() == bottom_->backPtr().refCore() && top.ptr().key() == bottom_->backPtr().key()) ||
	(topBckGood && bottomFwdGood && top.backPtr().refCore() == bottom_->ptr().refCore() && top.backPtr().key() == bottom_->ptr().key()) ||
	(topBckGood && bottomBckGood && top.backPtr().refCore() == bottom_->backPtr().refCore() && top.backPtr().key() == bottom_->backPtr().key())
	;
      if ( !matched ) {
	for ( unsigned isource = 0; isource < top->numberOfSourceCandidatePtrs(); ++isource ) {
	  reco::CandidatePtr const & topSrcPtr = top->sourceCandidatePtr(isource);
	  bool topSrcGood = topSrcPtr.isNonnull() && topSrcPtr.isAvailable();
	  if ( (topSrcGood && bottomFwdGood && topSrcPtr.refCore() == bottom_->ptr().refCore() && topSrcPtr.key() == bottom_->ptr().key())|| 
	       (topSrcGood && bottomBckGood && topSrcPtr.refCore() == bottom_->backPtr().refCore() && topSrcPtr.key() == bottom_->backPtr().key())
	       ) {
	    matched = true;
	    break;
	  }
	}
      }
      if ( !matched ) {
	for ( unsigned isource = 0; isource < (*bottom_)->numberOfSourceCandidatePtrs(); ++isource ) {
	  reco::CandidatePtr const & bottomSrcPtr = (*bottom_)->sourceCandidatePtr(isource);
	  bool bottomSrcGood = bottomSrcPtr.isNonnull() && bottomSrcPtr.isAvailable();
	  if ( (topFwdGood && bottomSrcGood && bottomSrcPtr.refCore() == top.ptr().refCore() && bottomSrcPtr.key() == top.ptr().key() )|| 
	       (topBckGood && bottomSrcGood && bottomSrcPtr.refCore() == top.backPtr().refCore() && bottomSrcPtr.key() == top.backPtr().key() )
	       ) {
	    matched = true;
	    break;
	  }
	}
      }

      return matched;
      
    }

 protected :
    BottomFwdPtr const * bottom_; 
 
};


/// This checks matching based on delta R
template < class Top, class Bottom>
  class TopProjectorDeltaROverlap : public std::unary_function<edm::FwdPtr<Top>, bool > { 

  public:
    typedef edm::FwdPtr<Top> TopFwdPtr;
    typedef edm::FwdPtr<Bottom> BottomFwdPtr;

    explicit TopProjectorDeltaROverlap() {bottom_ = 0;}
    explicit TopProjectorDeltaROverlap(edm::ParameterSet const & config ) :
    deltaR2_( config.getParameter<double>("deltaR") ),
      bottom_(0),bottomCPtr_(0),botEta_(-999.f),botPhi_(0.f)
      {deltaR2_*=deltaR2_;}


      inline void setBottom(BottomFwdPtr const & bottom ) { 
	bottom_ = &bottom; bottomCPtr_=&**bottom_;
	botEta_ = bottomCPtr_->eta();
	botPhi_ = bottomCPtr_->phi();
      }

    bool operator() ( TopFwdPtr const & top ) const{
      const Top& oTop = *top; float topEta = oTop.eta(); float topPhi = oTop.phi();
      bool matched = reco::deltaR2( topEta, topPhi, botEta_, botPhi_) < deltaR2_;
      return matched;
    }

 protected :
    double deltaR2_;
    BottomFwdPtr const * bottom_; 
    const Bottom* bottomCPtr_; 
    float botEta_,botPhi_;
};

template< class Top, class Bottom, class Matcher = TopProjectorFwdPtrOverlap<Top,Bottom> >
class TopProjector : public edm::EDProducer {

 public:

  typedef std::vector<Top> TopCollection;
  typedef edm::Handle< std::vector<Top> > TopHandle;
  typedef edm::FwdPtr<Top> TopFwdPtr;
  typedef std::vector<TopFwdPtr> TopFwdPtrCollection; 
  typedef edm::Handle< TopFwdPtrCollection > TopFwdPtrHandle; 

  typedef std::vector<Bottom> BottomCollection;
  typedef edm::Handle< std::vector<Bottom> > BottomHandle;
  typedef edm::Ptr<Bottom> BottomPtr; 
  typedef edm::Ref<BottomCollection> BottomRef; 
  typedef edm::FwdPtr<Bottom> BottomFwdPtr;
  typedef std::vector<BottomFwdPtr> BottomFwdPtrCollection;
  typedef edm::Handle< BottomFwdPtrCollection > BottomFwdPtrHandle; 

  TopProjector(const edm::ParameterSet&);

  ~TopProjector() {};

  
  void produce(edm::Event&, const edm::EventSetup&);


 private:
  /// Matching method. 
  Matcher         match_;

  /// enable? if not, all candidates in the bottom collection are copied to the output collection
  bool            enable_;

  /// name of the top projection
  std::string     name_;
 
  /// input tag for the top (masking) collection
  edm::InputTag   inputTagTop_;

  /// input tag for the masked collection. 
  edm::InputTag   inputTagBottom_;
};




template< class Top, class Bottom, class Matcher >
TopProjector< Top, Bottom, Matcher>::TopProjector(const edm::ParameterSet& iConfig) : 
  match_(iConfig),
  enable_(iConfig.getParameter<bool>("enable")) {
  name_ = iConfig.getUntrackedParameter<std::string>("name","No Name");
  inputTagTop_ = iConfig.getParameter<edm::InputTag>("topCollection");
  inputTagBottom_ = iConfig.getParameter<edm::InputTag>("bottomCollection");

  // will produce a collection of the unmasked candidates in the 
  // bottom collection 
  produces< BottomFwdPtrCollection > ();
}


template< class Top, class Bottom, class Matcher >
void TopProjector< Top, Bottom, Matcher >::produce(edm::Event& iEvent,
					  const edm::EventSetup& iSetup) {
  // get the various collections

  // Access the masking collection
  TopFwdPtrHandle tops;
  iEvent.getByLabel(  inputTagTop_, tops );
  std::list<  TopFwdPtr > topsList;

  for ( typename TopFwdPtrCollection::const_iterator ibegin = tops->begin(), 
	  iend = tops->end(), i = ibegin; i != iend; ++i ) {
    topsList.push_back( *i );
  }


/*   if( !tops.isValid() ) { */
/*     std::ostringstream  err; */
/*     err<<"The top collection must be supplied."<<std::endl */
/*        <<"It is now set to : "<<inputTagTop_<<std::endl; */
/*     edm::LogError("PFPAT")<<err.str(); */
/*     throw cms::Exception( "MissingProduct", err.str()); */
/*   } */


  

  // Access the collection to
  // be masked by the other ones
  BottomFwdPtrHandle bottoms;
  iEvent.getByLabel(  inputTagBottom_, bottoms );

/*   if( !bottoms.isValid() ) { */
/*     std::ostringstream  err; */
/*     err<<"The bottom collection must be supplied."<<std::endl */
/*        <<"It is now set to : "<<inputTagBottom_<<std::endl; */
/*     edm::LogError("PFPAT")<<err.str(); */
/*     throw cms::Exception( "MissingProduct", err.str()); */
/*   } */

 
  /* if(verbose_) { */
  /*   const edm::Provenance& topProv = iEvent.getProvenance(tops.id()); */
  /*   const edm::Provenance& bottomProv = iEvent.getProvenance(bottoms.id()); */

  /*   edm::LogDebug("TopProjection")<<"Top projector: event "<<iEvent.id().event()<<std::endl; */
  /*   edm::LogDebug("TopProjection")<<"Inputs --------------------"<<std::endl; */
  /*   edm::LogDebug("TopProjection")<<"Top      :  " */
  /* 	     <<tops.id()<<"\t"<<tops->size()<<std::endl */
  /* 	     <<topProv.branchDescription()<<std::endl; */

  /*   for(unsigned i=0; i<tops->size(); i++) { */
  /*     TopFwdPtr top = (*tops)[i]; */
  /*     edm::LogDebug("TopProjection") << "< " << i << " " << top.key() << " : " << *top << std::endl; */
  /*   } */


  /*   edm::LogDebug("TopProjection")<<"Bottom   :  " */
  /* 	     <<bottoms.id()<<"\t"<<bottoms->size()<<std::endl */
  /* 	     <<bottomProv.branchDescription()<<std::endl; */

  /*   for(unsigned i=0; i<bottoms->size(); i++) { */
  /*     BottomFwdPtr bottom = (*bottoms)[i]; */
  /*     edm::LogDebug("TopProjection") << "> " << i << " " << bottom.key() << " : " << *bottom << std::endl; */
  /*   } */

  /* } */


  // output collection of FwdPtrs to objects,
  // selected from the Bottom collection
  std::auto_ptr< BottomFwdPtrCollection > 
    pBottomFwdPtrOutput( new BottomFwdPtrCollection );

  LogDebug("TopProjection")<<" Remaining candidates in the bottom collection ------ ";
  
  for(typename BottomFwdPtrCollection::const_iterator i=bottoms->begin(),
	iend = bottoms->end(), ibegin = i; i != iend; ++i) {

    BottomFwdPtr const & bottom = *i;
    match_.setBottom(bottom);
    typename std::list< TopFwdPtr >::iterator found = topsList.end();
    if ( enable_ ) {
      found = std::find_if( topsList.begin(), topsList.end(), match_ );
    }

    // If this is masked in the top projection, we remove it. 
    if( found != topsList.end() ) {
      LogDebug("TopProjection")<<"X "<<i-ibegin << **i;
      topsList.erase(found);
      continue;
    }
    // otherwise, we keep it. 
    else {
      LogDebug("TopProjection")<<"O "<<i-ibegin << **i;
      pBottomFwdPtrOutput->push_back( bottom );
    }
  }

  iEvent.put( pBottomFwdPtrOutput ); 
}


#endif
