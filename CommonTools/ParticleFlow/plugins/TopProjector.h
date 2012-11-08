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

/**\class TopProjector 
\brief 

\author Colin Bernet
\date   february 2008
*/

#include <iostream>


/// This checks a slew of possible overlaps for FwdPtr<Candidate> and derivatives. 
template < class Top, class Bottom>
  class TopProjectorFwdPtrOverlap : public std::binary_function<edm::FwdPtr<Top>, edm::FwdPtr<Bottom>, bool > { 

  public:
    typedef edm::FwdPtr<Top> TopFwdPtr;
    typedef edm::FwdPtr<Bottom> BottomFwdPtr;

    explicit TopProjectorFwdPtrOverlap() {}
    explicit TopProjectorFwdPtrOverlap(edm::ParameterSet const & ) {}

    bool operator() ( TopFwdPtr const & top, BottomFwdPtr const & bottom ) const{
      bool matched = 
	(top.ptr().refCore() == bottom.ptr().refCore() && top.ptr().key() == bottom.ptr().key()) ||
	(top.ptr().refCore() == bottom.backPtr().refCore() && top.ptr().key() == bottom.backPtr().key()) ||
	(top.backPtr().refCore() == bottom.ptr().refCore() && top.backPtr().key() == bottom.ptr().key()) ||
	(top.backPtr().refCore() == bottom.backPtr().refCore() && top.backPtr().key() == bottom.backPtr().key())
	;
      if ( !matched ) {
	for ( unsigned isource = 0; isource < top->numberOfSourceCandidatePtrs(); ++isource ) {
	  reco::CandidatePtr const & topSrcPtr = top->sourceCandidatePtr(isource);
	  if ( (topSrcPtr.refCore() == bottom.ptr().refCore() && topSrcPtr.key() == bottom.ptr().key())|| 
	       (topSrcPtr.refCore() == bottom.backPtr().refCore() && topSrcPtr.key() == bottom.backPtr().key())
	       ) {
	    matched = true;
	    break;
	  }
	}
      }
      if ( !matched ) {
	for ( unsigned isource = 0; isource < bottom->numberOfSourceCandidatePtrs(); ++isource ) {
	  reco::CandidatePtr const & bottomSrcPtr = bottom->sourceCandidatePtr(isource);
	  if ( (bottomSrcPtr.refCore() == top.ptr().refCore() && bottomSrcPtr.key() == top.ptr().key() )|| 
	       (bottomSrcPtr.refCore() == top.backPtr().refCore() && bottomSrcPtr.key() == top.backPtr().key() )
	       ) {
	    matched = true;
	    break;
	  }
	}
      }

      return matched;
      
    }
 
};


/// This checks matching based on delta R
template < class Top, class Bottom>
  class TopProjectorDeltaROverlap : public std::binary_function<edm::FwdPtr<Top>, edm::FwdPtr<Bottom>, bool > { 

  public:
    typedef edm::FwdPtr<Top> TopFwdPtr;
    typedef edm::FwdPtr<Bottom> BottomFwdPtr;

    explicit TopProjectorDeltaROverlap() {}
    explicit TopProjectorDeltaROverlap(edm::ParameterSet const & config ) :
    deltaR_( config.getParameter<double>("deltaR") )
    {}

    bool operator() ( TopFwdPtr const & top, BottomFwdPtr const & bottom ) const{
      bool matched = reco::deltaR( top->p4(), bottom->p4() ) < deltaR_;
      return matched;
    }

 protected :
    double deltaR_;
 
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

  Matcher         match_;

  /// enable? if not, all candidates in the bottom collection are copied to the output collection
  bool            enable_;

  /// verbose ?
  bool            verbose_;

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

  verbose_ = iConfig.getUntrackedParameter<bool>("verbose",false);
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
  
  if( verbose_)
    std::cout<<"Event -------------------- "<<iEvent.id().event()<<std::endl;
  
  // get the various collections

  // Access the masking collection
  TopFwdPtrHandle tops;
  iEvent.getByLabel(  inputTagTop_, tops );


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

 
  if(verbose_) {
    const edm::Provenance& topProv = iEvent.getProvenance(tops.id());
    const edm::Provenance& bottomProv = iEvent.getProvenance(bottoms.id());

    std::cout<<"Top projector: event "<<iEvent.id().event()<<std::endl;
    std::cout<<"Inputs --------------------"<<std::endl;
    std::cout<<"Top      :  "
	<<tops.id()<<"\t"<<tops->size()<<std::endl
	<<topProv.branchDescription()<<std::endl
	<<"Bottom   :  "
	<<bottoms.id()<<"\t"<<bottoms->size()<<std::endl
	<<bottomProv.branchDescription()<<std::endl;
  }


  // output collection of FwdPtrs to objects,
  // selected from the Bottom collection
  std::auto_ptr< BottomFwdPtrCollection > 
    pBottomFwdPtrOutput( new BottomFwdPtrCollection );

  if(verbose_)
    std::cout<<" Remaining candidates in the bottom collection ------ "<<std::endl;
  
  for(unsigned i=0; i<bottoms->size(); i++) {

    BottomFwdPtr masked;
    
    BottomFwdPtr bottom = (*bottoms)[i];
    
    if ( enable_ ) {
      // Find any references in the "bottom" collection that match one
      // in the "top" collection. If that is found, it's masked.
      for ( unsigned j=0; j<tops->size(); j++ ) {
	TopFwdPtr top = (*tops)[j];
	if ( match_(top,bottom) ) {
	  masked = (*bottoms)[i];
	  break; 
	}
      }
    }

    // If this is masked in the top projection, we remove it. 
    if( enable_ && masked.isNonnull() && masked.isAvailable() ) {
      if(verbose_)
	std::cout<<"X "<<i<<" "<< *masked <<std::endl;
      continue;
    }
    // otherwise, we keep it. 
    else {
      if(verbose_)
	std::cout<<"O "<<i<<" "<< *((*bottoms)[i]) <<std::endl;
      pBottomFwdPtrOutput->push_back( (*bottoms)[i] );
    }
  }

  iEvent.put( pBottomFwdPtrOutput ); 
}


#endif
