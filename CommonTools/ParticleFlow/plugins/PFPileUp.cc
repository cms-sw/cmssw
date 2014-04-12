#include "CommonTools/ParticleFlow/plugins/PFPileUp.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;

PFPileUp::PFPileUp(const edm::ParameterSet& iConfig) {

  tokenPFCandidates_
    = consumes<PFCollection>(iConfig.getParameter<InputTag>("PFCandidates"));

  tokenVertices_
    = consumes<VertexCollection>(iConfig.getParameter<InputTag>("Vertices"));

  enable_ = iConfig.getParameter<bool>("Enable");

  verbose_ =
    iConfig.getUntrackedParameter<bool>("verbose",false);


  if ( iConfig.exists("checkClosestZVertex") ) {
    checkClosestZVertex_ = iConfig.getParameter<bool>("checkClosestZVertex");
  } else {
    checkClosestZVertex_ = false;
  }

  // Configure the algo
  pileUpAlgo_.setVerbose(verbose_);
  pileUpAlgo_.setCheckClosestZVertex(checkClosestZVertex_);

  //produces<reco::PFCandidateCollection>();
  produces< PFCollection > ();
  // produces< PFCollectionByValue > ();
}



PFPileUp::~PFPileUp() { }



void PFPileUp::produce(Event& iEvent,
			  const EventSetup& iSetup) {

//   LogDebug("PFPileUp")<<"START event: "<<iEvent.id().event()
// 			 <<" in run "<<iEvent.id().run()<<endl;


  // get PFCandidates

  auto_ptr< PFCollection >
    pOutput( new PFCollection );

  auto_ptr< PFCollectionByValue >
    pOutputByValue ( new PFCollectionByValue );

  if(enable_) {


    // get vertices
    Handle<VertexCollection> vertices;
    iEvent.getByToken( tokenVertices_, vertices);

    // get PF Candidates
    Handle<PFCollection> pfCandidates;
    PFCollection const * pfCandidatesRef = 0;
    PFCollection usedIfNoFwdPtrs;
    bool getFromFwdPtr = iEvent.getByToken( tokenPFCandidates_, pfCandidates);
    if ( getFromFwdPtr ) {
      pfCandidatesRef = pfCandidates.product();
    }
    // Maintain backwards-compatibility.
    // If there is no vector of FwdPtr<PFCandidate> found, then
    // make a dummy vector<FwdPtr<PFCandidate> > for the PFPileupAlgo,
    // set the pointer "pfCandidatesRef" to point to it, and
    // then we can pass it to the PFPileupAlgo.
    else {
      Handle<PFView> pfView;
      bool getFromView = iEvent.getByToken( tokenPFCandidates_, pfView );
      if ( ! getFromView ) {
	throw cms::Exception("PFPileUp is misconfigured. This needs to be either vector<FwdPtr<PFCandidate> >, or View<PFCandidate>");
      }
      for ( edm::View<reco::PFCandidate>::const_iterator viewBegin = pfView->begin(),
	      viewEnd = pfView->end(), iview = viewBegin;
	    iview != viewEnd; ++iview ) {
	usedIfNoFwdPtrs.push_back( edm::FwdPtr<reco::PFCandidate>( pfView->ptrAt(iview-viewBegin), pfView->ptrAt(iview-viewBegin)  ) );
      }
      pfCandidatesRef = &usedIfNoFwdPtrs;
    }

    if ( pfCandidatesRef == 0 ) {
      throw cms::Exception("Something went dreadfully wrong with PFPileUp. pfCandidatesRef should never be zero, so this is a logic error.");
    }



    pileUpAlgo_.process(*pfCandidatesRef,*vertices);
    pOutput->insert(pOutput->end(),pileUpAlgo_.getPFCandidatesFromPU().begin(),pileUpAlgo_.getPFCandidatesFromPU().end());

    // for ( PFCollection::const_iterator byValueBegin = pileUpAlgo_.getPFCandidatesFromPU().begin(),
    // 	    byValueEnd = pileUpAlgo_.getPFCandidatesFromPU().end(), ibyValue = byValueBegin;
    // 	  ibyValue != byValueEnd; ++ibyValue ) {
    //   pOutputByValue->push_back( **ibyValue );
    // }

  } // end if enabled
  // outsize of the loop to fill the collection anyway even when disabled
  iEvent.put( pOutput );
  // iEvent.put( pOutputByValue );
}

