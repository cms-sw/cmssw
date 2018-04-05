#ifndef PhysicsTools_FWLite_WSelectorFast_h
#define PhysicsTools_FWLite_WSelectorFast_h
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"

/**
   \class WSelector WSelector.h "PhysicsTools/FWLite/interface/WSelector.h"
   \brief Example class of an EventSelector to apply a simple W Boson selection

   Example class for of an EventSelector as defined in the SelectorUtils package. 
   EventSelectors may be used to facilitate cutflows and to implement selections
   independent from the event loop in FWLite or full framework.
*/


class WSelector : public EventSelector {

public:
  /// constructor
  WSelector(edm::ParameterSet const& params) :
    muonSrc_(params.getParameter<edm::InputTag>("muonSrc")),
    metSrc_ (params.getParameter<edm::InputTag>("metSrc")) 
  {
    double muonPtMin = params.getParameter<double>("muonPtMin");
    double metMin    = params.getParameter<double>("metMin");
    push_back("Muon Pt", muonPtMin );
    push_back("MET"    , metMin    );
    set("Muon Pt"); set("MET");
    wMuon_ = 0; met_ = 0;
    if ( params.exists("cutsToIgnore") ){
      setIgnoredCuts( params.getParameter<std::vector<std::string> >("cutsToIgnore") );
    }
    retInternal_ = getBitTemplate();
  }
  /// destructor
  virtual ~WSelector() {}
  /// return muon candidate of W boson
  pat::Muon const& wMuon() const { return *wMuon_;}
  /// return MET of W boson
  pat::MET  const& met()   const { return *met_;  }

  /// here is where the selection occurs
  virtual bool operator()( edm::EventBase const & event, pat::strbitset & ret){
    ret.set(false);
    // Handle to the muon collection
    edm::Handle<std::vector<pat::Muon> > muons;    
    // Handle to the MET collection
    edm::Handle<std::vector<pat::MET> > met;
    // get the objects from the event
    bool gotMuons = event.getByLabel(muonSrc_, muons);
    bool gotMET   = event.getByLabel(metSrc_, met   );
    // get the MET, require to be > minimum
    if( gotMET ){
      met_ = &met->at(0);
      if( met_->pt() > cut(metIndex_, double()) || ignoreCut(metIndex_) ) 
	passCut(ret, metIndex_);
    }
    // get the highest pt muon, require to have pt > minimum
    if( gotMuons ){
      if( !ignoreCut(muonPtIndex_) ){
	if( muons->size() > 0 ){
	  wMuon_ = &muons->at(0);
	  if ( wMuon_->pt() > cut(muonPtIndex_, double()) || ignoreCut(muonPtIndex_) ) 
	    passCut(ret, muonPtIndex_);
	}
      } 
      else{
	passCut( ret, muonPtIndex_);
      }
    }
    setIgnored(ret);
    return (bool)ret;
  }

protected:
  /// muon input
  edm::InputTag muonSrc_;
  /// met input
  edm::InputTag metSrc_;
  /// muon candidate from W boson
  pat::Muon const* wMuon_;
  /// MET from W boson
  pat::MET const* met_;
  /// index for muon Pt cut
  index_type muonPtIndex_;
  /// index for MET cut
  index_type metIndex_;
};
#endif // PhysicsTools_FWLite_WSelectorFast_h
