#ifndef PhysicsTools_FWLite_WSelector_h
#define PhysicsTools_FWLite_WSelector_h
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
  WSelector(edm::ParameterSet const& params)
      : muonSrc_(params.getParameter<edm::InputTag>("muonSrc")), metSrc_(params.getParameter<edm::InputTag>("metSrc")) {
    double muonPtMin = params.getParameter<double>("muonPtMin");
    double metMin = params.getParameter<double>("metMin");
    push_back("Muon Pt", muonPtMin);
    push_back("MET", metMin);
    set("Muon Pt");
    set("MET");
    wMuon_ = nullptr;
    met_ = nullptr;
    if (params.exists("cutsToIgnore")) {
      setIgnoredCuts(params.getParameter<std::vector<std::string> >("cutsToIgnore"));
    }
    retInternal_ = getBitTemplate();
  }
  /// destructor
  ~WSelector() override {}
  /// return muon candidate of W boson
  pat::Muon const& wMuon() const { return *wMuon_; }
  /// return MET of W boson
  pat::MET const& met() const { return *met_; }

  /// here is where the selection occurs
  bool operator()(edm::EventBase const& event, pat::strbitset& ret) override {
    ret.set(false);
    // Handle to the muon collection
    edm::Handle<std::vector<pat::Muon> > muons;
    // Handle to the MET collection
    edm::Handle<std::vector<pat::MET> > met;
    // get the objects from the event
    bool gotMuons = event.getByLabel(muonSrc_, muons);
    bool gotMET = event.getByLabel(metSrc_, met);
    // get the MET, require to be > minimum
    if (gotMET) {
      met_ = &met->at(0);
      if (met_->pt() > cut("MET", double()) || ignoreCut("MET"))
        passCut(ret, "MET");
    }
    // get the highest pt muon, require to have pt > minimum
    if (gotMuons) {
      if (!ignoreCut("Muon Pt")) {
        if (!muons->empty()) {
          wMuon_ = &muons->at(0);
          if (wMuon_->pt() > cut("Muon Pt", double()) || ignoreCut("Muon Pt"))
            passCut(ret, "Muon Pt");
        }
      } else {
        passCut(ret, "Muon Pt");
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
};
#endif  // PhysicsTools_FWLite_WSelector_h
