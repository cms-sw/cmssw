#ifndef Analysis_AnalysisFilters_interface_PVSelector_h
#define Analysis_AnalysisFilters_interface_PVSelector_h

#include "FWCore/Common/interface/EventBase.h"
#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "DataFormats/Common/interface/Handle.h"

#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"
#include "PhysicsTools/SelectorUtils/interface/PVObjectSelector.h"


// make a selector for this selection
class PVSelector : public Selector<edm::EventBase> {
public:

  PVSelector() {}

  PVSelector( edm::ParameterSet const & params ) :
    pvSrc_ (params.getParameter<edm::InputTag>("pvSrc") ),
    pvSel_ (params)
  {
    push_back("NPV", params.getParameter<int>("NPV") );
    set("NPV");
    retInternal_ = getBitTemplate();
    indexNPV_ = index_type(&bits_, "NPV");
  }

#ifndef __GCCXML__
  PVSelector( edm::ParameterSet const & params, edm::ConsumesCollector&& iC ) :
    PVSelector(params)
  {
    pvSrcToken_ = iC.consumes<std::vector<reco::Vertex> >(pvSrc_);
  }
#endif

  bool operator() ( edm::EventBase const & event,  pat::strbitset & ret ) {
    ret.set(false);
    event.getByLabel(pvSrc_, h_primVtx);

    // check if there is a good primary vertex

    if ( h_primVtx->size() < 1 ) return false;

    // Loop over PV's and count those that pass
    int npv = 0;
    int _ntotal = 0;
    mvSelPvs.clear();
    for ( std::vector<reco::Vertex>::const_iterator ibegin = h_primVtx->begin(),
          iend = h_primVtx->end(), i = ibegin; i != iend; ++i ) {
      reco::Vertex const & pv = *i;
      bool ipass = pvSel_(pv);
      if ( ipass ) {
        ++npv;
        mvSelPvs.push_back(edm::Ptr<reco::Vertex>(h_primVtx,_ntotal));
      }
      ++_ntotal;
    }

    // cache npv
    mNpv = npv;

    // Set the strbitset
    if ( npv >= cut(indexNPV_, int() ) || ignoreCut(indexNPV_) ) {
      passCut(ret, indexNPV_);
    }

    // Check if there is anything to ignore
    setIgnored(ret);

    // Return status
    bool pass = (bool)ret;
    return pass;
  }

  using EventSelector::operator();

  edm::Handle<std::vector<reco::Vertex> > const & vertices() const { return h_primVtx; }



  // get NPV from the last check
  int GetNpv(void){return mNpv;}



  std::vector<edm::Ptr<reco::Vertex> >  const &
    GetSelectedPvs() const { return mvSelPvs; }



private:
  edm::InputTag                           pvSrc_;
#ifndef __GCCXML__
  edm::EDGetTokenT<std::vector<reco::Vertex> >                           pvSrcToken_;
#endif
  PVObjectSelector                        pvSel_;
  edm::Handle<std::vector<reco::Vertex> > h_primVtx;
  std::vector<edm::Ptr<reco::Vertex> >    mvSelPvs; // selected vertices
  index_type                              indexNPV_;
  int                                     mNpv; // cache number of PVs
};

#endif
