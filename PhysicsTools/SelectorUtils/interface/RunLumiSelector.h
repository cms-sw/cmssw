#ifndef Analysis_AnalysisFilters_interface_RunLumiSelector_h
#define Analysis_AnalysisFilters_interface_RunLumiSelector_h

#ifndef __GCCXML__
#include "FWCore/Framework/interface/ConsumesCollector.h"
#endif
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/Common/interface/Handle.h"

#include "PhysicsTools/SelectorUtils/interface/EventSelector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>
#include <string>

// make a selector for this selection
class RunLumiSelector : public EventSelector {
public:
  RunLumiSelector() {}

#ifndef __GCCXML__
  RunLumiSelector( edm::ParameterSet const & params, edm::ConsumesCollector&& iC ) :
    RunLumiSelector( params )
  {}
#endif

  RunLumiSelector( edm::ParameterSet const & params ) {

    push_back("RunLumi");

    if ( params.exists("lumisToProcess") ) {
      lumis_ = params.getUntrackedParameter<std::vector<edm::LuminosityBlockRange> > ("lumisToProcess");
      set("RunLumi" );
    }
    else {
      lumis_.clear();
      set("RunLumi", false);
    }

    retInternal_ = getBitTemplate();
  }

  bool operator() ( edm::EventBase const & ev,  pat::strbitset & ret ) {

    if ( !ignoreCut("RunLumi") ) {
      bool goodLumi = false;
      for ( std::vector<edm::LuminosityBlockRange>::const_iterator lumisBegin = lumis_.begin(),
	      lumisEnd = lumis_.end(), ilumi = lumisBegin;
	    ilumi != lumisEnd; ++ilumi ) {
	if ( ev.id().run() >= ilumi->startRun() && ev.id().run() <= ilumi->endRun()  &&
	     ev.id().luminosityBlock() >= ilumi->startLumi() && ev.id().luminosityBlock() <= ilumi->endLumi() )  {
	  goodLumi = true;
	  break;
	}
      }
      if ( goodLumi ) passCut(ret, "RunLumi" );
    } else {
      passCut(ret, "RunLumi");
    }

    setIgnored(ret);
    return (bool)ret;
  }

  using EventSelector::operator();

private:

  std::vector<edm::LuminosityBlockRange> lumis_;

};

#endif
