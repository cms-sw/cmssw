#include "RecoPixelVertexing/PixelVertexFinding/interface/SkipBadEvents.h"

SkipBadEvents::SkipBadEvents(const edm::ParameterSet& config) {
  std::vector<int> badrunevent = config.getParameter<std::vector<int> >("RunEvent");
  // Convert to a map for easy lookup
  for (unsigned int i=0; i<badrunevent.size(); i+=2) {
    skip_[ badrunevent[i] ].insert( badrunevent[i+1] );
  }
}

SkipBadEvents::~SkipBadEvents(){}

bool SkipBadEvents::filter(edm::Event& e, const edm::EventSetup& s) {
  int run = e.id().run();
  int evt = e.id().event();

  bool pass = ( skip_[run].find(evt) == skip_[run].end() );

  return pass;
}
