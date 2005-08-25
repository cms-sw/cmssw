
#include <iostream>

#include "IOPool/Streamer/test/StreamThingAnalyzer.h"
#include "IOPool/Streamer/interface/StreamTestThing.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace edmtestprod;

namespace edmtest_thing
{
  StreamThingAnalyzer::StreamThingAnalyzer(edm::ParameterSet const& ps):
    name_(ps.getParameter<string>("product_to_get")),
    total_()
  {
  }
    
  StreamThingAnalyzer::~StreamThingAnalyzer()
  {
    cout << "total=" << total_ << endl;
  }

  void StreamThingAnalyzer::analyze(edm::Event const& e,
				    edm::EventSetup const&)
  {
    edm::Handle<StreamTestThing> prod;
    e.getByLabel(name_, prod);
    total_ = accumulate(prod->data_.begin(),prod->data_.end(),total_);
    //cout << tot << endl;
  }
}

using edmtest_thing::StreamThingAnalyzer;
DEFINE_FWK_MODULE(StreamThingAnalyzer)
