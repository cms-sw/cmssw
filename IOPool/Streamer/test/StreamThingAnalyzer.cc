
#include <iostream>

#include "IOPool/Streamer/test/StreamThingAnalyzer.h"

#if 1
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
typedef edmtestprod::StreamTestThing WriteThis;
#else
#include "FWCore/Integration/interface/IntArray.h"
typedef edmtestprod::IntArray WriteThis;
#endif

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace std;
using namespace edmtestprod;

namespace edmtest_thing
{
  StreamThingAnalyzer::StreamThingAnalyzer(edm::ParameterSet const& ps):
    name_(ps.getParameter<string>("product_to_get")),
    total_(),
    out_("gennums.txt"),
	cnt_()
  {
    if(!out_)
    {
	cerr << "cannot open file gennums.txt" << endl;
        abort();
    }
	out_ << "event instance value" << endl;

	//LogDebug("StreamThing") << "ctor completing"; // << endl;
	//edm::LogInfo("stuff") << "again, ctor completing";
  }
    
  StreamThingAnalyzer::~StreamThingAnalyzer()
  {
    cout << "\nSTREAMTHING_CHECKSUM " << total_ << "\n" << endl;
  }

  void StreamThingAnalyzer::analyze(edm::Event const& e,
				    edm::EventSetup const&)
  {
    edm::ModuleLabelSelector all(name_);
    typedef vector<edm::Handle<WriteThis> > ProdList;
    ProdList prod;
    e.getMany(all, prod);
    ProdList::iterator i(prod.begin()),end(prod.end());
    for(; i != end; ++i)
      total_ = accumulate((*i)->data_.begin(),(*i)->data_.end(),total_);
    //cout << tot << endl;

#if 0
    for(i = prod.begin();i != end; ++i) {
	  vector<int>::const_iterator ii((*i)->data_.begin()),
	     ib((*i)->data_.end());
	  for(; ii != ib; ++ii) {
             out_ << cnt_ << " " << i->id() << " " << *ii << "\n" ;
	  }
    }
#endif

    ++cnt_;
  }
}

using edmtest_thing::StreamThingAnalyzer;
DEFINE_FWK_MODULE(StreamThingAnalyzer);
