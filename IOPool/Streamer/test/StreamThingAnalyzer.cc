
#include <iostream>

#include "IOPool/Streamer/test/StreamThingAnalyzer.h"

#if 1
#include "IOPool/Streamer/interface/StreamTestThing.h"
typedef edmtestprod::StreamTestThing WriteThis;
#else
#include "FWCore/Integration/interface/IntArray.h"
typedef edmtestprod::IntArray WriteThis;
#endif

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace edmtestprod;

namespace 
{
  
  class AllSelector : public edm::Selector {
  public:
    AllSelector(const std::string& name):name_(name) {}
    
    virtual bool doMatch(const edm::Provenance& p) const {
      return p.product.module.moduleLabel_==name_;
    }

  private:
    string name_;
  };
}

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
    AllSelector all(name_);
    typedef vector<edm::Handle<WriteThis> > ProdList;
    ProdList prod;
    e.getMany(all, prod);
    ProdList::iterator i(prod.begin()),end(prod.end());
    for(;i!=end;++i)
      total_ = accumulate((*i)->data_.begin(),(*i)->data_.end(),total_);
    //cout << tot << endl;
  }
}

using edmtest_thing::StreamThingAnalyzer;
DEFINE_FWK_MODULE(StreamThingAnalyzer)
