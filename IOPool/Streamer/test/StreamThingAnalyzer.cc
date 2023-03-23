
#include <iostream>

#include "IOPool/Streamer/test/StreamThingAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ModuleLabelMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace edmtestprod;

namespace edmtest_thing {
  StreamThingAnalyzer::StreamThingAnalyzer(edm::ParameterSet const& ps)
      : name_(ps.getParameter<std::string>("product_to_get")),
        total_(),
        out_("gennums.txt"),
        inChecksumFile_(ps.getUntrackedParameter<std::string>("inChecksum", "")),
        outChecksumFile_(ps.getUntrackedParameter<std::string>("outChecksum", "")),
        cnt_(),
        getterUsingLabel_(edm::ModuleLabelMatch(name_), this) {
    callWhenNewProductsRegistered(getterUsingLabel_);
    if (!out_) {
      std::cerr << "cannot open file gennums.txt" << std::endl;
      abort();
    }
    out_ << "event instance value" << std::endl;

    //LogDebug("StreamThing") << "ctor completing"; // << std::endl;
    //edm::LogInfo("stuff") << "again, ctor completing";
  }

  StreamThingAnalyzer::~StreamThingAnalyzer() { std::cout << "\nSTREAMTHING_CHECKSUM " << total_ << "\n" << std::endl; }

  void StreamThingAnalyzer::endJob() {
    edm::LogInfo("StreamThingAnalyzer") << "STREAMTHING_CHECKSUM " << total_;
    if (!outChecksumFile_.empty()) {
      edm::LogInfo("StreamThingAnalyzer") << "Writing checksum to " << outChecksumFile_;
      std::ofstream outChecksum(outChecksumFile_);
      outChecksum << total_;
    } else if (!inChecksumFile_.empty()) {
      edm::LogInfo("StreamThingAnalyzer") << "Reading checksum from " << inChecksumFile_;
      int totalIn = 0;
      std::ifstream inChecksum(inChecksumFile_);
      inChecksum >> totalIn;
      if (totalIn != total_)
        throw cms::Exception("StreamThingAnalyzer")
            << "Checksum mismatch input: " << totalIn << " compared to: " << total_;
    }
  }

  void StreamThingAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    typedef std::vector<edm::Handle<WriteThis> > ProdList;
    ProdList prod;
    getterUsingLabel_.fillHandles(e, prod);
    ProdList::iterator i(prod.begin()), end(prod.end());
    for (; i != end; ++i)
      total_ = accumulate((*i)->data_.begin(), (*i)->data_.end(), total_);
      //std::cout << tot << std::endl;

#if 0
    for(i = prod.begin();i != end; ++i) {
	  std::vector<int>::const_iterator ii((*i)->data_.begin()),
	     ib((*i)->data_.end());
	  for(; ii != ib; ++ii) {
             out_ << cnt_ << " " << i->id() << " " << *ii << "\n" ;
	  }
    }
#endif

    ++cnt_;
  }
}  // namespace edmtest_thing

using edmtest_thing::StreamThingAnalyzer;
DEFINE_FWK_MODULE(StreamThingAnalyzer);
