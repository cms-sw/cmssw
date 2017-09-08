#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DataFormats/TestObjects/interface/TableTest.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"


namespace {
  std::vector<float> doublesToFloats(std::vector<double> const& iDoubles) {
    std::vector<float> t;
    t.reserve(iDoubles.size());
    for(double d: iDoubles) { t.push_back( static_cast<float>(d) ); }
    return t;
  }
}

namespace edmtest {

  class TableTestProducer : public edm::global::EDProducer<> {
  public:
    TableTestProducer(edm::ParameterSet const& iConfig):
      anInts_(iConfig.getParameter<std::vector<int>>("anInts")),
      aFloats_(doublesToFloats(iConfig.getParameter<std::vector<double>>("aFloats"))),
      aStrings_(iConfig.getParameter<std::vector<std::string>>("aStrings")) {
      produces<edmtest::TableTest>();
    }

    void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const final {
      iEvent.put( std::make_unique<TableTest>(anInts_,aFloats_,aStrings_) );      
    }
  private:
    const std::vector<int>  anInts_;
    const std::vector<float> aFloats_;
    const std::vector<std::string> aStrings_;
  };

  class TableTestAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    TableTestAnalyzer(edm::ParameterSet const& iConfig):
      anInts_(iConfig.getUntrackedParameter<std::vector<int>>("anInts")),
      aFloats_(doublesToFloats(iConfig.getUntrackedParameter<std::vector<double>>("aFloats"))),
      aStrings_(iConfig.getUntrackedParameter<std::vector<std::string>>("aStrings"))
    {
      tableToken_ = consumes<edmtest::TableTest>(iConfig.getUntrackedParameter<edm::InputTag>("table"));
      if(anInts_.size() != aFloats_.size() or anInts_.size() != aStrings_.size()) {
        throw cms::Exception("Configuration")<<"anInts_, aFloats_, and aStrings_ must have the same length";
      }
    }
    
    void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const final {
      edm::Handle<edmtest::TableTest> h;
      iEvent.getByToken(tableToken_, h);
      
      auto size = h->size();
      if(size != anInts_.size()) {
        throw cms::Exception("RuntimeError")<<"Table size ("<<size<<") does not equal expected size ("<<anInts_.size()<<")";
      }

      unsigned int index=0;
      for(auto const& row : *h) {
        if( anInts_[index] != row.get<edmtest::AnInt>() ) {
          throw cms::Exception("RuntimeError")<<"index "<<index<<" anInt ="<<row.get<edmtest::AnInt>()<<" expected "<<anInts_[index];
        }
        if( aFloats_[index] != row.get<edmtest::AFloat>() ) {
          throw cms::Exception("RuntimeError")<<"index "<<index<<" aFloat ="<<row.get<edmtest::AFloat>()<<" expected "<<aFloats_[index];
        }
        if( aStrings_[index] != row.get<edmtest::AString>() ) {
          throw cms::Exception("RuntimeError")<<"index "<<index<<" aString ="<<row.get<edmtest::AString>()<<" expected "<<aStrings_[index];
        }
        ++index;
      }
    }
    
  private:
    const std::vector<int>  anInts_;
    const std::vector<float> aFloats_;
    const std::vector<std::string> aStrings_;
    edm::EDGetTokenT<edmtest::TableTest> tableToken_;
  };   
}
DEFINE_FWK_MODULE(edmtest::TableTestProducer);
DEFINE_FWK_MODULE(edmtest::TableTestAnalyzer);
