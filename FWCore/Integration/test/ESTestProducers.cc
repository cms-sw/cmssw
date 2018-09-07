#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

#include <memory>

namespace edmtest {

  class ESTestProducerA : public edm::ESProducer {
  public:
    ESTestProducerA(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataA> produce(ESTestRecordA const&);
  private:
    std::shared_ptr<ESTestDataA> data_;
  };

  ESTestProducerA::ESTestProducerA(edm::ParameterSet const&) : data_(new ESTestDataA(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataA> ESTestProducerA::produce(ESTestRecordA const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerB : public edm::ESProducer {
  public:
    ESTestProducerB(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataB> produce(ESTestRecordB const&);
  private:
    std::shared_ptr<ESTestDataB> data_;
  };

  ESTestProducerB::ESTestProducerB(edm::ParameterSet const&) : data_(new ESTestDataB(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataB> ESTestProducerB::produce(ESTestRecordB const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  // This class is used to test ESProductHost
  class ESTestProducerBUsingHost : public edm::ESProducer {
  public:
    ESTestProducerBUsingHost(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataB> produce(ESTestRecordB const&);
  private:

    using HostType = edm::ESProductHost<ESTestDataB,
                                        ESTestRecordC,
                                        ESTestRecordD,
                                        ESTestRecordE,
                                        ESTestRecordF,
                                        ESTestRecordG,
                                        ESTestRecordH>;

    edm::ReusableObjectHolder<HostType> holder_;
  };

  ESTestProducerBUsingHost::ESTestProducerBUsingHost(edm::ParameterSet const&) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataB> ESTestProducerBUsingHost::produce(ESTestRecordB const& record) {

    auto host = holder_.makeOrGet([]() {
      return new HostType(100, 1000);
    });

    // Test that the numberOfRecordTypes and index functions are working properly
    if (host->numberOfRecordTypes() != 6 ||
        host->index<ESTestRecordC>() != 0 ||
        host->index<ESTestRecordD>() != 1 ||
        host->index<ESTestRecordE>() != 2 ||
        host->index<ESTestRecordF>() != 3 ||
        host->index<ESTestRecordG>() != 4 ||
        host->index<ESTestRecordH>() != 5) {

      throw cms::Exception("TestError") << "Either function numberOfRecordTypes or index returns incorrect value";
    }

    host->ifRecordChanges<ESTestRecordC>(record,
                                         [h=host.get()](auto const& rec) {
      ++h->value();
    });

    ++host->value();
    return host;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerC : public edm::ESProducer {
  public:
    ESTestProducerC(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataC> produce(ESTestRecordC const&);
  private:
    std::shared_ptr<ESTestDataC> data_;
  };

  ESTestProducerC::ESTestProducerC(edm::ParameterSet const&) : data_(new ESTestDataC(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataC> ESTestProducerC::produce(ESTestRecordC const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerD : public edm::ESProducer {
  public:
    ESTestProducerD(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataD> produce(ESTestRecordD const&);
  private:
    std::shared_ptr<ESTestDataD> data_;
  };

  ESTestProducerD::ESTestProducerD(edm::ParameterSet const&) : data_(new ESTestDataD(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataD> ESTestProducerD::produce(ESTestRecordD const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerE : public edm::ESProducer {
  public:
    ESTestProducerE(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataE> produce(ESTestRecordE const&);
  private:
    std::shared_ptr<ESTestDataE> data_;
  };

  ESTestProducerE::ESTestProducerE(edm::ParameterSet const&) : data_(new ESTestDataE(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataE> ESTestProducerE::produce(ESTestRecordE const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerF : public edm::ESProducer {
  public:
    ESTestProducerF(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataF> produce(ESTestRecordF const&);
  private:
    std::shared_ptr<ESTestDataF> data_;
  };

  ESTestProducerF::ESTestProducerF(edm::ParameterSet const&) : data_(new ESTestDataF(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataF> ESTestProducerF::produce(ESTestRecordF const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerG : public edm::ESProducer {
  public:
    ESTestProducerG(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataG> produce(ESTestRecordG const&);
  private:
    std::shared_ptr<ESTestDataG> data_;
  };

  ESTestProducerG::ESTestProducerG(edm::ParameterSet const&) : data_(new ESTestDataG(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataG> ESTestProducerG::produce(ESTestRecordG const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerH : public edm::ESProducer {
  public:
    ESTestProducerH(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataH> produce(ESTestRecordH const&);
  private:
    std::shared_ptr<ESTestDataH> data_;
  };

  ESTestProducerH::ESTestProducerH(edm::ParameterSet const&) : data_(new ESTestDataH(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataH> ESTestProducerH::produce(ESTestRecordH const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerI : public edm::ESProducer {
  public:
    ESTestProducerI(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataI> produce(ESTestRecordI const&);
  private:
    std::shared_ptr<ESTestDataI> data_;
  };

  ESTestProducerI::ESTestProducerI(edm::ParameterSet const&) : data_(new ESTestDataI(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataI> ESTestProducerI::produce(ESTestRecordI const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerJ : public edm::ESProducer {
  public:
    ESTestProducerJ(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataJ> produce(ESTestRecordJ const&);
  private:
    std::shared_ptr<ESTestDataJ> data_;
  };

  ESTestProducerJ::ESTestProducerJ(edm::ParameterSet const&) : data_(new ESTestDataJ(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataJ> ESTestProducerJ::produce(ESTestRecordJ const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerK : public edm::ESProducer {
  public:
    ESTestProducerK(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataK> produce(ESTestRecordK const&);
  private:
    std::shared_ptr<ESTestDataK> data_;
  };

  ESTestProducerK::ESTestProducerK(edm::ParameterSet const&) : data_(new ESTestDataK(0)) {
    setWhatProduced(this);
  }

  std::shared_ptr<ESTestDataK> ESTestProducerK::produce(ESTestRecordK const& rec) {
    ++data_->value();
    return data_;
  }

  // ---------------------------------------------------------------------

  class ESTestProducerAZ : public edm::ESProducer {
  public:
    ESTestProducerAZ(edm::ParameterSet const&);
    std::shared_ptr<ESTestDataA> produceA(ESTestRecordA const&);
    std::shared_ptr<ESTestDataZ> produceZ(ESTestRecordZ const&);
  private:
    std::shared_ptr<ESTestDataA> dataA_;
    std::shared_ptr<ESTestDataZ> dataZ_;
  };

  ESTestProducerAZ::ESTestProducerAZ(edm::ParameterSet const&) :
    dataA_(new ESTestDataA(0)),
    dataZ_(new ESTestDataZ(0)) {
    setWhatProduced(this, &edmtest::ESTestProducerAZ::produceA, edm::es::Label("foo"));
    setWhatProduced(this, &edmtest::ESTestProducerAZ::produceZ, edm::es::Label("foo"));
  }

  std::shared_ptr<ESTestDataA> ESTestProducerAZ::produceA(ESTestRecordA const& rec) {
    ++dataA_->value();
    return dataA_;
  }

  std::shared_ptr<ESTestDataZ> ESTestProducerAZ::produceZ(ESTestRecordZ const& rec) {
    ++dataZ_->value();
    return dataZ_;
  }

}

using namespace edmtest;
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerA);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerB);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerBUsingHost);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerC);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerD);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerE);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerF);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerG);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerH);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerI);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerJ);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerK);
DEFINE_FWK_EVENTSETUP_MODULE(ESTestProducerAZ);
