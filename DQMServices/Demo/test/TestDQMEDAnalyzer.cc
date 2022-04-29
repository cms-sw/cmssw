#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Histograms/interface/MonitorElementCollection.h"

template <typename BOOKERLIKE, typename ME, bool DOLUMI = false>
class BookerFiller {
public:
  BookerFiller(std::string folder, int howmany) {
    this->howmany = howmany;
    this->folder = folder;
  }

  BookerFiller(){};

  void bookall(BOOKERLIKE& ibooker) {
    mes_1D.clear();
    mes_2D.clear();
    mes_3D.clear();

    for (int i = 0; i < howmany; i++) {
      ibooker.setCurrentFolder(folder);
      auto num = std::to_string(i);
      // this does *not* test most of the booking API, just one ME of each kind.
      mes_1D.push_back(ibooker.bookFloat("float" + num));
      mes_1D.push_back(ibooker.bookInt("int" + num));
      mes_1D.push_back(ibooker.book1D("th1f" + num, "1D Float Histogram " + num, 101, -0.5, 100.5));
      mes_1D.push_back(ibooker.book1S("th1s" + num, "1D Short Histogram " + num, 101, -0.5, 100.5));
      mes_1D.push_back(ibooker.book1I("th1i" + num, "1D Integer Histogram " + num, 101, -0.5, 100.5));
      mes_1D.push_back(ibooker.book1DD("th1d" + num, "1D Double Histogram " + num, 101, -0.5, 100.5));

      mes_2D.push_back(ibooker.book2D("th2f" + num, "2D Float Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
      mes_2D.push_back(ibooker.book2S("th2s" + num, "2D Short Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
      mes_2D.push_back(ibooker.book2DD("th2d" + num, "2D Double Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
      mes_2D.push_back(ibooker.book2I("th2i" + num, "2D Integer Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
      mes_2D.push_back(
          ibooker.bookProfile("tprofile" + num, "1D Profile Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));

      mes_3D.push_back(
          ibooker.book3D("th3f" + num, "3D Float Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5, 3, -0.5, 2.5));
      mes_3D.push_back(ibooker.bookProfile2D(
          "thprofile2d" + num, "2D Profile Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5, 3, -0.5, 2.5));

      if (DOLUMI) {
        auto scope = typename BOOKERLIKE::UseLumiScope(ibooker);
        ibooker.setCurrentFolder(folder + "/lumi");

        mes_1D.push_back(ibooker.bookFloat("float" + num));
        mes_1D.push_back(ibooker.bookInt("int" + num));
        mes_1D.push_back(ibooker.book1D("th1f" + num, "1D Float Histogram " + num, 101, -0.5, 100.5));
        mes_1D.push_back(ibooker.book1S("th1s" + num, "1D Short Histogram " + num, 101, -0.5, 100.5));
        mes_1D.push_back(ibooker.book1I("th1i" + num, "1D Integer Histogram " + num, 101, -0.5, 100.5));
        mes_1D.push_back(ibooker.book1DD("th1d" + num, "1D Double Histogram " + num, 101, -0.5, 100.5));

        mes_2D.push_back(ibooker.book2D("th2f" + num, "2D Float Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
        mes_2D.push_back(ibooker.book2S("th2s" + num, "2D Short Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
        mes_2D.push_back(ibooker.book2DD("th2d" + num, "2D Double Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
        mes_2D.push_back(ibooker.book2I("th2i" + num, "2D Integer Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));
        mes_2D.push_back(
            ibooker.bookProfile("tprofile" + num, "1D Profile Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5));

        mes_3D.push_back(
            ibooker.book3D("th3f" + num, "3D Float Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5, 3, -0.5, 2.5));
        mes_3D.push_back(ibooker.bookProfile2D(
            "thprofile2d" + num, "2D Profile Histogram " + num, 101, -0.5, 100.5, 11, -0.5, 10.5, 3, -0.5, 2.5));
      }
    }
  }

  void fillall(double x, double y, double z) const {
    for (auto me : mes_1D) {
      me->Fill(x);
    }
    for (auto me : mes_2D) {
      me->Fill(x, y);
    }
    for (auto me : mes_3D) {
      me->Fill(x, y, z);
    }
  }

  std::vector<ME*> mes_1D;
  std::vector<ME*> mes_2D;
  std::vector<ME*> mes_3D;
  std::string folder;
  int howmany;
};

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
class TestDQMEDAnalyzer : public DQMEDAnalyzer {
public:
  explicit TestDQMEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestDQMEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Normal/test")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("test", desc);
  }

private:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override {
    mymes_.bookall(ibooker);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    mymes_.fillall(iEvent.luminosityBlock(), iEvent.run(), myvalue_);
  }

  BookerFiller<DQMStore::IBooker, MonitorElement, /* DOLUMI */ true> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMEDAnalyzer);

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
class TestDQMOneEDAnalyzer : public DQMOneEDAnalyzer<> {
public:
  explicit TestDQMOneEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestDQMOneEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "One/testone")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testone", desc);
  }

private:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override {
    mymes_.bookall(ibooker);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    mymes_.fillall(iEvent.luminosityBlock(), iEvent.run(), myvalue_);
  }

  BookerFiller<DQMStore::IBooker, MonitorElement, /* DOLUMI */ true> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMOneEDAnalyzer);

class TestDQMOneFillRunEDAnalyzer : public DQMOneEDAnalyzer<> {
public:
  explicit TestDQMOneFillRunEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestDQMOneFillRunEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "One/testonefillrun")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testonefillrun", desc);
  }

private:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override {
    mymes_.bookall(ibooker);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {}
  void dqmEndRun(edm::Run const& run, edm::EventSetup const&) override { mymes_.fillall(0, run.run(), myvalue_); }

  BookerFiller<DQMStore::IBooker, MonitorElement> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMOneFillRunEDAnalyzer);

class TestDQMOneLumiEDAnalyzer : public DQMOneLumiEDAnalyzer<> {
public:
  explicit TestDQMOneLumiEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestDQMOneLumiEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "One/testonelumi")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testonelumi", desc);
  }

private:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override {
    mymes_.bookall(ibooker);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    mymes_.fillall(iEvent.luminosityBlock(), iEvent.run(), myvalue_);
  }

  BookerFiller<DQMStore::IBooker, MonitorElement, /* DOLUMI */ true> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMOneLumiEDAnalyzer);

class TestDQMOneLumiFillLumiEDAnalyzer : public DQMOneLumiEDAnalyzer<> {
public:
  explicit TestDQMOneLumiFillLumiEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestDQMOneLumiFillLumiEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "One/testonelumifilllumi")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testonelumifilllumi", desc);
  }

private:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override {
    mymes_.bookall(ibooker);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {}
  void dqmBeginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) override {
    mymes_.fillall(lumi.luminosityBlock(), lumi.run(), myvalue_);
  }

  BookerFiller<DQMStore::IBooker, MonitorElement, /* DOLUMI */ true> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMOneLumiFillLumiEDAnalyzer);

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
typedef BookerFiller<dqm::reco::DQMStore::IBooker, dqm::reco::MonitorElement> TestHistograms;

class TestDQMGlobalEDAnalyzer : public DQMGlobalEDAnalyzer<TestHistograms> {
public:
  explicit TestDQMGlobalEDAnalyzer(const edm::ParameterSet& iConfig)
      : folder_(iConfig.getParameter<std::string>("folder")),
        howmany_(iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}
  ~TestDQMGlobalEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Global/testglobal")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testglobal", desc);
  }

private:
  void bookHistograms(DQMStore::IBooker& ibooker,
                      edm::Run const&,
                      edm::EventSetup const&,
                      TestHistograms& h) const override {
    h.folder = this->folder_;
    h.howmany = this->howmany_;
    h.bookall(ibooker);
  }

  void dqmAnalyze(edm::Event const& iEvent, edm::EventSetup const&, TestHistograms const& h) const override {
    h.fillall(iEvent.luminosityBlock(), iEvent.run(), myvalue_);
  }

private:
  std::string folder_;
  int howmany_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMGlobalEDAnalyzer);

class TestDQMGlobalRunSummaryEDAnalyzer : public DQMGlobalRunSummaryEDAnalyzer<TestHistograms, int> {
public:
  explicit TestDQMGlobalRunSummaryEDAnalyzer(const edm::ParameterSet& iConfig)
      : folder_(iConfig.getParameter<std::string>("folder")),
        howmany_(iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}
  ~TestDQMGlobalRunSummaryEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Global/testglobal")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testglobalrunsummary", desc);
  }

private:
  std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
    return std::make_shared<int>(0);
  }

  void bookHistograms(DQMStore::IBooker& ibooker,
                      edm::Run const&,
                      edm::EventSetup const&,
                      TestHistograms& h) const override {
    h.folder = this->folder_;
    h.howmany = this->howmany_;
    h.bookall(ibooker);
  }

  void dqmAnalyze(edm::Event const& iEvent, edm::EventSetup const&, TestHistograms const& h) const override {
    h.fillall(iEvent.luminosityBlock(), iEvent.run(), myvalue_);
  }

  void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int* runSummaryCache) const override {
    (*runSummaryCache) += 1;
  }

  void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int* runSummaryCache) const override {}

  void dqmEndRun(edm::Run const& run,
                 edm::EventSetup const& setup,
                 TestHistograms const& h,
                 int const& runSummaryCache) const override {}

private:
  std::string folder_;
  int howmany_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestDQMGlobalRunSummaryEDAnalyzer);

#include "FWCore/Framework/interface/EDAnalyzer.h"
class TestLegacyEDAnalyzer : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit TestLegacyEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestLegacyEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Legacy/testlegacy")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testlegacy", desc);
  }

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override {
    edm::Service<DQMStore> store;
    mymes_.bookall(*store);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    mymes_.fillall(iEvent.luminosityBlock(), iEvent.run(), myvalue_);
  }

  BookerFiller<DQMStore, MonitorElement, /* DOLUMI */ true> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestLegacyEDAnalyzer);

class TestLegacyFillRunEDAnalyzer : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit TestLegacyFillRunEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestLegacyFillRunEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Legacy/testlegacyfillrun")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testlegacyfillrun", desc);
  }

private:
  void beginRun(edm::Run const& run, edm::EventSetup const&) override {
    edm::Service<DQMStore> store;
    mymes_.bookall(*store);
    mymes_.fillall(0, run.run(), myvalue_);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {}

  BookerFiller<DQMStore, MonitorElement> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestLegacyFillRunEDAnalyzer);

class TestLegacyFillLumiEDAnalyzer : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit TestLegacyFillLumiEDAnalyzer(const edm::ParameterSet& iConfig)
      : mymes_(iConfig.getParameter<std::string>("folder"), iConfig.getParameter<int>("howmany")),
        myvalue_(iConfig.getParameter<double>("value")) {}

  ~TestLegacyFillLumiEDAnalyzer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("folder", "Legacy/testlegacyfilllumi")->setComment("Where to put all the histograms");
    desc.add<int>("howmany", 1)->setComment("How many copies of each ME to put");
    desc.add<double>("value", 1)->setComment("Which value to use on the third axis (first two are lumi and run)");
    descriptions.add("testlegacyfilllumi", desc);
  }

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override {
    edm::Service<DQMStore> store;
    mymes_.bookall(*store);
  }

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) override {
    mymes_.fillall(lumi.luminosityBlock(), lumi.run(), myvalue_);
  }

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {}

  BookerFiller<DQMStore, MonitorElement, /* DOLUMI */ true> mymes_;
  double myvalue_;
};
DEFINE_FWK_MODULE(TestLegacyFillLumiEDAnalyzer);
