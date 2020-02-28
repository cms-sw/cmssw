#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "string"

class DQMTestMultiThread : public DQMEDAnalyzer {
public:
  explicit DQMTestMultiThread(const edm::ParameterSet &);

  void analyze(const edm::Event &, const edm::EventSetup &) override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void dumpMe(MonitorElement const &, bool printStat = false);

private:
  MonitorElement *myHisto;
  std::string folder_;
  double fill_value_;
  bool debug_;
};

DQMTestMultiThread::DQMTestMultiThread(const edm::ParameterSet &pset)
    : folder_(pset.getUntrackedParameter<std::string>("folder")),
      fill_value_(pset.getUntrackedParameter<double>("fillValue", 1.)),
      debug_(pset.getUntrackedParameter<bool>("debug", false)) {}

void DQMTestMultiThread::bookHistograms(DQMStore::IBooker &b,
                                        edm::Run const & /* iRun*/,
                                        edm::EventSetup const & /* iSetup*/) {
  b.setCurrentFolder("");
  b.setCurrentFolder(folder_);
  myHisto = b.book1D("MyHisto", "MyHisto", 100, -0.5, 99.5);
  DQMStore *store = edm::Service<DQMStore>().operator->();
  if (debug_) {
    std::cout << std::endl;
    for (auto me : store->getAllContents("")) {
      dumpMe(*me);
    }
  }
}

void DQMTestMultiThread::analyze(const edm::Event &iEvent, const edm::EventSetup &) { myHisto->Fill(fill_value_); }

void DQMTestMultiThread::dumpMe(MonitorElement const &me, bool printStat /* = false */) {
  std::cout << " LumiFlag: " << me.getLumiFlag() << " fullpathname: " << me.getPathname();
  if (printStat)
    std::cout << " Mean: " << me.getMean() << " RMS: " << me.getRMS() << " Entries: " << std::setprecision(9)
              << me.getEntries();
  std::cout << std::endl;
}

// define this as a plug-in
DEFINE_FWK_MODULE(DQMTestMultiThread);
