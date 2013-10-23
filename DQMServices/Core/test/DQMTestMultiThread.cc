#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class DQMTestMultiThread
    : public DQMEDAnalyzer
{
 public:
  DQMTestMultiThread(void);
  DQMTestMultiThread(const edm::ParameterSet&);

  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;

  virtual void endRunSummary(edm::Run const&,
                             edm::EventSetup const&,
                             int*) const override;
  virtual void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         int*) const override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void bookHistograms(edm::Run const&,
                              uint32_t streamId,
                              uint32_t moduleId) override;

 private:
  void transferMEs(edm::Run const&, edm::EventSetup const&) const;

  MonitorElement * myHisto;
};


DQMTestMultiThread::DQMTestMultiThread(void){}
DQMTestMultiThread::DQMTestMultiThread(const edm::ParameterSet&){}

void DQMTestMultiThread::transferMEs(edm::Run const&,
                                     edm::EventSetup const&) const {
  // do nothing for the moment
}

void DQMTestMultiThread::bookHistograms(edm::Run const & iRun,
                                        uint32_t streamId,
                                        uint32_t moduleId) {
  DQMStore * store = edm::Service<DQMStore>().operator->();
  store->bookTransaction([&](DQMStore::IBooker & b) {
                           b.setCurrentFolder("/MyDetector");
                           myHisto = b.book1D("MyHisto",
                                              "MyHisto",
                                              100, 0., 100.);
                         }, iRun.run(),
                         streamId,
                         moduleId);
}


void DQMTestMultiThread::beginRun(edm::Run const &iRun,
                                 edm::EventSetup const &iSetup) {
  bookHistograms(iRun, streamId(), iRun.moduleCallingContext()->moduleDescription()->id());
}


void DQMTestMultiThread::endRunSummary(edm::Run const &iRun,
                                       edm::EventSetup const &iSetup,
                                       int*) const {
  transferMEs(iRun, iSetup);
}

void DQMTestMultiThread::endLuminosityBlockSummary(edm::LuminosityBlock const &iRun,
                                                   edm::EventSetup const &iSetup,
                                                   int*) const {
  //  transferMEs(iRun, iSetup);
}

void DQMTestMultiThread::analyze(const edm::Event&, const edm::EventSetup&)
{}

// define this as a plug-in
DEFINE_FWK_MODULE(DQMTestMultiThread);
