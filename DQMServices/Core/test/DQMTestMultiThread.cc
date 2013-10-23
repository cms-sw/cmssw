#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/MakerMacros.h"

class DQMTestMultiThread
    : public DQMEDAnalyzer
{
 public:
  DQMTestMultiThread(void);
  DQMTestMultiThread(const edm::ParameterSet&);

  virtual void endRunSummary(edm::Run const&,
                             edm::EventSetup const&,
                             int*) const override;
  virtual void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                         edm::EventSetup const&,
                                         int*) const override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  void transferMEs(edm::Run const&, edm::EventSetup const&) const;
};


DQMTestMultiThread::DQMTestMultiThread(void){}
DQMTestMultiThread::DQMTestMultiThread(const edm::ParameterSet&){}

void DQMTestMultiThread::transferMEs(edm::Run const&,
                                     edm::EventSetup const&) const {
  // do nothing for the moment
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
