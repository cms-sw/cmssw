#ifndef ESIntegrityTask_H
#define ESIntegrityTask_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

struct ESLSCache {
  int ievtLS_;
  int DIErrorsLS_[2][2][40][40];
};

class ESIntegrityTask : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<ESLSCache>> {
public:
  ESIntegrityTask(const edm::ParameterSet& ps);
  ~ESIntegrityTask() override {}

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// EndJob
  void endJob(void) override;

  /// EndRun
  void dqmEndRun(const edm::Run& r, const edm::EventSetup& c) override;

  /// Begin Lumi
  std::shared_ptr<ESLSCache> globalBeginLuminosityBlock(const edm::LuminosityBlock& lumi,
                                                        const edm::EventSetup& c) const override;

  /// End Lumi
  void globalEndLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& c) override;

  /// Calculate Data Integrity Fraction
  void calculateDIFraction(const edm::LuminosityBlock& lumi, const edm::EventSetup& c);

private:
  int ievt_;

  std::string prefixME_;

  edm::EDGetTokenT<ESRawDataCollection> dccCollections_;
  edm::EDGetTokenT<ESLocalRawDataCollection> kchipCollections_;

  MonitorElement* meGain_;
  MonitorElement* meFED_;
  MonitorElement* meSLinkCRCErr_;
  MonitorElement* meDCCErr_;
  MonitorElement* meDCCCRCErr_;
  MonitorElement* meOptoRX_;
  MonitorElement* meOptoBC_;
  MonitorElement* meFiberBadStatus_;
  MonitorElement* meFiberErrCode_;
  MonitorElement* meFiberOff_;
  MonitorElement* meEVDR_;
  MonitorElement* meKF1_;
  MonitorElement* meKF2_;
  MonitorElement* meKBC_;
  MonitorElement* meKEC_;
  MonitorElement* meDIErrors_[2][2];
  MonitorElement* meDIErrorsLS_[2][2];
  MonitorElement* meDIFraction_;

  edm::FileInPath lookup_;

  int runNum_, eCount_, runtype_, seqtype_, dac_, gain_, precision_;
  int firstDAC_, nDAC_, isPed_, vDAC_[5];
  int fed_[2][2][40][40], kchip_[2][2][40][40], fiber_[2][2][40][40];
  bool doLumiAnalysis_;
};

#endif
