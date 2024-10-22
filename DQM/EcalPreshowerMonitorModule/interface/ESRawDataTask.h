#ifndef ESRawDataTask_H
#define ESRawDataTask_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

struct ESRawLSCache {
  int ievtLS_;
};

class ESRawDataTask : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<ESRawLSCache>> {
public:
  ESRawDataTask(const edm::ParameterSet& ps);
  ~ESRawDataTask() override {}

protected:
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// Setup
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  /// Begin Lumi
  std::shared_ptr<ESRawLSCache> globalBeginLuminosityBlock(const edm::LuminosityBlock& lumi,
                                                           const edm::EventSetup& c) const override;
  /// End Lumi
  void globalEndLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& c) override;

private:
  int ievt_;

  std::string prefixME_;

  edm::EDGetTokenT<ESRawDataCollection> dccCollections_;
  edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;

  MonitorElement* meL1ADCCErrors_;
  MonitorElement* meL1ADCCErrorsByLS_;
  MonitorElement* meBXDCCErrors_;
  MonitorElement* meBXDCCErrorsByLS_;
  MonitorElement* meOrbitNumberDCCErrors_;
  MonitorElement* meOrbitNumberDCCErrorsByLS_;
  MonitorElement* meL1ADiff_;
  MonitorElement* meBXDiff_;
  MonitorElement* meOrbitNumberDiff_;

  int runNum_;
};

#endif
