#ifndef ESRawDataTask_H
#define ESRawDataTask_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class ESRawDataTask : public DQMEDAnalyzer {
public:
  ESRawDataTask(const edm::ParameterSet& ps);
  ~ESRawDataTask() override {}

protected:
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// Setup
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  int ievt_;

  std::string prefixME_;

  edm::EDGetTokenT<ESRawDataCollection> dccCollections_;
  edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;

  //MonitorElement* meRunNumberErrors_;
  MonitorElement* meL1ADCCErrors_;
  MonitorElement* meBXDCCErrors_;
  MonitorElement* meOrbitNumberDCCErrors_;
  MonitorElement* meL1ADiff_;
  MonitorElement* meBXDiff_;
  MonitorElement* meOrbitNumberDiff_;

  int runNum_;
};

#endif
