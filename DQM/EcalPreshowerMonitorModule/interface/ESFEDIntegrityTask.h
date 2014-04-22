#ifndef ESFEDIntegrityTask_H
#define ESFEDIntegrityTask_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class MonitorElement;

class ESFEDIntegrityTask : public DQMEDAnalyzer {
  
 public:
  
  ESFEDIntegrityTask(const edm::ParameterSet& ps);
  virtual ~ESFEDIntegrityTask() {}
  
 protected:

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void endJob();
  
 private:
  
  int ievt_;
  
  std::string prefixME_;
  std::string fedDirName_;
  bool debug_;

  edm::EDGetTokenT<ESRawDataCollection> dccCollections_;
  edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_; 
  
  MonitorElement* meESFedsEntries_;
  MonitorElement* meESFedsFatal_;
  MonitorElement* meESFedsNonFatal_;

};

#endif
