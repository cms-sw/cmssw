#ifndef EcalFEDMonitor_H
#define EcalFEDMonitor_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <iostream>

// Making the class templated temporarily, until HLT sequence can be fixed (is using EBHltTask and EEHltTask currently)
template<int SUBDET>
class EcalFEDMonitorTemp : public DQMEDAnalyzer{
 public:
  EcalFEDMonitorTemp(edm::ParameterSet const&);
  ~EcalFEDMonitorTemp() {}

 private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  enum MEs {
    kEBOccupancy,
    kEBFatal,
    kEBNonFatal,
    kEEOccupancy,
    kEEFatal,
    kEENonFatal,
    nMEs
  };

  std::string folderName_;

  edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataToken_;
  edm::EDGetTokenT<EBDetIdCollection> ebGainErrorsToken_;
  edm::EDGetTokenT<EEDetIdCollection> eeGainErrorsToken_;
  edm::EDGetTokenT<EBDetIdCollection> ebChIdErrorsToken_;
  edm::EDGetTokenT<EEDetIdCollection> eeChIdErrorsToken_;
  edm::EDGetTokenT<EBDetIdCollection> ebGainSwitchErrorsToken_;
  edm::EDGetTokenT<EEDetIdCollection> eeGainSwitchErrorsToken_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> towerIdErrorsToken_;
  edm::EDGetTokenT<EcalElectronicsIdCollection> blockSizeErrorsToken_;

  std::vector<MonitorElement*> MEs_;
};

#endif
