/**
 * Module which outputs a root file of ADC counts (all three gains)
 *   of user-selected channels (defaults to channel 1) for
 *   user-selected samples (defaults to samples 1,2,3) for
 *   user-selected supermodules.
 *
 * \author S. Cooper
 *
 */

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "TDirectory.h"
#include "TFile.h"
#include "TH1.h"
#include <iostream>
#include <set>
#include <vector>

typedef std::map<std::string, TH1F *> stringHistMap;

class EcalPedHists : public edm::EDAnalyzer {
public:
  EcalPedHists(const edm::ParameterSet &ps);
  ~EcalPedHists() override;

protected:
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void beginRun(edm::Run const &, edm::EventSetup const &c) override;
  void endJob(void) override;

private:
  std::string intToString(int num);
  void readEBdigis(edm::Handle<EBDigiCollection> digis);
  void readEEdigis(edm::Handle<EEDigiCollection> digis);
  void initHists(int FED);

  int runNum_;
  bool inputIsOk_;
  bool allFEDsSelected_;
  bool histsFilled_;
  std::string fileName_;
  edm::InputTag barrelDigiCollection_;
  edm::InputTag endcapDigiCollection_;
  edm::InputTag headerProducer_;
  std::vector<int> listChannels_;
  std::vector<int> listSamples_;
  std::vector<int> listFEDs_;
  std::vector<std::string> listEBs_;
  std::map<int, stringHistMap> FEDsAndHistMaps_;
  std::set<int> theRealFedSet_;
  EcalFedMap *fedMap_;
  TFile *root_file_;
  const EcalElectronicsMapping *ecalElectronicsMap_;
};
