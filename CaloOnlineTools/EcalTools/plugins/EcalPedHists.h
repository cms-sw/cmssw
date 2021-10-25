#ifndef CaloOnlineTools_EcalTools_EcalPedHists_h
#define CaloOnlineTools_EcalTools_EcalPedHists_h
/**
 * Module which outputs a root file of ADC counts (all three gains)
 *   of user-selected channels (defaults to channel 1) for 
 *   user-selected samples (defaults to samples 1,2,3) for
 *   user-selected supermodules.
 * 
 * \author S. Cooper
 *
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include <iostream>
#include <vector>
#include <set>
#include "TFile.h"
#include "TH1.h"
#include "TDirectory.h"

typedef std::map<std::string, TH1F*> stringHistMap;

class EcalPedHists : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  EcalPedHists(const edm::ParameterSet& ps);
  ~EcalPedHists() override;

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void beginRun(edm::Run const&, edm::EventSetup const& c) override;
  void endRun(edm::Run const&, edm::EventSetup const& c) override;
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
  const edm::InputTag barrelDigiCollection_;
  const edm::InputTag endcapDigiCollection_;
  const edm::InputTag headerProducer_;
  std::vector<int> listChannels_;
  std::vector<int> listSamples_;
  std::vector<int> listFEDs_;
  std::vector<std::string> listEBs_;
  std::map<int, stringHistMap> FEDsAndHistMaps_;
  std::set<int> theRealFedSet_;
  EcalFedMap* fedMap_;
  TFile* root_file_;

  const edm::EDGetTokenT<EcalRawDataCollection> rawDataToken_;
  const edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;
  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> ecalMappingToken_;

  const EcalElectronicsMapping* ecalElectronicsMap_;
};

#endif
