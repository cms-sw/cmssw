// -*- C++ -*-
//
// Package:   EcalMipGraphs
// Class:     EcalMipGraphs
//
/**\class EcalMipGraphs EcalMipGraphs.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
//
//

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TTree.h"

//
// class declaration
//

class EcalMipGraphs : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit EcalMipGraphs(const edm::ParameterSet&);
  ~EcalMipGraphs() override;

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override;
  std::string intToString(int num);
  std::string floatToString(float num);
  void writeGraphs();
  void initHists(int);
  void selectHits(edm::Handle<EcalRecHitCollection> hits, int ievt);
  TGraph* selectDigi(DetId det, int ievt);
  int getEEIndex(EcalElectronicsId elecId);

  // ----------member data ---------------------------

  const edm::InputTag EBRecHitCollection_;
  const edm::InputTag EERecHitCollection_;
  const edm::InputTag EBDigis_;
  const edm::InputTag EEDigis_;
  const edm::InputTag headerProducer_;

  edm::Handle<EBDigiCollection> EBdigisHandle;
  edm::Handle<EEDigiCollection> EEdigisHandle;

  const edm::EDGetTokenT<EcalRawDataCollection> rawDataToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHitToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHitToken_;
  const edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;

  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> ecalMappingToken_;
  const edm::ESGetToken<CaloTopology, CaloTopologyRecord> topologyToken_;

  int runNum_;
  const int side_;
  const double threshold_;
  const double minTimingAmp_;

  std::set<EBDetId> listEBChannels;
  std::set<EEDetId> listEEChannels;

  int abscissa[10];
  int ordinate[10];

  static float gainRatio[3];
  static edm::Service<TFileService> fileService;

  std::vector<std::string>* names;
  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<int> seedCrys_;
  std::vector<std::string> maskedEBs_;
  std::map<int, TH1F*> FEDsAndTimingHists_;
  std::map<int, float> crysAndAmplitudesMap_;
  std::map<int, EcalDCCHeaderBlock> FEDsAndDCCHeaders_;
  std::map<std::string, int> seedFrequencyMap_;

  TH1F* allFedsTimingHist_;

  TFile* file_;
  TTree* canvasNames_;
  EcalFedMap* fedMap_;
  const EcalElectronicsMapping* ecalElectronicsMap_;
  const CaloTopology* caloTopo_;

  int naiveEvtNum_;
};
