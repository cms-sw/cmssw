#ifndef ECALRAWTODIGI_ECALDIGIDISPLAY_h
#define ECALRAWTODIGI_ECALDIGIDISPLAY_h

#include <memory>
#include <vector>
#include <string>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

// class declaration

class EcalDigiDisplay : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  //Constractor
  EcalDigiDisplay(const edm::ParameterSet& ps);
  //Distractor
  ~EcalDigiDisplay() override;

private:
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void beginRun(edm::Run const&, edm::EventSetup const& c) override;
  void endRun(edm::Run const&, edm::EventSetup const& c) override;
  void endJob() override;

protected:
  void readEBDigis(edm::Handle<EBDigiCollection> digis, int Mode);
  void readEEDigis(edm::Handle<EEDigiCollection> digis, int Mode);
  void readPNDigis(edm::Handle<EcalPnDiodeDigiCollection> PNs, int Mode);

  EcalFedMap* fedMap;

  const std::string ebDigiCollection_;
  const std::string eeDigiCollection_;
  const std::string digiProducer_;

  const edm::EDGetTokenT<EcalRawDataCollection> rawDataToken_;
  const edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;
  const edm::EDGetTokenT<EcalPnDiodeDigiCollection> pnDiodeDigiToken_;
  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> ecalMappingToken_;

  std::vector<int> requestedFeds_;
  std::vector<std::string> requestedEbs_;

  bool inputIsOk;
  bool cryDigi;
  bool ttDigi;
  bool pnDigi;

  int mode;

  std::vector<int> listChannels;
  std::vector<int> listTowers;
  std::vector<int> listPns;

  const EcalElectronicsMapping* ecalElectronicsMap_;
};
#endif
