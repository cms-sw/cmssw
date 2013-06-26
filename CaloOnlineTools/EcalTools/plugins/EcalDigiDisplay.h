#ifndef ECALRAWTODIGI_ECALDIGIDISPLAY_h
#define ECALRAWTODIGI_ECALDIGIDISPLAY_h

#include <memory>
#include <vector>
#include <string>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

// class declaration

class EcalDigiDisplay : public edm::EDAnalyzer {
 public:
  //Constractor
  EcalDigiDisplay(const edm::ParameterSet& ps);
  //Distractor
  ~EcalDigiDisplay();
  
 private:
  virtual void analyze( edm::Event const & e, edm::EventSetup const & c);
  virtual void beginRun(edm::Run const &, edm::EventSetup const & c);
  virtual void endJob();
  
 protected:
  void readEBDigis (edm::Handle<EBDigiCollection> digis, int Mode);
  void readEEDigis (edm::Handle<EEDigiCollection> digis, int Mode);
  void readPNDigis (edm::Handle<EcalPnDiodeDigiCollection> PNs, int Mode);
  
  EcalFedMap* fedMap;

  std::string ebDigiCollection_;
  std::string eeDigiCollection_;
  std::string digiProducer_;

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
