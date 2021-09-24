#ifndef CaloOnlineTools_EcalTools_EcalURecHitHists_h
#define CaloOnlineTools_EcalTools_EcalURecHitHists_h
// -*- C++ -*-
//
// Package:   EcalURecHitHists
// Class:     EcalURecHitHists
//
/**\class EcalURecHitHists EcalURecHitHists.cc

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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TNtuple.h"

//
// class declaration
//

class EcalURecHitHists : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit EcalURecHitHists(const edm::ParameterSet&);
  ~EcalURecHitHists() override;

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override;
  std::string intToString(int num);
  void initHists(int);

  // ----------member data ---------------------------

  const edm::InputTag ebUncalibratedRecHitCollection_;
  const edm::InputTag eeUncalibratedRecHitCollection_;

  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> ebUncalibRecHitsToken_;
  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> eeUncalibRecHitsToken_;
  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> ecalMappingToken_;

  int runNum_;
  const double histRangeMax_, histRangeMin_;
  std::string fileName_;

  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<std::string> maskedEBs_;
  std::map<int, TH1F*> FEDsAndHists_;
  std::map<int, TH1F*> FEDsAndTimingHists_;

  TH1F* allFedsHist_;
  TH1F* allFedsTimingHist_;

  TFile* file;
  EcalFedMap* fedMap_;
  const EcalElectronicsMapping* ecalElectronicsMap_;
};

#endif
