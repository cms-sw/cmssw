#ifndef ECALSIMPLE2007H4TBANALYZER_H
#define ECALSIMPLE2007H4TBANALYZER_H

/**\class EcalSimple2007H4TBAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <string>
#include "TH1.h"
#include "TGraph.h"
#include "TH2.h"
#include <fstream>
#include <map>

class EcalSimple2007H4TBAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit EcalSimple2007H4TBAnalyzer(const edm::ParameterSet&);
  ~EcalSimple2007H4TBAnalyzer() override;

  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;

private:
  const std::string rootfile_;
  const std::string digiCollection_;
  const std::string digiProducer_;
  const std::string hitCollection_;
  const std::string hitProducer_;
  const std::string hodoRecInfoCollection_;
  const std::string hodoRecInfoProducer_;
  const std::string tdcRecInfoCollection_;
  const std::string tdcRecInfoProducer_;
  const std::string eventHeaderCollection_;
  const std::string eventHeaderProducer_;

  const edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;
  const edm::EDGetTokenT<EEUncalibratedRecHitCollection> eeUncalibratedRecHitToken_;
  const edm::EDGetTokenT<EcalTBHodoscopeRecInfo> tbHodoscopeRecInfoToken_;
  const edm::EDGetTokenT<EcalTBTDCRecInfo> tbTDCRecInfoToken_;
  const edm::EDGetTokenT<EcalTBEventHeader> tbEventHeaderToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;

  // Amplitude vs TDC offset
  TH2F* h_ampltdc;

  TH2F* h_Shape_;

  // Reconstructed energies
  TH1F* h_tableIsMoving;
  TH1F* h_e1x1;
  TH1F* h_e3x3;
  TH1F* h_e5x5;

  TH1F* h_e1x1_center;
  TH1F* h_e3x3_center;
  TH1F* h_e5x5_center;

  TH1F* h_e1e9;
  TH1F* h_e1e25;
  TH1F* h_e9e25;

  TH1F* h_S6;
  TH1F* h_bprofx;
  TH1F* h_bprofy;

  TH1F* h_qualx;
  TH1F* h_qualy;

  TH1F* h_slopex;
  TH1F* h_slopey;

  TH2F* h_mapx[25];
  TH2F* h_mapy[25];

  TH2F* h_e1e9_mapx;
  TH2F* h_e1e9_mapy;

  TH2F* h_e1e25_mapx;
  TH2F* h_e1e25_mapy;

  TH2F* h_e9e25_mapx;
  TH2F* h_e9e25_mapy;

  EEDetId xtalInBeam_;
  EBDetId xtalInBeamTmp;
  EEDetId Xtals5x5[25];

  const CaloGeometry* theTBGeometry_;
};

#endif
