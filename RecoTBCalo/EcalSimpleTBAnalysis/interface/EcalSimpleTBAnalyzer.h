#ifndef ECALSIMPLETBANALYZER_H
#define ECALSIMPLETBANALYZER_H

/**\class EcalSimpleTBAnalyzer

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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <string>
//#include "TTree.h"
#include "TH1.h"
#include "TGraph.h"
#include "TH2.h"
#include <fstream>
#include <map>
//#include<stl_pair>

class EcalSimpleTBAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit EcalSimpleTBAnalyzer(const edm::ParameterSet&);
  ~EcalSimpleTBAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  std::string rootfile_;
  std::string digiCollection_;
  std::string digiProducer_;
  std::string hitCollection_;
  std::string hitProducer_;
  std::string hodoRecInfoCollection_;
  std::string hodoRecInfoProducer_;
  std::string tdcRecInfoCollection_;
  std::string tdcRecInfoProducer_;
  std::string eventHeaderCollection_;
  std::string eventHeaderProducer_;

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

  EBDetId xtalInBeam_;
};

#endif
