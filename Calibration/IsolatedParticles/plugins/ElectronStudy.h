#ifndef SimG4CMS_Calo_ElectronStudy_H
#define SimG4CMS_Calo_ElectronStudy_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <TH1F.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class ElectronStudy: public edm::EDAnalyzer {

public:

  ElectronStudy(const edm::ParameterSet& ps);
  ~ElectronStudy() {}

  void analyze  (const edm::Event& e, const edm::EventSetup& c);

private:

  static const int NEtaBins = 3;
  static const int NPBins   = 8; 
  double           pBins[NPBins+1], etaBins[NEtaBins+1];

  std::string      sourceLabel, g4Label, hitLabEB, hitLabEE;
  int              hotZone, verbose;
  bool             histos;
  TH1F             *histoR1[NPBins+1][NEtaBins+1], *histoR2[NPBins+1][NEtaBins+1];
  TH1F             *histoR3[NPBins+1][NEtaBins+1], *histoE1x1[NPBins+1][NEtaBins+1];
  TH1F             *histoE3x3[NPBins+1][NEtaBins+1], *histoE5x5[NPBins+1][NEtaBins+1];
};

#endif
