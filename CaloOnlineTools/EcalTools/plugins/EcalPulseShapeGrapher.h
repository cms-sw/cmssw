// -*- C++ -*-
//
// Package:    EcalPulseShapeGrapher
// Class:      EcalPulseShapeGrapher
//
/**\class EcalPulseShapeGrapher EcalPulseShapeGrapher.cc EcalPulseShapeGrapher.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth Cooper
//         Created:  Tue Feb  5 11:35:45 CST 2008
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
#include "TFile.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TH1F.h"
#include <vector>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

//
// class decleration
//

class EcalPulseShapeGrapher : public edm::one::EDAnalyzer<> {
public:
  explicit EcalPulseShapeGrapher(const edm::ParameterSet&);
  ~EcalPulseShapeGrapher() override = default;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::string intToString(int);

  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> EBUncalibratedRecHitCollection_;
  const edm::EDGetTokenT<EBDigiCollection> EBDigis_;
  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> EEUncalibratedRecHitCollection_;
  const edm::EDGetTokenT<EEDigiCollection> EEDigis_;

  int abscissa[10];
  int ordinate[10];
  std::vector<int> listChannels_;
  std::map<int, TH1F*> ampHistMap_;
  std::map<int, TH2F*> pulseShapeHistMap_;
  std::map<int, TH1F*> firstSampleHistMap_;
  std::map<int, TH2F*> rawPulseShapeHistMap_;
  std::map<int, TH1F*> cutAmpHistMap_;

  int ampCut_;
  std::string rootFilename_;

  TFile* file_;

  EcalFedMap* fedMap_;
  // ----------member data ---------------------------
};
