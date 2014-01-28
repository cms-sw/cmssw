#ifndef CSCTrackFinder_CSCTFEfficiencies_h
#define CSCTrackFinder_CSCTFEfficiencies_h

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"

//ROOT
#include <TH1F.h>
#include <TH2D.h>
#include <TH1I.h>
#include <TFile.h>
#include <TTree.h>
#include <TStyle.h>
#include <TCanvas.h>

class CSCTFEfficiencies : public edm::EDAnalyzer {
 public:
  explicit CSCTFEfficiencies(edm::ParameterSet const& conf);
  virtual ~CSCTFEfficiencies() {}
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();
  virtual void beginJob();
  int ghosts;
  int haloGhosts;
  int lostTracks;
  int DebugCounter;
  unsigned haloTrigger;
 private:
  // variables persistent across events should be declared here.
  //
	CSCTFDTReceiver* my_dtrc;
	CSCSectorReceiverLUT *srLUTs_[5][6][2];
	
  edm::InputTag lctProducer;
  std::string outFile;//c
  TFile* fAnalysis;//c

  void DeleteHistos();
  Int_t cnttrk, cntGen;

	TH1F* modeOcc;
  TH1F* simEta, *simPhi, *simPt, *simPz, *simP, *simEHalo, *fidPtDen;
  TH1F* trackedEta, *trackedPhi, *trackedPt, *trackedBx, *trackedEHalo, *trackedPtHalo;
  TH1F* matchedPhi, *matchedEta, *matchedPt, *Radius, *HaloPRes;
  TH1F* EffEtaAll, *EffEtaQ3, *EffEtaQ2, *EffEtaQ1, *EffPhi, *EffPt, *EffEn;
  TH1F* EtaQ3, *EtaQ2, *EtaQ1;
  TH1F* LostPhi, *LostEta;
  TH1F* ghostPhi, *ghostEta, *ghostPt, *ghostRadius;
  TH1F* etaResolution, *phiResolution, *ptResolution;
  TH1F* matchedPt10, *matchedPt20, *matchedPt40, *matchedPt60;
  TH1F* EffPt10, *EffPt20, *EffPt40, *EffPt60;
  TH1F* ptResolutionEtaLow, *ptResolutionEtaHigh, *ptResolutionQ3;
  TH1F* numEScat, *ghostDelPhi, *ghostDelEta, *ghostTrackRad, *ghostselectPtRes, *ghostdropPtRes ;
  TH1F* simHaloPipeOff, *trackedHaloPipeOff, *EffHaloPipeOff, *simHaloPipeOff2, *LostHaloPipeOff;
  TH2F* simHaloPosition, *trackHaloPosition, *lostHaloPosition;
  TH2F* PhiResVPt;
  TH2F* PtResVPt, *PtResVEta;
	TH1F* dtStubBx;
	TH1F* overDeleta12, *overDelphi12, *overDeleta25, *overDelphi25, *overDeleta15, *overDelphi15;
  TLegend* TrackerLeg1, *TrackerLeg2;
  
};

DEFINE_FWK_MODULE(CSCTFEfficiencies);

#endif
