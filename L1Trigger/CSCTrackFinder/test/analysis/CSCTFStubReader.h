#ifndef CSCTrackFinder_CSCTFStubReader_h
#define CSCTrackFinder_CSCTFStubReader_h

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

#include "L1Trigger/CSCTriggerPrimitives/src/CSCTriggerPrimitivesProducer.h"
#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>

#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"//
#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1I.h>
#include <TTree.h>//
#include <TF1.h>//
#include <TH1F.h>
#include <TH2F.h>//
#include <TString.h>
#include <TStyle.h>//

  //==========ported from ORCA L1CSCTriggerStudy==================================

class CSCGeometry;
class TFile;
enum {MAX_STATIONS = 4, CSC_TYPES = 10, MAX_SECTORS = 6, MAX_CHAMBERS =9};
const TString stationLabel[MAX_STATIONS+1] = {"1a","1b","2","3","4"};
const TString sectorLabel[MAX_SECTORS] = {"1","2","3","4","5","6"};
const TString chamberLabel[MAX_CHAMBERS] = {"1","2","3","4","5","6","7","8","9"};
const std::string FPGAs[5] = {"F1","F2","F3","F4","F5"};//

class CSCTFStubReader : public edm::EDAnalyzer
{
 public:
  // Constructors
  explicit CSCTFStubReader(const edm::ParameterSet& conf);

  // Destructor
  virtual ~CSCTFStubReader();
  /// Does the job
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

  /// Write to ROOT file, make plots, etc.
  void endJob();

 private:
  int event;
  // Cache geometry for current event
  const CSCGeometry* geom_;
  bool debug;               // on/off switch
  // Module labels
  std::string   lctProducer_;
  edm::InputTag wireDigiProducer_;
  edm::InputTag compDigiProducer_;

 
  enum {MAXPAGES = 50};      // max. number of pages in postscript files
  static const double TWOPI; // 2.*pi
  static const std::string csc_type[CSC_TYPES];
  static const Int_t MAX_WG[CSC_TYPES];
  static const Int_t MAX_HS[CSC_TYPES];
  static const int ptype[CSCTFConstants::NUM_CLCT_PATTERNS];

  static bool bookedMuSimHitsVsMuDigis;

  // File to store the generated hisograms
  std::string outFile;
  TFile* fAnalysis;
  void setRootStyle() ;
  Double_t getHsPerRad(const Int_t i);

  void MCStudies(const edm::Event& ev,
		 const CSCCorrelatedLCTDigiCollection* lcts,
		 const CSCALCTDigiCollection* alcts,
		 const CSCCLCTDigiCollection* clcts);
  void fillMuSimHitsVsMuDigis(const edm::Event& ev,
			      const CSCCorrelatedLCTDigiCollection* lcts,
			      const CSCALCTDigiCollection* alcts,
			      const CSCCLCTDigiCollection* clcts,
			      const CSCWireDigiCollection* wiredc,
			      const CSCComparatorDigiCollection* compdc,
			      const edm::PSimHitContainer* allSimHits);
  int    getCSCType(const CSCDetId& id);

 
    //those for stubs
  void bookMuSimHitsVsMuDigis();
  void drawMuSimHitsVsMuDigis();
  void drawALCTHistos();
  void drawCLCTHistos();
  void deleteMuSimHitsVsMuDigis();

  // ALCTs
  TH1F *hAlctPerEvent, *hAlctPerCSC;
  TH1F *hAlctValid, *hAlctQuality, *hAlctAccel, *hAlctCollis, *hAlctKeyGroup;
  TH1F *hAlctBXN;
  // CLCTs
  TH1F *hClctPerEvent, *hClctPerCSC;
  TH1F *hClctValid, *hClctQuality, *hClctStripType, *hClctSign, *hClctCFEB;
  TH1F *hClctBXN;
  TH1F *hClctKeyStrip[2], *hClctPattern[2];
  TH1F *hClctPatternCsc[CSC_TYPES][2], *hClctKeyStripCsc[CSC_TYPES];

  // Correlated LCTs in MPC
  TH1F *hLctMPCPerEvent, *hLctMPCPerCSC, *hCorrLctMPCPerCSC;
  TH1F *hLctMPCEndcap, *hLctMPCStation, *hLctMPCSector, *hLctMPCRing;
  TH1F *hLctMPCChamber[MAX_STATIONS];
  TH1F *hLctMPCValid, *hLctMPCQuality, *hLctMPCKeyGroup;
  TH1F *hLctMPCKeyStrip, *hLctMPCStripType;
  TH1F *hLctMPCPattern, *hLctMPCBend, *hLctMPCBXN;

  TH1F *LctVsEta[MAX_STATIONS][2];
  TH1F *LctVsPhi[MAX_STATIONS];
  TH1F *EtaDiffVsEta[MAX_STATIONS], *EtaDiffVsPhi[MAX_STATIONS];
  TH1F *PhiDiffVsEta[MAX_STATIONS], *PhiDiffVsPhi[MAX_STATIONS];

  TH2F *EtaRecVsSim;
  TH1F *EtaDiff[2], *EtaDiffCsc[CSC_TYPES][6], *LctVsEtaCsc[CSC_TYPES];
  TH2F *EtaDiffVsWireCsc[CSC_TYPES], *EtaDiffVsStripCsc[CSC_TYPES][4];

  TH2F *PhiRecVsSim;
  TH1F *PhiDiff[2], *PhiDiffCsc[CSC_TYPES][9], *PhiDiffPattern[9];
  TH2F *PhiDiffVsWireCsc[CSC_TYPES], *PhiDiffVsStripCsc[CSC_TYPES][2];
  TH1F *KeyStripCsc[CSC_TYPES], *PatternCsc[CSC_TYPES][2];
};

DEFINE_FWK_MODULE(CSCTFStubReader)

#endif // CSCTFSTUBREADER_H
