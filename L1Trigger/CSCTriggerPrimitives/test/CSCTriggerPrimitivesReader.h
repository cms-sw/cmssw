#ifndef CSCTriggerPrimitives_CSCTriggerPrimitivesReader_h
#define CSCTriggerPrimitives_CSCTriggerPrimitivesReader_h

/** \class CSCTriggerPrimitivesReader
 *
 * Basic analyzer class which accesses ALCTs, CLCTs, and correlated LCTs
 * and plot various quantities.
 *
 * \author Slava Valuev, UCLA.
 *
 * $Date: 2006/06/07 10:52:28 $
 * $Revision: 1.1 $
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>

class TFile;
class TH1F;

class CSCTriggerPrimitivesReader : public edm::EDAnalyzer
{
 public:
  /// Constructor
  explicit CSCTriggerPrimitivesReader(const edm::ParameterSet& conf);

  /// Destructor
  virtual ~CSCTriggerPrimitivesReader();

  /// Does the job
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

  /// Write to ROOT file, make plots, etc.
  void endJob();

 private:
  int eventsAnalyzed;       // event number
  bool debug;               // on/off switch
  //std::string rootFileName; // root file name

  // Module label
  std::string lctProducer_;

  // The file which will store the histos
  // TFile *theFile;

  enum trig_cscs {MAX_STATIONS = 4, CSC_TYPES = 10};
  enum {MAXPAGES = 20};     // max. number of pages in postscript file

  // Various useful constants
  static const std::string csc_type[CSC_TYPES];
  static const int MAX_HS[CSC_TYPES];
  static const int ptype[CSCConstants::NUM_CLCT_PATTERNS];

  // LCT counters
  static int numALCT;
  static int numCLCT;
  static int numLCT;

  static bool bookedALCTHistos;
  static bool bookedCLCTHistos;
  static bool bookedLCTHistos;

  void setRootStyle();

  void bookALCTHistos();
  void bookCLCTHistos();
  void bookLCTHistos();
  void fillALCTHistos(const CSCALCTDigiCollection* alcts);
  void fillCLCTHistos(const CSCCLCTDigiCollection* clcts);
  void fillLCTHistos(const CSCCorrelatedLCTDigiCollection* lcts);
  void drawALCTHistos();
  void drawCLCTHistos();
  void drawLCTHistos();

  int getCSCType(const CSCDetId& id);

  // Histograms
  TH1F *hAlctPerEvent, *hAlctPerCSC;
  TH1F *hAlctValid, *hAlctQuality, *hAlctAccel, *hAlctCollis, *hAlctKeyGroup;
  TH1F *hAlctBXN;
  TH1F *hClctPerEvent, *hClctPerCSC;
  TH1F *hClctValid, *hClctQuality, *hClctStripType, *hClctSign, *hClctBXN;
  TH1F *hClctKeyStrip[2], *hClctPattern[2];
  TH1F *hClctPatternCsc[CSC_TYPES][2], *hClctKeyStripCsc[CSC_TYPES];
  TH1F *hLctPerEvent, *hLctPerCSC;
  TH1F *hLctEndcap, *hLctStation, *hLctSector, *hLctRing;
  TH1F *hLctChamber[MAX_STATIONS];
  TH1F *hLctValid, *hLctQuality, *hLctKeyGroup, *hLctKeyStrip, *hLctStripType;
  TH1F *hLctPattern, *hLctBend, *hLctBXN;
};

#endif
