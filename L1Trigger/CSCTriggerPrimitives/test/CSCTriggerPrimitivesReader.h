#ifndef CSCTriggerPrimitives_CSCTriggerPrimitivesReader_h
#define CSCTriggerPrimitives_CSCTriggerPrimitivesReader_h

/** \class CSCTriggerPrimitivesReader
 *
 * Basic analyzer class which accesses ALCTs, CLCTs, and correlated LCTs
 * and plot various quantities.
 *
 * \author Slava Valuev, UCLA.
 *
 * $Date: 2009/05/20 15:01:34 $
 * $Revision: 1.15 $
 *
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>

#include <L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h> // TMB07

#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>

#include <TH1.h>
#include <TH2.h>

class CSCGeometry;
class CSCBadChambers;
class TFile;

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

  // Cache geometry for current event
  const CSCGeometry* geom_;

  // Cache conditions data for bad chambers
  const CSCBadChambers* badChambers_;

  // Define which LCTs are present in the input file.  This will determine the
  // workflow of the Reader.
  bool dataLctsIn_;
  bool emulLctsIn_;

  // Flag to indicate MTCC data (used only when dataLctsIn_ = true).
  bool isMTCCData_;

  bool isTMB07;

  // Flag to plot or not plot ME1/A as a separate chamber.
  bool plotME1A;

  // Flag to plot or not plot ME4/2.
  bool plotME42;

  // Producer's labels
  std::string   lctProducerData_;
  std::string   lctProducerEmul_;
  edm::InputTag simHitProducer_;
  edm::InputTag wireDigiProducer_;
  edm::InputTag compDigiProducer_;

  // The file which will store the histos
  // TFile *theFile;

  enum trig_cscs {MAX_ENDCAPS = 2, MAX_STATIONS = 4, CSC_TYPES = 10};
  enum {MAXPAGES = 20};      // max. number of pages in postscript files
  static const double TWOPI; // 2.*pi

  // Various useful constants
  static const std::string csc_type[CSC_TYPES];
  static const std::string csc_type_plus[CSC_TYPES];
  static const std::string csc_type_minus[CSC_TYPES];
  static const int NCHAMBERS[CSC_TYPES];
  static const int MAX_WG[CSC_TYPES];
  static const int MAX_HS[CSC_TYPES];
  static const int ptype[CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07];
  static const int ptype_TMB07[CSCConstants::NUM_CLCT_PATTERNS];

  // LCT counters
  static int numALCT;
  static int numCLCT;
  static int numLCTTMB;
  static int numLCTMPC;

  static bool bookedALCTHistos;
  static bool bookedCLCTHistos;
  static bool bookedLCTTMBHistos;
  static bool bookedLCTMPCHistos;

  static bool bookedCompHistos;

  static bool bookedResolHistos;
  static bool bookedEfficHistos;

  void setRootStyle();

  void bookALCTHistos();
  void bookCLCTHistos();
  void bookLCTTMBHistos();
  void bookLCTMPCHistos();
  void fillALCTHistos(const CSCALCTDigiCollection* alcts);
  void fillCLCTHistos(const CSCCLCTDigiCollection* clcts);
  void fillLCTTMBHistos(const CSCCorrelatedLCTDigiCollection* lcts);
  void fillLCTMPCHistos(const CSCCorrelatedLCTDigiCollection* lcts);
  void drawALCTHistos();
  void drawCLCTHistos();
  void drawLCTTMBHistos();
  void drawLCTMPCHistos();

  void bookResolHistos();
  void drawResolHistos();
  void bookEfficHistos();
  void drawEfficHistos();
  void drawHistosForTalks();

  int    getCSCType(const CSCDetId& id);
  double getHsPerRad(const int idh);

  void compare(const CSCALCTDigiCollection* alcts_data,
	       const CSCALCTDigiCollection* alcts_emul,
	       const CSCCLCTDigiCollection* clcts_data,
	       const CSCCLCTDigiCollection* clcts_emul,
	       const CSCCorrelatedLCTDigiCollection* lcts_data,
	       const CSCCorrelatedLCTDigiCollection* lcts_emul);
  void bookCompHistos();
  void compareALCTs(const CSCALCTDigiCollection* alcts_data,
		    const CSCALCTDigiCollection* alcts_emul);
  void compareCLCTs(const CSCCLCTDigiCollection* clcts_data,
		    const CSCCLCTDigiCollection* clcts_emul);
  void compareLCTs(const CSCCorrelatedLCTDigiCollection* lcts_data,
		   const CSCCorrelatedLCTDigiCollection* lcts_emul,
		   const CSCALCTDigiCollection* alcts_data,
		   const CSCCLCTDigiCollection* clcts_data);
  int  convertBXofLCT(const int emul_bx, const CSCDetId& detid,
		      const CSCALCTDigiCollection* alcts_data,
		      const CSCCLCTDigiCollection* clcts_data);
  void drawCompHistos();

  void MCStudies(const edm::Event& ev,
		 const CSCALCTDigiCollection* alcts,
		 const CSCCLCTDigiCollection* clcts);

  void calcResolution(const CSCALCTDigiCollection* alcts,
		      const CSCCLCTDigiCollection* clcts,
		      const CSCWireDigiCollection* wiredc,
		      const CSCComparatorDigiCollection* compdc,
		      const edm::PSimHitContainer* allSimHits);

  void calcEfficiency(const CSCALCTDigiCollection* alcts,
		      const CSCCLCTDigiCollection* clcts,
		      const edm::PSimHitContainer* allSimHits);

  // Histograms
  // ALCTs
  TH1F *hAlctPerEvent, *hAlctPerChamber, *hAlctPerCSC;
  TH1F *hAlctCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hAlctValid, *hAlctQuality, *hAlctAccel, *hAlctCollis, *hAlctKeyGroup;
  TH1F *hAlctBXN;
  // CLCTs
  TH1F *hClctPerEvent, *hClctPerChamber, *hClctPerCSC;
  TH1F *hClctCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hClctValid, *hClctQuality, *hClctStripType, *hClctSign, *hClctCFEB;
  TH1F *hClctBXN;
  TH1F *hClctKeyStrip[2], *hClctPattern[2];
  TH1F *hClctBendCsc[CSC_TYPES][2], *hClctKeyStripCsc[CSC_TYPES];
  // Correlated LCTs in TMB
  TH1F *hLctTMBPerEvent, *hLctTMBPerChamber;
  TH1F *hLctTMBPerCSC, *hCorrLctTMBPerCSC;
  TH1F *hLctTMBEndcap, *hLctTMBStation, *hLctTMBSector, *hLctTMBRing;
  TH1F *hLctTMBChamber[MAX_STATIONS], *hLctTMBCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hLctTMBValid, *hLctTMBQuality, *hLctTMBKeyGroup;
  TH1F *hLctTMBKeyStrip, *hLctTMBStripType;
  TH1F *hLctTMBPattern, *hLctTMBBend, *hLctTMBBXN;
  // Correlated LCTs in MPC
  TH1F *hLctMPCPerEvent, *hLctMPCPerCSC, *hCorrLctMPCPerCSC;
  TH1F *hLctMPCEndcap, *hLctMPCStation, *hLctMPCSector, *hLctMPCRing;
  TH1F *hLctMPCChamber[MAX_STATIONS];
  TH1F *hLctMPCValid, *hLctMPCQuality, *hLctMPCKeyGroup;
  TH1F *hLctMPCKeyStrip, *hLctMPCStripType;
  TH1F *hLctMPCPattern, *hLctMPCBend, *hLctMPCBXN;

  // Histograms for firmware-emulator comparisons
  // ALCTs
  TH1F *hAlctCompFoundCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hAlctCompSameNCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hAlctCompTotalCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hAlctCompMatchCsc[MAX_ENDCAPS][CSC_TYPES];
  // CLCTs
  TH1F *hClctCompFoundCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hClctCompSameNCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hClctCompTotalCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hClctCompMatchCsc[MAX_ENDCAPS][CSC_TYPES];
  // Correlated LCTs
  TH1F *hLctCompFoundCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hLctCompSameNCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hLctCompTotalCsc[MAX_ENDCAPS][CSC_TYPES];
  TH1F *hLctCompMatchCsc[MAX_ENDCAPS][CSC_TYPES];

  // Resolution histograms
  // ALCT
  TH2F *hEtaRecVsSim;
  TH1F *hResolDeltaWG, *hResolDeltaEta;
  TH1F *hAlctVsEta[MAX_STATIONS];
  TH1F *hEtaDiffVsEta[MAX_STATIONS];
  TH1F *hEtaDiffCsc[CSC_TYPES][3];
  TH2F *hEtaDiffVsWireCsc[CSC_TYPES];
  // CLCT
  TH2F *hPhiRecVsSim;
  TH1F *hResolDeltaHS, *hResolDeltaDS;
  TH1F *hResolDeltaPhi, *hResolDeltaPhiHS, *hResolDeltaPhiDS;
  TH1F *hClctVsPhi[MAX_STATIONS];
  TH1F *hPhiDiffVsPhi[MAX_STATIONS];
  TH1F *hPhiDiffCsc[CSC_TYPES][5];
  TH2F *hPhiDiffVsStripCsc[CSC_TYPES][2];
  TH1F *hTrueBendCsc[CSC_TYPES];
  TH1F *hPhiDiffPattern[CSCConstants::NUM_CLCT_PATTERNS];
  // Efficiency histograms
  TH1F *hEfficHitsEta[MAX_STATIONS];
  TH1F *hEfficALCTEta[MAX_STATIONS], *hEfficCLCTEta[MAX_STATIONS];
  TH1F *hEfficHitsEtaCsc[CSC_TYPES];
  TH1F *hEfficALCTEtaCsc[CSC_TYPES], *hEfficCLCTEtaCsc[CSC_TYPES];
};

#endif
