//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesReader
//
//   Description: Basic analyzer class which accesses ALCTs, CLCTs, and
//                correlated LCTs and plot various quantities.
//
//   Author List: S. Valuev, UCLA.
//
//   $Date: 2006/06/27 15:05:07 $
//   $Revision: 1.3 $
//
//   Modifications:
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//----------------------- 
#include "CSCTriggerPrimitivesReader.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

// MC particles
#include <SimDataFormats/HepMCProduct/interface/HepMCProduct.h>

// MC tests
#include <L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h>
#include <L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TText.h"
#include "TPaveLabel.h"
#include "TPostScript.h"
#include "TStyle.h"

using namespace std;

//-----------------
// Static variables
//-----------------

// Various useful constants
const string CSCTriggerPrimitivesReader::csc_type[CSC_TYPES] = {
  "ME1/1", "ME1/2", "ME1/3", "ME1/A", "ME2/1", "ME2/2", "ME3/1", "ME3/2",
  "ME4/1", "ME4/2"};
const int CSCTriggerPrimitivesReader::MAX_HS[CSC_TYPES] = {
  128, 160, 128,  96, 160, 160, 160, 160, 160, 160}; // max. # of halfstrips
const int CSCTriggerPrimitivesReader::ptype[CSCConstants::NUM_CLCT_PATTERNS]= {
  -999,  3, -3,  2,  -2,  1, -1,  0};  // "signed" pattern (== phiBend)

// LCT counters
int  CSCTriggerPrimitivesReader::numALCT   = 0;
int  CSCTriggerPrimitivesReader::numCLCT   = 0;
int  CSCTriggerPrimitivesReader::numLCTTMB = 0;
int  CSCTriggerPrimitivesReader::numLCTMPC = 0;

bool CSCTriggerPrimitivesReader::bookedALCTHistos   = false;
bool CSCTriggerPrimitivesReader::bookedCLCTHistos   = false;
bool CSCTriggerPrimitivesReader::bookedLCTTMBHistos = false;
bool CSCTriggerPrimitivesReader::bookedLCTMPCHistos = false;

bool CSCTriggerPrimitivesReader::bookedResolHistos  = false;

//----------------
// Constructor  --
//----------------
CSCTriggerPrimitivesReader::CSCTriggerPrimitivesReader(const edm::ParameterSet& conf) : eventsAnalyzed(0) {

  // Various input parameters.
  lctProducer_ = conf.getUntrackedParameter<string>("CSCTriggerPrimitivesProducer");
  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  debug        = conf.getUntrackedParameter<bool>("debug", false);
  //rootFileName = conf.getUntrackedParameter<string>("rootFileName");

  // Create the root file.
  // Not sure we really need it - comment out for now. -Slava.
  //theFile = new TFile(rootFileName.c_str(), "RECREATE");
  //theFile->cd();

  // My favourite ROOT settings.
  setRootStyle();
}

//----------------
// Destructor   --
//----------------
CSCTriggerPrimitivesReader::~CSCTriggerPrimitivesReader() {
  //delete theFile;
}

void CSCTriggerPrimitivesReader::analyze(const edm::Event& ev,
					 const edm::EventSetup& setup) {
  ++eventsAnalyzed;
  //if (ev.id().event()%10 == 0)
  LogDebug("CSCTriggerPrimitivesReader")
    << "\n** CSCTriggerPrimitivesReader: processing run #"
    << ev.id().run() << " event #" << ev.id().event()
    << "; events so far: " << eventsAnalyzed << " **";

  // Find the geometry for this event & cache it.  Needed in LCTAnalyzer
  // modules.
  edm::ESHandle<CSCGeometry> cscGeom;
  setup.get<MuonGeometryRecord>().get(cscGeom);
  geom_ = &*cscGeom;

  // Get the collections of ALCTs, CLCTs, and correlated LCTs from event.
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_tmb;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_mpc;
  if (lctProducer_ == "cscunpacker") {
    // Data
    ev.getByLabel(lctProducer_, "MuonCSCALCTDigi", alcts);
    ev.getByLabel(lctProducer_, "MuonCSCCLCTDigi", clcts);
    ev.getByLabel(lctProducer_, "MuonCSCCorrelatedLCTDigi", lcts_tmb);
  }
  else {
    // Emulator
    ev.getByLabel(lctProducer_,              alcts);
    ev.getByLabel(lctProducer_,              clcts);
    ev.getByLabel(lctProducer_,              lcts_tmb);
    ev.getByLabel(lctProducer_, "MPCSORTED", lcts_mpc);
  }

  // Fill histograms with reconstructed or emulated quantities.
  fillALCTHistos(alcts.product());
  fillCLCTHistos(clcts.product());
  fillLCTTMBHistos(lcts_tmb.product());
  if (lctProducer_ != "cscunpacker") fillLCTMPCHistos(lcts_mpc.product());

  // Compare LCTs in the data with the ones produced by the emulator.
  //compare(ev);

  // Fill MC-based resolution/efficiency histograms, if needed.
  MCStudies(ev, alcts.product(), clcts.product());
}

void CSCTriggerPrimitivesReader::endJob() {
  // Note: all operations involving ROOT should be placed here and not in the
  // destructor.
  // Plot histos if they were booked/filled.
  if (bookedALCTHistos)    drawALCTHistos();
  if (bookedCLCTHistos)    drawCLCTHistos();
  if (bookedLCTTMBHistos)  drawLCTTMBHistos();
  if (bookedLCTMPCHistos)  drawLCTMPCHistos();

  if (bookedResolHistos)   drawResolHistos();
  //drawHistosForTalks();

  //theFile->cd();
  //theFile->Write();
  //theFile->Close();

  // Job summary.
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "\n  Average number of ALCTs/event = "
    << static_cast<float>(numALCT)/eventsAnalyzed << endl;
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "  Average number of CLCTs/event = "
    << static_cast<float>(numCLCT)/eventsAnalyzed << endl;
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "  Average number of TMB LCTs/event = "
    << static_cast<float>(numLCTTMB)/eventsAnalyzed << endl;
  edm::LogInfo("CSCTriggerPrimitivesReader")
    << "  Average number of MPC LCTs/event = "
    << static_cast<float>(numLCTMPC)/eventsAnalyzed << endl;
}

//---------------
// ROOT settings
//---------------
void CSCTriggerPrimitivesReader::setRootStyle() {
  TH1::AddDirectory(false);

  gROOT->SetStyle("Plain");
  gStyle->SetFillColor(0);
  gStyle->SetOptDate();
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(1111);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetMarkerSize(0.5);
  gStyle->SetMarkerStyle(8);
  gStyle->SetGridStyle(3);
  gStyle->SetPaperSize(TStyle::kA4);
  gStyle->SetStatW(0.25); // width of statistics box; default is 0.19
  gStyle->SetStatH(0.10); // height of statistics box; default is 0.1
  gStyle->SetStatFormat("6.4g");  // leave default format for now
  gStyle->SetTitleSize(0.055, "");   // size for pad title; default is 0.02
  gStyle->SetLabelSize(0.05, "XYZ"); // size for axis labels; default is 0.04
  gStyle->SetStatFontSize(0.06);     // size for stat. box
  gStyle->SetTitleFont(32, "XYZ"); // times-bold-italic font (p. 153) for axes
  gStyle->SetTitleFont(32, "");    // same for pad title
  gStyle->SetLabelFont(32, "XYZ"); // same for axis labels
  gStyle->SetStatFont(32);         // same for stat. box
  gStyle->SetLabelOffset(0.006, "Y"); // default is 0.005
}

//---------------------
// Histograms for LCTs
//---------------------
void CSCTriggerPrimitivesReader::bookALCTHistos() {
  hAlctPerEvent = new TH1F("", "ALCTs per event",     11, -0.5,  10.5);
  hAlctPerCSC   = new TH1F("", "ALCTs per CSC type",  10, -0.5,   9.5);
  hAlctValid    = new TH1F("", "ALCT validity",        3, -0.5,   2.5);
  hAlctQuality  = new TH1F("", "ALCT quality",         5, -0.5,   4.5);
  hAlctAccel    = new TH1F("", "ALCT accel. flag",     3, -0.5,   2.5);
  hAlctCollis   = new TH1F("", "ALCT collision. flag", 3, -0.5,   2.5);
  hAlctKeyGroup = new TH1F("", "ALCT key wiregroup", 120, -0.5, 119.5);
  hAlctBXN      = new TH1F("", "ALCT bx",             20, -0.5,  19.5);

  bookedALCTHistos = true;
}

void CSCTriggerPrimitivesReader::bookCLCTHistos() {
  hClctPerEvent  = new TH1F("", "CLCTs per event",    11, -0.5, 10.5);
  hClctPerCSC    = new TH1F("", "CLCTs per CSC type", 10, -0.5,  9.5);
  hClctValid     = new TH1F("", "CLCT validity",       3, -0.5,  2.5);
  hClctQuality   = new TH1F("", "CLCT layers hit",     8, -0.5,  7.5);
  hClctStripType = new TH1F("", "CLCT strip type",     3, -0.5,  2.5);
  hClctSign      = new TH1F("", "CLCT sign (L/R)",     3, -0.5,  2.5);
  hClctCFEB      = new TH1F("", "CLCT cfeb #",         6, -0.5,  5.5);
  hClctBXN       = new TH1F("", "CLCT bx",            20, -0.5, 19.5);

  hClctKeyStrip[0] = new TH1F("","CLCT keystrip, distrips",   40, -0.5,  39.5);
  //hClctKeyStrip[0] = new TH1F("","CLCT keystrip, distrips",  160, -0.5, 159.5);
  hClctKeyStrip[1] = new TH1F("","CLCT keystrip, halfstrips",160, -0.5, 159.5);
  hClctPattern[0]  = new TH1F("","CLCT pattern, distrips",    10, -0.5,   9.5);
  hClctPattern[1]  = new TH1F("","CLCT pattern, halfstrips",  10, -0.5,   9.5);

  for (Int_t i = 0; i < CSC_TYPES; i++) {
    string s1 = "Pattern number, " + csc_type[i];
    hClctPatternCsc[i][0] = new TH1F("", s1.c_str(),  9, -4.5, 4.5);
    hClctPatternCsc[i][1] = new TH1F("", s1.c_str(),  9, -4.5, 4.5);

    string s2 = "CLCT keystrip, " + csc_type[i];
    int max_ds = MAX_HS[i]/4;
    hClctKeyStripCsc[i]   = new TH1F("", s2.c_str(), max_ds, 0., max_ds);
  }

  bookedCLCTHistos = true;
}

void CSCTriggerPrimitivesReader::bookLCTTMBHistos() {
  hLctTMBPerEvent  = new TH1F("", "LCTs per event",    11, -0.5, 10.5);
  hLctTMBPerCSC    = new TH1F("", "LCTs per CSC type", 10, -0.5,  9.5);
  hCorrLctTMBPerCSC= new TH1F("", "Corr. LCTs per CSC type", 10, -0.5, 9.5);
  hLctTMBEndcap    = new TH1F("", "Endcap",             4, -0.5,  3.5);
  hLctTMBStation   = new TH1F("", "Station",            6, -0.5,  5.5);
  hLctTMBSector    = new TH1F("", "Sector",             8, -0.5,  7.5);
  hLctTMBRing      = new TH1F("", "Ring",               5, -0.5,  4.5);

  hLctTMBValid     = new TH1F("", "LCT validity",        3, -0.5,   2.5);
  hLctTMBQuality   = new TH1F("", "LCT quality",        17, -0.5,  16.5);
  hLctTMBKeyGroup  = new TH1F("", "LCT key wiregroup", 120, -0.5, 119.5);
  hLctTMBKeyStrip  = new TH1F("", "LCT key strip",     160, -0.5, 159.5);
  hLctTMBStripType = new TH1F("", "LCT strip type",      3, -0.5,   2.5);
  hLctTMBPattern   = new TH1F("", "LCT pattern",        10, -0.5,   9.5);
  hLctTMBBend      = new TH1F("", "LCT L/R bend",        3, -0.5,   2.5);
  hLctTMBBXN       = new TH1F("", "LCT bx",             20, -0.5,  19.5);

  // LCT quantities per station
  char histname[60];
  for (Int_t istat = 0; istat < MAX_STATIONS; istat++) {
    sprintf(histname, "CSCId, station %d", istat+1);
    hLctTMBChamber[istat] = new TH1F("", histname,  10, -0.5, 9.5);
  }

  bookedLCTTMBHistos = true;
}

void CSCTriggerPrimitivesReader::bookLCTMPCHistos() {
  hLctMPCPerEvent  = new TH1F("", "LCTs per event",    11, -0.5, 10.5);
  hLctMPCPerCSC    = new TH1F("", "LCTs per CSC type", 10, -0.5,  9.5);
  hCorrLctMPCPerCSC= new TH1F("", "Corr. LCTs per CSC type", 10, -0.5,9.5);
  hLctMPCEndcap    = new TH1F("", "Endcap",             4, -0.5,  3.5);
  hLctMPCStation   = new TH1F("", "Station",            6, -0.5,  5.5);
  hLctMPCSector    = new TH1F("", "Sector",             8, -0.5,  7.5);
  hLctMPCRing      = new TH1F("", "Ring",               5, -0.5,  4.5);

  hLctMPCValid     = new TH1F("", "LCT validity",        3, -0.5,   2.5);
  hLctMPCQuality   = new TH1F("", "LCT quality",        17, -0.5,  16.5);
  hLctMPCKeyGroup  = new TH1F("", "LCT key wiregroup", 120, -0.5, 119.5);
  hLctMPCKeyStrip  = new TH1F("", "LCT key strip",     160, -0.5, 159.5);
  hLctMPCStripType = new TH1F("", "LCT strip type",      3, -0.5,   2.5);
  hLctMPCPattern   = new TH1F("", "LCT pattern",        10, -0.5,   9.5);
  hLctMPCBend      = new TH1F("", "LCT L/R bend",        3, -0.5,   2.5);
  hLctMPCBXN       = new TH1F("", "LCT bx",             20, -0.5,  19.5);

  // LCT quantities per station
  char histname[60];
  for (Int_t istat = 0; istat < MAX_STATIONS; istat++) {
    sprintf(histname, "CSCId, station %d", istat+1);
    hLctMPCChamber[istat] = new TH1F("", histname,  10, -0.5, 9.5);
  }

  bookedLCTMPCHistos = true;
}

void CSCTriggerPrimitivesReader::bookResolHistos() {
  hResolDeltaWG = new TH1F("", "Delta key wiregroup",         10, -5., 5.);

  hResolDeltaHS = new TH1F("", "Delta key strip, halfstrips", 10, -5., 5.);
  hResolDeltaDS = new TH1F("", "Delta key strip, distrips",   10, -5., 5.);

  hResolDeltaEta   = new TH1F("", "Delta eta",               100, -0.1, 0.1);
  hResolDeltaPhiHS = new TH1F("", "Delta phi (mrad), halfstrips",
			      100, -10., 10.);
  hResolDeltaPhiDS = new TH1F("", "Delta phi (mrad), distrips",
			      100, -10., 10.);

  bookedResolHistos = true;
}

void CSCTriggerPrimitivesReader::fillALCTHistos(const CSCALCTDigiCollection* alcts) {
  // Book histos when called for the first time.
  if (!bookedALCTHistos) bookALCTHistos();

  int nValidALCTs = 0;
  CSCALCTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = alcts->begin(); detUnitIt != alcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    const CSCALCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool alct_valid = (*digiIt).isValid();
      hAlctValid->Fill(alct_valid);
      if (alct_valid) {
        hAlctQuality->Fill((*digiIt).getQuality());
        hAlctAccel->Fill((*digiIt).getAccelerator());
        hAlctCollis->Fill((*digiIt).getCollisionB());
        hAlctKeyGroup->Fill((*digiIt).getKeyWG());
        hAlctBXN->Fill((*digiIt).getBX());

	hAlctPerCSC->Fill(getCSCType(id));

        nValidALCTs++;

	if (debug) LogDebug("CSCTriggerPrimitivesReader")
	  << (*digiIt) << " found in endcap " <<  id.endcap()
	  << " station " << id.station() << " sector " << id.triggerSector()
	  << " ring " << id.ring() << " chamber " << id.chamber()
	  << " (trig id. " << id.triggerCscId() << ")";
	//cout << "raw id = " << id.rawId() << endl;
      }
    }
  }
  hAlctPerEvent->Fill(nValidALCTs);
  if (debug) LogDebug("CSCTriggerPrimitivesReader")
    << nValidALCTs << " valid ALCTs found in this event";
  numALCT += nValidALCTs;
}

void CSCTriggerPrimitivesReader::fillCLCTHistos(const CSCCLCTDigiCollection* clcts) {
  // Book histos when called for the first time.
  if (!bookedCLCTHistos) bookCLCTHistos();

  int nValidCLCTs = 0;
  CSCCLCTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = clcts->begin(); detUnitIt != clcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    const CSCCLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCLCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool clct_valid = (*digiIt).isValid();
      hClctValid->Fill(clct_valid);
      if (clct_valid) {
	int striptype = (*digiIt).getStripType();
	int keystrip  = (*digiIt).getKeyStrip(); // halfstrip #
        if (striptype == 0) keystrip /= 4;       // distrip # for distrip ptns
        hClctQuality->Fill((*digiIt).getQuality());
        hClctStripType->Fill(striptype);
        hClctSign->Fill((*digiIt).getBend());
        hClctCFEB->Fill((*digiIt).getCFEB());
        hClctBXN->Fill((*digiIt).getBX());
        hClctKeyStrip[striptype]->Fill(keystrip);
        hClctPattern[striptype]->Fill((*digiIt).getPattern());

	int csctype = getCSCType(id);
	hClctPerCSC->Fill(csctype);
        hClctPatternCsc[csctype][striptype]->Fill(ptype[(*digiIt).getPattern()]);
	if (striptype == 0) // distrips
	  hClctKeyStripCsc[csctype]->Fill(keystrip);

        nValidCLCTs++;

	if (debug) LogDebug("CSCTriggerPrimitivesReader")
	  << (*digiIt) << " found in endcap " <<  id.endcap()
	  << " station " << id.station() << " sector " << id.triggerSector()
	  << " ring " << id.ring() << " chamber " << id.chamber()
	  << " (trig id. " << id.triggerCscId() << ")";
      }
    }
  }
  hClctPerEvent->Fill(nValidCLCTs);
  if (debug) LogDebug("CSCTriggerPrimitivesReader")
    << nValidCLCTs << " valid CLCTs found in this event";
  numCLCT += nValidCLCTs;
}

void CSCTriggerPrimitivesReader::fillLCTTMBHistos(const CSCCorrelatedLCTDigiCollection* lcts) {
  // Book histos when called for the first time.
  if (!bookedLCTTMBHistos) bookLCTTMBHistos();

  int nValidLCTs = 0;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool lct_valid = (*digiIt).isValid();
      hLctTMBValid->Fill(lct_valid);
      if (lct_valid) {
        hLctTMBEndcap->Fill(id.endcap());
        hLctTMBStation->Fill(id.station());
	hLctTMBSector->Fill(id.triggerSector());
	hLctTMBRing->Fill(id.ring());
	hLctTMBChamber[id.station()-1]->Fill(id.triggerCscId());

	int quality = (*digiIt).getQuality();
        hLctTMBQuality->Fill(quality);
        hLctTMBBXN->Fill((*digiIt).getBX());

	bool alct_valid = (quality != 4 && quality != 5);
	if (alct_valid) {
	  hLctTMBKeyGroup->Fill((*digiIt).getKeyWG());
	}

	bool clct_valid = (quality != 1 && quality != 3);
	if (clct_valid) {
	  hLctTMBKeyStrip->Fill((*digiIt).getStrip());
	  hLctTMBStripType->Fill((*digiIt).getStripType());
	  hLctTMBPattern->Fill((*digiIt).getCLCTPattern());
	  hLctTMBBend->Fill((*digiIt).getBend());
	}

	int csctype = getCSCType(id);
	hLctTMBPerCSC->Fill(csctype);
	// Truly correlated LCTs; for DAQ
	if (alct_valid && clct_valid) hCorrLctTMBPerCSC->Fill(csctype); 

        nValidLCTs++;

	if (debug) LogDebug("CSCTriggerPrimitivesReader")
	  << (*digiIt) << " found in endcap " <<  id.endcap()
	  << " station " << id.station() << " sector " << id.triggerSector()
	  << " ring " << id.ring() << " chamber " << id.chamber()
	  << " (trig id. " << id.triggerCscId() << ")";
      }
    }
  }
  hLctTMBPerEvent->Fill(nValidLCTs);
  if (debug) LogDebug("CSCTriggerPrimitivesReader")
    << nValidLCTs << " valid LCTs found in this event";
  numLCTTMB += nValidLCTs;
}

void CSCTriggerPrimitivesReader::fillLCTMPCHistos(const CSCCorrelatedLCTDigiCollection* lcts) {
  // Book histos when called for the first time.
  if (!bookedLCTMPCHistos) bookLCTMPCHistos();

  int nValidLCTs = 0;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool lct_valid = (*digiIt).isValid();
      hLctMPCValid->Fill(lct_valid);
      if (lct_valid) {
        hLctMPCEndcap->Fill(id.endcap());
        hLctMPCStation->Fill(id.station());
	hLctMPCSector->Fill(id.triggerSector());
	hLctMPCRing->Fill(id.ring());
	hLctMPCChamber[id.station()-1]->Fill(id.triggerCscId());

	int quality = (*digiIt).getQuality();
        hLctMPCQuality->Fill(quality);
        hLctMPCBXN->Fill((*digiIt).getBX());

	bool alct_valid = (quality != 4 && quality != 5);
	if (alct_valid) {
	  hLctMPCKeyGroup->Fill((*digiIt).getKeyWG());
	}

	bool clct_valid = (quality != 1 && quality != 3);
	if (clct_valid) {
	  hLctMPCKeyStrip->Fill((*digiIt).getStrip());
	  hLctMPCStripType->Fill((*digiIt).getStripType());
	  hLctMPCPattern->Fill((*digiIt).getCLCTPattern());
	  hLctMPCBend->Fill((*digiIt).getBend());
	}

	int csctype = getCSCType(id);
	hLctMPCPerCSC->Fill(csctype);
	// Truly correlated LCTs; for DAQ
	if (alct_valid && clct_valid) hCorrLctMPCPerCSC->Fill(csctype); 

        nValidLCTs++;

	if (debug) LogDebug("CSCTriggerPrimitivesReader")
	  << "MPC " << (*digiIt) << " found in endcap " <<  id.endcap()
	  << " station " << id.station() << " sector " << id.triggerSector()
	  << " ring " << id.ring() << " chamber " << id.chamber()
	  << " (trig id. " << id.triggerCscId() << ")";
      }
    }
  }
  hLctMPCPerEvent->Fill(nValidLCTs);
  if (debug) LogDebug("CSCTriggerPrimitivesReader")
    << nValidLCTs << " MPC LCTs found in this event";
  numLCTMPC += nValidLCTs;
}

void CSCTriggerPrimitivesReader::compare(const edm::Event& ev) {

  // Get the collections of ALCTs, CLCTs, and correlated LCTs from event.
  edm::Handle<CSCALCTDigiCollection> alcts_data;
  edm::Handle<CSCCLCTDigiCollection> clcts_data;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_data;
  edm::Handle<CSCALCTDigiCollection> alcts_emul;
  edm::Handle<CSCCLCTDigiCollection> clcts_emul;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_emul;

  // Data
  ev.getByLabel("cscunpacker", "MuonCSCALCTDigi", alcts_data);
  ev.getByLabel("cscunpacker", "MuonCSCCLCTDigi", clcts_data);
  ev.getByLabel("cscunpacker", "MuonCSCCorrelatedLCTDigi", lcts_data);

  // Emulator
  ev.getByLabel("lctproducer", alcts_emul);
  ev.getByLabel("lctproducer", clcts_emul);
  ev.getByLabel("lctproducer",  lcts_emul);

  // Comparisons
  compareALCTs(alcts_data.product(), alcts_emul.product());
  //compareCLCTs(clcts_data.product(), clcts_emul.product());
  //compareLCTs(lcts_data.product(), clcts_emul.product());
}

void CSCTriggerPrimitivesReader::compareALCTs(
                                 const CSCALCTDigiCollection* alcts_data,
				 const CSCALCTDigiCollection* alcts_emul) {
  CSCALCTDigiCollection::DigiRangeIterator detUnitIt;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= 3; ring++) {
        for (int cham = 1; cham <= 36; cham++) {
	  // Calculate DetId.  0th layer means whole chamber.
	  CSCDetId detid(endc, stat, ring, cham, 0);

	  std::vector<CSCALCTDigi> alctV_data, alctV_emul;
	  std::vector<CSCALCTDigi>::iterator pd, pe;
	  for (detUnitIt = alcts_data->begin();
	       detUnitIt != alcts_data->end(); detUnitIt++) {
	    if ((*detUnitIt).first == detid) {
	      const CSCALCTDigiCollection::Range& range = (*detUnitIt).second;
	      for (CSCALCTDigiCollection::const_iterator digiIt = range.first;
		   digiIt != range.second; digiIt++) {
		alctV_data.push_back(*digiIt);
	      }
	    }
	  }

	  for (detUnitIt = alcts_emul->begin();
	       detUnitIt != alcts_emul->end(); detUnitIt++) {
	    if ((*detUnitIt).first == detid) {
	      const CSCALCTDigiCollection::Range& range = (*detUnitIt).second;
	      for (CSCALCTDigiCollection::const_iterator digiIt = range.first;
		   digiIt != range.second; digiIt++) {
		alctV_emul.push_back(*digiIt);
	      }
	    }
	  }

	  int ndata = alctV_data.size();
	  int nemul = alctV_emul.size();
	  if (ndata == 0 && nemul == 0) continue;

	  if (debug) {
	    ostringstream strstrm;
	    strstrm << "\n --- Endcap "  << detid.endcap()
		    << " station " << detid.station()
		    << " sector "  << detid.triggerSector()
		    << " ring "    << detid.ring()
		    << " chamber " << detid.chamber()
		    << " (trig id. " << detid.triggerCscId() << "):";
	    strstrm << "  * " << ndata << " data ALCTs found: \n";
	    for (pd = alctV_data.begin(); pd != alctV_data.end(); pd++) {
	      strstrm << "     " << (*pd) << "\n";
	    }
	    strstrm << "  * " << nemul << " emul ALCTs found: \n";
	    for (pe = alctV_emul.begin(); pe != alctV_emul.end(); pe++) {
	      strstrm << "     " << (*pe) << "\n";
	    }
	    LogDebug("CSCTriggerPrimitivesReader") << strstrm.str();
	  }

	  if (ndata != nemul) {
	    LogDebug("CSCTriggerPrimitivesReader")
	      << "    +++ Different numbers of ALCTs found: data = " << ndata
	      << " emulator = " << nemul << " +++";
	  }

	  for (pd = alctV_data.begin(); pd != alctV_data.end(); pd++) {
	    int wire_data = (*pd).getKeyWG();
	    for (pe = alctV_emul.begin(); pe != alctV_emul.end(); pe++) {
	      if ((*pe).getKeyWG() == wire_data) {
		if ((*pd).isValid()        == (*pe).isValid() &&
		    (*pd).getQuality()     == (*pe).getQuality() &&
		    (*pd).getAccelerator() == (*pe).getAccelerator() &&
		    (*pd).getCollisionB()  == (*pe).getCollisionB()  &&
		    (*pd).getBX()          == (*pe).getBX()) {
		  LogDebug("CSCTriggerPrimitivesReader")
		    << "        Identical ALCTs on key wire = " << wire_data;
		}
		else {
		  LogDebug("CSCTriggerPrimitivesReader")
		    << "        Different ALCTs on key wire = " << wire_data;
		}
	      }
	    }
	  }
	}
      }
    }
  }
}

void CSCTriggerPrimitivesReader::MCStudies(const edm::Event& ev,
                                 const CSCALCTDigiCollection* alcts,
                                 const CSCCLCTDigiCollection* clcts) {
  // MC particles, if any.
  //edm::Handle<edm::HepMCProduct> mcp;
  //ev.getByLabel("source", mcp);
  //ev.getByType(mcp);
  vector<edm::Handle<edm::HepMCProduct> > allhepmcp;
  // Use "getManyByType" to be able to check the existence of MC info.
  ev.getManyByType(allhepmcp);
  //cout << "HepMC info: " << allhepmcp.size() << endl;
  if (allhepmcp.size() > 0) {
    const HepMC::GenEvent& mc = allhepmcp[0]->getHepMCData();
    int i = 0;
    for (HepMC::GenEvent::particle_const_iterator p = mc.particles_begin();
	 p != mc.particles_end(); ++p) {
      int id = (*p)->pdg_id();
      double phitmp = (*p)->momentum().phi();
      if (phitmp < 0) phitmp += 2.*M_PI;
      if (debug) LogDebug("CSCTriggerPrimitivesReader") 
	<< "MC part #" << ++i << ": id = "  << id
	<< ", status = " << (*p)->status()
	<< ", pT = " << (*p)->momentum().perp() << " GeV"
	<< ", eta = " << (*p)->momentum().pseudoRapidity()
	<< ", phi = " << phitmp*180./M_PI << " deg";
    }

    // If hepMC info is there, try to get wire and comparator digis,
    // and SimHits.
    edm::Handle<CSCWireDigiCollection>       wireDigis;
    edm::Handle<CSCComparatorDigiCollection> compDigis;
    edm::Handle<edm::PSimHitContainer>       simHits;
    ev.getByLabel(wireDigiProducer_.label(), wireDigiProducer_.instance(),
		  wireDigis);
    ev.getByLabel(compDigiProducer_.label(), compDigiProducer_.instance(),
		  compDigis);
    ev.getByLabel("g4SimHits", "MuonCSCHits", simHits);
    if (debug) LogDebug("CSCTriggerPrimitivesReader")
      << "   #CSC SimHits: " << simHits->size();

    // MC-based resolution studies.
    calcResolution(alcts, clcts, wireDigis.product(), compDigis.product(),
		   simHits.product());
  }
}

void CSCTriggerPrimitivesReader::calcResolution(
    const CSCALCTDigiCollection* alcts, const CSCCLCTDigiCollection* clcts,
    const CSCWireDigiCollection* wiredc,
    const CSCComparatorDigiCollection* compdc,
    const edm::PSimHitContainer* allSimHits) {

  // Book histos when called for the first time.
  if (!bookedResolHistos) bookResolHistos();

  // ALCT resolution
  CSCAnodeLCTAnalyzer alct_analyzer;
  alct_analyzer.setGeometry(geom_);
  CSCALCTDigiCollection::DigiRangeIterator adetUnitIt;
  for (adetUnitIt = alcts->begin(); adetUnitIt != alcts->end(); adetUnitIt++) {
    const CSCDetId& id = (*adetUnitIt).first;
    const CSCALCTDigiCollection::Range& range = (*adetUnitIt).second;
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool alct_valid = (*digiIt).isValid();
      if (alct_valid) {
	vector<CSCAnodeLayerInfo> alctInfo =
	  alct_analyzer.getSimInfo(*digiIt, id, wiredc, allSimHits);

	double hitPhi = -999.0, hitEta = -999.0;
	int hitWG = alct_analyzer.nearestWG(alctInfo, hitPhi, hitEta);
	if (hitWG >= 0.) {
	  // Key wire group and key layer id.
	  int wiregroup = (*digiIt).getKeyWG();
	  CSCDetId layerId(id.endcap(), id.station(), id.ring(),
			   id.chamber(), 3);

	  double alctEta  = alct_analyzer.getWGEta(layerId, wiregroup+1);
	  double deltaEta = alctEta - hitEta;
	  hResolDeltaEta->Fill(deltaEta);

	  double deltaWG = wiregroup - hitWG;
	  if (debug) LogDebug("CSCTriggerPrimitivesReader")
	    << "WG: MC = " << hitWG << " rec = " << wiregroup
	    << " delta = " << deltaWG;
	  hResolDeltaWG->Fill(deltaWG);
	}
	else {
	  edm::LogWarning("CSCTriggerPrimitivesReader")
	    << "+++ Warning in calcResolution(): no matched SimHit"
	    << " found! +++\n";
	}
      }
    }
  }

  // CLCT resolution
  CSCCathodeLCTAnalyzer clct_analyzer;
  clct_analyzer.setGeometry(geom_);
  CSCCLCTDigiCollection::DigiRangeIterator cdetUnitIt;
  for (cdetUnitIt = clcts->begin(); cdetUnitIt != clcts->end(); cdetUnitIt++) {
    const CSCDetId& id = (*cdetUnitIt).first;
    const CSCCLCTDigiCollection::Range& range = (*cdetUnitIt).second;
    for (CSCCLCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool clct_valid = (*digiIt).isValid();
      if (clct_valid) {
	vector<CSCCathodeLayerInfo> clctInfo =
	  clct_analyzer.getSimInfo(*digiIt, id, compdc, allSimHits);

	double hitPhi = -999.0, hitEta = -999.0, deltaStrip = -999.0;
	int hitHS = clct_analyzer.nearestHS(clctInfo, hitPhi, hitEta);
	if (hitHS >= 0.) {
	  // Key strip and key layer id.
	  int halfstrip = (*digiIt).getKeyStrip();
	  int strip     = halfstrip/2;
	  CSCDetId layerId(id.endcap(), id.station(), id.ring(),
			   id.chamber(), CSCConstants::KEY_LAYER);

	  double clctPhi = clct_analyzer.getStripPhi(layerId, strip+1);
	  double deltaPhi = clctPhi - hitPhi;
	  if      (deltaPhi < -M_PI) deltaPhi += 2.*M_PI;
	  else if (deltaPhi >  M_PI) deltaPhi -= 2.*M_PI;
	  deltaPhi *= 1000; // in mrad

	  if ((*digiIt).getStripType() == 0) { // di-strip CLCT
	    deltaStrip = halfstrip/4 - hitHS/4;
	    hResolDeltaDS->Fill(deltaStrip);
	    hResolDeltaPhiDS->Fill(deltaPhi);
	  }
	  else {                              // half-strip CLCT
	    deltaStrip = halfstrip - hitHS;
	    hResolDeltaHS->Fill(deltaStrip);
	    hResolDeltaPhiHS->Fill(deltaPhi);
	  }
	  if (debug) LogDebug("CSCTriggerPrimitivesReader")
	    << "Half-strip: MC = " << hitHS << " rec = " << halfstrip
	    << " pattern type = " << (*digiIt).getStripType()
	    << " delta = " << deltaStrip;
	}
	else {
	  edm::LogWarning("CSCTriggerPrimitivesReader")
	    << "+++ Warning in calcResolution(): no matched SimHit"
	    << " found! +++\n";
	}
      }
    }
  }
}


void CSCTriggerPrimitivesReader::drawALCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("alcts.ps", 111);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  Char_t pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of ALCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,2);
  pad[page]->cd(1);  hAlctPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hAlctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hAlctPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hAlctValid->Draw();
  pad[page]->cd(2);  hAlctQuality->Draw();
  pad[page]->cd(3);  hAlctAccel->Draw();
  pad[page]->cd(4);  hAlctCollis->Draw();
  pad[page]->cd(5);  hAlctKeyGroup->Draw();
  pad[page]->cd(6);  hAlctBXN->Draw();
  page++;  c1->Update();

  ps->Close();
}

void CSCTriggerPrimitivesReader::drawCLCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("clcts.ps", 111);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  Char_t pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of CLCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,2);
  pad[page]->cd(1);  hClctPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hClctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hClctPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hClctValid->Draw();
  pad[page]->cd(2);  hClctQuality->Draw();
  pad[page]->cd(3);  hClctSign->Draw();
  TH1F* hClctPatternTot = (TH1F*)hClctPattern[0]->Clone();
  hClctPatternTot->SetTitle("CLCT pattern #");
  hClctPatternTot->Add(hClctPattern[0], hClctPattern[1], 1., 1.);
  pad[page]->cd(4);  hClctPatternTot->Draw();
  hClctPattern[0]->SetLineStyle(2);  hClctPattern[0]->Draw("same");
  hClctPattern[1]->SetLineStyle(3);  hClctPattern[1]->Draw("same");
  pad[page]->cd(5);  hClctCFEB->Draw();
  pad[page]->cd(6);  hClctBXN->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hClctStripType->Draw();
  pad[page]->cd(2);  hClctKeyStrip[0]->Draw();
  pad[page]->cd(3);  hClctKeyStrip[1]->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT halfstrip pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);
    hClctPatternCsc[idh][1]->GetXaxis()->SetTitle("Pattern number");
    hClctPatternCsc[idh][1]->GetYaxis()->SetTitle("Number of LCTs");
    hClctPatternCsc[idh][1]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT distrip pattern types");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);
    hClctPatternCsc[idh][0]->GetXaxis()->SetTitle("Pattern number");
    hClctPatternCsc[idh][0]->GetYaxis()->SetTitle("Number of LCTs");
    hClctPatternCsc[idh][0]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98,
			 "CLCT keystrip, distrip patterns only");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,5);
  for (Int_t idh = 0; idh < CSC_TYPES; idh++) {
    pad[page]->cd(idh+1);
    hClctKeyStripCsc[idh]->GetXaxis()->SetTitle("Key distrip");
    hClctKeyStripCsc[idh]->GetXaxis()->SetTitleOffset(1.2);
    hClctKeyStripCsc[idh]->GetYaxis()->SetTitle("Number of LCTs");
    hClctKeyStripCsc[idh]->Draw();
  }
  page++;  c1->Update();

  ps->Close();
}

void CSCTriggerPrimitivesReader::drawLCTTMBHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("lcts_tmb.ps", 111);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  Char_t pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of LCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hLctTMBPerEvent->Draw();
  c1->Update();
  for (int i = 0; i < CSC_TYPES; i++) {
    hLctTMBPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hCorrLctTMBPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hLctTMBPerCSC->Draw();
  pad[page]->cd(3);  hCorrLctTMBPerCSC->Draw();
  gStyle->SetOptStat(1110);
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT geometry");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctTMBEndcap->Draw();
  pad[page]->cd(2);  hLctTMBStation->Draw();
  pad[page]->cd(3);  hLctTMBSector->Draw();
  pad[page]->cd(4);  hLctTMBRing->Draw();
  for (Int_t istat = 0; istat < MAX_STATIONS; istat++) {
    pad[page]->cd(istat+5);  hLctTMBChamber[istat]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctTMBValid->Draw();
  pad[page]->cd(2);  hLctTMBQuality->Draw();
  pad[page]->cd(3);  hLctTMBKeyGroup->Draw();
  pad[page]->cd(4);  hLctTMBKeyStrip->Draw();
  pad[page]->cd(5);  hLctTMBStripType->Draw();
  pad[page]->cd(6);  hLctTMBPattern->Draw();
  pad[page]->cd(7);  hLctTMBBend->Draw();
  pad[page]->cd(8);  hLctTMBBXN->Draw();
  page++;  c1->Update();

  ps->Close();
}

void CSCTriggerPrimitivesReader::drawLCTMPCHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("lcts_mpc.ps", 111);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  Char_t pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "Number of LCTs");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,3);
  pad[page]->cd(1);  hLctMPCPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hLctMPCPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hCorrLctMPCPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hLctMPCPerCSC->Draw();
  pad[page]->cd(3);  hCorrLctMPCPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT geometry");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctMPCEndcap->Draw();
  pad[page]->cd(2);  hLctMPCStation->Draw();
  pad[page]->cd(3);  hLctMPCSector->Draw();
  pad[page]->cd(4);  hLctMPCRing->Draw();
  for (Int_t istat = 0; istat < MAX_STATIONS; istat++) {
    pad[page]->cd(istat+5);  hLctMPCChamber[istat]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctMPCValid->Draw();
  pad[page]->cd(2);  hLctMPCQuality->Draw();
  pad[page]->cd(3);  hLctMPCKeyGroup->Draw();
  pad[page]->cd(4);  hLctMPCKeyStrip->Draw();
  pad[page]->cd(5);  hLctMPCStripType->Draw();
  pad[page]->cd(6);  hLctMPCPattern->Draw();
  pad[page]->cd(7);  hLctMPCBend->Draw();
  pad[page]->cd(8);  hLctMPCBXN->Draw();
  page++;  c1->Update();

  ps->Close();
}

void CSCTriggerPrimitivesReader::drawResolHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("lcts_resol.ps", 111);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TText t;
  t.SetTextFont(32);
  t.SetTextSize(0.025);
  Char_t pagenum[6];
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "ALCT resolution");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(1,2);
  pad[page]->cd(1);  hResolDeltaWG->Draw();
  pad[page]->cd(2);  hResolDeltaEta->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT resolution");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(111110);
  pad[page]->Draw();
  pad[page]->Divide(2,2);
  pad[page]->cd(1);  hResolDeltaHS->Draw();
  pad[page]->cd(2);  hResolDeltaDS->Draw();
  pad[page]->cd(3);  hResolDeltaPhiHS->Draw();
  pad[page]->cd(4);  hResolDeltaPhiDS->Draw();
  page++;  c1->Update();

  ps->Close();
}

void CSCTriggerPrimitivesReader::drawHistosForTalks() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("clcts.eps", 113);

  TPad *pad[MAXPAGES];
  for (Int_t i_page = 0; i_page < MAXPAGES; i_page++) {
    pad[i_page] = new TPad("", "", .05, .05, .93, .93);
  }

  Int_t page = 1;
  TPaveLabel *title;

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "CLCT quantities");
  title->SetFillColor(10);  title->Draw();
  pad[page]->Draw();
  pad[page]->Divide(2,3);
  pad[page]->cd(1);  hClctQuality->Draw();
  pad[page]->cd(2);  hClctSign->Draw();
  TH1F* hClctPatternTot = (TH1F*)hClctPattern[0]->Clone();
  hClctPatternTot->SetTitle("CLCT pattern #");
  hClctPatternTot->Add(hClctPattern[0], hClctPattern[1], 1., 1.);
  pad[page]->cd(3);  hClctPatternTot->Draw();
  hClctPattern[0]->SetLineStyle(2);  hClctPattern[0]->Draw("same");
  hClctPattern[1]->SetLineStyle(3);  hClctPattern[1]->Draw("same");
  pad[page]->cd(4);  hClctCFEB->Draw();
  pad[page]->cd(5);  hClctStripType->Draw();
  TH1F* hClctKeyStripTot = (TH1F*)hClctKeyStrip[0]->Clone();
  hClctKeyStripTot->SetTitle("CLCT key strip #");
  hClctKeyStripTot->Add(hClctKeyStrip[0], hClctKeyStrip[1], 1., 1.);
  pad[page]->cd(6);  hClctKeyStripTot->Draw();
  page++;  c1->Update();

  ps->Close();
}

// Returns chamber type (0-9) according to the station and ring number
int CSCTriggerPrimitivesReader::getCSCType(const CSCDetId& id) {
  int type = -999;

  if (id.station() == 1) {
    type = id.triggerCscId()/3;
  }
  else { // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }

  assert(type >= 0 && type < CSC_TYPES-1); // no ME4/2
  return type;
}

DEFINE_FWK_MODULE(CSCTriggerPrimitivesReader)
