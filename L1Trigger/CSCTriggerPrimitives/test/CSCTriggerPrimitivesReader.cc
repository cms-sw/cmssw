//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesReader
//
//   Description: Basic analyzer class which accesses ALCTs, CLCTs, and
//                correlated LCTs and plot various quantities.
//
//   Author List: S. Valuev, UCLA.
//
//   $Date: 2006/06/08 16:02:21 $
//   $Revision: 1.1 $
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

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>

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
int  CSCTriggerPrimitivesReader::numALCT = 0;
int  CSCTriggerPrimitivesReader::numCLCT = 0;
int  CSCTriggerPrimitivesReader::numLCT  = 0;

bool CSCTriggerPrimitivesReader::bookedALCTHistos = false;
bool CSCTriggerPrimitivesReader::bookedCLCTHistos = false;
bool CSCTriggerPrimitivesReader::bookedLCTHistos  = false;

//----------------
// Constructor  --
//----------------
CSCTriggerPrimitivesReader::CSCTriggerPrimitivesReader(const edm::ParameterSet& conf) : eventsAnalyzed(0) {

  // Various input parameters.
  lctProducer_ = conf.getUntrackedParameter<string>("CSCTriggerPrimitivesProducer");
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
  if (ev.id().event()%10 == 0)
    cout << "\n** CSCTriggerPrimitivesReader: processing run #"
	 << ev.id().run() << " event #" << ev.id().event()
	 << "; events so far: " << eventsAnalyzed << " **" << endl;

  // Find the geometry for this event & cache it in CSCTriggerGeometry.
  // Is it really needed?  If not, check need for L1Trigger/CSCCommonTrigger
  // in BuildFile.
  edm::ESHandle<CSCGeometry> h;
  setup.get<MuonGeometryRecord>().get(h);
  CSCTriggerGeometry::setGeometry(h);

  // Get the collections of ALCTs, CLCTs, and correlated LCTs from event.
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;
  if (lctProducer_ == "cscunpacker") {
    // Data
    ev.getByLabel(lctProducer_, "MuonCSCALCTDigi", alcts);
    ev.getByLabel(lctProducer_, "MuonCSCCLCTDigi", clcts);
    ev.getByLabel(lctProducer_, "MuonCSCCorrelatedLCTDigi", lcts);
  }
  else {
    // Emulator
    ev.getByLabel(lctProducer_, alcts);
    ev.getByLabel(lctProducer_, clcts);
    ev.getByLabel(lctProducer_,  lcts);
  }

  // Fill histograms.
  fillALCTHistos(alcts.product());
  fillCLCTHistos(clcts.product());
  fillLCTHistos(lcts.product());
}

void CSCTriggerPrimitivesReader::endJob() {
  // Note: all operations involving ROOT should be placed here and not in the
  // destructor.
  // Plot histos if they were booked/filled.
  if (bookedALCTHistos) drawALCTHistos();
  if (bookedCLCTHistos) drawCLCTHistos();
  if (bookedLCTHistos)  drawLCTHistos();
  //drawHistosForTalks();

  //theFile->cd();
  //theFile->Write();
  //theFile->Close();

  // Job summary.
  cout << "\n  Average number of ALCTs/event = "
       << static_cast<float>(numALCT)/eventsAnalyzed << endl;
  cout << "  Average number of CLCTs/event = "
       << static_cast<float>(numCLCT)/eventsAnalyzed << endl;
  cout << "  Average number of correlated LCTs/event = "
       << static_cast<float>(numLCT)/eventsAnalyzed  << endl;
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

void CSCTriggerPrimitivesReader::bookLCTHistos() {
  hLctPerEvent  = new TH1F("", "LCTs per event",    11, -0.5, 10.5);
  hLctPerCSC    = new TH1F("", "LCTs per CSC type", 10, -0.5,  9.5);
  hCorrLctPerCSC= new TH1F("", "Corr. LCTs per CSC type", 10, -0.5, 9.5);
  hLctEndcap    = new TH1F("", "Endcap",             4, -0.5,  3.5);
  hLctStation   = new TH1F("", "Station",            6, -0.5,  5.5);
  hLctSector    = new TH1F("", "Sector",             8, -0.5,  7.5);
  hLctRing      = new TH1F("", "Ring",               5, -0.5,  4.5);

  hLctValid     = new TH1F("", "LCT validity",        3, -0.5,   2.5);
  hLctQuality   = new TH1F("", "LCT quality",        17, -0.5,  16.5);
  hLctKeyGroup  = new TH1F("", "LCT key wiregroup", 120, -0.5, 119.5);
  hLctKeyStrip  = new TH1F("", "LCT key strip",     160, -0.5, 159.5);
  hLctStripType = new TH1F("", "LCT strip type",      3, -0.5,   2.5);
  hLctPattern   = new TH1F("", "LCT pattern",        10, -0.5,   9.5);
  hLctBend      = new TH1F("", "LCT L/R bend",        3, -0.5,   2.5);
  hLctBXN       = new TH1F("", "LCT bx",             20, -0.5,  19.5);

  // LCT quantities per station
  char histname[60];
  for (Int_t istat = 0; istat < MAX_STATIONS; istat++) {
    sprintf(histname, "CSCId, station %d", istat+1);
    hLctChamber[istat] = new TH1F("", histname,  10, -0.5, 9.5);
  }

  bookedLCTHistos = true;
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

	if (debug) 
	  cout << (*digiIt) << " found in endcap " <<  id.endcap()
	       << " station " << id.station()
	       << " sector " << id.triggerSector()
	       << " ring " << id.ring() << " chamber " << id.chamber()
	       << " (trig id. " << id.triggerCscId() << ")" << endl;
      }
    }
  }
  hAlctPerEvent->Fill(nValidALCTs);
  if (debug) cout << nValidALCTs << " valid ALCTs found in this event" << endl;
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

	if (debug) 
	  cout << (*digiIt) << " found in endcap " <<  id.endcap()
	       << " station " << id.station()
	       << " sector " << id.triggerSector()
	       << " ring " << id.ring() << " chamber " << id.chamber()
	       << " (trig id. " << id.triggerCscId() << ")" << endl;
      }
    }
  }
  hClctPerEvent->Fill(nValidCLCTs);
  if (debug) cout << nValidCLCTs << " valid CLCTs found in this event" << endl;
  numCLCT += nValidCLCTs;
}

void CSCTriggerPrimitivesReader::fillLCTHistos(const CSCCorrelatedLCTDigiCollection* lcts) {
  // Book histos when called for the first time.
  if (!bookedLCTHistos) bookLCTHistos();

  int nValidLCTs = 0;
  CSCCorrelatedLCTDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt = lcts->begin(); detUnitIt != lcts->end(); detUnitIt++) {
    const CSCDetId& id = (*detUnitIt).first;
    const CSCCorrelatedLCTDigiCollection::Range& range = (*detUnitIt).second;
    for (CSCCorrelatedLCTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; digiIt++) {

      bool lct_valid = (*digiIt).isValid();
      hLctValid->Fill(lct_valid);
      if (lct_valid) {
        hLctEndcap->Fill(id.endcap());
        hLctStation->Fill(id.station());
	hLctSector->Fill(id.triggerSector());
	hLctRing->Fill(id.ring());
	hLctChamber[id.station()-1]->Fill(id.triggerCscId());

	int quality = (*digiIt).getQuality();
        hLctQuality->Fill(quality);
        hLctBXN->Fill((*digiIt).getBX());

	bool alct_valid = (quality != 4 && quality != 5);
	if (alct_valid) {
	  hLctKeyGroup->Fill((*digiIt).getKeyWG());
	}

	bool clct_valid = (quality != 1 && quality != 3);
	if (clct_valid) {
	  hLctKeyStrip->Fill((*digiIt).getStrip());
	  hLctStripType->Fill((*digiIt).getStripType());
	  hLctPattern->Fill((*digiIt).getCLCTPattern());
	  hLctBend->Fill((*digiIt).getBend());
	}

	int csctype = getCSCType(id);
	hLctPerCSC->Fill(csctype);
	// Truly correlated LCTs; for DAQ
	if (alct_valid && clct_valid) hCorrLctPerCSC->Fill(csctype); 

        nValidLCTs++;

	if (debug) 
	  cout << (*digiIt) << " found in endcap " <<  id.endcap()
	       << " station " << id.station()
	       << " sector " << id.triggerSector()
	       << " ring " << id.ring() << " chamber " << id.chamber()
	       << " (trig id. " << id.triggerCscId() << ")" << endl;
      }
    }
  }
  hLctPerEvent->Fill(nValidLCTs);
  if (debug) cout << nValidLCTs << " valid LCTs found in this event" << endl;
  numLCT += nValidLCTs;
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

void CSCTriggerPrimitivesReader::drawLCTHistos() {
  TCanvas *c1 = new TCanvas("c1", "", 0, 0, 500, 640);
  TPostScript *ps = new TPostScript("lcts.ps", 111);

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
  pad[page]->cd(1);  hLctPerEvent->Draw();
  for (int i = 0; i < CSC_TYPES; i++) {
    hLctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
    hCorrLctPerCSC->GetXaxis()->SetBinLabel(i+1, csc_type[i].c_str());
  }
  // Should be multiplied by 40/nevents to convert to MHz
  pad[page]->cd(2);  hLctPerCSC->Draw();
  pad[page]->cd(3);  hCorrLctPerCSC->Draw();
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT geometry");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  gStyle->SetOptStat(110110);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctEndcap->Draw();
  pad[page]->cd(2);  hLctStation->Draw();
  pad[page]->cd(3);  hLctSector->Draw();
  pad[page]->cd(4);  hLctRing->Draw();
  for (Int_t istat = 0; istat < MAX_STATIONS; istat++) {
    pad[page]->cd(istat+5);  hLctChamber[istat]->Draw();
  }
  page++;  c1->Update();

  ps->NewPage();
  c1->Clear();  c1->cd(0);
  title = new TPaveLabel(0.1, 0.94, 0.9, 0.98, "LCT quantities");
  title->SetFillColor(10);  title->Draw();
  sprintf(pagenum, "- %d -", page);  t.DrawText(0.9, 0.02, pagenum);
  pad[page]->Draw();
  pad[page]->Divide(2,4);
  pad[page]->cd(1);  hLctValid->Draw();
  pad[page]->cd(2);  hLctQuality->Draw();
  pad[page]->cd(3);  hLctKeyGroup->Draw();
  pad[page]->cd(4);  hLctKeyStrip->Draw();
  pad[page]->cd(5);  hLctStripType->Draw();
  pad[page]->cd(6);  hLctPattern->Draw();
  pad[page]->cd(7);  hLctBend->Draw();
  pad[page]->cd(8);  hLctBXN->Draw();
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
