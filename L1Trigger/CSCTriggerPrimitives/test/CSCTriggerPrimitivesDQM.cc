//-------------------------------------------------
//
//   Class: CSCTriggerPrimitivesReader
//
//   Description: Basic analyzer class which accesses ALCTs, CLCTs, and
//                correlated LCTs and plot various quantities.
//
//   Author List: S. Valuev, UCLA.
//
//   $Date: 2012/09/27 15:47:22 $
//   $Revision: 1.3 $
//
//   Modifications:
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//----------------------- 
#include "CSCTriggerPrimitivesDQM.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

// MC particles
#include <SimDataFormats/GeneratorProducts/interface/HepMCProduct.h>

// MC tests
#include <L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h>
#include <L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TText.h"
#include "TPaveLabel.h"
#include "TPostScript.h"
#include "TStyle.h"
#include "TROOT.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

using namespace std;

//-----------------
// Static variables
//-----------------

// Various useful constants
const double CSCTriggerPrimitivesDQM::TWOPI = 2.*M_PI;
const string CSCTriggerPrimitivesDQM::csc_type[CSC_TYPES] = {
  "ME1/1",  "ME1/2",  "ME1/3",  "ME1/A",  "ME2/1",  "ME2/2",
  "ME3/1",  "ME3/2",  "ME4/1",  "ME4/2"};
const string CSCTriggerPrimitivesDQM::csc_type_plus[CSC_TYPES] = {
  "ME+1/1", "ME+1/2", "ME+1/3", "ME+1/A", "ME+2/1", "ME+2/2",
  "ME+3/1", "ME+3/2", "ME+4/1", "ME+4/2"};
const string CSCTriggerPrimitivesDQM::csc_type_minus[CSC_TYPES] = {
  "ME-1/1", "ME-1/2", "ME-1/3", "ME-1/A", "ME-2/1", "ME-2/2",
  "ME-3/1", "ME-3/2", "ME-4/1", "ME-4/2"};
const int CSCTriggerPrimitivesDQM::NCHAMBERS[CSC_TYPES] = {
  36, 36, 36, 36, 18, 36, 18, 36, 18, 36};
const int CSCTriggerPrimitivesDQM::MAX_WG[CSC_TYPES] = {
   48,  64,  32,  48, 112,  64,  96,  64,  96,  64};//max. number of wiregroups
const int CSCTriggerPrimitivesDQM::MAX_HS[CSC_TYPES] = {
  128, 160, 128,  96, 160, 160, 160, 160, 160, 160}; // max. # of halfstrips
const int CSCTriggerPrimitivesDQM::ptype[CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07]= {
  -999,  3, -3,  2, -2,  1, -1,  0};  // "signed" pattern (== phiBend)
const int CSCTriggerPrimitivesDQM::ptype_TMB07[CSCConstants::NUM_CLCT_PATTERNS]= {
  -999,  -5,  4, -4,  3, -3,  2, -2,  1, -1,  0}; // "signed" pattern (== phiBend)

// LCT counters
int  CSCTriggerPrimitivesDQM::numALCT   = 0;
int  CSCTriggerPrimitivesDQM::numCLCT   = 0;
int  CSCTriggerPrimitivesDQM::numLCTTMB = 0;
int  CSCTriggerPrimitivesDQM::numLCTMPC = 0;

bool CSCTriggerPrimitivesDQM::bookedHotWireHistos = false;
bool CSCTriggerPrimitivesDQM::bookedALCTHistos    = false;
bool CSCTriggerPrimitivesDQM::bookedCLCTHistos    = false;
bool CSCTriggerPrimitivesDQM::bookedLCTTMBHistos  = false;
bool CSCTriggerPrimitivesDQM::bookedLCTMPCHistos  = false;

bool CSCTriggerPrimitivesDQM::bookedCompHistos    = false;

bool CSCTriggerPrimitivesDQM::bookedResolHistos   = false;
bool CSCTriggerPrimitivesDQM::bookedEfficHistos   = false;

bool CSCTriggerPrimitivesDQM::printps = false;

//----------------
// Constructor  --
//----------------
CSCTriggerPrimitivesDQM::CSCTriggerPrimitivesDQM(const edm::ParameterSet& conf) : eventsAnalyzed(0) {

  printps = conf.getParameter<bool>("printps");
  dataLctsIn_ = conf.getParameter<bool>("dataLctsIn");
  emulLctsIn_ = conf.getParameter<bool>("emulLctsIn");
  isMTCCData_ = conf.getParameter<bool>("isMTCCData");
  isTMB07 = true;
  plotME1A = false;
  plotME42 = true;
  lctProducerData_ = conf.getUntrackedParameter<string>("CSCLCTProducerData",
							"cscunpacker");
  lctProducerEmul_ = conf.getUntrackedParameter<string>("CSCLCTProducerEmul",
							"cscTriggerPrimitiveDigis");

  simHitProducer_ = conf.getParameter<edm::InputTag>("CSCSimHitProducer");
  wireDigiProducer_ = conf.getParameter<edm::InputTag>("CSCWireDigiProducer");
  compDigiProducer_ = conf.getParameter<edm::InputTag>("CSCComparatorDigiProducer");
  debug = conf.getUntrackedParameter<bool>("debug", false);
  bad_chambers = conf.getUntrackedParameter< std::vector<std::string> >("bad_chambers");
  bad_wires = conf.getUntrackedParameter< std::vector<std::string> >("bad_wires");
  bad_strips = conf.getUntrackedParameter< std::vector<std::string> >("bad_strips");
  dbe = edm::Service<DQMStore>().operator->();
}

//----------------
// Destructor   --
//----------------
CSCTriggerPrimitivesDQM::~CSCTriggerPrimitivesDQM() {
  //  histos->writeHists(theFile);
  //  theFile->Close();
  //delete theFile;
}

void CSCTriggerPrimitivesDQM::analyze(const edm::Event& ev,
					 const edm::EventSetup& setup) {
  ++eventsAnalyzed;
  //if (ev.id().event()%10 == 0)
  // Find the geometry for this event & cache it.  Needed in LCTAnalyzer
  // modules.
  edm::ESHandle<CSCGeometry> cscGeom;
  setup.get<MuonGeometryRecord>().get(cscGeom);
  geom_ = &*cscGeom;

  // Find conditions data for bad chambers & cache it.  Needed for efficiency
  // calculations.
  edm::ESHandle<CSCBadChambers> pBad;
  setup.get<CSCBadChambersRcd>().get(pBad);
  badChambers_=pBad.product();

  // Get the collections of ALCTs, CLCTs, and correlated LCTs from event.
  edm::Handle<CSCALCTDigiCollection> alcts_data;
  edm::Handle<CSCCLCTDigiCollection> clcts_data;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_tmb_data;
  edm::Handle<CSCALCTDigiCollection> alcts_emul;
  edm::Handle<CSCCLCTDigiCollection> clcts_emul;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_tmb_emul;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts_mpc_emul;

  // Data
  HotWires(ev);
  if (dataLctsIn_) {
    ev.getByLabel(lctProducerData_, "MuonCSCALCTDigi", alcts_data);
    ev.getByLabel(lctProducerData_, "MuonCSCCLCTDigi", clcts_data);
    ev.getByLabel(lctProducerData_, "MuonCSCCorrelatedLCTDigi", lcts_tmb_data);

    if (!alcts_data.isValid()) {
      return;
    }
    if (!clcts_data.isValid()) {
      return;
    }
    if (!lcts_tmb_data.isValid()) {
      return;
    }
  }

  // Emulator
  if (emulLctsIn_) {
    ev.getByLabel(lctProducerEmul_,              alcts_emul);
    ev.getByLabel(lctProducerEmul_,              clcts_emul);
    ev.getByLabel(lctProducerEmul_,              lcts_tmb_emul);
    ev.getByLabel(lctProducerEmul_, "MPCSORTED", lcts_mpc_emul);

    if (!alcts_emul.isValid()) {
      return;
    }
    if (!clcts_emul.isValid()) {
      return;
    }
    if (!lcts_tmb_emul.isValid()) {
      return;
    }
    if (!lcts_mpc_emul.isValid()) {
      return;
    }
  }

  // Fill histograms with reconstructed or emulated quantities.  If both are
  // present, plot LCTs in data.

  // Compare LCTs in the data with the ones produced by the emulator.
  if (dataLctsIn_ && emulLctsIn_) {
    compare(alcts_data.product(),    alcts_emul.product(),
	    clcts_data.product(),    clcts_emul.product(),
	    lcts_tmb_data.product(), lcts_tmb_emul.product());
  }

}

void CSCTriggerPrimitivesDQM::endJob() {

}

// Histograms for LCTs
//---------------------
void CSCTriggerPrimitivesDQM::bookHotWireHistos() {
  //  edm::Service<TFileService> fs;
  //  hHotWire1  = fs->make<TH1F>("hHotWire1", "hHotWire1",570*6*112,0,570*6*112);
  //  hHotCham1  = fs->make<TH1F>("hHotCham1", "hHotCham1",570,0,570);
  bookedHotWireHistos = true;
}

int CSCTriggerPrimitivesDQM::chamberIXi(CSCDetId id) {  
  //    1/1 1/2 1/3 2/1 2/2 3/1 3/2 4/1 4/2 -1/1 -1/2 -1/3 -2/1 -2/2 -3/1 -3/2 -4/1 -4/2
  //ix= 0   1   2   3   4   5   6   7   8    9    10   11   12   13   14   15   16   17
  int ix=0;
  if(id.station()!=1) {

    ix=(id.station()-2)*2+3;
  }
  ix+=id.ring()-1;
  if(id.endcap()==2) { 
    ix+=9;
  }
  return ix;
}

int CSCTriggerPrimitivesDQM::chamberIX(CSCDetId id) {
  int ix=1;
  if(id.station()!=1) {
    ix=(id.station()-2)*2+3+1;
  }
  ix+=id.ring()-1;
  if(id.endcap()==2) { 
    ix*=-1;    
  }
  return ix;
}

int CSCTriggerPrimitivesDQM::chamberSerial( CSCDetId id ) {
  int st = id.station();
  int ri = id.ring();
  int ch = id.chamber();
  int ec = id.endcap();
  int kSerial = ch;
  if (st == 1 && ri == 1) kSerial = ch;
  if (st == 1 && ri == 2) kSerial = ch + 36;
  if (st == 1 && ri == 3) kSerial = ch + 72;
  if (st == 1 && ri == 4) kSerial = ch;
  if (st == 2 && ri == 1) kSerial = ch + 108;
  if (st == 2 && ri == 2) kSerial = ch + 126;
  if (st == 3 && ri == 1) kSerial = ch + 162;
  if (st == 3 && ri == 2) kSerial = ch + 180;
  if (st == 4 && ri == 1) kSerial = ch + 216;
  if (st == 4 && ri == 2) kSerial = ch + 234;  // one day...
  if (ec == 2) kSerial = kSerial + 300;
  return kSerial;
}


void CSCTriggerPrimitivesDQM::bookCompHistos() {
  string s;

  dbe->setCurrentFolder("CSC/TriggerPrimitivesEmulator");

  //Chad's improved historgrams

  /*
  hAlctCompFound2i = dbe->book2D("h_ALCT_found2i","h_ALCT_found2i",18,0,18,36,0.5,36.5);
  hAlctCompSameN2i = dbe->book2D("h_ALCT_SameN2i","h_ALCT_SameN2i",18,0,18,36,0.5,36.5);
  hAlctCompMatch2i = dbe->book2D("h_ALCT_match2i","h_ALCT_match2i",18,0,18,36,0.5,36.5);
  hAlctCompTotal2i = dbe->book2D("h_ALCT_total2i","h_ALCT_total2i",18,0,18,36,0.5,36.5);
  hClctCompFound2i = dbe->book2D("h_CLCT_found2i","h_CLCT_found2i",18,0,18,36,0.5,36.5);  		   
  hClctCompSameN2i = dbe->book2D("h_CLCT_SameN2i","h_CLCT_SameN2i",18,0,18,36,0.5,36.5);
  hClctCompMatch2i = dbe->book2D("h_CLCT_match2i","h_CLCT_match2i",18,0,18,36,0.5,36.5);
  hClctCompTotal2i = dbe->book2D("h_CLCT_total2i","h_CLCT_total2i",18,0,18,36,0.5,36.5);
  hLCTCompFound2i = dbe->book2D("h_LCT_found2i","h_LCT_found2i",18,0,18,36,0.5,36.5);
  hLCTCompSameN2i = dbe->book2D("h_LCT_SameN2i","h_LCT_SameN2i",18,0,18,36,0.5,36.5);
  hLCTCompMatch2i = dbe->book2D("h_LCT_match2i","h_LCT_match2i",18,0,18,36,0.5,36.5);
  hLCTCompTotal2i = dbe->book2D("h_LCT_total2i","h_LCT_total2i",18,0,18,36,0.5,36.5);
  */
  hCompAll = dbe->book3D("h_All","h_All",15,0.5,15.5,18,0,18,36,0.5,36.5);
  std::vector<std::string> lines[3];
  lines[0]=bad_wires;
  lines[1]=bad_strips;
  lines[2]=bad_chambers;
  char asdf[3][256]={"wires","strips","chambers"};
  
  for(int j=0; j<3; j++) {
    for(uint i=0; i<lines[j].size(); i++) {
      string line=lines[j][i];
      int endcap=1;
      if(line[2]=='-') endcap=2;
      int station=line[3]-48;
      int ring=line[5]-48;
      int chamber=line[7]-48;
      if(line.size()==9) chamber=10*(line[7]-48)+line[8]-48;
      
      int ix=0;
      if(station==1) ix=ring-1;
      else {
	ix=3+2*(station-2)+ring-1;
      }
      if(endcap==2) ix+=9;
      printf("%s %s\n",asdf[j],bad_chambers[i].c_str());
      printf("endcap %i, station %i, ring %i, chamber %i\n",endcap,station,ring,chamber);
      printf("ix %i, chamber %i\n",ix,chamber);
      hCompAll->Fill(13+j,ix+0.5,chamber);
    }
  }
    bookedCompHistos = true;
}

void CSCTriggerPrimitivesDQM::compare(
			 const CSCALCTDigiCollection* alcts_data,
			 const CSCALCTDigiCollection* alcts_emul,
			 const CSCCLCTDigiCollection* clcts_data,
			 const CSCCLCTDigiCollection* clcts_emul,
			 const CSCCorrelatedLCTDigiCollection* lcts_data,
			 const CSCCorrelatedLCTDigiCollection* lcts_emul) {

  // Book histos when called for the first time.
  if (!bookedCompHistos) bookCompHistos();

  // Comparisons
  compareALCTs(alcts_data, alcts_emul);
  compareCLCTs(clcts_data, clcts_emul);
  compareLCTs(lcts_data,  lcts_emul, alcts_data, clcts_data);
}

void CSCTriggerPrimitivesDQM::compareALCTs(
                                 const CSCALCTDigiCollection* alcts_data,
				 const CSCALCTDigiCollection* alcts_emul) {
  int emul_corr_bx;

  // (Empirical) offset between 12-bit fullBX and Tbin0 of raw anode hits.
  //int tbin_anode_offset = 4; // 2007, starting with run 539.
  int tbin_anode_offset = 5; // 2007, run 14419.
  if (isMTCCData_) tbin_anode_offset = 10; // MTCC-II. Why not 6???

  // Should be taken from config. parameters.
  int fifo_pretrig     = 10;
  int fpga_latency     =  6;
  int l1a_window_width =  7;
  // Time offset of raw hits w.r.t. the full 12-bit BXN.
  int rawhit_tbin_offset =
    (fifo_pretrig - fpga_latency) + (l1a_window_width-1)/2;
  // Extra difference due to additional register stages; determined
  // empirically.
  int register_delay =  2;

  // Loop over all chambers in search for ALCTs.
  CSCALCTDigiCollection::const_iterator digiIt;
  std::vector<CSCALCTDigi>::const_iterator pd, pe;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= 3; ring++) {
        for (int cham = 1; cham <= 36; cham++) {
	  // Calculate DetId.  0th layer means whole chamber.
	  CSCDetId detid(endc, stat, ring, cham, 0);

	  std::vector<CSCALCTDigi> alctV_data, alctV_emul;
	  const CSCALCTDigiCollection::Range& drange = alcts_data->get(detid);
	  for (digiIt = drange.first; digiIt != drange.second; digiIt++) {
	    if ((*digiIt).isValid()) {
	      alctV_data.push_back(*digiIt);
	    }
	  }

	  const CSCALCTDigiCollection::Range& erange = alcts_emul->get(detid);
	  for (digiIt = erange.first; digiIt != erange.second; digiIt++) {
	    if ((*digiIt).isValid()) {
	      alctV_emul.push_back(*digiIt);
	    }
	  }

	  int ndata = alctV_data.size();
	  int nemul = alctV_emul.size();
	  if (ndata == 0 && nemul == 0) continue;


	  //	  int csctype = getCSCType(detid);
	  //	  int mychamber = chamberSerial(detid);
	  //	  int ix = chamberIX(detid);
	  int ix2 = chamberIXi(detid);
	  //	  hAlctCompFound2i->Fill(ix2,detid.chamber());
	  hCompAll->Fill(1,ix2,detid.chamber());
	  if (ndata != nemul) {
	  }
	  else {
	    //	    hAlctCompSameN2i->Fill(ix2,detid.chamber());
	    hCompAll->Fill(2,ix2,detid.chamber());
	  }

	  for (int i = 0; i < ndata; i++) {
	    if (alctV_data[i].isValid() == 0) continue;
	    int data_trknmb    = alctV_data[i].getTrknmb();
	    int data_quality   = alctV_data[i].getQuality();
	    int data_accel     = alctV_data[i].getAccelerator();
	    int data_collB     = alctV_data[i].getCollisionB();
	    int data_wiregroup = alctV_data[i].getKeyWG();
	    int data_bx        = alctV_data[i].getBX();
	    int fullBX = alctV_data[i].getFullBX(); // full 12-bit BX

	    if (i < nemul) {
	      if (alctV_emul[i].isValid() == 0) continue;
	      int emul_trknmb    = alctV_emul[i].getTrknmb();
	      int emul_quality   = alctV_emul[i].getQuality();
	      int emul_accel     = alctV_emul[i].getAccelerator();
	      int emul_collB     = alctV_emul[i].getCollisionB();
	      int emul_wiregroup = alctV_emul[i].getKeyWG();
	      int emul_bx        = alctV_emul[i].getBX();

	      // Emulator BX re-calculated for comparison with BX in the data.
	      if (!isTMB07)
		emul_corr_bx = (fullBX + emul_bx - tbin_anode_offset) & 0x1f;
	      else
		emul_corr_bx = emul_bx - rawhit_tbin_offset + register_delay;
	      if (ndata == nemul) {
		if(detid.station()>1 && detid.ring()==1) {
		}	   
		else {
		}
		//                hAlctCompTotal2i->Fill(ix2,detid.chamber());
		hCompAll->Fill(3,ix2,detid.chamber());
	      }
	      if (data_trknmb    == emul_trknmb    &&
		  data_quality   == emul_quality   &&
		  data_accel     == emul_accel     &&
		  data_collB     == emul_collB     &&
		  data_wiregroup == emul_wiregroup &&
		  data_bx        == emul_corr_bx) {
		if (ndata == nemul) {
		  if(detid.station()>1 && detid.ring()==1) {
		  }	   
		  else {
		  }
		  //		  hAlctCompMatch2i->Fill(ix2,detid.chamber());
		  hCompAll->Fill(4,ix2,detid.chamber());
		}
	      }
	      else {
	      }
	    }
	  }
	}
      }
    }
  }
}

void CSCTriggerPrimitivesDQM::compareCLCTs(
                                 const CSCCLCTDigiCollection* clcts_data,
				 const CSCCLCTDigiCollection* clcts_emul) {
  // Number of Tbins before pre-trigger for raw cathode hits.
  const int tbin_cathode_offset = 7;

  // Loop over all chambers in search for CLCTs.
  CSCCLCTDigiCollection::const_iterator digiIt;
  std::vector<CSCCLCTDigi>::const_iterator pd, pe;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= 3; ring++) {
        for (int cham = 1; cham <= 36; cham++) {
	  // Calculate DetId.  0th layer means whole chamber.
	  CSCDetId detid(endc, stat, ring, cham, 0);

	  std::vector<CSCCLCTDigi> clctV_data, clctV_emul;
	  const CSCCLCTDigiCollection::Range& drange = clcts_data->get(detid);
	  for (digiIt = drange.first; digiIt != drange.second; digiIt++) {
	    if ((*digiIt).isValid()) {
	      clctV_data.push_back(*digiIt);
	    }
	  }

	  const CSCCLCTDigiCollection::Range& erange = clcts_emul->get(detid);
	  for (digiIt = erange.first; digiIt != erange.second; digiIt++) {
	    if ((*digiIt).isValid()) {
	      clctV_emul.push_back(*digiIt);
	    }
	  }

	  int ndata = clctV_data.size();
	  int nemul = clctV_emul.size();
	  if (ndata == 0 && nemul == 0) continue;
	  //	  int csctype = getCSCType(detid);
	  //	  int ix = chamberIX(detid);
	  int ix2 = chamberIXi(detid);
	  //	  hClctCompFound2i->Fill(ix2,detid.chamber());
	  hCompAll->Fill(5,ix2,detid.chamber());
	  if (ndata != nemul) {
	  }
	  else {
	    //	    hClctCompSameN2i->Fill(ix2,detid.chamber());
	    hCompAll->Fill(6,ix2,detid.chamber());
	  }

	  for (pd = clctV_data.begin(); pd != clctV_data.end(); pd++) {
	    if ((*pd).isValid() == 0) continue;
	    int data_trknmb    = (*pd).getTrknmb();
	    int data_quality   = (*pd).getQuality();
	    int data_pattern   = (*pd).getPattern();
	    int data_striptype = (*pd).getStripType();
	    int data_bend      = (*pd).getBend();
	    int data_keystrip  = (*pd).getKeyStrip();
	    int data_cfeb      = (*pd).getCFEB();
	    int data_bx        = (*pd).getBX();
	    int fullBX = (*pd).getFullBX(); // 12-bit full BX

	    for (pe = clctV_emul.begin(); pe != clctV_emul.end(); pe++) {
	      if ((*pe).isValid() == 0) continue;
	      int emul_trknmb    = (*pe).getTrknmb();
	      int emul_quality   = (*pe).getQuality();
	      int emul_pattern   = (*pe).getPattern();
	      int emul_striptype = (*pe).getStripType();
	      int emul_bend      = (*pe).getBend();
	      int emul_keystrip  = (*pe).getKeyStrip();
	      int emul_cfeb      = (*pe).getCFEB();
	      int emul_bx        = (*pe).getBX();

	      if (data_trknmb == emul_trknmb) {
		// Emulator BX re-calculated using 12-bit full BX number.
		// Used for comparison with BX in the data.
		int emul_corr_bx =
		  (fullBX + emul_bx - tbin_cathode_offset) & 0x03;
		if (ndata == nemul) {
		  if(detid.station()>1 && detid.ring()==1) {
		  }	   
		  else {
		  }
		  //		  hClctCompTotal2i->Fill(ix2,detid.chamber());
		  hCompAll->Fill(7,ix2,detid.chamber());
		}
		if (data_quality   == emul_quality   &&
		    data_pattern   == emul_pattern   &&
		    data_striptype == emul_striptype &&
		    data_bend      == emul_bend      &&
		    data_keystrip  == emul_keystrip  &&
		    data_cfeb      == emul_cfeb      &&
		    // BX comparison cannot be performed for MTCC data.
		    (isMTCCData_ || (data_bx == emul_corr_bx))) {
		  if (ndata == nemul) {
		    if(detid.station()>1 && detid.ring()==1) {
		    }	   
		    else {
		    }
		    //		    hClctCompMatch2i->Fill(ix2,detid.chamber());
		    hCompAll->Fill(8,ix2,detid.chamber());
		  }
		}
		break;
	      }
	    }
	  }
	}
      }
    }
  }
}

void CSCTriggerPrimitivesDQM::compareLCTs(
                             const CSCCorrelatedLCTDigiCollection* lcts_data,
			     const CSCCorrelatedLCTDigiCollection* lcts_emul,
			     const CSCALCTDigiCollection* alcts_data,
			     const CSCCLCTDigiCollection* clcts_data) {
  // Need ALCT and CLCT digi collections to convert emulator bx into
  // hardware bx.
  // Loop over all chambers in search for correlated LCTs.
  CSCCorrelatedLCTDigiCollection::const_iterator digiIt;
  std::vector<CSCCorrelatedLCTDigi>::const_iterator pd, pe;
  for (int endc = 1; endc <= 2; endc++) {
    for (int stat = 1; stat <= 4; stat++) {
      for (int ring = 1; ring <= 3; ring++) {
        for (int cham = 1; cham <= 36; cham++) {
	  // Calculate DetId.  0th layer means whole chamber.
	  CSCDetId detid(endc, stat, ring, cham, 0);

	  std::vector<CSCCorrelatedLCTDigi> lctV_data, lctV_emul;
	  const CSCCorrelatedLCTDigiCollection::Range&
	    drange = lcts_data->get(detid);
	  for (digiIt = drange.first; digiIt != drange.second; digiIt++) {
	    if ((*digiIt).isValid()) {
	      lctV_data.push_back(*digiIt);
	    }
	  }

	  const CSCCorrelatedLCTDigiCollection::Range&
	    erange = lcts_emul->get(detid);
	  for (digiIt = erange.first; digiIt != erange.second; digiIt++) {
	    if ((*digiIt).isValid()) {
	      lctV_emul.push_back(*digiIt);
	    }
	  }

	  int ndata = lctV_data.size();
	  int nemul = lctV_emul.size();
	  if (ndata == 0 && nemul == 0) continue;

	  //	  int csctype = getCSCType(detid);
	  //	  int ix = chamberIX(detid);
	  int ix2 = chamberIXi(detid);
	  if(detid.station()>1 && detid.ring()==1) {
	  }	   
	  else {
	  }
	  //	  hLCTCompFound2i->Fill(ix2,detid.chamber());
	  hCompAll->Fill(9,ix2,detid.chamber());
	  if (ndata != nemul) {
	  }
	  else {
	    //	    hLCTCompSameN2i->Fill(ix2,detid.chamber());
	    hCompAll->Fill(10,ix2,detid.chamber());
	  }
	  for (pd = lctV_data.begin(); pd != lctV_data.end(); pd++) {
	    if ((*pd).isValid() == 0) continue;
	    int data_trknmb    = (*pd).getTrknmb();
	    int data_quality   = (*pd).getQuality();
	    int data_wiregroup = (*pd).getKeyWG();
	    int data_keystrip  = (*pd).getStrip();
	    int data_pattern   = (*pd).getCLCTPattern();
	    int data_striptype = (*pd).getStripType();
	    int data_bend      = (*pd).getBend();
	    int data_bx        = (*pd).getBX();

	    for (pe = lctV_emul.begin(); pe != lctV_emul.end(); pe++) {
	      if ((*pe).isValid() == 0) continue;
	      int emul_trknmb    = (*pe).getTrknmb();
	      int emul_quality   = (*pe).getQuality();
	      int emul_wiregroup = (*pe).getKeyWG();
	      int emul_keystrip  = (*pe).getStrip();
	      int emul_pattern   = (*pe).getCLCTPattern();
	      int emul_striptype = (*pe).getStripType();
	      int emul_bend      = (*pe).getBend();
	      int emul_bx        = (*pe).getBX();
	      if (data_trknmb == emul_trknmb) {
		// Convert emulator BX into hardware BX using full 12-bit
		// BX words in ALCT and CLCT digi collections.
		int emul_corr_bx = convertBXofLCT(emul_bx, detid,
						  alcts_data, clcts_data);
		if (ndata == nemul) {
		  if(detid.station()>1 && detid.ring()==1) {
		  }	   
		  else {
		  }
		  //		  hLCTCompTotal2i->Fill(ix2,detid.chamber());
		  hCompAll->Fill(11,ix2,detid.chamber());
		}
		if (data_quality   == emul_quality   &&
		    data_wiregroup == emul_wiregroup &&
		    data_keystrip  == emul_keystrip  &&
		    data_pattern   == emul_pattern   &&
		    data_striptype == emul_striptype &&
		    data_bend      == emul_bend      &&
		    data_bx        == emul_corr_bx) {
		  if (ndata == nemul) {
		    if(detid.station()>1 && detid.ring()==1) {
		    }	   
		    else {
		    }
		    //		    hLCTCompMatch2i->Fill(ix2,detid.chamber());
		    hCompAll->Fill(12,ix2,detid.chamber());
		  }
		}
		break;
	      }
	    }
	  }
	}
      }
    }
  }
}

int CSCTriggerPrimitivesDQM::convertBXofLCT(
                             const int emul_bx, const CSCDetId& detid,
			     const CSCALCTDigiCollection* alcts_data,
			     const CSCCLCTDigiCollection* clcts_data) {
  int full_anode_bx = -999;
  //int full_cathode_bx = -999;
  int lct_bx = -999;
  int tbin_anode_offset = 5; // 2007, run 14419.
  if (isMTCCData_) tbin_anode_offset = 10; // MTCC-II.  Why not 6???

  // Extract full 12-bit anode BX word from ALCT collections.
  const CSCALCTDigiCollection::Range& arange = alcts_data->get(detid);
  for (CSCALCTDigiCollection::const_iterator digiIt = arange.first;
       digiIt != arange.second; digiIt++) {
    if ((*digiIt).isValid()) {
      full_anode_bx = (*digiIt).getFullBX();
      break;
    }
  }

  // Extract full 12-bit cathode BX word from CLCT collections.
  const CSCCLCTDigiCollection::Range& crange = clcts_data->get(detid);
  for (CSCCLCTDigiCollection::const_iterator digiIt = crange.first;
       digiIt != crange.second; digiIt++) {
    if ((*digiIt).isValid()) {
      //full_cathode_bx = (*digiIt).getFullBX();
      break;
    }
  }

  // Use these 12-bit BX's to convert emulator BX into hardware BX.
  if (full_anode_bx == -999) {
  }
  else {
    // LCT BX has two bits: the least-significant bit is the LSB of ALCT BX;
    // the MSB is 1/0 depending on whether the 12-bit full cathode BX is 0
    // or not.
    lct_bx = (full_anode_bx + emul_bx - tbin_anode_offset) & 0x01;
    // SV, 12/Jun/08: it looks like this bit is never set - docu must be
    // wrong.
    //lct_bx = lct_bx | ((full_cathode_bx == 0) << 1);
  }

  return lct_bx;
}


void CSCTriggerPrimitivesDQM::HotWires(const edm::Event& iEvent) {
  if (!bookedHotWireHistos) bookHotWireHistos();
  edm::Handle<CSCWireDigiCollection> wires;
  iEvent.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
  
  int serial_old=-1;
  for (CSCWireDigiCollection::DigiRangeIterator dWDiter=wires->begin(); dWDiter!=wires->end(); dWDiter++) {
    CSCDetId id = (CSCDetId)(*dWDiter).first;
    int serial = chamberSerial(id)-1;
    //     printf("serial %i\n",serial);
    std::vector<CSCWireDigi>::const_iterator wireIter = (*dWDiter).second.first;
    std::vector<CSCWireDigi>::const_iterator lWire = (*dWDiter).second.second;
    bool has_layer=false;
    for( ; wireIter != lWire; ++wireIter) {
      has_layer=true;
      //      int i_layer= id.layer()-1;
      //      int i_wire = wireIter->getWireGroup()-1;
    }
    if(serial_old!=serial && has_layer) {
      serial_old=serial;
    }
  }
}

// Returns chamber type (0-9) according to the station and ring number
int CSCTriggerPrimitivesDQM::getCSCType(const CSCDetId& id) {
  int type = -999;

  if (id.station() == 1) {
    type = (id.triggerCscId()-1)/3;
  }
  else { // stations 2-4
    type = 3 + id.ring() + 2*(id.station()-2);
  }

  assert(type >= 0 && type < CSC_TYPES);
  return type;
}

// Returns halfstrips-per-radian for different CSC types
double CSCTriggerPrimitivesDQM::getHsPerRad(const int idh) {
  return (NCHAMBERS[idh]*MAX_HS[idh]/TWOPI);
}

DEFINE_FWK_MODULE(CSCTriggerPrimitivesDQM);
