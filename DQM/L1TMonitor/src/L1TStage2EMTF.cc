#include "DQM/L1TMonitor/interface/L1TStage2EMTF.h"


L1TStage2EMTF::L1TStage2EMTF(const edm::ParameterSet& ps) 
    : emtfToken(consumes<l1t::EMTFOutputCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2EMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  ibooker.setCurrentFolder(monitorDir);

  emtferrors = ibooker.book1D("emtferrors", "EMTF Errors", 6, 0, 6);
  emtferrors->setAxisTitle("Error type",1);
  emtferrors->setAxisTitle("Number of Errors",2);
  emtferrors->setBinLabel(1,"Corruptions [NOT IMPLEMENTED]",1);
  emtferrors->setBinLabel(2,"Synch. Err.",1);
  emtferrors->setBinLabel(3,"Synch. Mod.",1);
  emtferrors->setBinLabel(4,"BX mismatch",1);
  emtferrors->setBinLabel(5,"Time misalign.",1);
  emtferrors->setBinLabel(6,"FMM != Ready",1);

  emtflcts = ibooker.book2D("emtflcts", "EMTF LCTs", 9, -4.5, 4.5, 18, 0, 18);
  emtflcts->setAxisTitle("BX",1);

  int ihist = 0;
  for (int iEndcap = 0; iEndcap < 2; iEndcap++) {
    for (int iStation = 1; iStation < 5; iStation++) {
      for (int iRing = 1; iRing < 4; iRing++) {
        if (iStation != 1 && iRing > 2) continue;
        TString signEndcap = "+";
        if (iEndcap == 0) signEndcap = "-";

        char lcttitle[200];
        snprintf(lcttitle, 200, "ME%s%d/%d", signEndcap.Data(), iStation, iRing);
        if (ihist <= 8){
          emtflcts->setBinLabel(9-ihist,lcttitle,2);
        } else {
          emtflcts->setBinLabel(ihist+1,lcttitle,2);
        }
        ihist++;
      }
    }
  }

  emtfChamberOccupancy = ibooker.book2D("emtfChamberOccupancy", "EMTF Chamber Occupancy", 55, -0.5, 54.5, 10, -4.5, 5.5);
  emtfChamberOccupancy->setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  emtfChamberOccupancy->setBinLabel(1, "1", 1);
  emtfChamberOccupancy->setBinLabel(10, "2", 1);
  emtfChamberOccupancy->setBinLabel(19, "3", 1);
  emtfChamberOccupancy->setBinLabel(28, "4", 1);
  emtfChamberOccupancy->setBinLabel(37, "5", 1);
  emtfChamberOccupancy->setBinLabel(46, "6", 1);
  emtfChamberOccupancy->setBinLabel(1, "ME-4", 2);
  emtfChamberOccupancy->setBinLabel(2, "ME-3", 2);
  emtfChamberOccupancy->setBinLabel(3, "ME-2", 2);
  emtfChamberOccupancy->setBinLabel(4, "ME-1b", 2);
  emtfChamberOccupancy->setBinLabel(5, "ME-1a", 2);
  emtfChamberOccupancy->setBinLabel(6, "ME+1a", 2);
  emtfChamberOccupancy->setBinLabel(7, "ME+1b", 2);
  emtfChamberOccupancy->setBinLabel(8, "ME+2", 2);
  emtfChamberOccupancy->setBinLabel(9, "ME+3", 2);
  emtfChamberOccupancy->setBinLabel(10, "ME+4", 2);
  
  emtf_strip_ME11_POS = ibooker.book1D("emtf_strip_ME11_POS", "EMTF Strip ME+1/1", 230, 0, 230);
  emtf_strip_ME11_POS->setAxisTitle("HalfStrip, ME+1/1", 1);
  
  emtf_wire_ME11_POS = ibooker.book1D("emtf_wire_ME11_POS", "EMTF Wire ME+1/1", 120, 0, 120);
  emtf_wire_ME11_POS->setAxisTitle("Wire, ME+1/1", 1);
    
  emtf_strip_ME12_POS = ibooker.book1D("emtf_strip_ME12_POS", "EMTF Strip ME+1/2", 230, 0, 230);
  emtf_strip_ME12_POS->setAxisTitle("HalfStrip, ME+1/2", 1);
  
  emtf_wire_ME12_POS = ibooker.book1D("emtf_wire_ME12_POS", "EMTF Wire ME+1/2", 120, 0, 120);
  emtf_wire_ME12_POS->setAxisTitle("Wire, ME+1/2", 1);
    
  emtf_strip_ME13_POS = ibooker.book1D("emtf_strip_ME13_POS", "EMTF Strip ME+1/3", 230, 0, 230);
  emtf_strip_ME13_POS->setAxisTitle("HalfStrip, ME+1/3", 1);
  
  emtf_wire_ME13_POS = ibooker.book1D("emtf_wire_ME13_POS", "EMTF Wire ME+1/3", 120, 0, 120);
  emtf_wire_ME13_POS->setAxisTitle("Wire, ME+1/3", 1);
  
  emtf_strip_ME21_POS = ibooker.book1D("emtf_strip_ME21_POS", "EMTF Strip ME+2/1", 230, 0, 230);
  emtf_strip_ME21_POS->setAxisTitle("HalfStrip, ME+2/1", 1);
  
  emtf_wire_ME21_POS = ibooker.book1D("emtf_wire_ME21_POS", "EMTF Wire ME+2/1", 120, 0, 120);
  emtf_wire_ME21_POS->setAxisTitle("Wire, ME+2/1", 1);
  
  emtf_strip_ME22_POS = ibooker.book1D("emtf_strip_ME22_POS", "EMTF Strip ME+2/2", 230, 0, 230);
  emtf_strip_ME22_POS->setAxisTitle("HalfStrip, ME+2/2", 1);
  
  emtf_wire_ME22_POS = ibooker.book1D("emtf_wire_ME22_POS", "EMTF Wire ME+2/2", 120, 0, 120);
  emtf_wire_ME22_POS->setAxisTitle("Wire, ME+2/2", 1);
  
  emtf_strip_ME31_POS = ibooker.book1D("emtf_strip_ME31_POS", "EMTF Strip ME+3/1", 230, 0, 230);
  emtf_strip_ME31_POS->setAxisTitle("HalfStrip, ME+3/1", 1);
  
  emtf_wire_ME31_POS = ibooker.book1D("emtf_wire_ME31_POS", "EMTF Wire ME+3/1", 120, 0, 120);
  emtf_wire_ME31_POS->setAxisTitle("Wire, ME+3/1", 1);
  
  emtf_strip_ME32_POS = ibooker.book1D("emtf_strip_ME32_POS", "EMTF Strip ME+3/2", 230, 0, 230);
  emtf_strip_ME32_POS->setAxisTitle("HalfStrip, ME+3/2", 1);
  
  emtf_wire_ME32_POS = ibooker.book1D("emtf_wire_ME32_POS", "EMTF Wire ME+3/2", 120, 0, 120);
  emtf_wire_ME32_POS->setAxisTitle("Wire, ME+3/2", 1);
    
  emtf_strip_ME41_POS = ibooker.book1D("emtf_strip_ME41_POS", "EMTF Strip ME+4/1", 230, 0, 230);
  emtf_strip_ME41_POS->setAxisTitle("HalfStrip, ME+4/1", 1);
  
  emtf_wire_ME41_POS = ibooker.book1D("emtf_wire_ME41_POS", "EMTF Wire ME+4/1", 120, 0, 120);
  emtf_wire_ME41_POS->setAxisTitle("Wire, ME+4/1", 1);
  
  emtf_strip_ME42_POS = ibooker.book1D("emtf_strip_ME42_POS", "EMTF Strip ME+4/2", 230, 0, 230);
  emtf_strip_ME42_POS->setAxisTitle("HalfStrip, ME+4/2", 1);
  
  emtf_wire_ME42_POS = ibooker.book1D("emtf_wire_ME42_POS", "EMTF Wire ME+4/2", 120, 0, 120);
  emtf_wire_ME42_POS->setAxisTitle("Wire, ME+4/2", 1); 
    
  emtf_strip_ME11_NEG = ibooker.book1D("emtf_strip_ME11_NEG", "EMTF Strip ME-1/1", 230, 0, 230);
  emtf_strip_ME11_NEG->setAxisTitle("HalfStrip, ME-1/1", 1);
  
  emtf_wire_ME11_NEG = ibooker.book1D("emtf_wire_ME11_NEG", "EMTF Wire ME-1/1", 120, 0, 120);
  emtf_wire_ME11_NEG->setAxisTitle("Wire, ME-1/1", 1);
    
  emtf_strip_ME12_NEG = ibooker.book1D("emtf_strip_ME12_NEG", "EMTF Strip ME-1/2", 230, 0, 230);
  emtf_strip_ME12_NEG->setAxisTitle("HalfStrip, ME-1/2", 1);
  
  emtf_wire_ME12_NEG = ibooker.book1D("emtf_wire_ME12_NEG", "EMTF Wire ME-1/2", 120, 0, 120);
  emtf_wire_ME12_NEG->setAxisTitle("Wire, ME-1/2", 1);
    
  emtf_strip_ME13_NEG = ibooker.book1D("emtf_strip_ME13_NEG", "EMTF Strip ME-1/3", 230, 0, 230);
  emtf_strip_ME13_NEG->setAxisTitle("HalfStrip, ME-1/3", 1);
  
  emtf_wire_ME13_NEG = ibooker.book1D("emtf_wire_ME13_NEG", "EMTF Wire ME-1/3", 120, 0, 120);
  emtf_wire_ME13_NEG->setAxisTitle("Wire, ME-1/3", 1);
  
  emtf_strip_ME21_NEG = ibooker.book1D("emtf_strip_ME21_NEG", "EMTF Strip ME-2/1", 230, 0, 230);
  emtf_strip_ME21_NEG->setAxisTitle("HalfStrip, ME-2/1", 1);
  
  emtf_wire_ME21_NEG = ibooker.book1D("emtf_wire_ME21_NEG", "EMTF Wire ME-2/1", 120, 0, 120);
  emtf_wire_ME21_NEG->setAxisTitle("Wire, ME-2/1", 1);
  
  emtf_strip_ME22_NEG = ibooker.book1D("emtf_strip_ME22_NEG", "EMTF Strip ME-2/2", 230, 0, 230);
  emtf_strip_ME22_NEG->setAxisTitle("HalfStrip, ME-2/2", 1);
  
  emtf_wire_ME22_NEG = ibooker.book1D("emtf_wire_ME22_NEG", "EMTF Wire ME-2/2", 120, 0, 120);
  emtf_wire_ME22_NEG->setAxisTitle("Wire, ME-2/2", 1);
  
  emtf_strip_ME31_NEG = ibooker.book1D("emtf_strip_ME31_NEG", "EMTF Strip ME-3/1", 230, 0, 230);
  emtf_strip_ME31_NEG->setAxisTitle("HalfStrip, ME-3/1", 1);
  
  emtf_wire_ME31_NEG = ibooker.book1D("emtf_wire_ME31_NEG", "EMTF Wire ME-3/1", 120, 0, 120);
  emtf_wire_ME31_NEG->setAxisTitle("Wire, ME-3/1", 1);
  
  emtf_strip_ME32_NEG = ibooker.book1D("emtf_strip_ME32_NEG", "EMTF Strip ME-3/2", 230, 0, 230);
  emtf_strip_ME32_NEG->setAxisTitle("HalfStrip, ME-3/2", 1);
  
  emtf_wire_ME32_NEG = ibooker.book1D("emtf_wire_ME32_NEG", "EMTF Wire ME-3/2", 120, 0, 120);
  emtf_wire_ME32_NEG->setAxisTitle("Wire, ME-3/2", 1);
    
  emtf_strip_ME41_NEG = ibooker.book1D("emtf_strip_ME41_NEG", "EMTF Strip ME-4/1", 230, 0, 230);
  emtf_strip_ME41_NEG->setAxisTitle("HalfStrip, ME-4/1", 1);
  
  emtf_wire_ME41_NEG = ibooker.book1D("emtf_wire_ME41_NEG", "EMTF Wire ME-4/1", 120, 0, 120);
  emtf_wire_ME41_NEG->setAxisTitle("Wire, ME-4/1", 1);
  
  emtf_strip_ME42_NEG = ibooker.book1D("emtf_strip_ME42_NEG", "EMTF Strip ME-4/2", 230, 0, 230);
  emtf_strip_ME42_NEG->setAxisTitle("HalfStrip, ME-4/2", 1);
  
  emtf_wire_ME42_NEG = ibooker.book1D("emtf_wire_ME42_NEG", "EMTF Wire ME-4/2", 120, 0, 120);
  emtf_wire_ME42_NEG->setAxisTitle("Wire, ME-4/2", 1);
    
  emtf_chamberstrip_ME21_POS = ibooker.book2D("emtf_chamberstrip_ME21_POS", "EMTF Strip ME+2/1", 18, .5, 18.5, 160, 0, 160);
  emtf_chamberstrip_ME21_POS->setAxisTitle("Chamber, ME+2/1", 1);
  emtf_chamberstrip_ME21_POS->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME22_POS = ibooker.book2D("emtf_chamberstrip_ME22_POS", "EMTF Strip ME+2/2", 36, .5, 36.5, 160, 0, 160);
  emtf_chamberstrip_ME22_POS->setAxisTitle("Chamber, ME+2/2", 1);
  emtf_chamberstrip_ME22_POS->setAxisTitle("HalfStrip ME", 2);
  
  emtf_chamberstrip_ME31_POS = ibooker.book2D("emtf_chamberstrip_ME31_POS", "EMTF Strip ME+3/1", 18, .5, 18.5, 160, 0, 160);
  emtf_chamberstrip_ME31_POS->setAxisTitle("Chamber, ME+3/1", 1);
  emtf_chamberstrip_ME31_POS->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME32_POS = ibooker.book2D("emtf_chamberstrip_ME32_POS", "EMTF Strip ME+3/2", 36, .5, 36.5, 160, 0, 160);
  emtf_chamberstrip_ME32_POS->setAxisTitle("Chamber, ME+3/2", 1);
  emtf_chamberstrip_ME32_POS->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME41_POS = ibooker.book2D("emtf_chamberstrip_ME41_POS", "EMTF Strip ME+4/1", 18, .5, 18.5, 160, 0, 160);
  emtf_chamberstrip_ME41_POS->setAxisTitle("Chamber, ME+4/1", 1);
  emtf_chamberstrip_ME41_POS->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME42_POS = ibooker.book2D("emtf_chamberstrip_ME42_POS", "EMTF Strip ME+4/2", 36, .5, 36.5, 160, 0, 160);
  emtf_chamberstrip_ME42_POS->setAxisTitle("Chamber, ME+4/2", 1);
  emtf_chamberstrip_ME42_POS->setAxisTitle("HalfStrip", 2);  
    
  emtf_chamberstrip_ME21_NEG = ibooker.book2D("emtf_chamberstrip_ME21_NEG", "EMTF Strip ME-2/1", 18, .5, 18.5, 160, 0, 160);
  emtf_chamberstrip_ME21_NEG->setAxisTitle("Chamber, ME-2/1", 1);
  emtf_chamberstrip_ME21_NEG->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME22_NEG = ibooker.book2D("emtf_chamberstrip_ME22_NEG", "EMTF Strip ME-2/2", 36, .5, 36.5, 160, 0, 160);
  emtf_chamberstrip_ME22_NEG->setAxisTitle("Chamber, ME-2/2", 1);
  emtf_chamberstrip_ME22_NEG->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME31_NEG = ibooker.book2D("emtf_chamberstrip_ME31_NEG", "EMTF Strip ME-3/1", 18, .5, 18.5, 160, 0, 160);
  emtf_chamberstrip_ME31_NEG->setAxisTitle("Chamber, ME-3/1", 1);
  emtf_chamberstrip_ME31_NEG->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME32_NEG = ibooker.book2D("emtf_chamberstrip_ME32_NEG", "EMTF Strip ME-3/2", 36, .5, 36.5, 160, 0, 160);
  emtf_chamberstrip_ME32_NEG->setAxisTitle("Chamber, ME-3/2", 1);
  emtf_chamberstrip_ME32_NEG->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME41_NEG = ibooker.book2D("emtf_chamberstrip_ME41_NEG", "EMTF Strip ME-4/1", 18, .5, 18.5, 160, 0, 160);
  emtf_chamberstrip_ME41_NEG->setAxisTitle("Chamber, ME-4/1", 1);
  emtf_chamberstrip_ME41_NEG->setAxisTitle("HalfStrip", 2);
  
  emtf_chamberstrip_ME42_NEG = ibooker.book2D("emtf_chamberstrip_ME42_NEG", "EMTF Strip ME-4/2", 36, .5, 36.5, 160, 0, 160);
  emtf_chamberstrip_ME42_NEG->setAxisTitle("Chamber, ME-4/2", 1);
  emtf_chamberstrip_ME42_NEG->setAxisTitle("HalfStrip", 2);

  emtfnTracks = ibooker.book1D("emtfnTracks", "Number of EMTF Tracks per Event", 4, -0.5, 3.5);

  emtfnLCTs = ibooker.book1D("emtfnLCTs", "Number of LCTs per EMTF Track", 5, -0.5, 4.5);

  emtfTrackBX = ibooker.book2D("emtfTrackBX", "EMTF Track Bunch Crossings", 13, -6.5, 6.5, 7, -3.5, 3.5);
  emtfTrackBX->setAxisTitle("Sector (Endcap)", 1);
  emtfTrackBX->setBinLabel(1, "6 (-)", 1);
  emtfTrackBX->setBinLabel(2, "5 (-)", 1);
  emtfTrackBX->setBinLabel(3, "4 (-)", 1);
  emtfTrackBX->setBinLabel(4, "3 (-)", 1);
  emtfTrackBX->setBinLabel(5, "2 (-)", 1);
  emtfTrackBX->setBinLabel(6, "1 (-)", 1);
  emtfTrackBX->setBinLabel(7, "", 1);
  emtfTrackBX->setBinLabel(8, "1 (+)", 1);
  emtfTrackBX->setBinLabel(9, "2 (+)", 1);
  emtfTrackBX->setBinLabel(10, "3 (+)", 1);
  emtfTrackBX->setBinLabel(11, "4 (+)", 1);
  emtfTrackBX->setBinLabel(12, "5 (+)", 1);
  emtfTrackBX->setBinLabel(13, "6 (+)", 1);
  emtfTrackBX->setAxisTitle("Track BX", 2);
  
  emtfTrackPt = ibooker.book1D("emtfTrackPt", "EMTF Track p_{T}", 256, 0.5, 256.5);
  emtfTrackPt->setAxisTitle("Track p_{T} [GeV]", 1);

  emtfTrackEta = ibooker.book1D("emtfTrackEta", "EMTF Track #eta", 100, -2.5, 2.5);
  emtfTrackEta->setAxisTitle("Track #eta", 1);

  emtfTrackPhi = ibooker.book1D("emtfTrackPhi", "EMTF Track #phi", 126, -3.15, 3.15);
  emtfTrackPhi->setAxisTitle("Track #phi", 1);

  emtfTrackOccupancy = ibooker.book2D("emtfTrackOccupancy", "EMTF Track Occupancy", 100, -2.5, 2.5, 126, -3.15, 3.15);
  emtfTrackOccupancy->setAxisTitle("#eta", 1);
  emtfTrackOccupancy->setAxisTitle("#phi", 2);
}

void L1TStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2EMTF") << "L1TStage2EMTF: analyze..." << std::endl;

  edm::Handle<l1t::EMTFOutputCollection> EMTFOutputCollection;
  e.getByToken(emtfToken, EMTFOutputCollection);

  int nTracks = 0;
 
  for (std::vector<l1t::EMTFOutput>::const_iterator EMTFOutput = EMTFOutputCollection->begin(); EMTFOutput != EMTFOutputCollection->end(); ++EMTFOutput) {

    // Event Record Header
    l1t::emtf::EventHeader EventHeader = EMTFOutput->GetEventHeader();
    int Endcap = EventHeader.Endcap();
    int Sector = EventHeader.Sector();
    int RDY = EventHeader.Rdy(); //For csctferrors, check if FMM Signal was good

    // ME Data Record (LCTs)
    l1t::emtf::MECollection MECollection = EMTFOutput->GetMECollection();

    for (std::vector<l1t::emtf::ME>::const_iterator ME = MECollection.begin(); ME != MECollection.end(); ++ME) {
      int CSCID = ME->CSC_ID();
      int Station = ME->Station();
      int CSCID_offset = (Sector - 1) * 9;
      int strip = ME->CLCT_key_half_strip();
      int wire = ME->Key_wire_group();
      int CSCID_offset_ring1 = CSCID + (3 * (Sector-1)) + 1; 
      int CSCID_offset_ring2 = CSCID + (6 * (Sector-1)) - 2;
      int ring = 0;
      int bx = ME->Tbin_num() - 3;
      bool SE = ME->SE();
      bool SM = ME->SM();
      bool BXE = ME->BXE();
      bool AF = ME->AF();

      if (SE)     emtferrors->Fill(1.5);
      if (SM)     emtferrors->Fill(2.5);
      if (BXE)    emtferrors->Fill(3.5);
      if (AF)     emtferrors->Fill(4.5);
      if (RDY == 0) emtferrors->Fill(5.5);

      // Get the ring number
      if (Station == 1 || Station == 0) {
        if (CSCID > -1 && CSCID < 3) {
          ring = 1;
        } else if (CSCID > 2 && CSCID < 6) {
          ring = 2;
        } else if (CSCID > 5 && CSCID < 9) {
          ring = 3;
        }
      } else if (Station == 2 || Station == 3 || Station == 4) {
        if (CSCID > -1 && CSCID < 3) {
          ring = 1;
        } else if (CSCID > 2 && CSCID < 9) {
          ring = 2;
        }
      }

      if (Endcap < 0) {

        emtfChamberOccupancy->Fill(CSCID + CSCID_offset, Station * -1);

        if (Station == 0 || Station == 1) {
          if (ring == 1) {
            emtflcts->Fill(bx, 8.5);
          } else if (ring == 2) {
            emtflcts->Fill(bx, 7.5);
          } else {
            emtflcts->Fill(bx, 6.5);
          }
        } else if (Station==2) {
          if (ring == 1) {
            emtflcts->Fill(bx, 5.5);
          } else { 
            emtflcts->Fill(bx, 4.5);
          }
        } else if (Station == 3) {
          if (ring == 1) {
            emtflcts->Fill(bx, 3.5);
          } else{
            emtflcts->Fill(bx, 2.5);
          }
        } else if (Station == 4) {
          if (ring == 1) {
            emtflcts->Fill(bx, 1.5);
          } else {
            emtflcts->Fill(bx, 0.5);
          }
        }
      }

      if (Endcap > 0) {

        emtfChamberOccupancy->Fill(CSCID + CSCID_offset, Station + 1);

        if (Station == 0 || Station == 1) {
          if (ring == 1) {
            emtflcts->Fill(bx, 9.5);
          } else if (ring == 2) {
            emtflcts->Fill(bx, 10.5);
          } else {
            emtflcts->Fill(bx, 11.5);
          }
        } else if (Station == 2) {
          if (ring == 1) {
            emtflcts->Fill(bx, 12.5);
          } else {
            emtflcts->Fill(bx, 13.5);
          }
        } else if (Station == 3) {
          if (ring == 1){
            emtflcts->Fill(bx, 14.5);
          } else {
            emtflcts->Fill(bx, 15.5);
          }
        } else if (Station == 4) {
          if (ring == 1) {
            emtflcts->Fill(bx, 16.5);
          } else { 
            emtflcts->Fill(bx, 17.5);
          }
        }
      }

      // Positive Endcap, Ring 1
      if (Endcap > 0 && CSCID < 3) {
        if (Station < 2) {
          emtf_strip_ME11_POS->Fill(strip);
          emtf_wire_ME11_POS->Fill(wire);
        }
        if (Station == 2) {
          emtf_strip_ME21_POS->Fill(strip);
          emtf_wire_ME21_POS->Fill(wire);
        }
        if (Station == 3) {
          emtf_strip_ME31_POS->Fill(strip);
          emtf_wire_ME31_POS->Fill(wire);
        }
        if (Station == 4) {
          emtf_strip_ME41_POS->Fill(strip);
          emtf_wire_ME41_POS->Fill(wire);
        }
      }
      
      // Positive Endcap, Station 1
      if (Endcap > 0 && Station == 1) {
        if (CSCID > 2 && CSCID < 6) {
          emtf_strip_ME12_POS->Fill(strip);
          emtf_wire_ME12_POS->Fill(wire);
        }
	if (CSCID > 5) {
          emtf_strip_ME13_POS->Fill(strip);
          emtf_wire_ME13_POS->Fill(wire);
        }
      }
      
      // Positive Endcap, Ring 2
      if (Endcap > 0 && CSCID > 2) {
        if (Station == 2) {
          emtf_strip_ME22_POS->Fill(strip);
          emtf_wire_ME22_POS->Fill(wire);
        }
        if (Station == 3) {
          emtf_strip_ME32_POS->Fill(strip);
          emtf_wire_ME32_POS->Fill(wire);
        }
        if (Station == 4) {
          emtf_strip_ME42_POS->Fill(strip);
          emtf_wire_ME42_POS->Fill(wire);
        }
      }
      
      // Negative Endcap, Ring 1
      if (Endcap < 0 && CSCID < 3) {
        if (Station < 2) {
          emtf_strip_ME11_NEG->Fill(strip);
	  emtf_wire_ME11_NEG->Fill(wire);
        }
        if (Station == 2) {
          emtf_strip_ME21_NEG->Fill(strip);
          emtf_wire_ME21_NEG->Fill(wire);
        }
        if (Station == 3) {
          emtf_strip_ME31_NEG->Fill(strip);
          emtf_wire_ME31_NEG->Fill(wire);
        }
        if (Station == 4) {
          emtf_strip_ME41_NEG->Fill(strip);
          emtf_wire_ME41_NEG->Fill(wire);
        }
      }
      
      // Negative Endcap, Station 1
      if (Endcap < 0 && Station == 1) {
        if (CSCID > 2 && CSCID < 6) {
          emtf_strip_ME12_NEG->Fill(strip);
          emtf_wire_ME12_NEG->Fill(wire);
        }
        if (CSCID > 5) {
          emtf_strip_ME13_NEG->Fill(strip);
          emtf_wire_ME13_NEG->Fill(wire);
        }
      }
      
      // Negative Endcap, Ring 2
      if (Endcap < 0 && CSCID > 2) {
        if (Station == 2) {
          emtf_strip_ME22_NEG->Fill(strip);
          emtf_wire_ME22_NEG->Fill(wire);
        }
        if (Station == 3) {
          emtf_strip_ME32_NEG->Fill(strip);
          emtf_wire_ME32_NEG->Fill(wire);
        }
        if (Station == 4) {
          emtf_strip_ME42_NEG->Fill(strip);
          emtf_wire_ME42_NEG->Fill(wire);
        }
      }
   
      
      // Positive Endcap      
      if (Endcap > 0 && CSCID < 3) {
        if (Station == 2) emtf_chamberstrip_ME21_POS->Fill(CSCID_offset_ring1, strip);
        if (Station == 3) emtf_chamberstrip_ME31_POS->Fill(CSCID_offset_ring1, strip);
        if (Station == 4) emtf_chamberstrip_ME41_POS->Fill(CSCID_offset_ring1, strip);
      }
      
      if (Endcap > 0 && CSCID > 2) {
        if (Station == 2) emtf_chamberstrip_ME22_POS->Fill(CSCID_offset_ring2, strip);
        if (Station == 3) emtf_chamberstrip_ME32_POS->Fill(CSCID_offset_ring2, strip);
        if (Station == 4) emtf_chamberstrip_ME42_POS->Fill(CSCID_offset_ring2, strip);
      }   
      
      // Negative Endcap
      if (Endcap < 0 && CSCID < 3) {
        if (Station == 2) emtf_chamberstrip_ME21_NEG->Fill(CSCID_offset_ring1, strip);
        if (Station == 3) emtf_chamberstrip_ME31_NEG->Fill(CSCID_offset_ring1, strip);
        if (Station == 4) emtf_chamberstrip_ME41_NEG->Fill(CSCID_offset_ring1, strip);
      }
      
      if (Endcap < 0 && CSCID > 2) {
        if (Station==2) emtf_chamberstrip_ME22_NEG->Fill(CSCID_offset_ring2, strip);
        if (Station==3) emtf_chamberstrip_ME32_NEG->Fill(CSCID_offset_ring2, strip);
        if (Station==4) emtf_chamberstrip_ME42_NEG->Fill(CSCID_offset_ring2, strip);
      }  
    }

    // SP Output Data Record
    l1t::emtf::SPCollection SPCollection = EMTFOutput->GetSPCollection();

    for (std::vector<l1t::emtf::SP>::const_iterator SP = SPCollection.begin(); SP != SPCollection.end(); ++SP) {
      int Quality = SP->Quality();
      float Eta_GMT = SP->Eta_GMT();
      float Phi_GMT_global_rad = SP->Phi_GMT_global() * (M_PI/180);
      if (Phi_GMT_global_rad > M_PI) Phi_GMT_global_rad -= 2*M_PI;
      

      switch (Quality) {
        case 0: {
          emtfnLCTs->Fill(0);
          break;
        }
        case 1: case 2: case 4: case 8: {
          emtfnLCTs->Fill(1);
          break;
        }
        case 3: case 5: case 9: case 10: case 12: {
          emtfnLCTs->Fill(2);
          break;
        }
        case 7: case 11: case 13: case 14: {
          emtfnLCTs->Fill(3);
          break;
        }
        case 15: {
          emtfnLCTs->Fill(4);
          break;
        }
      }

      emtfTrackBX->Fill(Endcap * Sector, SP->TBIN_num() - 3);
      emtfTrackPt->Fill(SP->Pt());
      emtfTrackEta->Fill(Eta_GMT);
      emtfTrackPhi->Fill(Phi_GMT_global_rad);
      emtfTrackOccupancy->Fill(Eta_GMT, Phi_GMT_global_rad);

      nTracks++;
    }
  }

  emtfnTracks->Fill(nTracks);
}

