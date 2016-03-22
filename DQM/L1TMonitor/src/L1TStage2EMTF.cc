#include <string>
#include <vector>

#include "DQM/L1TMonitor/interface/L1TStage2EMTF.h"


L1TStage2EMTF::L1TStage2EMTF(const edm::ParameterSet& ps) 
    : emtfToken(consumes<l1t::EMTFOutputCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2EMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  int n_xbins;
  std::vector<std::string> name = {"42", "41", "32", "31", "22", "21", "13", "12", "11"};
  std::vector<std::string> label = {"4/2", "4/1", "3/2", "3/1", "2/2", "2/1", "1/3", "1/2", "1/1"};

  ibooker.setCurrentFolder(monitorDir);

  // ME (LCTs) Monitor Elements
  emtfErrors = ibooker.book1D("emtfErrors", "EMTF Errors", 6, 0, 6);
  emtfErrors->setAxisTitle("Error Type (Corruptions not implemented)", 1);
  emtfErrors->setAxisTitle("Number of Errors", 2);
  emtfErrors->setBinLabel(1, "Corruptions", 1);
  emtfErrors->setBinLabel(2, "Synch. Err.", 1);
  emtfErrors->setBinLabel(3, "Synch. Mod.", 1);
  emtfErrors->setBinLabel(4, "BX Mismatch", 1);
  emtfErrors->setBinLabel(5, "Time Misalign.", 1);
  emtfErrors->setBinLabel(6, "FMM != Ready", 1);

  emtfLCTBX = ibooker.book2D("emtfLCTBX", "EMTF LCT BX", 9, -1, 8, 18, -9, 9);
  emtfLCTBX->setAxisTitle("BX", 1);
  for (int bin = 1, j = -4; bin <= 9; ++bin, ++j) {
    emtfLCTBX->setBinLabel(bin, std::to_string(j), 1);
  }

  for (int i = 0; i < 9; ++i) {
    emtfLCTBX->setBinLabel(i + 1, "ME-" + label[i], 2);
    emtfLCTBX->setBinLabel(18 - i, "ME+" + label[i], 2);

    emtfLCTStrip[i] = ibooker.book1D("emtfLCTStripMENeg" + name[i], "EMTF Halfstrip ME-" + label[i], 256, 0, 256);
    emtfLCTStrip[i]->setAxisTitle("Cathode Halfstrip, ME-" + label[i], 1);

    emtfLCTStrip[17 - i] = ibooker.book1D("emtfLCTStripMEPos" + name[i], "EMTF Halfstrip ME+" + label[i], 256, 0, 256);
    emtfLCTStrip[17 - i]->setAxisTitle("Cathode Halfstrip, ME+" + label[i], 1);

    emtfLCTWire[i] = ibooker.book1D("emtfLCTWireMENeg" + name[i], "EMTF Wiregroup ME-" + label[i], 128, 0, 128);
    emtfLCTWire[i]->setAxisTitle("Anode Wiregroup, ME-" + label[i], 1);

    emtfLCTWire[17 - i] = ibooker.book1D("emtfLCTWireMEPos" + name[i], "EMTF Wiregroup ME+" + label[i], 128, 0, 128);
    emtfLCTWire[17 - i]->setAxisTitle("Anode Wiregroup, ME+" + label[i], 1);

    if (i < 6) {
      n_xbins = (i % 2) ? 18 : 36;

      emtfChamberStrip[i] = ibooker.book2D("emtfChamberStripMENeg" + name[i], "EMTF Halfstrip ME-" + label[i], n_xbins, 0, n_xbins, 256, 0, 256);
      emtfChamberStrip[i]->setAxisTitle("Chamber, ME-" + label[i], 1);
      emtfChamberStrip[i]->setAxisTitle("Cathode Halfstrip", 2);

      emtfChamberStrip[17 - i] = ibooker.book2D("emtfChamberStripMEPos" + name[i], "EMTF Halfstrip ME+" + label[i], n_xbins, 0, n_xbins, 256, 0, 256);
      emtfChamberStrip[17 - i]->setAxisTitle("Chamber, ME+" + label[i], 1);
      emtfChamberStrip[17 - i]->setAxisTitle("Cathode Halfstrip", 2);
    }
  }

  emtfChamberOccupancy = ibooker.book2D("emtfChamberOccupancy", "EMTF Chamber Occupancy", 55, 0, 55, 10, -5, 5);
  emtfChamberOccupancy->setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int bin = 1; bin <= 46; bin += 9) {
    emtfChamberOccupancy->setBinLabel(bin, std::to_string(bin % 8), 1);
  }
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

  // SP (Tracks) Monitor Elements
  emtfnTracks = ibooker.book1D("emtfnTracks", "Number of EMTF Tracks per Event", 4, 0, 4);
  for (int bin = 1; bin <= 4; ++bin) {
    emtfnTracks->setBinLabel(bin, std::to_string(bin - 1), 1);
  }

  emtfnLCTs = ibooker.book1D("emtfnLCTs", "Number of LCTs per EMTF Track", 5, 0, 5);
  for (int bin = 1; bin <= 5; ++bin) {
    emtfnLCTs->setBinLabel(bin, std::to_string(bin - 1), 1);
  }

  emtfTrackBX = ibooker.book2D("emtfTrackBX", "EMTF Track Bunch Crossings", 12, -6, 6, 7, 0, 7);
  emtfTrackBX->setAxisTitle("Sector (Endcap)", 1);
  for (int i = 0; i < 6; ++i) {
    emtfTrackBX->setBinLabel(i + 1, std::to_string(6 - i) + " (-)", 1);
    emtfTrackBX->setBinLabel(12 - i, std::to_string(6 - i) + " (+)", 1);
  }
  emtfTrackBX->setAxisTitle("Track BX", 2);
  for (int bin = 1, i = -3; bin <= 7; ++bin, ++i) {
    emtfTrackBX->setBinLabel(bin, std::to_string(i), 2);
  }
  
  emtfTrackPt = ibooker.book1D("emtfTrackPt", "EMTF Track p_{T}", 256, 1, 257);
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
    int Sector = EventHeader.SP_ts();
    if (Sector > 6) Sector -= 8;

    // Check if FMM Signal was good
    if (EventHeader.Rdy() == 0) emtfErrors->Fill(5);

    // ME (LCTs) Data Record
    l1t::emtf::MECollection MECollection = EMTFOutput->GetMECollection();

    for (std::vector<l1t::emtf::ME>::const_iterator ME = MECollection.begin(); ME != MECollection.end(); ++ME) {
      int Station = ME->Station();
      int CSCID = ME->CSC_ID();
      int half_strip = ME->CLCT_key_half_strip();
      int wire_group = ME->Key_wire_group();
      float bin_offset;
      int histogram_index;

      if (ME->SE()) emtfErrors->Fill(1);
      if (ME->SM()) emtfErrors->Fill(2);
      if (ME->BXE()) emtfErrors->Fill(3);
      if (ME->AF()) emtfErrors->Fill(4);

      if (Station == 0 || Station == 1) {
        if (CSCID < 3) {
          bin_offset = 0.5;
        } else if (CSCID > 2 && CSCID < 6) {
          bin_offset = 1.5;
        } else {
          bin_offset = 2.5;
        }
      } else {
        if (CSCID < 3) {
          if (Station == 2) {
            bin_offset = 3.5;
          } else if (Station == 3) {
            bin_offset = 5.5;
          } else {
            bin_offset = 7.5;
          }
        } else {
          if (Station == 2) {
            bin_offset = 4.5;
          } else if (Station == 3) {
            bin_offset = 6.5;
          } else {
            bin_offset = 8.5;
          }
        }
      }
   
      histogram_index = int(8.5 + Endcap * bin_offset);

      emtfLCTBX->Fill(ME->Tbin_num(), Endcap * bin_offset);
      emtfLCTStrip[histogram_index]->Fill(half_strip);
      emtfLCTWire[histogram_index]->Fill(wire_group);

      if (bin_offset > 3) {
        if (int(bin_offset) % 2) {
          emtfChamberStrip[histogram_index]->Fill((Sector * 3) + CSCID, half_strip);
        } else {
          emtfChamberStrip[histogram_index]->Fill((Sector * 6) + (CSCID - 3), half_strip);
        }
      }

      emtfChamberOccupancy->Fill((Sector * 9) + CSCID, Endcap * (Station + 0.5));
    }

    // SP (Tracks) Data Record
    l1t::emtf::SPCollection SPCollection = EMTFOutput->GetSPCollection();

    for (std::vector<l1t::emtf::SP>::const_iterator SP = SPCollection.begin(); SP != SPCollection.end(); ++SP) {
      float Pt = SP->Pt();
      float Eta = SP->Eta_GMT();
      float Phi_GMT_global_rad = SP->Phi_GMT_global() * (M_PI/180);
      if (Phi_GMT_global_rad > M_PI) Phi_GMT_global_rad -= 2*M_PI;
      int Quality = SP->Quality();

      if (Quality == 0) {
        emtfnLCTs->Fill(0);
      } else if (Quality == 1 || Quality == 2 || Quality == 4 || Quality == 8) {
        emtfnLCTs->Fill(1);
      } else if (Quality == 3 || Quality == 5 || Quality == 9 || Quality == 10 || Quality == 12) {
        emtfnLCTs->Fill(2);
      } else if (Quality == 7 || Quality == 11 || Quality == 13 || Quality == 14) {
        emtfnLCTs->Fill(3);
      } else {
        emtfnLCTs->Fill(4);
      }

      emtfTrackBX->Fill(Endcap * (Sector + 0.5), SP->TBIN_num());
      emtfTrackPt->Fill(Pt);
      emtfTrackEta->Fill(Eta);
      emtfTrackPhi->Fill(Phi_GMT_global_rad);
      emtfTrackOccupancy->Fill(Eta, Phi_GMT_global_rad);

      nTracks++;
    }
  }
  emtfnTracks->Fill(nTracks);
}

