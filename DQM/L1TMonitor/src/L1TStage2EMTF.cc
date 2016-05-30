#include <string>
#include <vector>

#include "DQM/L1TMonitor/interface/L1TStage2EMTF.h"


L1TStage2EMTF::L1TStage2EMTF(const edm::ParameterSet& ps)
    : inputToken(consumes<l1t::EMTFOutputCollection>(ps.getParameter<edm::InputTag>("emtfProducer"))),
      outputToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfProducer"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2EMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Monitor Elements for EMTF Output Collection
  ibooker.setCurrentFolder(monitorDir);

  int n_xbins;
  std::string name, label;
  std::vector<std::string> suffix_name = {"42", "41", "32", "31", "22", "21", "13", "12", "11"};
  std::vector<std::string> suffix_label = {"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"};

  // ME (LCTs)
  emtfErrors = ibooker.book1D("emtfErrors", "EMTF Errors", 6, 0, 6);
  emtfErrors->setAxisTitle("Error Type (Corruptions Not Implemented)", 1);
  emtfErrors->setAxisTitle("Number of Errors", 2);
  emtfErrors->setBinLabel(1, "Corruptions", 1);
  emtfErrors->setBinLabel(2, "Synch. Err.", 1);
  emtfErrors->setBinLabel(3, "Synch. Mod.", 1);
  emtfErrors->setBinLabel(4, "BX Mismatch", 1);
  emtfErrors->setBinLabel(5, "Time Misalign.", 1);
  emtfErrors->setBinLabel(6, "FMM != Ready", 1);

  emtfLCTBX = ibooker.book2D("emtfLCTBX", "EMTF LCT BX", 9, -1, 8, 18, 0, 18);
  emtfLCTBX->setAxisTitle("BX", 1);
  for (int bin = 1, bin_label = -4; bin <= 9; ++bin, ++bin_label) {
    emtfLCTBX->setBinLabel(bin, std::to_string(bin_label), 1);
    emtfLCTBX->setBinLabel(bin, "ME-" + suffix_label[bin - 1], 2);
    emtfLCTBX->setBinLabel(19 - bin, "ME+" + suffix_label[bin - 1], 2);
  }

  for (int hist = 0, i = 0; hist < 18; ++hist, i = hist % 9) {

    if (hist < 9) {
      name = "MENeg" + suffix_name[i];
      label = "ME-" + suffix_label[i];
    } else {
      name = "MEPos" + suffix_name[8 - i];
      label = "ME+" + suffix_label[8 - i];
    }

    if (hist < 6 || hist > 11) {
      n_xbins = (i % 2) ? 18 : 36;
    } else {
      n_xbins = 36;
    }

    emtfLCTStrip[hist] = ibooker.book1D("emtfLCTStrip" + name, "EMTF Halfstrip " + label, 256, 0, 256);
    emtfLCTStrip[hist]->setAxisTitle("Cathode Halfstrip, " + label, 1);

    emtfLCTWire[hist] = ibooker.book1D("emtfLCTWire" + name, "EMTF Wiregroup " + label, 128, 0, 128);
    emtfLCTWire[hist]->setAxisTitle("Anode Wiregroup, " + label, 1);

    emtfChamberStrip[hist] = ibooker.book2D("emtfChamberStrip" + name, "EMTF Halfstrip " + label, n_xbins, 1, 1+n_xbins, 256, 0, 256);
    emtfChamberStrip[hist]->setAxisTitle("Chamber, " + label, 1);
    emtfChamberStrip[hist]->setAxisTitle("Cathode Halfstrip", 2);

    emtfChamberWire[hist] = ibooker.book2D("emtfChamberWire" + name, "EMTF Wiregroup " + label, n_xbins, 1, 1+n_xbins, 128, 0, 128);
    emtfChamberWire[hist]->setAxisTitle("Chamber, " + label, 1);
    emtfChamberWire[hist]->setAxisTitle("Anode Wiregroup", 2);

    for (int bin = 1; bin <= n_xbins; ++bin) {
      emtfChamberStrip[hist]->setBinLabel(bin, std::to_string(bin), 1);
      emtfChamberWire[hist]->setBinLabel(bin, std::to_string(bin), 1);
    }
  }

  emtfChamberOccupancy = ibooker.book2D("emtfChamberOccupancy", "EMTF Chamber Occupancy", 54, 1, 55, 10, -5, 5);
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

  // SP (Tracks)
  emtfnTracksEvent = ibooker.book1D("emtfnTracksEvent", "Number of EMTF Tracks per Event", 11, 0, 11);
  for (int bin = 1; bin <= 10; ++bin) {
    emtfnTracksEvent->setBinLabel(bin, std::to_string(bin - 1), 1);
  }
  emtfnTracksEvent->setBinLabel(11, "Overflow", 1);

  emtfnTracksSP = ibooker.book1D("emtfnTracksSP", "Number of EMTF Tracks per Sector Processor", 6, 0, 6);
  for (int bin = 1; bin <= 5; ++bin) {
    emtfnTracksSP->setBinLabel(bin, std::to_string(bin - 1), 1);
  }
  emtfnTracksSP->setBinLabel(6, "Overflow", 1);

  emtfnLCTs = ibooker.book1D("emtfnLCTs", "Number of LCTs per EMTF Track", 5, 0, 5);
  for (int bin = 1; bin <= 5; ++bin) {
    emtfnLCTs->setBinLabel(bin, std::to_string(bin - 1), 1);
  }

  emtfTrackBX = ibooker.book2D("emtfTrackBX", "EMTF Track Bunch Crossing", 12, -6, 6, 7, 0, 7);
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

  emtfHQPhi = ibooker.book1D("emtfHQPhi", "EMTF High Quality #phi",126, -3.15,3.15);
  emtfHQPhi->setAxisTitle("Track #phi",1);

  emtfTrackOccupancy = ibooker.book2D("emtfTrackOccupancy", "EMTF Track Occupancy", 100, -2.5, 2.5, 126, -3.15, 3.15);
  emtfTrackOccupancy->setAxisTitle("#eta", 1);
  emtfTrackOccupancy->setAxisTitle("#phi", 2);

  emtfMode = ibooker.book1D("emtfMode", "EMTF Track Mode", 16, 0, 16);
  emtfMode->setAxisTitle("Mode", 1);

  emtfQuality = ibooker.book1D("emtfQuality", "EMTF Track Quality", 16, 0, 16);
  emtfQuality->setAxisTitle("Quality", 1);

  emtfQualityvsMode = ibooker.book2D("emtfQualityvsMode", "EMTF Track Quality vs Mode", 16, 0, 16, 16, 0, 16);
  emtfQualityvsMode->setAxisTitle("Mode", 1);
  emtfQualityvsMode->setAxisTitle("Quality", 2);

  for (int bin = 1; bin <= 16; ++bin) {
    emtfMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfQuality->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfQualityvsMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfQualityvsMode->setBinLabel(bin, std::to_string(bin - 1), 2);
  }

  // Monitor Elements for Muon Candidates (Output to uGMT)
  ibooker.setCurrentFolder(monitorDir + "/Output");

  emtfMuonBX = ibooker.book1D("emtfMuonBX", "EMTF Muon Cand BX", 7, -3, 4);
  emtfMuonBX->setAxisTitle("BX", 1);
  for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
    emtfMuonBX->setBinLabel(bin, std::to_string(bin_label), 1);
  }

  emtfMuonhwPt = ibooker.book1D("emtfMuonhwPt", "EMTF Muon Cand p_{T}", 512, 0, 512);
  emtfMuonhwPt->setAxisTitle("Hardware p_{T}", 1);

  emtfMuonhwEta = ibooker.book1D("emtfMuonhwEta", "EMTF Muon Cand #eta", 460, -230, 230);
  emtfMuonhwEta->setAxisTitle("Hardware #eta", 1);

  emtfMuonhwPhi = ibooker.book1D("emtfMuonhwPhi", "EMTF Muon Cand #phi", 125, -20, 105);
  emtfMuonhwPhi->setAxisTitle("Hardware #phi", 1);

  emtfMuonhwQual = ibooker.book1D("emtfMuonhwQual", "EMTF Muon Cand Quality", 16, 0, 16);
  emtfMuonhwQual->setAxisTitle("Quality", 1);
  for (int bin = 1; bin <= 16; ++bin) {
    emtfMuonhwQual->setBinLabel(bin, std::to_string(bin - 1), 1);
  }
}

void L1TStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose) edm::LogInfo("L1TStage2EMTF") << "L1TStage2EMTF: analyze..." << std::endl;

  edm::Handle<l1t::EMTFOutputCollection> EMTFOutputCollection;
  e.getByToken(inputToken, EMTFOutputCollection);

  int nTracksEvent = 0;

  for (std::vector<l1t::EMTFOutput>::const_iterator EMTFOutput = EMTFOutputCollection->begin(); EMTFOutput != EMTFOutputCollection->end(); ++EMTFOutput) {

    // Event Record Header
    const l1t::emtf::EventHeader* EventHeader = EMTFOutput->PtrEventHeader();
    int Endcap = EventHeader->Endcap();
    int Sector = EventHeader->Sector();

    if (!EventHeader->Rdy()) emtfErrors->Fill(5);

    // ME (LCTs) Data Record
    const l1t::emtf::MECollection* MECollection = EMTFOutput->PtrMECollection();

    for (std::vector<l1t::emtf::ME>::const_iterator ME = MECollection->begin(); ME != MECollection->end(); ++ME) {
      int Station = ME->Station();
      int Ring = ME->Ring();
      int Subsector = ME->Subsector();
      int CSC_ID = ME->CSC_ID();
      //int Neighbor = ME->Neighbor();
      int Strip = ME->Strip();
      int Wire = ME->Wire();

      // Evaluate histogram index and chamber number with respect to station and ring.
      int hist_index = 0, chamber_number = 0;

      if (Station == 1) {
        if (Ring == 1 || Ring == 4) {
          hist_index = 8;
          chamber_number = ((Sector-1) * 6) + CSC_ID + 2;
        } else if (Ring == 2) {
          hist_index = 7;
          chamber_number = ((Sector-1) * 6) + CSC_ID - 1;
        } else if (Ring == 3) {
          hist_index = 6;
          chamber_number = ((Sector-1) * 6) + CSC_ID - 4;
        }
        if (Subsector == 2) chamber_number += 3;
        if (chamber_number > 36) chamber_number -= 36;
      } else if (Ring == 1) {
        if (Station == 2) {
          hist_index = 5;
        } else if (Station == 3) {
          hist_index = 3;
        } else if (Station == 4) {
          hist_index = 1;
        }
        chamber_number = ((Sector-1) * 3) + CSC_ID + 1;
        if (chamber_number > 18) chamber_number -= 18;
      } else if (Ring == 2) {
        if (Station == 2) {
          hist_index = 4;
        } else if (Station == 3) {
          hist_index = 2;
        } else if (Station == 4) {
          hist_index = 0;
        }
        chamber_number = ((Sector-1) * 6) + CSC_ID + 2;
        if (chamber_number > 36) chamber_number -= 36;
      }

     if (Endcap > 0) hist_index = 17 - hist_index;

      if (ME->SE()) emtfErrors->Fill(1);
      if (ME->SM()) emtfErrors->Fill(2);
      if (ME->BXE()) emtfErrors->Fill(3);
      if (ME->AF()) emtfErrors->Fill(4);

      emtfLCTBX->Fill(ME->Tbin_num(), hist_index);

      emtfLCTStrip[hist_index]->Fill(Strip);
      emtfLCTWire[hist_index]->Fill(Wire);

      emtfChamberStrip[hist_index]->Fill(chamber_number, Strip);
      emtfChamberWire[hist_index]->Fill(chamber_number, Wire);

      if (Subsector == 1) {
        emtfChamberOccupancy->Fill(((Sector-1) * 9) + CSC_ID, Endcap * (Station - 0.5));
      } else {
        emtfChamberOccupancy->Fill(((Sector-1) * 9) + CSC_ID, Endcap * (Station + 0.5));
      }
    }

    // SP (Tracks) Data Record
    const l1t::emtf::SPCollection* SPCollection = EMTFOutput->PtrSPCollection();

    int nTracksSP = SPCollection->size();

    if (nTracksSP <= 6) {
      emtfnTracksSP->Fill(nTracksSP);
    } else {
      emtfnTracksSP->Fill(6);
    }

    for (std::vector<l1t::emtf::SP>::const_iterator SP = SPCollection->begin(); SP != SPCollection->end(); ++SP) {
      float Pt = SP->Pt();
      float Eta = SP->Eta_GMT();
      float Phi_GMT_global_rad = SP->Phi_GMT_global_rad();

      int Quality = SP->Quality();
      int Mode = SP->Mode();

      if (Mode == 0) {
        emtfnLCTs->Fill(0);
      } else if (Mode == 1 || Mode == 2 || Mode == 4 || Mode == 8) {
        emtfnLCTs->Fill(1);
      } else if (Mode == 3 || Mode == 5 || Mode == 9 || Mode == 10 || Mode == 12) {
        emtfnLCTs->Fill(2);
      } else if (Mode == 7 || Mode == 11 || Mode == 13 || Mode == 14) {
        emtfnLCTs->Fill(3);
      } else {
        emtfnLCTs->Fill(4);
      }

      emtfTrackBX->Fill(Endcap * (Sector - 0.5), SP->TBIN_num());
      emtfTrackPt->Fill(Pt);
      emtfTrackEta->Fill(Eta);
      emtfTrackPhi->Fill(Phi_GMT_global_rad);
      emtfTrackOccupancy->Fill(Eta, Phi_GMT_global_rad);
      emtfMode->Fill(Mode);
      emtfQuality->Fill(Quality);
      emtfQualityvsMode->Fill(Mode, Quality);
      if (Mode == 15) emtfHQPhi->Fill(Phi_GMT_global_rad);

      ++nTracksEvent;
    }
  }

  if (nTracksEvent <= 10) {
    emtfnTracksEvent->Fill(nTracksEvent);
  } else {
    emtfnTracksEvent->Fill(10);
  }

  edm::Handle<l1t::RegionalMuonCandBxCollection> MuonBxCollection;
  e.getByToken(outputToken, MuonBxCollection);

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {
      emtfMuonBX->Fill(itBX);
      emtfMuonhwPt->Fill(Muon->hwPt());
      emtfMuonhwEta->Fill(Muon->hwEta());
      emtfMuonhwPhi->Fill(Muon->hwPhi());
      emtfMuonhwQual->Fill(Muon->hwQual());
    }
  }
}

