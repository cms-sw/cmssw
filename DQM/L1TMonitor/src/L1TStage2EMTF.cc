#include <string>
#include <vector>

#include "DQM/L1TMonitor/interface/L1TStage2EMTF.h"


L1TStage2EMTF::L1TStage2EMTF(const edm::ParameterSet& ps)
    : daqToken(consumes<l1t::EMTFDaqOutCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      hitToken(consumes<l1t::EMTFHit2016Collection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      trackToken(consumes<l1t::EMTFTrack2016Collection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      muonToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TStage2EMTF::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  ibooker.setCurrentFolder(monitorDir);

  // DAQ Output Monitor Elements
  emtfErrors = ibooker.book1D("emtfErrors", "EMTF Errors", 6, 0, 6);
  emtfErrors->setAxisTitle("Error Type (Corruptions Not Implemented)", 1);
  emtfErrors->setAxisTitle("Number of Errors", 2);
  emtfErrors->setBinLabel(1, "Corruptions", 1);
  emtfErrors->setBinLabel(2, "Synch. Err.", 1);
  emtfErrors->setBinLabel(3, "Synch. Mod.", 1);
  emtfErrors->setBinLabel(4, "BX Mismatch", 1);
  emtfErrors->setBinLabel(5, "Time Misalign.", 1);
  emtfErrors->setBinLabel(6, "FMM != Ready", 1);

  // Hit (LCT) Monitor Elements
  int n_xbins;
  std::string name, label;
  std::vector<std::string> suffix_name = {"42", "41", "32", "31", "22", "21", "13", "12", "11"};
  std::vector<std::string> suffix_label = {"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"};

  emtfHitBX = ibooker.book2D("emtfHitBX", "EMTF Hit BX", 8, -3, 5, 18, 0, 18);
  emtfHitBX->setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 8; ++xbin, ++xbin_label) {
    emtfHitBX->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 9; ++ybin) {
    emtfHitBX->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    emtfHitBX->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
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

    emtfHitStrip[hist] = ibooker.book1D("emtfHitStrip" + name, "EMTF Halfstrip " + label, 256, 0, 256);
    emtfHitStrip[hist]->setAxisTitle("Cathode Halfstrip, " + label, 1);

    emtfHitWire[hist] = ibooker.book1D("emtfHitWire" + name, "EMTF Wiregroup " + label, 128, 0, 128);
    emtfHitWire[hist]->setAxisTitle("Anode Wiregroup, " + label, 1);

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

  emtfHitOccupancy = ibooker.book2D("emtfHitOccupancy", "EMTF Chamber Occupancy", 54, 1, 55, 10, -5, 5);
  emtfHitOccupancy->setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int bin = 1; bin <= 46; bin += 9) {
    emtfHitOccupancy->setBinLabel(bin, std::to_string(bin % 8), 1);
  }
  emtfHitOccupancy->setBinLabel(1, "ME-4", 2);
  emtfHitOccupancy->setBinLabel(2, "ME-3", 2);
  emtfHitOccupancy->setBinLabel(3, "ME-2", 2);
  emtfHitOccupancy->setBinLabel(4, "ME-1b", 2);
  emtfHitOccupancy->setBinLabel(5, "ME-1a", 2);
  emtfHitOccupancy->setBinLabel(6, "ME+1a", 2);
  emtfHitOccupancy->setBinLabel(7, "ME+1b", 2);
  emtfHitOccupancy->setBinLabel(8, "ME+2", 2);
  emtfHitOccupancy->setBinLabel(9, "ME+3", 2);
  emtfHitOccupancy->setBinLabel(10, "ME+4", 2);

  // Track Monitor Elements
  emtfnTracks = ibooker.book1D("emtfnTracks", "Number of EMTF Tracks per Event", 11, 0, 11);
  for (int bin = 1; bin <= 10; ++bin) {
    emtfnTracks->setBinLabel(bin, std::to_string(bin - 1), 1);
  }
  emtfnTracks->setBinLabel(11, "Overflow", 1);

  emtfTracknHits = ibooker.book1D("emtfTracknHits", "Number of Hits per EMTF Track", 5, 0, 5);
  for (int bin = 1; bin <= 5; ++bin) {
    emtfTracknHits->setBinLabel(bin, std::to_string(bin - 1), 1);
  }

  emtfTrackBX = ibooker.book2D("emtfTrackBX", "EMTF Track Bunch Crossing", 12, -6, 6, 8, -3, 5);
  emtfTrackBX->setAxisTitle("Sector (Endcap)", 1);
  for (int i = 0; i < 6; ++i) {
    emtfTrackBX->setBinLabel(i + 1, std::to_string(6 - i) + " (-)", 1);
    emtfTrackBX->setBinLabel(12 - i, std::to_string(6 - i) + " (+)", 1);
  }
  emtfTrackBX->setAxisTitle("Track BX", 2);
  for (int bin = 1, i = -3; bin <= 8; ++bin, ++i) {
    emtfTrackBX->setBinLabel(bin, std::to_string(i), 2);
  }

  emtfTrackPt = ibooker.book1D("emtfTrackPt", "EMTF Track p_{T}", 256, 1, 257);
  emtfTrackPt->setAxisTitle("Track p_{T} [GeV]", 1);

  emtfTrackEta = ibooker.book1D("emtfTrackEta", "EMTF Track #eta", 100, -2.5, 2.5);
  emtfTrackEta->setAxisTitle("Track #eta", 1);

  emtfTrackPhi = ibooker.book1D("emtfTrackPhi", "EMTF Track #phi", 126, -3.15, 3.15);
  emtfTrackPhi->setAxisTitle("Track #phi", 1);

  emtfTrackPhiHighQuality = ibooker.book1D("emtfTrackPhiHighQuality", "EMTF High Quality #phi", 126, -3.15, 3.15);
  emtfTrackPhiHighQuality->setAxisTitle("Track #phi (High Quality)", 1);

  emtfTrackOccupancy = ibooker.book2D("emtfTrackOccupancy", "EMTF Track Occupancy", 100, -2.5, 2.5, 126, -3.15, 3.15);
  emtfTrackOccupancy->setAxisTitle("#eta", 1);
  emtfTrackOccupancy->setAxisTitle("#phi", 2);

  emtfTrackMode = ibooker.book1D("emtfTrackMode", "EMTF Track Mode", 16, 0, 16);
  emtfTrackMode->setAxisTitle("Mode", 1);

  emtfTrackQuality = ibooker.book1D("emtfTrackQuality", "EMTF Track Quality", 16, 0, 16);
  emtfTrackQuality->setAxisTitle("Quality", 1);

  emtfTrackQualityVsMode = ibooker.book2D("emtfTrackQualityVsMode", "EMTF Track Quality vs Mode", 16, 0, 16, 16, 0, 16);
  emtfTrackQualityVsMode->setAxisTitle("Mode", 1);
  emtfTrackQualityVsMode->setAxisTitle("Quality", 2);

  for (int bin = 1; bin <= 16; ++bin) {
    emtfTrackMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQuality->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQualityVsMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQualityVsMode->setBinLabel(bin, std::to_string(bin - 1), 2);
  }

  // Regional Muon Candidate Monitor Elements
  ibooker.setCurrentFolder(monitorDir + "/MuonCand");

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

  // DAQ Output
  edm::Handle<l1t::EMTFDaqOutCollection> DaqOutCollection;
  e.getByToken(daqToken, DaqOutCollection);

  for (std::vector<l1t::EMTFDaqOut>::const_iterator DaqOut = DaqOutCollection->begin(); DaqOut != DaqOutCollection->end(); ++DaqOut) {
    const l1t::emtf::MECollection* MECollection = DaqOut->PtrMECollection();
    for (std::vector<l1t::emtf::ME>::const_iterator ME = MECollection->begin(); ME != MECollection->end(); ++ME) {
      if (ME->SE()) emtfErrors->Fill(1);
      if (ME->SM()) emtfErrors->Fill(2);
      if (ME->BXE()) emtfErrors->Fill(3);
      if (ME->AF()) emtfErrors->Fill(4);
    }

    const l1t::emtf::EventHeader* EventHeader = DaqOut->PtrEventHeader();
    if (!EventHeader->Rdy()) emtfErrors->Fill(5);
  }

  // Hits (LCTs)
  edm::Handle<l1t::EMTFHit2016Collection> HitCollection;
  e.getByToken(hitToken, HitCollection);

  for (std::vector<l1t::EMTFHit2016>::const_iterator Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
    int endcap = Hit->Endcap();
    int sector = Hit->Sector();
    int station = Hit->Station();
    int ring = Hit->Ring();
    int cscid = Hit->CSC_ID();
    int chamber = Hit->Chamber();
    int strip = Hit->Strip();
    int wire = Hit->Wire();

    int hist_index = 0;
    int cscid_offset = (sector - 1) * 9;

    // The following logic determines the index of the monitor element
    // to which a hit belongs, exploiting the symmetry of the endcaps.
    if (station == 1) {
      if (ring == 1 || ring == 4) {
        hist_index = 8;
      } else if (ring == 2) {
        hist_index = 7;
      } else if (ring == 3) {
        hist_index = 6;
      }
    } else if (ring == 1) {
      if (station == 2) {
        hist_index = 5;
      } else if (station == 3) {
        hist_index = 3;
      } else if (station == 4) {
        hist_index = 1;
      }
    } else if (ring == 2) {
      if (station == 2) {
        hist_index = 4;
      } else if (station == 3) {
        hist_index = 2;
      } else if (station == 4) {
        hist_index = 0;
      }
    }

    if (endcap > 0) hist_index = 17 - hist_index;

    emtfHitBX->Fill(Hit->BX(), hist_index);

    emtfHitStrip[hist_index]->Fill(strip);
    emtfHitWire[hist_index]->Fill(wire);

    emtfChamberStrip[hist_index]->Fill(chamber, strip);
    emtfChamberWire[hist_index]->Fill(chamber, wire);

    if (Hit->Subsector() == 1) {
      emtfHitOccupancy->Fill(cscid + cscid_offset, endcap * (station - 0.5));
    } else {
      emtfHitOccupancy->Fill(cscid + cscid_offset, endcap * (station + 0.5));
    }
  }

  // Tracks
  edm::Handle<l1t::EMTFTrack2016Collection> TrackCollection;
  e.getByToken(trackToken, TrackCollection);

  int nTracks = TrackCollection->size();

  if (nTracks <= 10) {
    emtfnTracks->Fill(nTracks);
  } else {
    emtfnTracks->Fill(10);
  }

  for (std::vector<l1t::EMTFTrack2016>::const_iterator Track = TrackCollection->begin(); Track != TrackCollection->end(); ++Track) {
    int endcap = Track->Endcap();
    int sector = Track->Sector();
    float eta = Track->Eta();
    float phi_glob_rad = Track->Phi_glob_rad();
    int mode = Track->Mode();
    int quality = Track->Quality();

    emtfTracknHits->Fill(Track->NumHits());
    emtfTrackBX->Fill(endcap * (sector - 0.5), Track->BX());
    emtfTrackPt->Fill(Track->Pt());
    emtfTrackEta->Fill(eta);
    emtfTrackPhi->Fill(phi_glob_rad);
    emtfTrackOccupancy->Fill(eta, phi_glob_rad);
    emtfTrackMode->Fill(mode);
    emtfTrackQuality->Fill(quality);
    emtfTrackQualityVsMode->Fill(mode, quality);
    if (mode == 15) emtfTrackPhiHighQuality->Fill(phi_glob_rad);
   }

  // Regional Muon Candidates
  edm::Handle<l1t::RegionalMuonCandBxCollection> MuonBxCollection;
  e.getByToken(muonToken, MuonBxCollection);

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

