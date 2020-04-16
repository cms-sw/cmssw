#include <string>
#include <vector>
#include <iostream>
#include <map>

#include "DQM/L1TMonitor/interface/L1TStage2EMTF.h"

L1TStage2EMTF::L1TStage2EMTF(const edm::ParameterSet& ps)
    : daqToken(consumes<l1t::EMTFDaqOutCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      hitToken(consumes<l1t::EMTFHitCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      trackToken(consumes<l1t::EMTFTrackCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      muonToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  // Monitor Dir
  ibooker.setCurrentFolder(monitorDir);

  const std::array<std::string, 6> binNamesErrors{
      {"Corruptions", "Synch. Err.", "Synch. Mod.", "BX Mismatch", "Time Misalign", "FMM != Ready"}};

  // DAQ Output Monitor Elements
  emtfErrors = ibooker.book1D("emtfErrors", "EMTF Errors", 6, 0, 6);
  emtfErrors->setAxisTitle("Error Type (Corruptions Not Implemented)", 1);
  emtfErrors->setAxisTitle("Number of Errors", 2);
  for (unsigned int bin = 0; bin < binNamesErrors.size(); ++bin) {
    emtfErrors->setBinLabel(bin + 1, binNamesErrors[bin], 1);
  }

  // CSC LCT Monitor Elements
  int nChambs, nWires, nStrips;  // Number of chambers, wiregroups, and halfstrips in each station/ring pair
  std::string name, label;
  const std::array<std::string, 10> suffix_name{{"42", "41", "32", "31", "22", "21", "13", "12", "11b", "11a"}};
  const std::array<std::string, 10> suffix_label{
      {"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1b", "1/1a"}};
  const std::array<std::string, 12> binNames{
      {"ME-N", "ME-4", "ME-3", "ME-2", "ME-1b", "ME-1a", "ME+1a", "ME+1b", "ME+2", "ME+3", "ME+4", "ME+N"}};

  cscLCTBX = ibooker.book2D("cscLCTBX", "CSC LCT BX", 7, -3, 4, 20, 0, 20);
  cscLCTBX->setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    cscLCTBX->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 10; ++ybin) {
    cscLCTBX->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    cscLCTBX->setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }

  cscLCTOccupancy = ibooker.book2D("cscLCTOccupancy", "CSC Chamber Occupancy", 54, 1, 55, 12, -6, 6);
  cscLCTOccupancy->setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int xbin = 1; xbin < 7; ++xbin) {
    cscLCTOccupancy->setBinLabel(xbin * 9 - 8, std::to_string(xbin), 1);
  }
  for (unsigned int ybin = 0; ybin < binNames.size(); ++ybin) {
    cscLCTOccupancy->setBinLabel(ybin + 1, binNames[ybin], 2);
  }

  //cscOccupancy designed to match the cscDQM plot
  cscDQMOccupancy = ibooker.book2D("cscDQMOccupancy", "CSC Chamber Occupancy", 42, 1, 43, 20, 0, 20);
  cscDQMOccupancy->setAxisTitle("10#circ Chamber (N=neighbor)", 1);
  int count = 0;
  for (int xbin = 1; xbin < 43; ++xbin) {
    cscDQMOccupancy->setBinLabel(xbin, std::to_string(xbin - count), 1);
    if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
      ++xbin;
      ++count;
      cscDQMOccupancy->setBinLabel(xbin, "N", 1);
    }
  }
  for (int ybin = 1; ybin <= 10; ++ybin) {
    cscDQMOccupancy->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    cscDQMOccupancy->setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
  cscDQMOccupancy->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing

  mpcLinkErrors = ibooker.book2D("mpcLinkErrors", "MPC Link Errors", 54, 1, 55, 12, -6, 6);
  mpcLinkErrors->setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int xbin = 1; xbin < 7; ++xbin) {
    mpcLinkErrors->setBinLabel(xbin * 9 - 8, std::to_string(xbin), 1);
  }
  for (unsigned int ybin = 0; ybin < binNames.size(); ++ybin) {
    mpcLinkErrors->setBinLabel(ybin + 1, binNames[ybin], 2);
  }

  mpcLinkGood = ibooker.book2D("mpcLinkGood", "MPC Good Links", 54, 1, 55, 12, -6, 6);
  mpcLinkGood->setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int xbin = 1; xbin < 7; ++xbin) {
    mpcLinkGood->setBinLabel(xbin * 9 - 8, std::to_string(xbin), 1);
  }
  for (unsigned int ybin = 0; ybin < binNames.size(); ++ybin) {
    mpcLinkGood->setBinLabel(ybin + 1, binNames[ybin], 2);
  }

  // RPC Monitor Elements
  const std::array<std::string, 6> rpc_name{{"43", "42", "33", "32", "22", "12"}};
  const std::array<std::string, 6> rpc_label{{"4/3", "4/2", "3/3", "3/2", "2/2", "1/2"}};

  rpcHitBX = ibooker.book2D("rpcHitBX", "RPC Hit BX", 7, -3, 4, 12, 0, 12);
  rpcHitBX->setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    rpcHitBX->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 6; ++ybin) {
    rpcHitBX->setBinLabel(ybin, "RE-" + rpc_label[ybin - 1], 2);
    rpcHitBX->setBinLabel(13 - ybin, "RE+" + rpc_label[ybin - 1], 2);
  }

  rpcHitOccupancy = ibooker.book2D("rpcHitOccupancy", "RPC Chamber Occupancy", 42, 1, 43, 12, 0, 12);
  rpcHitOccupancy->setAxisTitle("Sector (N=neighbor)", 1);
  for (int bin = 1; bin < 7; ++bin) {
    rpcHitOccupancy->setBinLabel(bin * 7 - 6, std::to_string(bin), 1);
    rpcHitOccupancy->setBinLabel(bin * 7, "N", 1);
    rpcHitOccupancy->setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
    rpcHitOccupancy->setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1], 2);
  }
  rpcHitOccupancy->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing

  // Track Monitor Elements
  emtfnTracks = ibooker.book1D("emtfnTracks", "Number of EMTF Tracks per Event", 11, 0, 11);
  for (int xbin = 1; xbin <= 10; ++xbin) {
    emtfnTracks->setBinLabel(xbin, std::to_string(xbin - 1), 1);
  }
  emtfnTracks->setBinLabel(11, "Overflow", 1);

  emtfTracknHits = ibooker.book1D("emtfTracknHits", "Number of Hits per EMTF Track", 5, 0, 5);
  for (int xbin = 1; xbin <= 5; ++xbin) {
    emtfTracknHits->setBinLabel(xbin, std::to_string(xbin - 1), 1);
  }

  emtfTrackBX = ibooker.book2D("emtfTrackBX", "EMTF Track Bunch Crossing", 12, -6, 6, 7, -3, 4);
  emtfTrackBX->setAxisTitle("Sector (Endcap)", 1);
  for (int xbin = 0; xbin < 6; ++xbin) {
    emtfTrackBX->setBinLabel(xbin + 1, std::to_string(6 - xbin) + " (-)", 1);
    emtfTrackBX->setBinLabel(12 - xbin, std::to_string(6 - xbin) + " (+)", 1);
  }
  emtfTrackBX->setAxisTitle("Track BX", 2);
  for (int ybin = 1, i = -3; ybin <= 7; ++ybin, ++i) {
    emtfTrackBX->setBinLabel(ybin, std::to_string(i), 2);
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

  emtfTrackMode = ibooker.book1D("emtfTrackMode", "EMTF Track Mode", 16, 0, 16);
  emtfTrackMode->setAxisTitle("Mode", 1);

  emtfTrackQuality = ibooker.book1D("emtfTrackQuality", "EMTF Track Quality", 16, 0, 16);
  emtfTrackQuality->setAxisTitle("Quality", 1);

  emtfTrackQualityVsMode = ibooker.book2D("emtfTrackQualityVsMode", "EMTF Track Quality vs Mode", 16, 0, 16, 16, 0, 16);
  emtfTrackQualityVsMode->setAxisTitle("Mode", 1);
  emtfTrackQualityVsMode->setAxisTitle("Quality", 2);

  RPCvsEMTFTrackMode = ibooker.book2D("RPCvsEMTFTrackMode", "RPC Mode vs EMTF TrackMode", 16, 0, 16, 16, 0, 16);
  RPCvsEMTFTrackMode->setAxisTitle("EMTF Mode", 1);
  RPCvsEMTFTrackMode->setAxisTitle("RPC Mode", 2);

  for (int bin = 1; bin <= 16; ++bin) {
    emtfTrackMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQuality->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQualityVsMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    emtfTrackQualityVsMode->setBinLabel(bin, std::to_string(bin - 1), 2);
    RPCvsEMTFTrackMode->setBinLabel(bin, std::to_string(bin - 1), 1);
    RPCvsEMTFTrackMode->setBinLabel(bin, std::to_string(bin - 1), 2);
  }

  //Chad Freer May 8 2018 (Selected Tracks)
  ibooker.setCurrentFolder(monitorDir + "/SelectedTracks");

  //Chad Freer May 8 2018 (High Quality Track Plots)
  emtfTrackPtHighQuality = ibooker.book1D("emtfTrackPtHighQuality", "EMTF High Quality Track p_{T}", 256, 1, 257);
  emtfTrackPtHighQuality->setAxisTitle("Track p_{T} [GeV] (Quality #geq 12)", 1);

  emtfTrackEtaHighQuality = ibooker.book1D("emtfTrackEtaHighQuality", "EMTF High Quality Track #eta", 100, -2.5, 2.5);
  emtfTrackEtaHighQuality->setAxisTitle("Track #eta (Quality #geq 12)", 1);

  emtfTrackPhiHighQuality = ibooker.book1D("emtfTrackPhiHighQuality", "EMTF High Quality #phi", 126, -3.15, 3.15);
  emtfTrackPhiHighQuality->setAxisTitle("Track #phi (Quality #geq 12)", 1);

  emtfTrackOccupancyHighQuality = ibooker.book2D(
      "emtfTrackOccupancyHighQuality", "EMTF High Quality Track Occupancy", 100, -2.5, 2.5, 126, -3.15, 3.15);
  emtfTrackOccupancyHighQuality->setAxisTitle("#eta", 1);
  emtfTrackOccupancyHighQuality->setAxisTitle("#phi", 2);

  //Chad Freer may 8 2018 (High Quality and High PT [22 GeV] Track Plots)
  emtfTrackPtHighQualityHighPT =
      ibooker.book1D("emtfTrackPtHighQualityHighPT", "EMTF High Quality High PT Track p_{T}", 256, 1, 257);
  emtfTrackPtHighQualityHighPT->setAxisTitle("Track p_{T} [GeV] (Quality #geq 12 and pT>22)", 1);

  emtfTrackEtaHighQualityHighPT =
      ibooker.book1D("emtfTrackEtaHighQualityHighPT", "EMTF High Quality High PT Track #eta", 100, -2.5, 2.5);
  emtfTrackEtaHighQualityHighPT->setAxisTitle("Track #eta (Quality #geq 12 and pT>22)", 1);

  emtfTrackPhiHighQualityHighPT =
      ibooker.book1D("emtfTrackPhiHighQualityHighPT", "EMTF High Quality High PT #phi", 126, -3.15, 3.15);
  emtfTrackPhiHighQualityHighPT->setAxisTitle("Track #phi (Quality #geq 12 and pT>22)", 1);

  emtfTrackOccupancyHighQualityHighPT = ibooker.book2D("emtfTrackOccupancyHighQualityHighPT",
                                                       "EMTF High Quality High PT Track Occupancy",
                                                       100,
                                                       -2.5,
                                                       2.5,
                                                       126,
                                                       -3.15,
                                                       3.15);
  emtfTrackOccupancyHighQualityHighPT->setAxisTitle("#eta", 1);
  emtfTrackOccupancyHighQualityHighPT->setAxisTitle("#phi", 2);
  //Chad Freer May 8 2018 (END new plots)

  // CSC Input
  ibooker.setCurrentFolder(monitorDir + "/CSCInput");

  for (int hist = 0, i = 0; hist < 20; ++hist, i = hist % 10) {
    if (hist < 10) {
      name = "MENeg" + suffix_name[i];
      label = "ME-" + suffix_label[i];
    } else {
      name = "MEPos" + suffix_name[9 - i];
      label = "ME+" + suffix_label[9 - i];
    }

    if (hist < 6) {
      nChambs = (i % 2) ? 18 : 36;
    } else if (hist > 13) {
      nChambs = (i % 2) ? 36 : 18;
    } else {
      nChambs = 36;
    }

    const std::array<int, 10> wiregroups{{64, 96, 64, 96, 64, 112, 32, 64, 48, 48}};
    const std::array<int, 10> halfstrips{{160, 160, 160, 160, 160, 160, 128, 160, 128, 96}};

    if (hist < 10) {
      nWires = wiregroups[hist];
      nStrips = halfstrips[hist];
    } else {
      nWires = wiregroups[19 - hist];
      nStrips = halfstrips[19 - hist];
    }

    cscLCTStrip[hist] = ibooker.book1D("cscLCTStrip" + name, "CSC Halfstrip " + label, nStrips, 0, nStrips);
    cscLCTStrip[hist]->setAxisTitle("Cathode Halfstrip, " + label, 1);

    cscLCTWire[hist] = ibooker.book1D("cscLCTWire" + name, "CSC Wiregroup " + label, nWires, 0, nWires);
    cscLCTWire[hist]->setAxisTitle("Anode Wiregroup, " + label, 1);

    cscChamberStrip[hist] = ibooker.book2D(
        "cscChamberStrip" + name, "CSC Halfstrip " + label, nChambs, 1, 1 + nChambs, nStrips, 0, nStrips);
    cscChamberStrip[hist]->setAxisTitle("Chamber, " + label, 1);
    cscChamberStrip[hist]->setAxisTitle("Cathode Halfstrip", 2);

    cscChamberWire[hist] =
        ibooker.book2D("cscChamberWire" + name, "CSC Wiregroup " + label, nChambs, 1, 1 + nChambs, nWires, 0, nWires);
    cscChamberWire[hist]->setAxisTitle("Chamber, " + label, 1);
    cscChamberWire[hist]->setAxisTitle("Anode Wiregroup", 2);

    for (int bin = 1; bin <= nChambs; ++bin) {
      cscChamberStrip[hist]->setBinLabel(bin, std::to_string(bin), 1);
      cscChamberWire[hist]->setBinLabel(bin, std::to_string(bin), 1);
    }
  }

  // RPC Input
  ibooker.setCurrentFolder(monitorDir + "/RPCInput");

  for (int hist = 0, i = 0; hist < 12; ++hist, i = hist % 6) {
    if (hist < 6) {
      name = "RENeg" + rpc_name[i];
      label = "RE-" + rpc_label[i];
    } else {
      name = "REPos" + rpc_name[5 - i];
      label = "RE+" + rpc_label[5 - i];
    }
    rpcHitPhi[hist] = ibooker.book1D("rpcHitPhi" + name, "RPC Hit Phi " + label, 1250, 0, 1250);
    rpcHitPhi[hist]->setAxisTitle("#phi", 1);
    rpcHitTheta[hist] = ibooker.book1D("rpcHitTheta" + name, "RPC Hit Theta " + label, 32, 0, 32);
    rpcHitTheta[hist]->setAxisTitle("#theta", 1);
    rpcChamberPhi[hist] = ibooker.book2D("rpcChamberPhi" + name, "RPC Chamber Phi " + label, 36, 1, 37, 1250, 0, 1250);
    rpcChamberPhi[hist]->setAxisTitle("Chamber", 1);
    rpcChamberPhi[hist]->setAxisTitle("#phi", 2);
    rpcChamberTheta[hist] =
        ibooker.book2D("rpcChamberTheta" + name, "RPC Chamber Theta " + label, 36, 1, 37, 32, 0, 32);
    rpcChamberTheta[hist]->setAxisTitle("Chamber", 1);
    rpcChamberTheta[hist]->setAxisTitle("#theta", 2);
    for (int xbin = 1; xbin < 37; ++xbin) {
      rpcChamberPhi[hist]->setBinLabel(xbin, std::to_string(xbin), 1);
      rpcChamberTheta[hist]->setBinLabel(xbin, std::to_string(xbin), 1);
    }
  }

  // CSC LCT and RPC Hit Timing
  ibooker.setCurrentFolder(monitorDir + "/Timing");

  cscTimingTot = ibooker.book2D("cscTimingTotal", "CSC Total BX ", 42, 1, 43, 20, 0, 20);
  cscTimingTot->setAxisTitle("10#circ Chamber (N=neighbor)", 1);

  rpcHitTimingTot = ibooker.book2D("rpcHitTimingTot", "RPC Chamber Occupancy ", 42, 1, 43, 12, 0, 12);
  rpcHitTimingTot->setAxisTitle("Sector (N=neighbor)", 1);
  const std::array<std::string, 5> nameBX{{"BXNeg1", "BXPos1", "BXNeg2", "BXPos2", "BX0"}};
  const std::array<std::string, 5> labelBX{{"BX -1", "BX +1", "BX -2", "BX +2", "BX 0"}};

  for (int hist = 0; hist < 5; ++hist) {
    count = 0;
    cscLCTTiming[hist] =
        ibooker.book2D("cscLCTTiming" + nameBX[hist], "CSC Chamber Occupancy " + labelBX[hist], 42, 1, 43, 20, 0, 20);
    cscLCTTiming[hist]->setAxisTitle("10#circ Chamber", 1);

    for (int xbin = 1; xbin < 43; ++xbin) {
      cscLCTTiming[hist]->setBinLabel(xbin, std::to_string(xbin - count), 1);
      if (hist == 0)
        cscTimingTot->setBinLabel(xbin, std::to_string(xbin - count), 1);  //only fill once in the loop
      if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
        ++xbin;
        ++count;
        cscLCTTiming[hist]->setBinLabel(xbin, "N", 1);
        if (hist == 0)
          cscTimingTot->setBinLabel(xbin, "N", 1);
      }
    }

    for (int ybin = 1; ybin <= 10; ++ybin) {
      cscLCTTiming[hist]->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
      cscLCTTiming[hist]->setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
      if (hist == 0)
        cscTimingTot->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
      if (hist == 0)
        cscTimingTot->setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    }
    if (hist == 0)
      cscTimingTot->getTH2F()->GetXaxis()->SetCanExtend(false);      // Needed to stop multi-thread summing
    cscLCTTiming[hist]->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing

    rpcHitTiming[hist] =
        ibooker.book2D("rpcHitTiming" + nameBX[hist], "RPC Chamber Occupancy " + labelBX[hist], 42, 1, 43, 12, 0, 12);
    rpcHitTiming[hist]->setAxisTitle("Sector (N=neighbor)", 1);
    for (int bin = 1; bin < 7; ++bin) {
      rpcHitTiming[hist]->setBinLabel(bin * 7 - 6, std::to_string(bin), 1);
      rpcHitTiming[hist]->setBinLabel(bin * 7, "N", 1);
      rpcHitTiming[hist]->setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
      rpcHitTiming[hist]->setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1], 2);
    }
    rpcHitTiming[hist]->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing
    if (hist == 0) {
      for (int bin = 1; bin < 7; ++bin) {
        rpcHitTimingTot->setBinLabel(bin * 7 - 6, std::to_string(bin), 1);
        rpcHitTimingTot->setBinLabel(bin * 7, "N", 1);
        rpcHitTimingTot->setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
        rpcHitTimingTot->setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1], 2);
      }
      rpcHitTimingTot->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing
    }
    //if (hist == 4) continue; // Don't book for BX = 0

    count = 0;
    cscLCTTimingFrac[hist] = ibooker.book2D(
        "cscLCTTimingFrac" + nameBX[hist], "CSC Chamber Occupancy " + labelBX[hist], 42, 1, 43, 20, 0, 20);
    cscLCTTimingFrac[hist]->setAxisTitle("10#circ Chambers", 1);
    for (int xbin = 1; xbin < 43; ++xbin) {
      cscLCTTimingFrac[hist]->setBinLabel(xbin, std::to_string(xbin - count), 1);
      if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
        ++xbin;
        ++count;
        cscLCTTimingFrac[hist]->setBinLabel(xbin, "N", 1);
      }
    }
    for (int ybin = 1; ybin <= 10; ++ybin) {
      cscLCTTimingFrac[hist]->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
      cscLCTTimingFrac[hist]->setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    }
    cscLCTTimingFrac[hist]->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing

    rpcHitTimingFrac[hist] = ibooker.book2D(
        "rpcHitTimingFrac" + nameBX[hist], "RPC Chamber Fraction in " + labelBX[hist], 42, 1, 43, 12, 0, 12);
    rpcHitTimingFrac[hist]->setAxisTitle("Sector (N=neighbor)", 1);
    for (int bin = 1; bin < 7; ++bin) {
      rpcHitTimingFrac[hist]->setBinLabel(bin * 7 - 6, std::to_string(bin), 1);
      rpcHitTimingFrac[hist]->setBinLabel(bin * 7, "N", 1);
      rpcHitTimingFrac[hist]->setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
      rpcHitTimingFrac[hist]->setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1], 2);
    }
  }

  rpcHitTimingInTrack =
      ibooker.book2D("rpcHitTimingInTrack", "RPC Hit Timing (matched to track in BX 0)", 7, -3, 4, 12, 0, 12);
  rpcHitTimingInTrack->setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    rpcHitTimingInTrack->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 6; ++ybin) {
    rpcHitTimingInTrack->setBinLabel(ybin, "RE-" + rpc_label[ybin - 1], 2);
    rpcHitTimingInTrack->setBinLabel(13 - ybin, "RE+" + rpc_label[ybin - 1], 2);
  }

  const std::array<std::string, 3> nameNumStation{{"4Station", "3Station", "2Station"}};
  const std::array<std::string, 3> labelNumStation{{"4 Station Track", "3 Station Track", "2 Station Track"}};

  for (int hist = 0; hist < 3; ++hist) {
    emtfTrackBXVsCSCLCT[hist] = ibooker.book2D("emtfTrackBXVsCSCLCT" + nameNumStation[hist],
                                               "EMTF " + labelNumStation[hist] + " BX vs CSC LCT BX",
                                               7,
                                               -3,
                                               4,
                                               7,
                                               -3,
                                               4);
    emtfTrackBXVsCSCLCT[hist]->setAxisTitle("LCT BX", 1);
    emtfTrackBXVsCSCLCT[hist]->setAxisTitle("Track BX", 2);
    for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
      emtfTrackBXVsCSCLCT[hist]->setBinLabel(bin, std::to_string(bin_label), 1);
      emtfTrackBXVsCSCLCT[hist]->setBinLabel(bin, std::to_string(bin_label), 2);
    }
    emtfTrackBXVsRPCHit[hist] = ibooker.book2D("emtfTrackBXVsRPCHit" + nameNumStation[hist],
                                               "EMTF " + labelNumStation[hist] + " BX vs RPC Hit BX",
                                               7,
                                               -3,
                                               4,
                                               7,
                                               -3,
                                               4);
    emtfTrackBXVsRPCHit[hist]->setAxisTitle("Hit BX", 1);
    emtfTrackBXVsRPCHit[hist]->setAxisTitle("Track BX", 2);
    for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
      emtfTrackBXVsRPCHit[hist]->setBinLabel(bin, std::to_string(bin_label), 1);
      emtfTrackBXVsRPCHit[hist]->setBinLabel(bin, std::to_string(bin_label), 2);
    }
  }

  // Muon Cand
  ibooker.setCurrentFolder(monitorDir + "/MuonCand");

  // Regional Muon Candidate Monitor Elements
  emtfMuonBX = ibooker.book1D("emtfMuonBX", "EMTF Muon Cand BX", 7, -3, 4);
  emtfMuonBX->setAxisTitle("BX", 1);
  for (int xbin = 1, bin_label = -3; xbin <= 7; ++xbin, ++bin_label) {
    emtfMuonBX->setBinLabel(xbin, std::to_string(bin_label), 1);
  }

  emtfMuonhwPt = ibooker.book1D("emtfMuonhwPt", "EMTF Muon Cand p_{T}", 512, 0, 512);
  emtfMuonhwPt->setAxisTitle("Hardware p_{T}", 1);

  emtfMuonhwEta = ibooker.book1D("emtfMuonhwEta", "EMTF Muon Cand #eta", 460, -230, 230);
  emtfMuonhwEta->setAxisTitle("Hardware #eta", 1);

  emtfMuonhwPhi = ibooker.book1D("emtfMuonhwPhi", "EMTF Muon Cand #phi", 145, -40, 105);
  emtfMuonhwPhi->setAxisTitle("Hardware #phi", 1);

  emtfMuonhwQual = ibooker.book1D("emtfMuonhwQual", "EMTF Muon Cand Quality", 16, 0, 16);
  emtfMuonhwQual->setAxisTitle("Quality", 1);
  for (int xbin = 1; xbin <= 16; ++xbin) {
    emtfMuonhwQual->setBinLabel(xbin, std::to_string(xbin - 1), 1);
  }
}

// CSCOccupancy chamber mapping for neighbor inclusive plots
int chamber_bin(int station, int ring, int chamber) {
  int chamber_bin_index = 0;
  if (station > 1 && (ring % 2) == 1) {
    chamber_bin_index = (chamber * 2) + ((chamber + 1) / 3);
  } else {
    chamber_bin_index = chamber + ((chamber + 3) / 6);
  }
  return chamber_bin_index;
};

void L1TStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose)
    edm::LogInfo("L1TStage2EMTF") << "L1TStage2EMTF: analyze..." << std::endl;

  // DAQ Output
  edm::Handle<l1t::EMTFDaqOutCollection> DaqOutCollection;
  e.getByToken(daqToken, DaqOutCollection);

  for (auto DaqOut = DaqOutCollection->begin(); DaqOut != DaqOutCollection->end(); ++DaqOut) {
    const l1t::emtf::MECollection* MECollection = DaqOut->PtrMECollection();
    for (auto ME = MECollection->begin(); ME != MECollection->end(); ++ME) {
      if (ME->SE())
        emtfErrors->Fill(1);
      if (ME->SM())
        emtfErrors->Fill(2);
      if (ME->BXE())
        emtfErrors->Fill(3);
      if (ME->AF())
        emtfErrors->Fill(4);
    }

    const l1t::emtf::EventHeader* EventHeader = DaqOut->PtrEventHeader();
    if (!EventHeader->Rdy())
      emtfErrors->Fill(5);

    // Fill MPC input link errors
    int offset = (EventHeader->Sector() - 1) * 9;
    int endcap = EventHeader->Endcap();
    l1t::emtf::Counters CO = DaqOut->GetCounters();
    const std::array<std::array<int, 9>, 5> counters{
        {{{CO.ME1a_1(),
           CO.ME1a_2(),
           CO.ME1a_3(),
           CO.ME1a_4(),
           CO.ME1a_5(),
           CO.ME1a_6(),
           CO.ME1a_7(),
           CO.ME1a_8(),
           CO.ME1a_9()}},
         {{CO.ME1b_1(),
           CO.ME1b_2(),
           CO.ME1b_3(),
           CO.ME1b_4(),
           CO.ME1b_5(),
           CO.ME1b_6(),
           CO.ME1b_7(),
           CO.ME1b_8(),
           CO.ME1b_9()}},
         {{CO.ME2_1(), CO.ME2_2(), CO.ME2_3(), CO.ME2_4(), CO.ME2_5(), CO.ME2_6(), CO.ME2_7(), CO.ME2_8(), CO.ME2_9()}},
         {{CO.ME3_1(), CO.ME3_2(), CO.ME3_3(), CO.ME3_4(), CO.ME3_5(), CO.ME3_6(), CO.ME3_7(), CO.ME3_8(), CO.ME3_9()}},
         {{CO.ME4_1(), CO.ME4_2(), CO.ME4_3(), CO.ME4_4(), CO.ME4_5(), CO.ME4_6(), CO.ME4_7(), CO.ME4_8(), CO.ME4_9()}}}};
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 9; j++) {
        if (counters.at(i).at(j) != 0)
          mpcLinkErrors->Fill(j + 1 + offset, endcap * (i + 0.5), counters.at(i).at(j));
        else
          mpcLinkGood->Fill(j + 1 + offset, endcap * (i + 0.5));
      }
    }
    if (CO.ME1n_3() == 1)
      mpcLinkErrors->Fill(1 + offset, endcap * 5.5);
    if (CO.ME1n_6() == 1)
      mpcLinkErrors->Fill(2 + offset, endcap * 5.5);
    if (CO.ME1n_9() == 1)
      mpcLinkErrors->Fill(3 + offset, endcap * 5.5);
    if (CO.ME2n_3() == 1)
      mpcLinkErrors->Fill(4 + offset, endcap * 5.5);
    if (CO.ME2n_9() == 1)
      mpcLinkErrors->Fill(5 + offset, endcap * 5.5);
    if (CO.ME3n_3() == 1)
      mpcLinkErrors->Fill(6 + offset, endcap * 5.5);
    if (CO.ME3n_9() == 1)
      mpcLinkErrors->Fill(7 + offset, endcap * 5.5);
    if (CO.ME4n_3() == 1)
      mpcLinkErrors->Fill(8 + offset, endcap * 5.5);
    if (CO.ME4n_9() == 1)
      mpcLinkErrors->Fill(9 + offset, endcap * 5.5);
    if (CO.ME1n_3() == 0)
      mpcLinkGood->Fill(1 + offset, endcap * 5.5);
    if (CO.ME1n_6() == 0)
      mpcLinkGood->Fill(2 + offset, endcap * 5.5);
    if (CO.ME1n_9() == 0)
      mpcLinkGood->Fill(3 + offset, endcap * 5.5);
    if (CO.ME2n_3() == 0)
      mpcLinkGood->Fill(4 + offset, endcap * 5.5);
    if (CO.ME2n_9() == 0)
      mpcLinkGood->Fill(5 + offset, endcap * 5.5);
    if (CO.ME3n_3() == 0)
      mpcLinkGood->Fill(6 + offset, endcap * 5.5);
    if (CO.ME3n_9() == 0)
      mpcLinkGood->Fill(7 + offset, endcap * 5.5);
    if (CO.ME4n_3() == 0)
      mpcLinkGood->Fill(8 + offset, endcap * 5.5);
    if (CO.ME4n_9() == 0)
      mpcLinkGood->Fill(9 + offset, endcap * 5.5);
  }

  // Hits (CSC LCTs and RPC hits)
  edm::Handle<l1t::EMTFHitCollection> HitCollection;
  e.getByToken(hitToken, HitCollection);

  // Maps CSC station and ring to the monitor element index and uses symmetry of the endcaps
  const std::map<std::pair<int, int>, int> histIndexCSC = {{{1, 4}, 9},
                                                           {{1, 1}, 8},
                                                           {{1, 2}, 7},
                                                           {{1, 3}, 6},
                                                           {{2, 1}, 5},
                                                           {{2, 2}, 4},
                                                           {{3, 1}, 3},
                                                           {{3, 2}, 2},
                                                           {{4, 1}, 1},
                                                           {{4, 2}, 0}};

  // Maps RPC staion and ring to the monitor element index and uses symmetry of the endcaps
  const std::map<std::pair<int, int>, int> histIndexRPC = {
      {{4, 3}, 0}, {{4, 2}, 1}, {{3, 3}, 2}, {{3, 2}, 3}, {{2, 2}, 4}, {{1, 2}, 5}};

  for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
    int endcap = Hit->Endcap();
    int sector = Hit->Sector();
    int station = Hit->Station();
    int ring = Hit->Ring();
    int cscid = Hit->CSC_ID();
    int chamber = Hit->Chamber();
    int strip = Hit->Strip();
    int wire = Hit->Wire();
    int cscid_offset = (sector - 1) * 9;

    int hist_index = 0;
    if (ring == 4 && strip >= 128)
      strip -= 128;

    if (Hit->Is_CSC() == true) {
      hist_index = histIndexCSC.at({station, ring});
      if (endcap > 0)
        hist_index = 19 - hist_index;
      cscLCTBX->Fill(Hit->BX(), hist_index);
      float evt_wgt = (Hit->Station() > 1 && Hit->Ring() == 1) ? 0.5 : 1.0;
      if (Hit->Neighbor() == false) {
        //Map for cscDQMOccupancy plot
        cscDQMOccupancy->Fill(chamber_bin(station, ring, chamber), hist_index, evt_wgt);
        if (station > 1 && (ring % 2) == 1) {
          cscDQMOccupancy->Fill(chamber_bin(station, ring, chamber) - 1, hist_index, evt_wgt);
        }
        cscLCTStrip[hist_index]->Fill(strip);
        cscLCTWire[hist_index]->Fill(wire);
        cscChamberStrip[hist_index]->Fill(chamber, strip);
        cscChamberWire[hist_index]->Fill(chamber, wire);
        if (Hit->Subsector() == 1) {
          cscLCTOccupancy->Fill(cscid + cscid_offset, endcap * (station - 0.5));
        } else {
          cscLCTOccupancy->Fill(cscid + cscid_offset, endcap * (station + 0.5));
        }
      } else {
        // Map neighbor chambers to "fake" CSC IDs: 1/3 --> 1, 1/6 --> 2, 1/9 --> 3, 2/3 --> 4, 2/9 --> 5, etc.
        int cscid_n = (station == 1 ? (cscid / 3) : (station * 2) + ((cscid - 3) / 6));
        cscLCTOccupancy->Fill(cscid_n + cscid_offset, endcap * 5.5);
      }
      if (Hit->Neighbor() == true) {
        cscDQMOccupancy->Fill(sector * 7 - 4, hist_index, evt_wgt);
      }
    }

    if (Hit->Is_RPC() == true) {
      hist_index = histIndexRPC.at({station, ring});
      if (endcap > 0)
        hist_index = 11 - hist_index;

      rpcHitBX->Fill(Hit->BX(), hist_index);

      if (Hit->Neighbor() == false) {
        rpcHitPhi[hist_index]->Fill(Hit->Phi_fp() / 4);
        rpcHitTheta[hist_index]->Fill(Hit->Theta_fp() / 4);
        rpcChamberPhi[hist_index]->Fill(chamber, Hit->Phi_fp() / 4);
        rpcChamberTheta[hist_index]->Fill(chamber, Hit->Theta_fp() / 4);
        rpcHitOccupancy->Fill((Hit->Sector_RPC() - 1) * 7 + Hit->Subsector(), hist_index + 0.5);
      } else if (Hit->Neighbor() == true) {
        rpcHitOccupancy->Fill((Hit->Sector_RPC() - 1) * 7 + 7, hist_index + 0.5);
      }
    }
  }

  // Tracks
  edm::Handle<l1t::EMTFTrackCollection> TrackCollection;
  e.getByToken(trackToken, TrackCollection);

  int nTracks = TrackCollection->size();

  emtfnTracks->Fill(std::min(nTracks, emtfnTracks->getTH1F()->GetNbinsX() - 1));

  for (auto Track = TrackCollection->begin(); Track != TrackCollection->end(); ++Track) {
    int endcap = Track->Endcap();
    int sector = Track->Sector();
    float eta = Track->Eta();
    float phi_glob_rad = Track->Phi_glob() * M_PI / 180.;
    int mode = Track->Mode();
    int quality = Track->GMT_quality();
    int numHits = Track->NumHits();
    int modeNeighbor = Track->Mode_neighbor();
    int modeRPC = Track->Mode_RPC();
    int singleMuQuality = 12;
    int singleMuPT = 22;

    // Only plot if there are <= 1 neighbor hits in the track to avoid spikes at sector boundaries
    if (modeNeighbor >= 2 && modeNeighbor != 4 && modeNeighbor != 8)
      continue;

    emtfTracknHits->Fill(numHits);
    emtfTrackBX->Fill(endcap * (sector - 0.5), Track->BX());
    emtfTrackPt->Fill(Track->Pt());
    emtfTrackEta->Fill(eta);

    emtfTrackOccupancy->Fill(eta, phi_glob_rad);
    emtfTrackMode->Fill(mode);
    emtfTrackQuality->Fill(quality);
    emtfTrackQualityVsMode->Fill(mode, quality);
    RPCvsEMTFTrackMode->Fill(mode, modeRPC);
    emtfTrackPhi->Fill(phi_glob_rad);

    if (quality >= singleMuQuality) {
      emtfTrackPtHighQuality->Fill(Track->Pt());
      emtfTrackEtaHighQuality->Fill(eta);
      emtfTrackPhiHighQuality->Fill(phi_glob_rad);
      emtfTrackOccupancyHighQuality->Fill(eta, phi_glob_rad);
      if (Track->Pt() >= singleMuPT) {
        emtfTrackPtHighQualityHighPT->Fill(Track->Pt());
        emtfTrackEtaHighQualityHighPT->Fill(eta);
        emtfTrackPhiHighQualityHighPT->Fill(phi_glob_rad);
        emtfTrackOccupancyHighQualityHighPT->Fill(eta, phi_glob_rad);
      }
    }

    ////////////////////////////////////////////////////
    ///  Begin block for CSC LCT and RPC hit timing  ///
    ////////////////////////////////////////////////////
    {
      // LCT and RPC Timing
      if (numHits < 2 || numHits > 4)
        continue;
      l1t::EMTFHitCollection tmp_hits = Track->Hits();
      int numHitsInTrack_BX0 = 0;
      unsigned int hist_index2 = 4 - numHits;

      for (const auto& iTrkHit : Track->Hits()) {
        if (iTrkHit.Is_CSC() == true) {
          emtfTrackBXVsCSCLCT[hist_index2]->Fill(iTrkHit.BX(), Track->BX());
        } else if (iTrkHit.Is_RPC() == true) {
          emtfTrackBXVsRPCHit[hist_index2]->Fill(iTrkHit.BX(), Track->BX());
        }
      }

      // Select well-timed tracks: >= 3 hits, with <= 1 in BX != 0
      if (numHits < 3)
        continue;
      for (const auto& jTrkHit : Track->Hits()) {
        if (jTrkHit.BX() == 0)
          numHitsInTrack_BX0++;
      }
      if (numHitsInTrack_BX0 < numHits - 1)
        continue;

      for (const auto& TrkHit : Track->Hits()) {
        int trackHitBX = TrkHit.BX();
        //int cscid        = TrkHit.CSC_ID();
        int ring = TrkHit.Ring();
        int station = TrkHit.Station();
        int sector = TrkHit.Sector();
        int subsector = TrkHit.Subsector();
        //int cscid_offset = (sector - 1) * 9;//no longer needed after new time plots (maybe useful for future plots)
        int neighbor = TrkHit.Neighbor();
        int endcap = TrkHit.Endcap();
        int chamber = TrkHit.Chamber();

        int hist_index = 0;
        float evt_wgt = (TrkHit.Station() > 1 && TrkHit.Ring() == 1) ? 0.5 : 1.0;
        // Maps CSC BX from -2 to 2 to monitor element cscLCTTIming
        const std::map<int, int> histIndexBX = {{0, 4}, {-1, 0}, {1, 1}, {-2, 2}, {2, 3}};
        if (std::abs(trackHitBX) > 2)
          continue;  // Should never happen, but just to be safe ...

        if (TrkHit.Is_CSC() == true) {
          hist_index = histIndexCSC.at({station, ring});
          if (endcap > 0)
            hist_index = 19 - hist_index;
          if (neighbor == false) {
            cscLCTTiming[histIndexBX.at(trackHitBX)]->Fill(chamber_bin(station, ring, chamber), hist_index, evt_wgt);
            cscTimingTot->Fill(chamber_bin(station, ring, chamber), hist_index, evt_wgt);
            if (station > 1 && (ring % 2) == 1) {
              cscLCTTiming[histIndexBX.at(trackHitBX)]->Fill(
                  chamber_bin(station, ring, chamber) - 1, hist_index, evt_wgt);
              cscTimingTot->Fill(chamber_bin(station, ring, chamber) - 1, hist_index, evt_wgt);
            }
          } else {
            // Map neighbor chambers to "fake" CSC IDs: 1/3 --> 1, 1/6 --> 2, 1/9 --> 3, 2/3 --> 4, 2/9 --> 5, etc.
            //int cscid_n = (station == 1 ? (cscid / 3) : (station * 2) + ((cscid - 3) / 6) );
            cscLCTTiming[histIndexBX.at(trackHitBX)]->Fill(sector * 7 - 4, hist_index, evt_wgt);
            cscTimingTot->Fill(sector * 7 - 4, hist_index, evt_wgt);
          }

          // Fill RPC timing with matched CSC LCTs
          if (trackHitBX == 0 && ring == 2) {
            for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
              if (Hit->Is_RPC() == false || neighbor == true)
                continue;
              if (std::abs(Track->Eta() - Hit->Eta()) > 0.1)
                continue;
              if (Hit->Endcap() != endcap || Hit->Station() != station || Hit->Chamber() != chamber)
                continue;
              if (std::abs(Hit->BX()) > 2)
                continue;

              hist_index = histIndexRPC.at({Hit->Station(), Hit->Ring()});
              if (Hit->Endcap() > 0)
                hist_index = 11 - hist_index;
              rpcHitTimingInTrack->Fill(Hit->BX(), hist_index + 0.5);
              rpcHitTiming[histIndexBX.at(Hit->BX())]->Fill((Hit->Sector_RPC() - 1) * 7 + Hit->Subsector(),
                                                            hist_index + 0.5);
              rpcHitTimingTot->Fill((Hit->Sector_RPC() - 1) * 7 + Hit->Subsector(), hist_index + 0.5);
            }  // End loop: for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit)
          }    // End conditional: if (trackHitBX == 0 && ring == 2)
        }      // End conditional: if (TrkHit.Is_CSC() == true)

        // Maps RPC station and ring to monitor element index
        const std::map<std::pair<int, int>, int> histIndexRPC = {
            {{4, 3}, 0}, {{4, 2}, 1}, {{3, 3}, 2}, {{3, 2}, 3}, {{2, 2}, 4}, {{1, 2}, 5}};

        if (TrkHit.Is_RPC() == true && neighbor == false) {
          hist_index = histIndexRPC.at({station, ring});
          if (endcap > 0)
            hist_index = 11 - hist_index;

          rpcHitTimingInTrack->Fill(trackHitBX, hist_index + 0.5);
          rpcHitTiming[histIndexBX.at(trackHitBX)]->Fill((TrkHit.Sector_RPC() - 1) * 7 + subsector, hist_index + 0.5);
          rpcHitTimingTot->Fill((TrkHit.Sector_RPC() - 1) * 7 + subsector, hist_index + 0.5);
        }  // End conditional: if (TrkHit.Is_RPC() == true && neighbor == false)
        if (TrkHit.Is_RPC() == true && neighbor == true) {
          hist_index = histIndexRPC.at({station, ring});
          if (endcap > 0)
            hist_index = 11 - hist_index;
          rpcHitTiming[histIndexBX.at(trackHitBX)]->Fill((TrkHit.Sector_RPC() - 1) * 7, hist_index + 0.5);
        }
      }  // End loop: for (int iHit = 0; iHit < numHits; ++iHit)
    }
    //////////////////////////////////////////////////
    ///  End block for CSC LCT and RPC hit timing  ///
    //////////////////////////////////////////////////

  }  // End loop: for (auto Track = TrackCollection->begin(); Track != TrackCollection->end(); ++Track)

  // CSC LCT and RPC Hit Timing Efficieny
  for (int hist_index = 0; hist_index < 5; ++hist_index) {
    cscLCTTimingFrac[hist_index]->getTH2F()->Divide(cscLCTTiming[hist_index]->getTH2F(), cscTimingTot->getTH2F());
    rpcHitTimingFrac[hist_index]->getTH2F()->Divide(rpcHitTiming[hist_index]->getTH2F(), rpcHitTimingTot->getTH2F());
  }

  // Regional Muon Candidates
  edm::Handle<l1t::RegionalMuonCandBxCollection> MuonBxCollection;
  e.getByToken(muonToken, MuonBxCollection);

  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX);
         Muon != MuonBxCollection->end(itBX);
         ++Muon) {
      emtfMuonBX->Fill(itBX);
      emtfMuonhwPt->Fill(Muon->hwPt());
      emtfMuonhwEta->Fill(Muon->hwEta());
      emtfMuonhwPhi->Fill(Muon->hwPhi());
      emtfMuonhwQual->Fill(Muon->hwQual());
    }
  }
}
