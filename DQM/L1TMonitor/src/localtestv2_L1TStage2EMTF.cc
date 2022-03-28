#include <string>
#include <vector>
#include <iostream>
#include <map>

#include "DQM/L1TMonitor/interface/localtestv2_L1TStage2EMTF.h"

locv2_L1TStage2EMTF::locv2_L1TStage2EMTF(const edm::ParameterSet& ps)
    : daqToken(consumes<l1t::EMTFDaqOutCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      hitToken(consumes<l1t::EMTFHitCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      trackToken(consumes<l1t::EMTFTrackCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      muonToken(consumes<l1t::RegionalMuonCandBxCollection>(ps.getParameter<edm::InputTag>("emtfSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

locv2_L1TStage2EMTF::~locv2_L1TStage2EMTF() {}

void locv2_L1TStage2EMTF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
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

  // GEM Monitor Elements // Add GEM Oct 27 2020
  // test plots Nov 01 2020
  hitType = ibooker.book1D("hitType", "Hit Type", 4, 0.5,4.5);
  hitType->setBinLabel(1, "CSC", 1);
  hitType->setBinLabel(2, "RPC", 1);
  hitType->setBinLabel(3, "GEM", 1);
  hitType->setBinLabel(4, "Tot", 1);

  hitTypeBX = ibooker.book2D("hitTypeBX", "Hit Type BX", 4, 0.5, 4.5, 7, -3, 4);
  hitTypeBX->setBinLabel(1, "CSC", 1);
  hitTypeBX->setBinLabel(2, "RPC", 1);
  hitTypeBX->setBinLabel(3, "GEM", 1);
  hitTypeBX->setBinLabel(4, "Tot", 1);
  for (int ybin = 1; ybin < 8; ybin++ ) hitTypeBX->setBinLabel(ybin, std::to_string(ybin-4), 2);

  hitTypeSector = ibooker.book2D("hitTypeSector", "Hit Type Sector", 4, 0.5, 4.5, 6, 1, 7);
  hitTypeSector->setBinLabel(1, "CSC", 1);
  hitTypeSector->setBinLabel(2, "RPC", 1);
  hitTypeSector->setBinLabel(3, "GEM", 1);
  hitTypeSector->setBinLabel(4, "Tot", 1);
  for (int ybin = 1; ybin < 7; ybin++ ) hitTypeSector->setBinLabel(ybin, std::to_string(ybin), 2);

  // hitTypeNumber for gem cosmic debug
  hitTypeNumber = ibooker.book2D("hitTypeNumber", "Hit Type Number per Event", 4, 0.5, 4.5, 1000, 0.1, 1000.1);
  hitTypeNumber->setBinLabel(1, "CSC", 1);
  hitTypeNumber->setBinLabel(2, "RPC", 1);
  hitTypeNumber->setBinLabel(3, "GEM", 1);
  hitTypeNumber->setBinLabel(4, "Tot", 1);
//  for (int ybin = 1; ybin < 101; ybin++ ) hitTypeNumber->setBinLabel(ybin, std::to_string(ybin), 2);

  hitTypeNumSecGE11Pos = ibooker.book2D("hitTypeNumSecGE11Pos", "GE11 and ME11 Pos Hit Number per Event", 12, 0.5, 12.5, 1000, 0.1, 1000.1);
  hitTypeNumSecGE11Pos->setAxisTitle("hits in GE+1/1 sectors", 1);
  for (int xbin = 1; xbin < 7; xbin++) {
    hitTypeNumSecGE11Pos->setBinLabel(xbin*2-1, "GE+"+std::to_string(xbin), 1);
    hitTypeNumSecGE11Pos->setBinLabel(xbin*2,   "ME+"+std::to_string(xbin), 1);
  }

  hitTypeNumSecGE11Neg = ibooker.book2D("hitTypeNumSecGE11Neg", "GE11 and ME11 Neg Hit Number per Event", 12, 0.5, 12.5, 1000, 0.1, 1000.1);
  hitTypeNumSecGE11Neg->setAxisTitle("hits in GE-1/1 sectors", 1);
  for (int xbin = 1; xbin < 7; xbin++) {
    hitTypeNumSecGE11Neg->setBinLabel(xbin*2-1, "GE-"+std::to_string(xbin), 1);
    hitTypeNumSecGE11Neg->setBinLabel(xbin*2,   "ME-"+std::to_string(xbin), 1);
  }

  hitCoincideME11 = ibooker.book2D("hitCoincideME11", "Hits accompanied ME11 hits", 6, 1, 7, 15, 1, 16);
  hitCoincideME11->setAxisTitle("ME-1/1 sector", 1);
  hitCoincideME11->setAxisTitle("hit type and number", 2);
  hitCoincideME11->setBinLabel(1, "other ME11", 2);
  hitCoincideME11->setBinLabel(2, "other ME21", 2);
  hitCoincideME11->setBinLabel(3, "other ME12", 2);
  hitCoincideME11->setBinLabel(4, "GE11 all", 2);
  hitCoincideME11->setBinLabel(5, "GE11 same chamber", 2);
  hitCoincideME11->setBinLabel(6, "GE11 same sec = 0", 2);
  hitCoincideME11->setBinLabel(7, "GE11 same sec = 1", 2);
  hitCoincideME11->setBinLabel(8, "GE11 same sec 2-10", 2);
  hitCoincideME11->setBinLabel(9, "GE11 same sec > 10", 2);
  hitCoincideME11->setBinLabel(10, "GE part 0134 = 0", 2);
  hitCoincideME11->setBinLabel(11, "GE part 0134 = 1", 2);
  hitCoincideME11->setBinLabel(12, "GE part 0134 = 2", 2);
  hitCoincideME11->setBinLabel(13, "GE part 0134 3-5", 2);
  hitCoincideME11->setBinLabel(14, "GE part 0134 6-9", 2);
  hitCoincideME11->setBinLabel(15, "GE part 0134 > 9", 2);

  hitCoincideGE11 = ibooker.book2D("hitCoincideGE11", "Hits accompanied GE11 hits", 2, 2, 6, 9, 1, 10);
  hitCoincideGE11->setAxisTitle("GE-1/1 sector", 1);
  hitCoincideGE11->setAxisTitle("hit type and number", 2);
  hitCoincideGE11->setBinLabel(1, "any ME = 0", 2);
  hitCoincideGE11->setBinLabel(2, "any ME = 1", 2);
  hitCoincideGE11->setBinLabel(3, "any ME > 1", 2);
  hitCoincideGE11->setBinLabel(4, "sec ME = 0", 2);
  hitCoincideGE11->setBinLabel(5, "sec ME = 1", 2);
  hitCoincideGE11->setBinLabel(6, "sec ME > 1", 2);
  hitCoincideGE11->setBinLabel(7, "sec ME11 = 0", 2);
  hitCoincideGE11->setBinLabel(8, "sec ME11 = 1", 2);
  hitCoincideGE11->setBinLabel(9, "sec ME11 > 1", 2);

  SameSectorTimingCSCGEM = ibooker.book2D("SameSectorTimingCSCGEM", "Hits in same sector CSC vs GEM BX", 7, -3, 4, 7, -3, 4);
  SameSectorTimingCSCGEM->setAxisTitle("CSC BX", 1);
  SameSectorTimingCSCGEM->setAxisTitle("GEM BX", 2);
  for (int xybin = 1, xybin_label = -3; xybin <= 7; ++xybin, ++xybin_label) {
    SameSectorTimingCSCGEM->setBinLabel(xybin, std::to_string(xybin_label), 1);
    SameSectorTimingCSCGEM->setBinLabel(xybin, std::to_string(xybin_label), 2);
  }

  SameSectorChamberCSCGEM = ibooker.book2D("SameSectorChamberCSCGEM", "Hits in same sector CSC vs GEM BX", 36, 1, 37, 36, 1, 37);
  SameSectorChamberCSCGEM->setAxisTitle("CSC chamber", 1);
  SameSectorChamberCSCGEM->setAxisTitle("GEM chamber", 2);

  SameSectorGEMminusCSCfpThetaPhi = ibooker.book2D("SameSectorGEMminusCSCfpThetaPhi", "same sector hits (GEM - CSC) full precision theta phi", 256, -128, 128, 500, -5000, 5000);
  SameSectorGEMminusCSCfpThetaPhi->setAxisTitle("GEM - CSC fp_theta", 1);
  SameSectorGEMminusCSCfpThetaPhi->setAxisTitle("GEM - CSC fp_phi", 2);

  SameSectorGEMPadPartition = ibooker.book2D("SameSectorGEMPadPartition", "Pad and Partition of GEM hits in same sector as CSC", 9, 0, 9, 512, 0, 511);
  SameSectorGEMPadPartition->setAxisTitle("GEM Partition", 1); 
  SameSectorGEMPadPartition->setAxisTitle("GEM Pad", 2); 

  gemNegBXAddress0134 = ibooker.book2D("gemNegBXAddress0134", "GEM Hit BX vs Address in Partitions 0134", 9,-4,5, 2048, 0, 2048);
  gemNegBXAddress0134->setAxisTitle("BX", 1);
  gemNegBXAddress0134->setAxisTitle("GEM-1/1 Address", 2);

  gemHitBX = ibooker.book2D("gemHitBX", "GEM Hit BX", 7, -3, 4, 2, 0, 2);
  gemHitBX->setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    gemHitBX->setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  gemHitBX->setBinLabel(1, "GE-1/1", 2);
  gemHitBX->setBinLabel(2, "GE+1/1", 2);

  gemHitOccupancy = ibooker.book2D("gemHitOccupancy", "GEM Chamber Occupancy", 42, 1, 43, 2, 0, 2);
  gemHitOccupancy->setAxisTitle("10#circ Chambers (N=neighbor)", 1);
  count = 0;
  for (int xbin = 1; xbin < 43; ++xbin) {
    gemHitOccupancy->setBinLabel(xbin, std::to_string(xbin - count), 1);
    if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
      ++xbin;
      ++count;
      gemHitOccupancy->setBinLabel(xbin, "N", 1);
    }
  }  

//  for (int bin = 1; bin < 7; ++bin) {
//    gemHitOccupancy->setBinLabel(bin * 7 -6, std::to_string(bin), 1);
//    gemHitOccupancy->setBinLabel(bin * 7 , "N", 1);
//  }
  gemHitOccupancy->setBinLabel(1, "GE-1/1", 2);
  gemHitOccupancy->setBinLabel(2, "GE+1/1", 2);
  gemHitOccupancy->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing

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

  // GEM Input Nov 03 2020
  ibooker.setCurrentFolder(monitorDir + "/GEMInput");
  for (int hist = 0; hist < 2; hist++) {
    if (hist == 0) {
      name = "GENeg11";
      label = "GE-1/1";
    }
    if (hist == 1) {
      name = "GEPos11";
      label = "GE+1/1";
    }
    nChambs = 36;
    nStrips = 192; //use nStrips for number of pads
//    nStrips = 512; // GEM cosmics debug 2021.05.21, 9 bits of pad max 512
    gemChamberPad[hist] = ibooker.book2D("gemChamberPad" + name, "GEM Chamber Pad " + label, nChambs, 1, 1 + nChambs, nStrips, 0, nStrips); // pads 0-191
    gemChamberPad[hist]->setAxisTitle("Chamber, " + label, 1);
    gemChamberPad[hist]->setAxisTitle("Pad", 2);
    gemChamberPartition[hist] = ibooker.book2D("gemChamberPartition" + name, "GEM Chamber Partition " + label, nChambs, 1, 1 + nChambs, 9, 0, 9); // partitions 1-8 // 0-8 for debug
    gemChamberPartition[hist]->setAxisTitle("Chamber, " + label, 1);
    gemChamberPartition[hist]->setAxisTitle("Partition", 2);
    for (int bin = 1; bin <= nChambs; ++bin) {
      gemChamberPad[hist]->setBinLabel(bin, std::to_string(bin), 1);
      gemChamberPartition[hist]->setBinLabel(bin, std::to_string(bin), 1);
    }
    gemChamberAddress[hist] = ibooker.book2D("gemChamberAddress" + name, "GEM Chamber Address " + label, nChambs, 1, 1 + nChambs, 2048, 0, 2048); // 11 bits 0-2047
    gemChamberAddress[hist]->setAxisTitle("Chamber, " + label, 1);
    gemChamberAddress[hist]->setAxisTitle("Address", 2);
    for (int bin = 1; bin <= nChambs; ++bin) {
      gemChamberAddress[hist]->setBinLabel(bin, std::to_string(bin), 1);
    }  

    gemChamberVFAT[hist] = ibooker.book2D("gemChamberVFAT" + name, "GEM Chamber VFAT " + label, 42, 1, 43, 24, 0, 24); // 8* (0-2) phi part + (0-7) eta part
    gemChamberVFAT[hist]->setAxisTitle("Chamber, " + label, 1);
    gemChamberVFAT[hist]->setAxisTitle("8*phi + eta", 2);
    count = 0;
    for (int xbin = 1; xbin < 43; ++xbin) {
      gemChamberVFAT[hist]->setBinLabel(xbin, std::to_string(xbin - count), 1);
      if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
        ++xbin;
        ++count;
        gemChamberVFAT[hist]->setBinLabel(xbin, "N", 1);
      }
    }
    for (int bin = 1; bin <= 24; ++bin) {
      gemChamberVFAT[hist]->setBinLabel(bin, std::to_string(bin-1), 2);
    }

    gemBXVFAT[hist] = ibooker.book2D("gemBXVFAT" + name, "GEM BX vs VFAT" + label, 7, -3, 4, 24, 0, 24);
    gemBXVFAT[hist]->setAxisTitle("BX", 1);
    gemBXVFAT[hist]->setAxisTitle("8*phi + eta, " + label, 2);
    for (int bin = 1; bin <= 24; ++bin) {
      gemBXVFAT[hist]->setBinLabel(bin, std::to_string(bin-1), 2);
    }

    gemBXVFATC91011[hist] = ibooker.book2D("gemBXVFATC91011" + name, "GEM BX vs VFAT in chamber 9 10 11" + label, 7, -3, 4, 24, 0, 24);
    gemBXVFATC91011[hist]->setAxisTitle("BX", 1);
    gemBXVFATC91011[hist]->setAxisTitle("8*phi + eta, " + label, 2);
    for (int bin = 1; bin <= 24; ++bin) {
      gemBXVFATC91011[hist]->setBinLabel(bin, std::to_string(bin-1), 2);
    }

    std::string layer = "Layer1";
    if (hist == 1) layer = "Layer2";
    gemBXVFATC9[hist] = ibooker.book2D("gemBXVFATChamber9" + layer, "GEM BX vs VFAT in GE-1/1 Chamber 9 " + layer, 7, -3, 4, 24, 0, 24);
    gemBXVFATC9[hist]->setAxisTitle("BX", 1);
    gemBXVFATC9[hist]->setAxisTitle("8*phi + eta, " + label, 2);
    gemBXVFATC10[hist] = ibooker.book2D("gemBXVFATChamber10" + layer, "GEM BX vs VFAT in GE-1/1 Chamber 10 " + layer, 7, -3, 4, 24, 0, 24);
    gemBXVFATC10[hist]->setAxisTitle("BX", 1);
    gemBXVFATC10[hist]->setAxisTitle("8*phi + eta, " + label, 2);
    gemBXVFATC11[hist] = ibooker.book2D("gemBXVFATChamber11" + layer, "GEM BX vs VFAT in GE-1/1 Chamber 11 " + layer, 7, -3, 4, 24, 0, 24);
    gemBXVFATC11[hist]->setAxisTitle("BX", 1);
    gemBXVFATC11[hist]->setAxisTitle("8*phi + eta, " + label, 2);
    for (int bin = 1; bin <= 24; ++bin) {
      gemBXVFATC9[hist]->setBinLabel(bin, std::to_string(bin-1), 2);
      gemBXVFATC10[hist]->setBinLabel(bin, std::to_string(bin-1), 2);
      gemBXVFATC11[hist]->setBinLabel(bin, std::to_string(bin-1), 2);
    }

    for (int ch = 0; ch < 36; ch++){
      for (int lyr = 0; lyr < 2; lyr++){
          gemBXVFATPerChamber[ch][hist][lyr] = ibooker.book2D("gemBXVFATPerChamber_" + std::to_string(ch) + "_" + std::to_string(hist) + "_" + std::to_string(lyr), "GEM BX vs VFAT in Chamber " + std::to_string(ch+1) + " " + label + " Layer " + std::to_string(lyr), 7, -3, 4, 48, 0, 48);
          gemBXVFATPerChamber[ch][hist][lyr]->setAxisTitle("BX", 1);
          gemBXVFATPerChamber[ch][hist][lyr]->setAxisTitle("8*phi + eta, " + label, 2);

          gemBXVFATPerChamberCoincidence[ch][hist][lyr] = ibooker.book2D("gemBXVFATPerChamberCoincidence_" + std::to_string(ch) + "_" + std::to_string(hist) + "_" + std::to_string(lyr), "GEM BX vs VFAT with CSC coincidence in Chamber " + std::to_string(ch+1) + " " + label + " Layer " + std::to_string(lyr), 7, -3, 4, 48, 0, 48);
          gemBXVFATPerChamberCoincidence[ch][hist][lyr]->setAxisTitle("BX", 1);
          gemBXVFATPerChamberCoincidence[ch][hist][lyr]->setAxisTitle("8*phi + eta, " + label, 2);
          for (int bin = 1; bin <= 48; ++bin) {
            gemBXVFATPerChamberCoincidence[ch][hist][lyr]->setBinLabel(bin, std::to_string(bin-1), 2);
            gemBXVFATPerChamber[ch][hist][lyr]->setBinLabel(bin, std::to_string(bin-1), 2);
          }
      }
    }


    gemChamberVFATBX[hist] = ibooker.book2D("gemChamberVFATBX" + name, "GEM Chamber vs VFAT * BX" + label, 42, 1, 43, 210, 0, 210); // 8* (0-2) phi part + (0-7) eta part
    gemChamberVFATBX[hist]->setAxisTitle("Chamber, " + label, 1);
    count = 0;
    for (int xbin = 1; xbin < 43; ++xbin) {
      gemChamberVFATBX[hist]->setBinLabel(xbin, std::to_string(xbin - count), 1);
      if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
        ++xbin;
        ++count;
        gemChamberVFATBX[hist]->setBinLabel(xbin, "N", 1);
      }
    }
    for (int bxbin = -3; bxbin <= 3; ++bxbin) {
      gemChamberVFATBX[hist]->setBinLabel( (bxbin+3) * 30 + 1, "BX" + std::to_string(bxbin) + ",VFAT0", 2);
      gemChamberVFATBX[hist]->setBinLabel( (bxbin+3) * 30 + 24 , "BX" + std::to_string(bxbin) + ",VFAT23", 2);
    }
  }


  nStrips = 512;
  gemPosCham32S5NPadPart = ibooker.book2D("gemPosCham32S5NPadPart", "GEM+1/1 hits in Chamber 32 and Neighbor of Sector 5", 18, 0, 18, nStrips, 0, nStrips);
  gemPosCham02S6NPadPart = ibooker.book2D("gemPosCham02S6NPadPart", "GEM+1/1 hits in Chamber 02 and Neighbor of Sector 6", 18, 0, 18, nStrips, 0, nStrips);
  gemNegCham08S1NPadPart = ibooker.book2D("gemNegCham08S1NPadPart", "GEM-1/1 hits in Chamber 08 and Neighbor of Sector 1", 18, 0, 18, nStrips, 0, nStrips);
  gemNegCham20S3NPadPart = ibooker.book2D("gemNegCham20S3NPadPart", "GEM-1/1 hits in Chamber 20 and Neighbor of Sector 3", 18, 0, 18, nStrips, 0, nStrips);
  gemPosCham32S5NPadPart->setAxisTitle("Chamber, Partition", 1); 
  gemPosCham02S6NPadPart->setAxisTitle("Chamber, Partition", 1); 
  gemNegCham08S1NPadPart->setAxisTitle("Chamber, Partition", 1); 
  gemNegCham20S3NPadPart->setAxisTitle("Chamber, Partition", 1);
 
  gemPosCham32S5NPadPart->setAxisTitle("Pad", 2);  
  gemPosCham02S6NPadPart->setAxisTitle("Pad", 2); 
  gemNegCham08S1NPadPart->setAxisTitle("Pad", 2); 
  gemNegCham20S3NPadPart->setAxisTitle("Pad", 2); 
  for (int bin = 1; bin <= 9; ++bin) {
    gemPosCham32S5NPadPart->setBinLabel(bin, "C32P"+std::to_string(bin-1), 1); 
    gemPosCham02S6NPadPart->setBinLabel(bin, "C02P"+std::to_string(bin-1), 1);
    gemNegCham08S1NPadPart->setBinLabel(bin, "C08P"+std::to_string(bin-1), 1); 
    gemNegCham20S3NPadPart->setBinLabel(bin, "C20P"+std::to_string(bin-1), 1); 
    gemPosCham32S5NPadPart->setBinLabel(bin+9, "NS5P"+std::to_string(bin-1), 1);
    gemPosCham02S6NPadPart->setBinLabel(bin+9, "NS6P"+std::to_string(bin-1), 1);
    gemNegCham08S1NPadPart->setBinLabel(bin+9, "NS1P"+std::to_string(bin-1), 1);
    gemNegCham20S3NPadPart->setBinLabel(bin+9, "NS3P"+std::to_string(bin-1), 1);
  } 

  gemNegCham12PadPart = ibooker.book2D("gemNegCham12PadPart", "GEM+1/1 hits in Chamber 12", 9, 0, 9, nStrips, 0, nStrips);
  gemNegCham12PadPart->setAxisTitle("Partition", 1);
  gemNegCham12PadPart->setAxisTitle("Pad", 2);


  // CSC LCT and RPC Hit Timing
  ibooker.setCurrentFolder(monitorDir + "/Timing");

  cscTimingTot = ibooker.book2D("cscTimingTotal", "CSC Total BX ", 42, 1, 43, 20, 0, 20);
  cscTimingTot->setAxisTitle("10#circ Chamber (N=neighbor)", 1);

  rpcHitTimingTot = ibooker.book2D("rpcHitTimingTot", "RPC Chamber Occupancy ", 42, 1, 43, 12, 0, 12);
  rpcHitTimingTot->setAxisTitle("Sector (N=neighbor)", 1);

  gemHitTimingTot = ibooker.book2D("gemHitTimingTot", "GEM Chamber Occupancy ", 42, 1, 43, 2, 0, 2);  // Add GEM Timing Oct 27 2020
  gemHitTimingTot->setAxisTitle("10#circ Chamber (N=neighbor)", 1);
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

    gemHitTiming[hist] =
        ibooker.book2D("gemHitTiming" + nameBX[hist], "GEM Chamber Occupancy " + labelBX[hist], 42, 1, 43, 2, 0, 2);
    gemHitTiming[hist]->setAxisTitle("10#circ Chamber", 1);
    count = 0;
    for (int xbin = 1; xbin < 43; ++xbin) {
      gemHitTiming[hist]->setBinLabel(xbin, std::to_string(xbin - count), 1);
      if (hist == 0)
        gemHitTimingTot->setBinLabel(xbin, std::to_string(xbin - count), 1);  //only fill once in the loop
      if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
        ++xbin;
        ++count;
        gemHitTiming[hist]->setBinLabel(xbin, "N", 1);
        if (hist == 0)
          gemHitTimingTot->setBinLabel(xbin, "N", 1);
      }
    }

    gemHitTiming[hist]->setBinLabel(1, "GE-1/1", 2);
    gemHitTiming[hist]->setBinLabel(2, "GE+1/1", 2);
    if (hist == 0) {
      gemHitTimingTot->setBinLabel(1, "GE-1/1", 2);
      gemHitTimingTot->setBinLabel(2, "GE+1/1", 2);
      gemHitTimingTot->getTH2F()->GetXaxis()->SetCanExtend(false);      // Needed to stop multi-thread summing
    }
    gemHitTiming[hist]->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing


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

    gemHitTimingFrac[hist] = ibooker.book2D(
        "gemHitTimingFrac" + nameBX[hist], "GEM Chamber Occupancy " + labelBX[hist], 42, 1, 43, 2, 0, 2);
    gemHitTimingFrac[hist]->setAxisTitle("10#circ Chambers", 1);
    count = 0;
    for (int xbin = 1; xbin < 43; ++xbin) {
      gemHitTimingFrac[hist]->setBinLabel(xbin, std::to_string(xbin - count), 1);
      if (xbin == 2 || xbin == 9 || xbin == 16 || xbin == 23 || xbin == 30 || xbin == 37) {
        ++xbin;
        ++count;
        gemHitTimingFrac[hist]->setBinLabel(xbin, "N", 1);
      }
    }
    gemHitTimingFrac[hist]->setBinLabel(1, "GE-1/1", 2);
    gemHitTimingFrac[hist]->setBinLabel(2, "GE+1/1", 2);
    gemHitTimingFrac[hist]->getTH2F()->GetXaxis()->SetCanExtend(false);  // Needed to stop multi-thread summing
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
    // Add GEM vs track BX 2020.12.05
    emtfTrackBXVsGEMHit[hist] = ibooker.book2D("emtfTrackBXVsGEMHit" + nameNumStation[hist],
                                             "EMTF " + labelNumStation[hist] + " BX vs GEM Hit BX",
                                             7,
                                             -3,
                                             4,
                                             7,
                                             -3,
                                             4);
    emtfTrackBXVsGEMHit[hist]->setAxisTitle("Hit BX", 1);
    emtfTrackBXVsGEMHit[hist]->setAxisTitle("Track BX", 2);
    for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
      emtfTrackBXVsGEMHit[hist]->setBinLabel(bin, std::to_string(bin_label), 1);
      emtfTrackBXVsGEMHit[hist]->setBinLabel(bin, std::to_string(bin_label), 2);
    }
  } // End loop: for (int hist = 0; hist < 3; ++hist)

  // Add mode vs BXdiff comparison 2020.12.07
  const std::array<std::string, 2> nameGEMStation{{"GENeg11", "GEPos11"}};
  const std::array<std::string, 2> labelGEMStation{{"GE-1/1", "GE+1/1"}};
  const std::array<std::string, 8> nameCSCStation{{"MENeg4", "MENeg3", "MENeg2", "MENeg1", "MEPos1", "MEPos2", "MEPos3", "MEPos4"}};
  const std::array<std::string, 8> labelCSCStation{{"ME-4", "ME-3", "ME-2", "ME-1", "ME+1", "ME+2", "ME+3", "ME+4"}}; 
  const std::array<std::string, 6> nameRPCStation{{"RENeg4", "RENeg3", "RENeg2", "REPos2", "REPos3", "REPos4"}};
  const std::array<std::string, 6> labelRPCStation{{"RE-4", "RE-3", "RE-2", "RE+2", "RE+3", "RE+4"}};  
 
  for (int iGEM = 0; iGEM < 2; iGEM++) {
    emtfTrackModeVsGEMBXDiff[iGEM] = ibooker.book2D("emtfTrackModeVsGEMBXDiff" + nameGEMStation[iGEM], 
                                                    "EMTF Track Mode vs (Track BX - GEM BX) " + labelGEMStation[iGEM],
                                                    9, -4, 5, 16, 0, 16);
    emtfTrackModeVsGEMBXDiff[iGEM]->setAxisTitle("Track BX - GEM BX", 1);
    emtfTrackModeVsGEMBXDiff[iGEM]->setAxisTitle("Track Mode", 2);
  }
  for (int iCSC = 0; iCSC < 8; iCSC++) {
    emtfTrackModeVsCSCBXDiff[iCSC] = ibooker.book2D("emtfTrackModeVsCSCBXDiff" + nameCSCStation[iCSC],
                                                    "EMTF Track Mode vs (Track BX - LCT BX) " + labelCSCStation[iCSC],
                                                    9, -4, 5, 16, 0, 16);
    emtfTrackModeVsCSCBXDiff[iCSC]->setAxisTitle("Track BX - LCT BX", 1);
    emtfTrackModeVsCSCBXDiff[iCSC]->setAxisTitle("Track Mode", 2);
  }
  for (int iRPC = 0; iRPC < 6; iRPC++) {
    emtfTrackModeVsRPCBXDiff[iRPC] = ibooker.book2D("emtfTrackModeVsRPCBXDiff" + nameRPCStation[iRPC],
                                                    "EMTF Track Mode vs (Track BX - RPC BX) " + labelRPCStation[iRPC],
                                                    9, -4, 5, 16, 0, 16);
    emtfTrackModeVsRPCBXDiff[iRPC]->setAxisTitle("Track BX - RPC BX", 1);
    emtfTrackModeVsRPCBXDiff[iRPC]->setAxisTitle("Track Mode", 2);
  }

  // GEM vs CSC 2020.12.06
  ibooker.setCurrentFolder(monitorDir + "/GEMVsCSC");
  for (int hist = 0; hist < 2; hist++){
    gemHitPhi[hist] = ibooker.book2D("gemHitPhi" + nameGEMStation[hist], "GEM Hit Phi " + labelGEMStation[hist], 4921, 0, 4921, 6, 1, 7);
    gemHitTheta[hist] = ibooker.book2D("gemHitTheta" + nameGEMStation[hist], "GEM Hit Theta " + labelGEMStation[hist], 128, 0, 128, 6, 1, 7);
    gemHitVScscLCTPhi[hist] = ibooker.book2D("gemHitVScscLCTPhi" + nameGEMStation[hist], "GEM Hit Phi - CSC LCT Phi " + labelGEMStation[hist], 
                                             1200, -600, 600, 36, 1, 37); // one chamber is 10 degrees, 60 integer phi per degree
    gemHitVScscLCTTheta[hist] = ibooker.book2D("gemHitVScscLCTTheta" + nameGEMStation[hist], "GEM Hit Theta - CSC LCT Theta " + labelGEMStation[hist], 
                                             20, -10, 10, 36, 1, 37); // 0.1 eta is at most 9.5 integer theta (between eta 1.5 and 1.6)
    gemHitVScscLCTBX[hist] = ibooker.book2D("gemHitVScscLCTBX" + nameGEMStation[hist], "GEM Hit BX - CSC LCT BX " + labelGEMStation[hist],
                                            9, -4, 5, 36, 1, 37); 

    gemHitPhi[hist]->setAxisTitle("Integer #phi", 1);
    gemHitTheta[hist]->setAxisTitle("Integer #theta", 1);
    gemHitVScscLCTPhi[hist]->setAxisTitle("Integer #phi", 1);
    gemHitVScscLCTTheta[hist]->setAxisTitle("Integer #theta", 1);
    gemHitVScscLCTBX[hist]->setAxisTitle("GEM BX - CSC BX", 1);

    gemHitPhi[hist]->setAxisTitle("Sector", 2);
    gemHitTheta[hist]->setAxisTitle("Sector", 2);
    gemHitVScscLCTPhi[hist]->setAxisTitle("Chamber", 2);
    gemHitVScscLCTTheta[hist]->setAxisTitle("Chamber", 2);
    gemHitVScscLCTBX[hist]->setAxisTitle("Chamber", 2);
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
int thisv2_chamber_bin(int station, int ring, int chamber) {
  int chamber_bin_index = 0;
  if (station > 1 && (ring % 2) == 1) {
    chamber_bin_index = (chamber * 2) + ((chamber + 1) / 3);
  } else {
    chamber_bin_index = chamber + ((chamber + 3) / 6);
  }
  return chamber_bin_index;
};

void locv2_L1TStage2EMTF::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose)
    edm::LogInfo("L1TStage2EMTF") << "L1TStage2EMTF: analyze..." << std::endl;

  // DAQ Output
//  edm::Handle<l1t::EMTFDaqOutCollection> DaqOutCollection;
//  e.getByToken(daqToken, DaqOutCollection);
//
//  for (auto DaqOut = DaqOutCollection->begin(); DaqOut != DaqOutCollection->end(); ++DaqOut) {
//    const l1t::emtf::MECollection* MECollection = DaqOut->PtrMECollection();
//    for (auto ME = MECollection->begin(); ME != MECollection->end(); ++ME) {
//      if (ME->SE())
//        emtfErrors->Fill(1);
//      if (ME->SM())
//        emtfErrors->Fill(2);
//      if (ME->BXE())
//        emtfErrors->Fill(3);
//      if (ME->AF())
//        emtfErrors->Fill(4);
//    }
//
//    const l1t::emtf::EventHeader* EventHeader = DaqOut->PtrEventHeader();
//    if (!EventHeader->Rdy())
//      emtfErrors->Fill(5);
//
//    // Fill MPC input link errors
//    int offset = (EventHeader->Sector() - 1) * 9;
//    int endcap = EventHeader->Endcap();
//    l1t::emtf::Counters CO = DaqOut->GetCounters();
//    const std::array<std::array<int, 9>, 5> counters{
//        {{{CO.ME1a_1(),
//           CO.ME1a_2(),
//           CO.ME1a_3(),
//           CO.ME1a_4(),
//           CO.ME1a_5(),
//           CO.ME1a_6(),
//           CO.ME1a_7(),
//           CO.ME1a_8(),
//           CO.ME1a_9()}},
//         {{CO.ME1b_1(),
//           CO.ME1b_2(),
//           CO.ME1b_3(),
//           CO.ME1b_4(),
//           CO.ME1b_5(),
//           CO.ME1b_6(),
//           CO.ME1b_7(),
//           CO.ME1b_8(),
//           CO.ME1b_9()}},
//         {{CO.ME2_1(), CO.ME2_2(), CO.ME2_3(), CO.ME2_4(), CO.ME2_5(), CO.ME2_6(), CO.ME2_7(), CO.ME2_8(), CO.ME2_9()}},
//         {{CO.ME3_1(), CO.ME3_2(), CO.ME3_3(), CO.ME3_4(), CO.ME3_5(), CO.ME3_6(), CO.ME3_7(), CO.ME3_8(), CO.ME3_9()}},
//         {{CO.ME4_1(), CO.ME4_2(), CO.ME4_3(), CO.ME4_4(), CO.ME4_5(), CO.ME4_6(), CO.ME4_7(), CO.ME4_8(), CO.ME4_9()}}}};
//    for (int i = 0; i < 5; i++) {
//      for (int j = 0; j < 9; j++) {
//        if (counters.at(i).at(j) != 0)
//          mpcLinkErrors->Fill(j + 1 + offset, endcap * (i + 0.5), counters.at(i).at(j));
//        else
//          mpcLinkGood->Fill(j + 1 + offset, endcap * (i + 0.5));
//      }
//    }
//    if (CO.ME1n_3() == 1)
//      mpcLinkErrors->Fill(1 + offset, endcap * 5.5);
//    if (CO.ME1n_6() == 1)
//      mpcLinkErrors->Fill(2 + offset, endcap * 5.5);
//    if (CO.ME1n_9() == 1)
//      mpcLinkErrors->Fill(3 + offset, endcap * 5.5);
//    if (CO.ME2n_3() == 1)
//      mpcLinkErrors->Fill(4 + offset, endcap * 5.5);
//    if (CO.ME2n_9() == 1)
//      mpcLinkErrors->Fill(5 + offset, endcap * 5.5);
//    if (CO.ME3n_3() == 1)
//      mpcLinkErrors->Fill(6 + offset, endcap * 5.5);
//    if (CO.ME3n_9() == 1)
//      mpcLinkErrors->Fill(7 + offset, endcap * 5.5);
//    if (CO.ME4n_3() == 1)
//      mpcLinkErrors->Fill(8 + offset, endcap * 5.5);
//    if (CO.ME4n_9() == 1)
//      mpcLinkErrors->Fill(9 + offset, endcap * 5.5);
//    if (CO.ME1n_3() == 0)
//      mpcLinkGood->Fill(1 + offset, endcap * 5.5);
//    if (CO.ME1n_6() == 0)
//      mpcLinkGood->Fill(2 + offset, endcap * 5.5);
//    if (CO.ME1n_9() == 0)
//      mpcLinkGood->Fill(3 + offset, endcap * 5.5);
//    if (CO.ME2n_3() == 0)
//      mpcLinkGood->Fill(4 + offset, endcap * 5.5);
//    if (CO.ME2n_9() == 0)
//      mpcLinkGood->Fill(5 + offset, endcap * 5.5);
//    if (CO.ME3n_3() == 0)
//      mpcLinkGood->Fill(6 + offset, endcap * 5.5);
//    if (CO.ME3n_9() == 0)
//      mpcLinkGood->Fill(7 + offset, endcap * 5.5);
//    if (CO.ME4n_3() == 0)
//      mpcLinkGood->Fill(8 + offset, endcap * 5.5);
//    if (CO.ME4n_9() == 0)
//      mpcLinkGood->Fill(9 + offset, endcap * 5.5);
//  }

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

  // Reverse 'rotate by 2' for RPC subsector
  auto get_subsector_rpc_cppf = [](int subsector_rpc) { return ((subsector_rpc + 3) % 6) + 1; };

  // count hit number for debugging
  int CSCtot = 1;
  int RPCtot = 1;
  int GEMtot = 1;
  int CSCGEM = 1;
  std::array<int, 6> GEpos11  {{1, 1, 1, 1, 1, 1}};
  std::array<int, 6> GEneg11  {{1, 1, 1, 1, 1, 1}};
  std::array<int, 6> MEpos11  {{1, 1, 1, 1, 1, 1}};
  std::array<int, 6> MEneg11  {{1, 1, 1, 1, 1, 1}};
  std::array<int, 6> GEneg11p0134  {{1, 1, 1, 1, 1, 1}};
  for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
    if (Hit->Neighbor() == true) continue;
    if      (Hit->Is_RPC() == true) RPCtot++;
    else if (Hit->Is_CSC() == true) {
      CSCtot++;
      if (Hit->Station()==1 and Hit->Ring()==1) {
        if (Hit->Endcap()>0) MEpos11[Hit->Sector()-1]++;
        else                 MEneg11[Hit->Sector()-1]++;
      }
    }
    else if (Hit->Is_GEM() == true) {
      GEMtot++;
      if (Hit->Station()==1 and Hit->Ring()==1) {
        if (Hit->Endcap()>0) GEpos11[Hit->Sector()-1]++;
        else {
          GEneg11[Hit->Sector()-1]++;
          if (Hit->Partition() == 0 or Hit->Partition() == 1 or Hit->Partition() == 3 or Hit->Partition() == 4) GEneg11p0134[Hit->Sector()-1]++;
        }
      }
    }
    if (Hit->Is_CSC() == true and Hit->Is_GEM() == true) CSCGEM++;
  }
  // only look at "good GEM" events
//  if (GEMtot != 2) return;
//
//  for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
//    if (Hit->Neighbor() == true) continue;
//    if (Hit->Is_GEM() == true) {
//      for (auto other_hit = HitCollection->begin(); other_hit != HitCollection->end(); ++other_hit) {
//        if (other_hit->Neighbor() == true) continue;
//        if (other_hit->Is_CSC() == true and other_hit->Endcap() == Hit->Endcap() and other_hit->Station() == 1 and other_hit->Ring()==1 and other_hit->Sector() == Hit->Sector()) {
//          SameSectorTimingCSCGEM->Fill(other_hit->BX(), Hit->BX());
//          SameSectorChamberCSCGEM->Fill(other_hit->Chamber(), Hit->Chamber());
//        }
//      }
//    }
//  }

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
        cscDQMOccupancy->Fill(thisv2_chamber_bin(station, ring, chamber), hist_index, evt_wgt);
        if (station > 1 && (ring % 2) == 1) {
          cscDQMOccupancy->Fill(thisv2_chamber_bin(station, ring, chamber) - 1, hist_index, evt_wgt);
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
        rpcHitOccupancy->Fill((Hit->Sector_RPC() - 1) * 7 + get_subsector_rpc_cppf(Hit->Subsector_RPC()),
                              hist_index + 0.5);
      } else if (Hit->Neighbor() == true) {
        rpcHitOccupancy->Fill((Hit->Sector_RPC() - 1) * 7 + 7, hist_index + 0.5);
      }
    }

    // Add GEM Oct 27 2020
    //test plots Nov 01 2020
    hitType->Fill(4);
    hitTypeBX->Fill(4, Hit->BX());
    hitTypeSector->Fill(4, sector);
    if (Hit->Is_CSC() == true) {
      hitType->Fill(1);
      hitTypeBX->Fill(1, Hit->BX());
      hitTypeSector->Fill(1, sector);
    }
    else if (Hit->Is_RPC() == true) {
      hitType->Fill(2);
      hitTypeBX->Fill(2, Hit->BX());
      hitTypeSector->Fill(2, Hit->Sector_RPC());
    }
    else if (Hit->Is_GEM() == true) {
      hitType->Fill(3);
      hitTypeBX->Fill(3, Hit->BX());
      hitTypeSector->Fill(3, sector);
    }


    if (Hit->Is_GEM() == true) {
      gemHitBX->Fill(Hit->BX(), (endcap > 0) ? 1.5 : 0.5);
      hist_index = (endcap > 0) ? 1 : 0;
      int layer = Hit->Layer();
      int phi_part = Hit->Pad() / 64; // 0-2
      int vfat = phi_part * 8 + Hit->Partition(); // 0-7, 8-15, 16-23
      // if (phi_part > 2) std::cout << "weird case: Pad = " << Hit->Pad() << std::endl;
      // if (phi_part < 0) std::cout << "weird case: Pad = " << Hit->Pad() << std::endl;
      // if (phi_part < 0 or phi_part > 2) continue;
      if (Hit->Neighbor() == false) {
        gemChamberPad[hist_index]->Fill(chamber, Hit->Pad());
        gemChamberPartition[hist_index]->Fill(chamber, Hit->Partition());
        gemChamberAddress[hist_index]->Fill(chamber, Hit->Pad() + Hit->Partition()*192);

        gemChamberVFAT[hist_index]->Fill( thisv2_chamber_bin(1, 1, chamber), vfat);
        gemBXVFAT[hist_index]->Fill( Hit->BX(), vfat);
        if (chamber >= 9 and chamber <= 11) {
          gemBXVFATC91011[hist_index]->Fill( Hit->BX(), vfat);
          if (endcap > 0) continue;
          if (layer != 0 and layer != 1) std::cout << "Weird case: chamber = " << chamber << ", layer = " << layer << std::endl;
          if (chamber == 9) gemBXVFATC9[layer]->Fill( Hit->BX(), vfat);
          if (chamber == 10) gemBXVFATC10[layer]->Fill( Hit->BX(), vfat);
          if (chamber == 11) gemBXVFATC11[layer]->Fill( Hit->BX(), vfat);
        }

        gemBXVFATPerChamber[chamber-1][hist_index][layer]->Fill(Hit->BX(), vfat);

        bool coincideME11 = false;
        for (auto other_hit = HitCollection->begin(); other_hit != HitCollection->end(); ++other_hit) {
          if (other_hit->Neighbor() == true) continue;
          // other CSC hits
          if (other_hit->Is_CSC() == true and other_hit->Endcap() == endcap and other_hit->Chamber() == chamber and other_hit->Station() == 1 and other_hit->Ring()==1) {
            // in ME11
            coincideME11 = true;
          }
        }

        if (coincideME11)
          gemBXVFATPerChamberCoincidence[chamber-1][hist_index][layer]->Fill(Hit->BX(), vfat);

        gemChamberVFATBX[hist_index]->Fill( thisv2_chamber_bin(1, 1, chamber), (Hit->BX() + 3) * 30 + vfat);

//        gemHitOccupancy->Fill((Hit->Sector() - 1) * 7 + (Hit->Chamber() + 4) % 6, (endcap > 0) ? 1.5 : 0.5);
        gemHitOccupancy->Fill(thisv2_chamber_bin(1, 1, chamber), (endcap > 0) ? 1.5 : 0.5); // follow CSC convention
      }
      else {
        gemChamberPad[hist_index]->Fill( (Hit->Sector()%6)*6+2 , Hit->Pad());
        gemChamberPartition[hist_index]->Fill( (Hit->Sector()%6)*6+2, Hit->Partition());
        gemChamberAddress[hist_index]->Fill( (Hit->Sector()%6)*6+2 , Hit->Pad() + Hit->Partition()*192);

        gemChamberVFAT[hist_index]->Fill( (Hit->Sector()%6+1) * 7 - 4, vfat);
        gemChamberVFATBX[hist_index]->Fill( (Hit->Sector()%6+1) * 7 - 4, (Hit->BX() + 3) * 30 + vfat);
//        gemHitOccupancy->Fill((((Hit->Sector() - 2) % 6) + 1) * 7, (endcap > 0) ? 1.5 : 0.5);
        gemHitOccupancy->Fill( (Hit->Sector()%6+1) * 7 - 4, (endcap > 0) ? 1.5 : 0.5); // follow CSC convention
      }

      // GEM cosmics debug 2021.05.25
      if (endcap>0) {
        if (Hit->Neighbor() == false and Hit->Chamber() == 32)      gemPosCham32S5NPadPart->Fill( Hit->Partition(),   Hit->Pad() );
        else if (Hit->Neighbor() == true and Hit->Sector() == 5)    gemPosCham32S5NPadPart->Fill( Hit->Partition()+9, Hit->Pad() );
        if (Hit->Neighbor() == false and Hit->Chamber() == 2)       gemPosCham02S6NPadPart->Fill( Hit->Partition(),   Hit->Pad() );
        else if (Hit->Neighbor() == true and Hit->Sector() == 6)    gemPosCham02S6NPadPart->Fill( Hit->Partition()+9, Hit->Pad() );
      }
      else {
        if (Hit->Neighbor() == false and Hit->Chamber() == 8)       gemNegCham08S1NPadPart->Fill( Hit->Partition(),   Hit->Pad() );
        else if (Hit->Neighbor() == true and Hit->Sector() == 1)    gemNegCham08S1NPadPart->Fill( Hit->Partition()+9, Hit->Pad() );
        if (Hit->Neighbor() == false and Hit->Chamber() == 20)      gemNegCham20S3NPadPart->Fill( Hit->Partition(),   Hit->Pad() );
        else if (Hit->Neighbor() == true and Hit->Sector() == 3)    gemNegCham20S3NPadPart->Fill( Hit->Partition()+9, Hit->Pad() );

        if (Hit->Neighbor() == false and Hit->Chamber() == 12) gemNegCham12PadPart->Fill( Hit->Partition(),   Hit->Pad() );
      }

    } // End of if (Hit->Is_GEM() == true)
  }

  // ***** GEM cosmics debug 2021.05.21
  // loop hits for rate plots
  // start from 1 not 0, as the binning is 0.1-1.1 for 0, 1.1-2.1 for 1, etc.

  hitTypeNumber->Fill(1, CSCtot);
  hitTypeNumber->Fill(2, RPCtot);
  hitTypeNumber->Fill(3, GEMtot);
  hitTypeNumber->Fill(4, CSCGEM); 
  for (int sec = 1; sec < 7; sec++) {
    hitTypeNumSecGE11Pos->Fill(sec*2-1, GEpos11[sec-1]);
    hitTypeNumSecGE11Pos->Fill(sec*2,   MEpos11[sec-1]);
    hitTypeNumSecGE11Neg->Fill(sec*2-1, GEneg11[sec-1]);
    hitTypeNumSecGE11Neg->Fill(sec*2,   MEneg11[sec-1]);
  }

  if (MEneg11[0] < 3 and MEneg11[1] < 3 and MEneg11[2] < 3 and MEneg11[3] < 3 and MEneg11[4] < 3 and MEneg11[5] < 3) {
    for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
      if (Hit->Neighbor() == true) continue;
      int endcap = Hit->Endcap();
      int sector = Hit->Sector();
      int station = Hit->Station();
      int ring = Hit->Ring();
      int chamber = Hit->Chamber();
      
      bool have_ME11 = false;
      bool have_ME21 = false;
      bool have_ME12 = false;
      bool have_GE11 = false;
      bool have_GE_chamber = false;
      int  GE11_sec = 0;
      int  GE_part0134 = 0;
      if (Hit->Is_CSC() == true and endcap <=0 and station == 1 and ring == 1) {
        for (auto other_hit = HitCollection->begin(); other_hit != HitCollection->end(); ++other_hit) {
          if (other_hit->Neighbor() == true) continue;
          // other CSC hits
          if (other_hit->Is_CSC() == true and other_hit->Endcap() == endcap) {
            // in ME11
            if (other_hit->Station() == 1 and other_hit->Ring()==1) {
              if (other_hit->Sector() != sector) have_ME11 = true;
            }
            // in ME21 
            else if (other_hit->Station() == 1 and other_hit->Ring() == 2) have_ME21 = true;
            else if (other_hit->Station() == 2 and other_hit->Ring() == 1) have_ME12 = true;
          }
          // other GEM hits
          else if (other_hit->Is_GEM() == true and other_hit->Endcap() == endcap and other_hit->Station() == 1 and other_hit->Ring()==1) {
            have_GE11 = true;
            if (other_hit->Sector() == sector) {
              GE11_sec++;
              if (other_hit->Partition() == 0 or other_hit->Partition() == 1 or other_hit->Partition()==3 or other_hit->Partition()==4) GE_part0134++;
            }
            if (other_hit->Chamber() == chamber) have_GE_chamber = true;
          }  
        }
        if (have_ME11) hitCoincideME11->Fill(sector, 1);
        if (have_ME21) hitCoincideME11->Fill(sector, 2);
        if (have_ME12) hitCoincideME11->Fill(sector, 3);
        if (have_GE11) hitCoincideME11->Fill(sector, 4);
        if (have_GE_chamber) hitCoincideME11->Fill(sector, 5);
        if (GE11_sec == 0) hitCoincideME11->Fill(sector, 6);
        else if (GE11_sec == 1) hitCoincideME11->Fill(sector, 7);
        else if (GE11_sec < 11) hitCoincideME11->Fill(sector, 8);
        else hitCoincideME11->Fill(sector, 9);
        if (GE_part0134 == 0) hitCoincideME11->Fill(sector, 10);
        else if (GE_part0134 == 1) hitCoincideME11->Fill(sector, 11);
        else if (GE_part0134 == 2) hitCoincideME11->Fill(sector, 12);
        else if (GE_part0134 < 6) hitCoincideME11->Fill(sector, 13);
        else if (GE_part0134 < 10) hitCoincideME11->Fill(sector, 14);
        else hitCoincideME11->Fill(sector, 15);

        if (GE_part0134>0 and GE_part0134 < 10) {
          for (auto other_hit = HitCollection->begin(); other_hit != HitCollection->end(); ++other_hit) {
            if (other_hit->Neighbor() == true) continue;
            if (other_hit->Is_GEM() == true and other_hit->Endcap() == endcap and other_hit->Station() == 1 and other_hit->Ring()==1 and other_hit->Sector() == sector)
              if (other_hit->Partition() == 0 or other_hit->Partition() == 1 or other_hit->Partition()==3 or other_hit->Partition()==4) {
//                SameSectorTimingCSCGEM->Fill(Hit->BX(), other_hit->BX());
//                SameSectorChamberCSCGEM->Fill(chamber, other_hit->Chamber());
                SameSectorGEMPadPartition->Fill(other_hit->Partition(), other_hit->Pad());
                SameSectorGEMminusCSCfpThetaPhi->Fill( other_hit->Theta_fp() - 0, other_hit->Phi_fp() - 0 );
              }
          }
        }
      }
    }
  }

  if (GEneg11p0134[1] < 3 and GEneg11p0134[3] < 3) {
    for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
      if (Hit->Neighbor() == true) continue;
      int endcap = Hit->Endcap();
      int sector = Hit->Sector();
      int station = Hit->Station();
      int ring = Hit->Ring();
      int partition = Hit->Partition();

      int ME11_sec = 0;
      int MEall_sec = 0;
      int MEall = 0;
      if (Hit->Is_GEM() == true and endcap <= 0 and station == 1 and ring == 1 and (sector == 2 or sector == 4) and (partition == 0 or partition == 1 or partition == 3 or partition == 4)) {
        gemNegBXAddress0134->Fill(Hit->BX(), Hit->Pad() + Hit->Partition()*192);
        for (auto other_hit = HitCollection->begin(); other_hit != HitCollection->end(); ++other_hit) {
          if (other_hit->Neighbor() == true) continue;
          if (other_hit->Is_CSC() == true and other_hit->Endcap() == endcap) {
            MEall++;
            if (other_hit->Sector() == sector) {
              MEall_sec++;
              if (other_hit->Station() == 1 and other_hit->Ring() == 1) ME11_sec++;
            }
          }
        } // End for (auto other_hit = HitCollection->begin(); other_hit != HitCollection->end(); ++other_hit)
        if (MEall == 0 )     hitCoincideGE11->Fill(sector, 1);
        else if (MEall == 1) hitCoincideGE11->Fill(sector, 2);
        else                 hitCoincideGE11->Fill(sector, 3);
        if (MEall_sec == 0)      hitCoincideGE11->Fill(sector, 4);
        else if (MEall_sec == 1) hitCoincideGE11->Fill(sector, 5); 
        else                     hitCoincideGE11->Fill(sector, 6);
        if (ME11_sec == 0)      hitCoincideGE11->Fill(sector, 7);
        else if (ME11_sec == 1) hitCoincideGE11->Fill(sector, 8);
        else                    hitCoincideGE11->Fill(sector, 9);
      }
    }
  }
  // ***** GEM cosmics debug 2021.05.21 end


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
          int iCSC = (endcap > 0) ? (iTrkHit.Station() + 3) : (4 - iTrkHit.Station());
          emtfTrackModeVsCSCBXDiff[iCSC]->Fill( Track->BX() - iTrkHit.BX(), mode); // Add mode vs BXdiff comparison 2020.12.07
        } else if (iTrkHit.Is_RPC() == true) {
          emtfTrackBXVsRPCHit[hist_index2]->Fill(iTrkHit.BX(), Track->BX());
          int iRPC = (endcap > 0) ? (iTrkHit.Station() + 2) : (4 - iTrkHit.Station());
          emtfTrackModeVsRPCBXDiff[iRPC]->Fill( Track->BX() - iTrkHit.BX(), mode); // Add mode vs BXdiff comparison 2020.12.07
        } else if (iTrkHit.Is_GEM() == true) {
          emtfTrackBXVsGEMHit[hist_index2]->Fill(iTrkHit.BX(), Track->BX());
          int iGEM = (endcap > 0) ? 1 : 0;
          emtfTrackModeVsGEMBXDiff[iGEM]->Fill( Track->BX() - iTrkHit.BX(), mode); // Add mode vs BXdiff comparison 2020.12.07
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
            cscLCTTiming[histIndexBX.at(trackHitBX)]->Fill(thisv2_chamber_bin(station, ring, chamber), hist_index, evt_wgt);
            cscTimingTot->Fill(thisv2_chamber_bin(station, ring, chamber), hist_index, evt_wgt);
            if (station > 1 && (ring % 2) == 1) {
              cscLCTTiming[histIndexBX.at(trackHitBX)]->Fill(
                  thisv2_chamber_bin(station, ring, chamber) - 1, hist_index, evt_wgt);
              cscTimingTot->Fill(thisv2_chamber_bin(station, ring, chamber) - 1, hist_index, evt_wgt);
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
              rpcHitTiming[histIndexBX.at(Hit->BX())]->Fill(
                  (Hit->Sector_RPC() - 1) * 7 + get_subsector_rpc_cppf(Hit->Subsector_RPC()), hist_index + 0.5);
              rpcHitTimingTot->Fill((Hit->Sector_RPC() - 1) * 7 + get_subsector_rpc_cppf(Hit->Subsector_RPC()),
                                    hist_index + 0.5);
            }  // End loop: for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit)
          }    // End conditional: if (trackHitBX == 0 && ring == 2)

          // Fill GEM timing with matched CSC LCTs
          if (trackHitBX == 0 && station == 1 && ring == 1) { // GEM only in station 1
            for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
              if (Hit->Is_GEM() == false)
                continue;
              if (std::abs(Track->Eta() - Hit->Eta()) > 0.1)
                continue;
              if (Hit->Endcap() != endcap || Hit->Station() != station || Hit->Chamber() != chamber || Hit->Neighbor() != neighbor) //different neighbor requirement from RPC
                continue;
              if (std::abs(Hit->BX()) > 2)
                continue;

              if (neighbor == false) {
                gemHitTiming[histIndexBX.at(Hit->BX())]->Fill(thisv2_chamber_bin(1, 1, chamber), (endcap > 0) ? 1.5 : 0.5);
                gemHitTimingTot->Fill(thisv2_chamber_bin(1, 1, chamber), (endcap > 0) ? 1.5 : 0.5);
                int ihist = (endcap > 0)? 1 : 0;
                gemHitPhi[ihist]->Fill(Hit->Phi_fp(), sector);
                gemHitTheta[ihist]->Fill(Hit->Theta_fp(), sector);
                gemHitVScscLCTPhi[ihist]->Fill( Hit->Phi_fp() - TrkHit.Phi_fp(), chamber);  // GEM vs CSC 2020.12.06
                gemHitVScscLCTTheta[ihist]->Fill( Hit->Theta_fp() - TrkHit.Theta_fp(), chamber);
                gemHitVScscLCTBX[ihist]->Fill( Hit->BX() - TrkHit.BX(), chamber);
              } else {
                gemHitTiming[histIndexBX.at(Hit->BX())]->Fill(sector * 7 - 4, (endcap > 0) ? 1.5 : 0.5);
                gemHitTimingTot->Fill(sector * 7 - 4, (endcap > 0) ? 1.5 : 0.5);
              }

            }  // End loop: for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit)
          }    // End conditional: if (trackHitBX == 0 && station == 1 && ring == 1)
        }      // End conditional: if (TrkHit.Is_CSC() == true)

        // Maps RPC station and ring to monitor element index
        const std::map<std::pair<int, int>, int> histIndexRPC = {
            {{4, 3}, 0}, {{4, 2}, 1}, {{3, 3}, 2}, {{3, 2}, 3}, {{2, 2}, 4}, {{1, 2}, 5}};

        if (TrkHit.Is_RPC() == true && neighbor == false) {
          hist_index = histIndexRPC.at({station, ring});
          if (endcap > 0)
            hist_index = 11 - hist_index;

          rpcHitTimingInTrack->Fill(trackHitBX, hist_index + 0.5);
          rpcHitTiming[histIndexBX.at(trackHitBX)]->Fill(
              (TrkHit.Sector_RPC() - 1) * 7 + get_subsector_rpc_cppf(TrkHit.Subsector_RPC()), hist_index + 0.5);
          rpcHitTimingTot->Fill((TrkHit.Sector_RPC() - 1) * 7 + get_subsector_rpc_cppf(TrkHit.Subsector_RPC()),
                                hist_index + 0.5);
        }  // End conditional: if (TrkHit.Is_RPC() == true && neighbor == false)
        if (TrkHit.Is_RPC() == true && neighbor == true) {
          hist_index = histIndexRPC.at({station, ring});
          if (endcap > 0)
            hist_index = 11 - hist_index;
          rpcHitTiming[histIndexBX.at(trackHitBX)]->Fill((TrkHit.Sector_RPC() - 1) * 7, hist_index + 0.5);
        }

        // Add GEM Timing Oct 27 2020
        // Decide whether needs to match to CSC LCT
        if (TrkHit.Is_GEM() == true) {
          if (neighbor == false) {
            gemHitTiming[histIndexBX.at(trackHitBX)]->Fill(thisv2_chamber_bin(1, 1, chamber), (endcap > 0) ? 1.5 : 0.5);
            gemHitTimingTot->Fill(thisv2_chamber_bin(1, 1, chamber), (endcap > 0) ? 1.5 : 0.5);
          } else {
            gemHitTiming[histIndexBX.at(trackHitBX)]->Fill(sector * 7 - 4, (endcap > 0) ? 1.5 : 0.5);
            gemHitTimingTot->Fill(sector * 7 - 4, (endcap > 0) ? 1.5 : 0.5);
          }
        } // End condition: if (TrkHit.Is_GEM() == true)
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
    gemHitTimingFrac[hist_index]->getTH2F()->Divide(gemHitTiming[hist_index]->getTH2F(), gemHitTimingTot->getTH2F());
  }

  // Regional Muon Candidates
//  edm::Handle<l1t::RegionalMuonCandBxCollection> MuonBxCollection;
//  e.getByToken(muonToken, MuonBxCollection);
//
//  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
//    for (l1t::RegionalMuonCandBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX);
//         Muon != MuonBxCollection->end(itBX);
//         ++Muon) {
//      emtfMuonBX->Fill(itBX);
//      emtfMuonhwPt->Fill(Muon->hwPt());
//      emtfMuonhwEta->Fill(Muon->hwEta());
//      emtfMuonhwPhi->Fill(Muon->hwPhi());
//      emtfMuonhwQual->Fill(Muon->hwQual());
//    }
//  }
}
