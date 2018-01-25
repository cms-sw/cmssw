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
      verbose(ps.getUntrackedParameter<bool>("verbose", false)),
      emtfnTracksNbins(11) {}

L1TStage2EMTF::~L1TStage2EMTF() {}

void L1TStage2EMTF::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c, emtfdqm::Histograms& histograms) const {}

void L1TStage2EMTF::bookHistograms(DQMStore::ConcurrentBooker& booker, const edm::Run&, const edm::EventSetup&, emtfdqm::Histograms& histograms) const {
  
  // Monitor Dir
  booker.setCurrentFolder(monitorDir);
  
  const std::array<std::string, 6> binNamesErrors{{"Corruptions","Synch. Err.","Synch. Mod.","BX Mismatch","Time Misalign","FMM != Ready"}};
  
  // DAQ Output Monitor Elements
  histograms.emtfErrors = booker.book1D("emtfErrors", "EMTF Errors", 6, 0, 6);
  histograms.emtfErrors.setAxisTitle("Error Type (Corruptions Not Implemented)", 1);
  histograms.emtfErrors.setAxisTitle("Number of Errors", 2);
  for (unsigned int bin = 0; bin < binNamesErrors.size(); ++bin) { 
    histograms.emtfErrors.setBinLabel(bin+1, binNamesErrors[bin], 1);
  }

  // CSC LCT Monitor Elements
  int nChambs, nWires, nStrips;  // Number of chambers, wiregroups, and halfstrips in each station/ring pair
  std::string name, label;
  const std::array<std::string, 10> suffix_name{{"42", "41", "32", "31", "22", "21", "13", "12", "11b", "11a"}};
  const std::array<std::string, 10> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1b", "1/1a"}};
  const std::array<std::string, 12> binNames{{"ME-N", "ME-4", "ME-3", "ME-2", "ME-1b", "ME-1a", "ME+1a", "ME+1b", "ME+2", "ME+3", "ME+4", "ME+N"}};

  histograms.cscLCTBX = booker.book2D("cscLCTBX", "CSC LCT BX", 7, -3, 4, 20, 0, 20);
  histograms.cscLCTBX.setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    histograms.cscLCTBX.setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 10; ++ybin) {
    histograms.cscLCTBX.setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    histograms.cscLCTBX.setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
  
  histograms.cscLCTOccupancy = booker.book2D("cscLCTOccupancy", "CSC Chamber Occupancy", 54, 1, 55, 12, -6, 6);
  histograms.cscLCTOccupancy.setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int xbin = 1; xbin < 7; ++xbin) {
    histograms.cscLCTOccupancy.setBinLabel(xbin * 9 - 8, std::to_string(xbin), 1);
  }
  for (unsigned int ybin = 0; ybin < binNames.size(); ++ybin) {
    histograms.cscLCTOccupancy.setBinLabel(ybin+1, binNames[ybin], 2);
  }

  //cscOccupancy designed to match the cscDQM plot  
  histograms.cscDQMOccupancy = booker.book2D("cscDQMOccupancy", "CSC Chamber Occupancy", 42, 1, 43, 20, 0, 20);
  histograms.cscDQMOccupancy.setAxisTitle("10#circ Chamber (N=neighbor)", 1);
  int count=0;
  for (int xbin=1; xbin < 43; ++xbin) {
  histograms.cscDQMOccupancy.setBinLabel(xbin, std::to_string(xbin-count), 1);
    if (xbin==2 || xbin==9 || xbin==16 || xbin==23 || xbin==30 ||xbin==37 ) {
       ++xbin;
       ++count;
       histograms.cscDQMOccupancy.setBinLabel(xbin, "N", 1);
    }
  }
  for (int ybin = 1; ybin <= 10; ++ybin) {
    histograms.cscDQMOccupancy.setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    histograms.cscDQMOccupancy.setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
  cscDQMOccupancy->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing

  histograms.mpcLinkErrors = booker.book2D("mpcLinkErrors", "MPC Link Errors", 54, 1, 55, 12, -6, 6);
  histograms.mpcLinkErrors.setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int xbin = 1; xbin < 7; ++xbin) {
    histograms.mpcLinkErrors.setBinLabel(xbin * 9 - 8, std::to_string(xbin), 1);
  }
  for (unsigned int ybin = 0; ybin < binNames.size(); ++ybin) {
    histograms.mpcLinkErrors.setBinLabel(ybin+1, binNames[ybin], 2);
  }

  histograms.mpcLinkGood = booker.book2D("mpcLinkGood", "MPC Good Links", 54, 1, 55, 12, -6, 6);
  histograms.mpcLinkGood.setAxisTitle("Sector (CSCID 1-9 Unlabelled)", 1);
  for (int xbin = 1; xbin < 7; ++xbin) {
    histograms.mpcLinkGood.setBinLabel(xbin * 9 - 8, std::to_string(xbin), 1);
  }
  for (unsigned int ybin = 0; ybin < binNames.size(); ++ybin) {
    histograms.mpcLinkGood.setBinLabel(ybin+1, binNames[ybin], 2);
  }

  // RPC Monitor Elements
  const std::array<std::string, 6> rpc_name{{"43", "42", "33", "32", "22", "12"}};
  const std::array<std::string, 6> rpc_label{{"4/3", "4/2", "3/3", "3/2", "2/2", "1/2"}};

  histograms.rpcHitBX = booker.book2D("rpcHitBX", "RPC Hit BX", 7, -3, 4, 12, 0, 12);
  histograms.rpcHitBX.setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    histograms.rpcHitBX.setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 6; ++ybin) {
    histograms.rpcHitBX.setBinLabel(ybin, "RE-" + rpc_label[ybin - 1], 2);
    histograms.rpcHitBX.setBinLabel(13 - ybin, "RE+" + rpc_label[ybin - 1], 2);
  }
  
  histograms.rpcHitOccupancy = booker.book2D("rpcHitOccupancy", "RPC Chamber Occupancy", 42, 1, 43, 12, 0, 12);
  histograms.rpcHitOccupancy.setAxisTitle("Sector (N=neighbor)", 1);
  for (int bin = 1; bin < 7; ++bin) {
    histograms.rpcHitOccupancy.setBinLabel(bin*7 - 6, std::to_string(bin), 1);
    histograms.rpcHitOccupancy.setBinLabel(bin*7, "N", 1);
    histograms.rpcHitOccupancy.setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
    histograms.rpcHitOccupancy.setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1],2);
  }  
  rpcHitOccupancy->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing

  // Track Monitor Elements
  histograms.emtfnTracks = booker.book1D("emtfnTracks", "Number of EMTF Tracks per Event", emtfnTracksNbins, 0, emtfnTracksNbins);
  for (int xbin = 1; xbin <= emtfnTracksNbins-1; ++xbin) {
    histograms.emtfnTracks.setBinLabel(xbin, std::to_string(xbin - 1), 1);
  }
  histograms.emtfnTracks.setBinLabel(11, "Overflow", 1);

  histograms.emtfTracknHits = booker.book1D("emtfTracknHits", "Number of Hits per EMTF Track", 5, 0, 5);
  for (int xbin = 1; xbin <= 5; ++xbin) {
    histograms.emtfTracknHits.setBinLabel(xbin, std::to_string(xbin - 1), 1);
  }

  histograms.emtfTrackBX = booker.book2D("emtfTrackBX", "EMTF Track Bunch Crossing", 12, -6, 6, 7, -3, 4);
  histograms.emtfTrackBX.setAxisTitle("Sector (Endcap)", 1);
  for (int xbin = 0; xbin < 6; ++xbin) {
    histograms.emtfTrackBX.setBinLabel(xbin + 1, std::to_string(6 - xbin) + " (-)", 1);
    histograms.emtfTrackBX.setBinLabel(12 - xbin, std::to_string(6 - xbin) + " (+)", 1);
  }
  histograms.emtfTrackBX.setAxisTitle("Track BX", 2);
  for (int ybin = 1, i = -3; ybin <= 7; ++ybin, ++i) {
    histograms.emtfTrackBX.setBinLabel(ybin, std::to_string(i), 2);
  }
  
  histograms.emtfTrackPt = booker.book1D("emtfTrackPt", "EMTF Track p_{T}", 256, 1, 257);
  histograms.emtfTrackPt.setAxisTitle("Track p_{T} [GeV]", 1);

  histograms.emtfTrackEta = booker.book1D("emtfTrackEta", "EMTF Track #eta", 100, -2.5, 2.5);
  histograms.emtfTrackEta.setAxisTitle("Track #eta", 1);

  histograms.emtfTrackPhi = booker.book1D("emtfTrackPhi", "EMTF Track #phi", 126, -3.15, 3.15);
  histograms.emtfTrackPhi.setAxisTitle("Track #phi", 1);

  histograms.emtfTrackPhiHighQuality = booker.book1D("emtfTrackPhiHighQuality", "EMTF High Quality #phi", 126, -3.15, 3.15);
  histograms.emtfTrackPhiHighQuality.setAxisTitle("Track #phi (Quality #geq 12)", 1);

  histograms.emtfTrackOccupancy = booker.book2D("emtfTrackOccupancy", "EMTF Track Occupancy", 100, -2.5, 2.5, 126, -3.15, 3.15);
  histograms.emtfTrackOccupancy.setAxisTitle("#eta", 1);
  histograms.emtfTrackOccupancy.setAxisTitle("#phi", 2);

  histograms.emtfTrackMode = booker.book1D("emtfTrackMode", "EMTF Track Mode", 16, 0, 16);
  histograms.emtfTrackMode.setAxisTitle("Mode", 1);

  histograms.emtfTrackQuality = booker.book1D("emtfTrackQuality", "EMTF Track Quality", 16, 0, 16);
  histograms.emtfTrackQuality.setAxisTitle("Quality", 1);

  histograms.emtfTrackQualityVsMode = booker.book2D("emtfTrackQualityVsMode", "EMTF Track Quality vs Mode", 16, 0, 16, 16, 0, 16);
  histograms.emtfTrackQualityVsMode.setAxisTitle("Mode", 1);
  histograms.emtfTrackQualityVsMode.setAxisTitle("Quality", 2);

  for (int bin = 1; bin <= 16; ++bin) {
    histograms.emtfTrackMode.setBinLabel(bin, std::to_string(bin - 1), 1);
    histograms.emtfTrackQuality.setBinLabel(bin, std::to_string(bin - 1), 1);
    histograms.emtfTrackQualityVsMode.setBinLabel(bin, std::to_string(bin - 1), 1);
    histograms.emtfTrackQualityVsMode.setBinLabel(bin, std::to_string(bin - 1), 2);
  }

  // CSC Input
  booker.setCurrentFolder(monitorDir + "/CSCInput");

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
    } else if (hist >13) {
      nChambs = (i % 2) ? 36 : 18;
    } else {
      nChambs = 36;
    }
    
    const std::array<int, 10> wiregroups{{64, 96, 64, 96, 64, 112, 32, 64, 48, 48}};
    const std::array<int, 10> halfstrips{{160, 160, 160, 160, 160, 160, 128, 160, 128, 96}};
    
    if (hist < 10) {
      nWires  = wiregroups[hist];
      nStrips = halfstrips[hist];
    } else {
      nWires  = wiregroups[19 - hist];
      nStrips = halfstrips[19 - hist];
    }

    histograms.cscLCTStrip[hist] = booker.book1D("cscLCTStrip" + name, "CSC Halfstrip " + label, nStrips, 0, nStrips);
    histograms.cscLCTStrip[hist].setAxisTitle("Cathode Halfstrip, " + label, 1);

    histograms.cscLCTWire[hist] = booker.book1D("cscLCTWire" + name, "CSC Wiregroup " + label, nWires, 0, nWires);
    histograms.cscLCTWire[hist].setAxisTitle("Anode Wiregroup, " + label, 1);

    histograms.cscChamberStrip[hist] = booker.book2D("cscChamberStrip" + name, "CSC Halfstrip " + label, nChambs, 1, 1+nChambs, nStrips, 0, nStrips);
    histograms.cscChamberStrip[hist].setAxisTitle("Chamber, " + label, 1);
    histograms.cscChamberStrip[hist].setAxisTitle("Cathode Halfstrip", 2);

    histograms.cscChamberWire[hist] = booker.book2D("cscChamberWire" + name, "CSC Wiregroup " + label, nChambs, 1, 1+nChambs, nWires, 0, nWires);
    histograms.cscChamberWire[hist].setAxisTitle("Chamber, " + label, 1);
    histograms.cscChamberWire[hist].setAxisTitle("Anode Wiregroup", 2);
    
    for (int bin = 1; bin <= nChambs; ++bin) {
      histograms.cscChamberStrip[hist].setBinLabel(bin, std::to_string(bin), 1);
      histograms.cscChamberWire[hist].setBinLabel(bin, std::to_string(bin), 1);
    }
  }

  // RPC Input
  booker.setCurrentFolder(monitorDir + "/RPCInput");

  for (int hist = 0, i = 0; hist < 12; ++hist, i = hist % 6) {
    if (hist < 6) {
      name = "RENeg" + rpc_name[i];
      label = "RE-" + rpc_label[i];
    } else {
      name = "REPos" + rpc_name[5 - i];
      label = "RE+" + rpc_label[5 - i];
    }
    histograms.rpcHitPhi[hist] = booker.book1D("rpcHitPhi" + name, "RPC Hit Phi " + label, 1250, 0, 1250);
    histograms.rpcHitPhi[hist].setAxisTitle("#phi", 1);
    histograms.rpcHitTheta[hist] = booker.book1D("rpcHitTheta" + name, "RPC Hit Theta " + label, 32, 0, 32);
    histograms.rpcHitTheta[hist].setAxisTitle("#theta", 1);
    histograms.rpcChamberPhi[hist] = booker.book2D("rpcChamberPhi" + name, "RPC Chamber Phi " + label, 36, 1, 37, 1250, 0, 1250);
    histograms.rpcChamberPhi[hist].setAxisTitle("Chamber", 1);
    histograms.rpcChamberPhi[hist].setAxisTitle("#phi", 2);
    histograms.rpcChamberTheta[hist] = booker.book2D("rpcChamberTheta" + name, "RPC Chamber Theta " + label, 36, 1, 37, 32, 0, 32);
    histograms.rpcChamberTheta[hist].setAxisTitle("Chamber", 1);
    histograms.rpcChamberTheta[hist].setAxisTitle("#theta", 2);
    for (int xbin = 1; xbin < 37; ++xbin) {
      histograms.rpcChamberPhi[hist].setBinLabel(xbin, std::to_string(xbin), 1);
      histograms.rpcChamberTheta[hist].setBinLabel(xbin, std::to_string(xbin), 1);
    }
  }

  // CSC LCT and RPC Hit Timing
  booker.setCurrentFolder(monitorDir + "/Timing");
 
  histograms.cscTimingTot = booker.book2D("cscTimingTotal", "CSC Total BX ", 42, 1, 43, 20, 0, 20);    
  histograms.cscTimingTot.setAxisTitle("10#circ Chamber (N=neighbor)", 1);

  histograms.rpcHitTimingTot = booker.book2D("rpcHitTimingTot", "RPC Chamber Occupancy ", 42, 1, 43, 12, 0, 12);
  histograms.rpcHitTimingTot.setAxisTitle("Sector (N=neighbor)", 1);
  const std::array<std::string, 5> nameBX{{"BXNeg1","BXPos1","BXNeg2","BXPos2","BX0"}};
  const std::array<std::string, 5> labelBX{{"BX -1","BX +1","BX -2","BX +2","BX 0"}};

  for (int hist = 0;  hist < 5; ++hist) {

    count = 0;
    histograms.cscLCTTiming[hist] = booker.book2D("cscLCTTiming" + nameBX[hist], "CSC Chamber Occupancy " + labelBX[hist], 42, 1, 43, 20, 0, 20);
    histograms.cscLCTTiming[hist].setAxisTitle("10#circ Chamber", 1);

    for (int xbin=1; xbin < 43; ++xbin) {
      histograms.cscLCTTiming[hist].setBinLabel(xbin, std::to_string(xbin-count), 1);
      if (hist==0) histograms.cscTimingTot.setBinLabel(xbin, std::to_string(xbin-count), 1);//only fill once in the loop
      if (xbin==2 || xbin==9 || xbin==16 || xbin==23 || xbin==30 ||xbin==37 ) {
        ++xbin;
        ++count;
        histograms.cscLCTTiming[hist].setBinLabel(xbin, "N", 1);
        if (hist==0) histograms.cscTimingTot.setBinLabel(xbin, "N", 1);
      }
    }

    for (int ybin = 1; ybin <= 10; ++ybin) {
      histograms.cscLCTTiming[hist].setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
      histograms.cscLCTTiming[hist].setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
      if (hist==0) histograms.cscTimingTot.setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
      if (hist==0) histograms.cscTimingTot.setBinLabel(21 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    }
    if (hist==0) cscTimingTot->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing
    cscLCTTiming[hist]->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing
      
    histograms.rpcHitTiming[hist] = booker.book2D("rpcHitTiming" + nameBX[hist], "RPC Chamber Occupancy " + labelBX[hist], 42, 1, 43, 12, 0, 12);
    histograms.rpcHitTiming[hist].setAxisTitle("Sector (N=neighbor)", 1);
    for (int bin = 1; bin < 7; ++bin) {
      histograms.rpcHitTiming[hist].setBinLabel(bin*7 - 6, std::to_string(bin), 1);
      histograms.rpcHitTiming[hist].setBinLabel(bin*7, "N", 1);
      histograms.rpcHitTiming[hist].setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
      histograms.rpcHitTiming[hist].setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1],2);
    }
    rpcHitTiming[hist]->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing
    if (hist==0) {
      for (int bin = 1; bin < 7; ++bin) {
        histograms.rpcHitTimingTot.setBinLabel(bin*7 - 6, std::to_string(bin), 1);
        histograms.rpcHitTimingTot.setBinLabel(bin*7, "N", 1);
        histograms.rpcHitTimingTot.setBinLabel(bin, "RE-" + rpc_label[bin - 1], 2);
        histograms.rpcHitTimingTot.setBinLabel(13 - bin, "RE+" + rpc_label[bin - 1],2);
      }
      rpcHitTimingTot->getTH2F()->GetXaxis()->SetCanExtend(false); // Needed to stop multi-thread summing
    }
    //if (hist == 4) continue; // Don't book for BX = 0

  }
      
  histograms.rpcHitTimingInTrack = booker.book2D("rpcHitTimingInTrack", "RPC Hit Timing (matched to track in BX 0)", 7, -3, 4, 12, 0, 12);
  histograms.rpcHitTimingInTrack.setAxisTitle("BX", 1);
  for (int xbin = 1, xbin_label = -3; xbin <= 7; ++xbin, ++xbin_label) {
    histograms.rpcHitTimingInTrack.setBinLabel(xbin, std::to_string(xbin_label), 1);
  }
  for (int ybin = 1; ybin <= 6; ++ybin) {
    histograms.rpcHitTimingInTrack.setBinLabel(ybin, "RE-" + rpc_label[ybin - 1], 2);
    histograms.rpcHitTimingInTrack.setBinLabel(13 - ybin, "RE+" + rpc_label[ybin - 1], 2);
  }
  
  const std::array<std::string, 3> nameNumStation{{"4Station","3Station","2Station"}};
  const std::array<std::string, 3> labelNumStation{{"4 Station Track","3 Station Track","2 Station Track"}};
    
  for (int hist = 0; hist < 3; ++hist) {
    histograms.emtfTrackBXVsCSCLCT[hist] = booker.book2D("emtfTrackBXVsCSCLCT" + nameNumStation[hist],
                                               "EMTF " + labelNumStation[hist] + " BX vs CSC LCT BX", 7, -3, 4, 7, -3, 4);
    histograms.emtfTrackBXVsCSCLCT[hist].setAxisTitle("LCT BX", 1);
    histograms.emtfTrackBXVsCSCLCT[hist].setAxisTitle("Track BX", 2);
    for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
      histograms.emtfTrackBXVsCSCLCT[hist].setBinLabel(bin, std::to_string(bin_label), 1);
      histograms.emtfTrackBXVsCSCLCT[hist].setBinLabel(bin, std::to_string(bin_label), 2);
    }
    histograms.emtfTrackBXVsRPCHit[hist] = booker.book2D("emtfTrackBXVsRPCHit" + nameNumStation[hist],
                                               "EMTF " + labelNumStation[hist] + " BX vs RPC Hit BX", 7, -3, 4, 7, -3, 4);
    histograms.emtfTrackBXVsRPCHit[hist].setAxisTitle("Hit BX", 1);
    histograms.emtfTrackBXVsRPCHit[hist].setAxisTitle("Track BX", 2);
    for (int bin = 1, bin_label = -3; bin <= 7; ++bin, ++bin_label) {
      histograms.emtfTrackBXVsRPCHit[hist].setBinLabel(bin, std::to_string(bin_label), 1);
      histograms.emtfTrackBXVsRPCHit[hist].setBinLabel(bin, std::to_string(bin_label), 2);
    }
  }

  // Muon Cand
  booker.setCurrentFolder(monitorDir + "/MuonCand");

  // Regional Muon Candidate Monitor Elements
  histograms.emtfMuonBX = booker.book1D("emtfMuonBX", "EMTF Muon Cand BX", 7, -3, 4);
  histograms.emtfMuonBX.setAxisTitle("BX", 1);
  for (int xbin = 1, bin_label = -3; xbin <= 7; ++xbin, ++bin_label) {
    histograms.emtfMuonBX.setBinLabel(xbin, std::to_string(bin_label), 1);
  }

  histograms.emtfMuonhwPt = booker.book1D("emtfMuonhwPt", "EMTF Muon Cand p_{T}", 512, 0, 512);
  histograms.emtfMuonhwPt.setAxisTitle("Hardware p_{T}", 1);

  histograms.emtfMuonhwEta = booker.book1D("emtfMuonhwEta", "EMTF Muon Cand #eta", 460, -230, 230);
  histograms.emtfMuonhwEta.setAxisTitle("Hardware #eta", 1);

  histograms.emtfMuonhwPhi = booker.book1D("emtfMuonhwPhi", "EMTF Muon Cand #phi", 145, -40, 105);
  histograms.emtfMuonhwPhi.setAxisTitle("Hardware #phi", 1);

  histograms.emtfMuonhwQual = booker.book1D("emtfMuonhwQual", "EMTF Muon Cand Quality", 16, 0, 16);
  histograms.emtfMuonhwQual.setAxisTitle("Quality", 1);
  for (int xbin = 1; xbin <= 16; ++xbin) {
    histograms.emtfMuonhwQual.setBinLabel(xbin, std::to_string(xbin - 1), 1);
  }
}

// CSCOccupancy chamber mapping for neighbor inclusive plots
int chamber_bin (int station, int ring, int chamber) {
  int chamber_bin_index = 0;
  if (station > 1 && (ring % 2) == 1) {
    chamber_bin_index = (chamber * 2) + ((chamber + 1) / 3);
  } else {
    chamber_bin_index = chamber + ((chamber + 3) / 6);
  }
  return chamber_bin_index;
};


void L1TStage2EMTF::dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, emtfdqm::Histograms const& histograms) const {

  if (verbose) edm::LogInfo("L1TStage2EMTF") << "L1TStage2EMTF: analyze..." << std::endl;

  // DAQ Output
  edm::Handle<l1t::EMTFDaqOutCollection> DaqOutCollection;
  e.getByToken(daqToken, DaqOutCollection);
  
  for (auto DaqOut = DaqOutCollection->begin(); DaqOut != DaqOutCollection->end(); ++DaqOut ) {
    const l1t::emtf::MECollection* MECollection = DaqOut->PtrMECollection();
    for (auto ME = MECollection->begin(); ME != MECollection->end(); ++ME ) {
      if (ME->SE())  histograms.emtfErrors.fill(1);
      if (ME->SM())  histograms.emtfErrors.fill(2);
      if (ME->BXE()) histograms.emtfErrors.fill(3);
      if (ME->AF())  histograms.emtfErrors.fill(4);
    }

    const l1t::emtf::EventHeader* EventHeader = DaqOut->PtrEventHeader();
    if (!EventHeader->Rdy()) histograms.emtfErrors.fill(5);
    
    // Fill MPC input link errors
    int offset = (EventHeader->Sector() - 1) * 9;
    int endcap = EventHeader->Endcap();
    l1t::emtf::Counters CO = DaqOut->GetCounters();
    const std::array<std::array<int,9>,5> counters {{
      {{CO.ME1a_1(), CO.ME1a_2(), CO.ME1a_3(), CO.ME1a_4(), CO.ME1a_5(), CO.ME1a_6(), CO.ME1a_7(), CO.ME1a_8(), CO.ME1a_9()}},
      {{CO.ME1b_1(), CO.ME1b_2(), CO.ME1b_3(), CO.ME1b_4(), CO.ME1b_5(), CO.ME1b_6(), CO.ME1b_7(), CO.ME1b_8(), CO.ME1b_9()}},
      {{CO.ME2_1(), CO.ME2_2(), CO.ME2_3(), CO.ME2_4(), CO.ME2_5(), CO.ME2_6(), CO.ME2_7(), CO.ME2_8(), CO.ME2_9()}},
      {{CO.ME3_1(), CO.ME3_2(), CO.ME3_3(), CO.ME3_4(), CO.ME3_5(), CO.ME3_6(), CO.ME3_7(), CO.ME3_8(), CO.ME3_9()}},
      {{CO.ME4_1(), CO.ME4_2(), CO.ME4_3(), CO.ME4_4(), CO.ME4_5(), CO.ME4_6(), CO.ME4_7(), CO.ME4_8(), CO.ME4_9()}}
      }};
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 9; j++) {
        if (counters.at(i).at(j) != 0) histograms.mpcLinkErrors.fill(j + 1 + offset, endcap * (i + 0.5), counters.at(i).at(j));
        else histograms.mpcLinkGood.fill(j + 1 + offset, endcap * (i  + 0.5));
      }
    }
    if (CO.ME1n_3() == 1) histograms.mpcLinkErrors.fill(1 + offset, endcap * 5.5);
    if (CO.ME1n_6() == 1) histograms.mpcLinkErrors.fill(2 + offset, endcap * 5.5);
    if (CO.ME1n_9() == 1) histograms.mpcLinkErrors.fill(3 + offset, endcap * 5.5);
    if (CO.ME2n_3() == 1) histograms.mpcLinkErrors.fill(4 + offset, endcap * 5.5);
    if (CO.ME2n_9() == 1) histograms.mpcLinkErrors.fill(5 + offset, endcap * 5.5);
    if (CO.ME3n_3() == 1) histograms.mpcLinkErrors.fill(6 + offset, endcap * 5.5);
    if (CO.ME3n_9() == 1) histograms.mpcLinkErrors.fill(7 + offset, endcap * 5.5);
    if (CO.ME4n_3() == 1) histograms.mpcLinkErrors.fill(8 + offset, endcap * 5.5);
    if (CO.ME4n_9() == 1) histograms.mpcLinkErrors.fill(9 + offset, endcap * 5.5);
    if (CO.ME1n_3() == 0) histograms.mpcLinkGood.fill(1 + offset, endcap * 5.5);
    if (CO.ME1n_6() == 0) histograms.mpcLinkGood.fill(2 + offset, endcap * 5.5);
    if (CO.ME1n_9() == 0) histograms.mpcLinkGood.fill(3 + offset, endcap * 5.5);
    if (CO.ME2n_3() == 0) histograms.mpcLinkGood.fill(4 + offset, endcap * 5.5);
    if (CO.ME2n_9() == 0) histograms.mpcLinkGood.fill(5 + offset, endcap * 5.5);
    if (CO.ME3n_3() == 0) histograms.mpcLinkGood.fill(6 + offset, endcap * 5.5);
    if (CO.ME3n_9() == 0) histograms.mpcLinkGood.fill(7 + offset, endcap * 5.5);
    if (CO.ME4n_3() == 0) histograms.mpcLinkGood.fill(8 + offset, endcap * 5.5);
    if (CO.ME4n_9() == 0) histograms.mpcLinkGood.fill(9 + offset, endcap * 5.5);
  }

  // Hits (CSC LCTs and RPC hits)
  edm::Handle<l1t::EMTFHitCollection> HitCollection;
  e.getByToken(hitToken, HitCollection);
  
  // Maps CSC station and ring to the monitor element index and uses symmetry of the endcaps    
  const std::map<std::pair<int,int>,int> histIndexCSC = { {{1,4}, 9}, {{1,1}, 8}, {{1,2}, 7}, {{1,3}, 6},
                                                          {{2,1}, 5}, {{2,2}, 4},
                                                          {{3,1}, 3}, {{3,2}, 2},
                                                          {{4,1}, 1}, {{4,2}, 0} };
  
  // Maps RPC staion and ring to the monitor element index and uses symmetry of the endcaps
  const std::map<std::pair<int, int>, int> histIndexRPC = { {{4,3}, 0}, {{4,2}, 1}, {{3,3}, 2}, {{3,2}, 3}, {{2,2}, 4}, {{1,2}, 5} };

  for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
    int endcap  = Hit->Endcap();
    int sector  = Hit->Sector();
    int station = Hit->Station();
    int ring    = Hit->Ring();
    int cscid   = Hit->CSC_ID();
    int chamber = Hit->Chamber();
    int strip   = Hit->Strip();
    int wire    = Hit->Wire();
    int cscid_offset = (sector - 1) * 9;

    int hist_index = 0;
    if (ring == 4 && strip >= 128) strip -= 128;

    if (Hit->Is_CSC() == true) {
      hist_index = histIndexCSC.at( {station, ring} );
      if (endcap > 0) hist_index = 19 - hist_index;
      histograms.cscLCTBX.fill(Hit->BX(), hist_index);
      float evt_wgt = (Hit->Station() > 1 && Hit->Ring() == 1) ? 0.5 : 1.0;
      if (Hit->Neighbor() == false) {
        //Map for cscDQMOccupancy plot
        histograms.cscDQMOccupancy.fill(chamber_bin(station,ring,chamber),hist_index,evt_wgt);
        if (station>1 && (ring % 2)==1) {
          histograms.cscDQMOccupancy.fill(chamber_bin(station,ring,chamber)-1,hist_index,evt_wgt);
        }         
        histograms.cscLCTStrip[hist_index].fill(strip);
        histograms.cscLCTWire[hist_index].fill(wire);
        histograms.cscChamberStrip[hist_index].fill(chamber, strip);
        histograms.cscChamberWire[hist_index].fill(chamber, wire);
        if (Hit->Subsector() == 1) {
          histograms.cscLCTOccupancy.fill(cscid + cscid_offset, endcap * (station - 0.5));
        } else {
          histograms.cscLCTOccupancy.fill(cscid + cscid_offset, endcap * (station + 0.5));
        }
      } else {
        // Map neighbor chambers to "fake" CSC IDs: 1/3 --> 1, 1/6 --> 2, 1/9 --> 3, 2/3 --> 4, 2/9 --> 5, etc.
        int cscid_n = (station == 1 ? (cscid / 3) : (station * 2) + ((cscid - 3) / 6) );
        histograms.cscLCTOccupancy.fill(cscid_n + cscid_offset, endcap * 5.5);
      } 
      if (Hit->Neighbor() == true) {
        histograms.cscDQMOccupancy.fill(sector*7-4,hist_index,evt_wgt);
      }  
    }

    if (Hit->Is_RPC() == true) {
      hist_index = histIndexRPC.at( {station, ring} );
      if (endcap > 0) hist_index = 11 - hist_index;

      histograms.rpcHitBX.fill(Hit->BX(), hist_index);

      if (Hit->Neighbor() == false) {
        histograms.rpcHitPhi[hist_index].fill(Hit->Phi_fp() / 4);
        histograms.rpcHitTheta[hist_index].fill(Hit->Theta_fp() / 4);
        histograms.rpcChamberPhi[hist_index].fill(chamber, Hit->Phi_fp() / 4);
        histograms.rpcChamberTheta[hist_index].fill(chamber, Hit->Theta_fp() / 4);
        histograms.rpcHitOccupancy.fill((Hit->Sector_RPC() - 1) * 7 + Hit->Subsector(), hist_index + 0.5);
      } else if (Hit->Neighbor() == true) {
        histograms.rpcHitOccupancy.fill((Hit->Sector_RPC() - 1) * 7 + 7, hist_index + 0.5);
      }
    }
  }

  // Tracks
  edm::Handle<l1t::EMTFTrackCollection> TrackCollection;
  e.getByToken(trackToken, TrackCollection);

  int nTracks = TrackCollection->size();

  histograms.emtfnTracks.fill(std::min(nTracks, emtfnTracksNbins - 1));

  for (auto Track = TrackCollection->begin(); Track != TrackCollection->end(); ++Track) {
    int endcap = Track->Endcap();
    int sector = Track->Sector();
    float eta = Track->Eta();
    float phi_glob_rad = Track->Phi_glob() * M_PI / 180.;
    int mode = Track->Mode();
    int quality = Track->GMT_quality();
    int numHits = Track->NumHits();
    int modeNeighbor = Track->Mode_neighbor();

    histograms.emtfTracknHits.fill(numHits);
    histograms.emtfTrackBX.fill(endcap * (sector - 0.5), Track->BX());
    histograms.emtfTrackPt.fill(Track->Pt());
    histograms.emtfTrackEta.fill(eta);

    histograms.emtfTrackOccupancy.fill(eta, phi_glob_rad);
    histograms.emtfTrackMode.fill(mode);
    histograms.emtfTrackQuality.fill(quality);
    histograms.emtfTrackQualityVsMode.fill(mode, quality);

    // Only plot if there are <= 1 neighbor hits in the track to avoid spikes at sector boundaries
    if (modeNeighbor < 2 || modeNeighbor == 4 || modeNeighbor == 8) {
      histograms.emtfTrackPhi.fill(phi_glob_rad);
      if (quality >= 12) {
        histograms.emtfTrackPhiHighQuality.fill(phi_glob_rad);
      }
    }

    ////////////////////////////////////////////////////
    ///  Begin block for CSC LCT and RPC hit timing  ///
    ////////////////////////////////////////////////////
    { 
      // LCT and RPC Timing
      if (numHits < 2 || numHits > 4) continue;
      l1t::EMTFHitCollection tmp_hits = Track->Hits();
      int numHitsInTrack_BX0 = 0;
      unsigned int hist_index2 = 4 - numHits;

      for (const auto & iTrkHit: Track->Hits()) {
        if (iTrkHit.Is_CSC() == true) {
          histograms.emtfTrackBXVsCSCLCT[hist_index2].fill(iTrkHit.BX(), Track->BX());
        }
        else if (iTrkHit.Is_RPC() == true) {
          histograms.emtfTrackBXVsRPCHit[hist_index2].fill(iTrkHit.BX(), Track->BX());
        }
      }

      // Select well-timed tracks: >= 3 hits, with <= 1 in BX != 0
      if (numHits < 3) continue;
      for (const auto & jTrkHit: Track->Hits()) {
        if (jTrkHit.BX() == 0)
          numHitsInTrack_BX0++;
      }
      if (numHitsInTrack_BX0 < numHits - 1) continue;

      for (const auto & TrkHit: Track->Hits()) {

        int trackHitBX   = TrkHit.BX();
        //int cscid        = TrkHit.CSC_ID();
        int ring         = TrkHit.Ring();
        int station      = TrkHit.Station();
        int sector       = TrkHit.Sector();
        int subsector    = TrkHit.Subsector();
        //int cscid_offset = (sector - 1) * 9;//no longer needed after new time plots (maybe useful for future plots)
        int neighbor     = TrkHit.Neighbor();
        int endcap       = TrkHit.Endcap();
        int chamber      = TrkHit.Chamber();

        int hist_index = 0;
        float evt_wgt = (TrkHit.Station() > 1 && TrkHit.Ring() == 1) ? 0.5 : 1.0;
        // Maps CSC BX from -2 to 2 to monitor element cscLCTTIming
        const std::map<int, int> histIndexBX = {{0, 4}, {-1, 0}, {1, 1}, {-2, 2}, {2, 3}};
        if (std::abs(trackHitBX) > 2) continue; // Should never happen, but just to be safe ...

        if (TrkHit.Is_CSC() == true) {
          hist_index = histIndexCSC.at( {station, ring} );
          if (endcap > 0) hist_index = 19 - hist_index;
          if (neighbor == false) {
            histograms.cscLCTTiming[histIndexBX.at(trackHitBX)].fill(chamber_bin(station,ring,chamber),hist_index,evt_wgt);
            histograms.cscTimingTot.fill(chamber_bin(station,ring,chamber),hist_index,evt_wgt);
            if (station>1 && (ring % 2)==1) {
              histograms.cscLCTTiming[histIndexBX.at(trackHitBX)].fill(chamber_bin(station,ring,chamber)-1,hist_index,evt_wgt);
              histograms.cscTimingTot.fill(chamber_bin(station,ring,chamber)-1,hist_index,evt_wgt);
            }
          }
          else {
            // Map neighbor chambers to "fake" CSC IDs: 1/3 --> 1, 1/6 --> 2, 1/9 --> 3, 2/3 --> 4, 2/9 --> 5, etc.
            //int cscid_n = (station == 1 ? (cscid / 3) : (station * 2) + ((cscid - 3) / 6) );
            histograms.cscLCTTiming[histIndexBX.at(trackHitBX)].fill(sector*7-4,hist_index,evt_wgt);
            histograms.cscTimingTot.fill(sector*7-4,hist_index,evt_wgt);
          }

          // Fill RPC timing with matched CSC LCTs
          if (trackHitBX == 0 && ring == 2) {
            for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit) {
              if ( Hit->Is_RPC() == false || neighbor == true ) continue;
              if ( std::abs(Track->Eta() - Hit->Eta()) > 0.1  ) continue;
              if ( Hit->Endcap()  != endcap  ||
                   Hit->Station() != station ||
                   Hit->Chamber() != chamber ) continue;
              if ( std::abs(Hit->BX()) > 2   ) continue;

              hist_index = histIndexRPC.at( {Hit->Station(), Hit->Ring()} );
              if (Hit->Endcap() > 0) hist_index = 11 - hist_index;
              histograms.rpcHitTimingInTrack.fill(Hit->BX(), hist_index + 0.5);
              histograms.rpcHitTiming[histIndexBX.at(Hit->BX())].fill((Hit->Sector_RPC() - 1) * 7 + Hit->Subsector(), hist_index + 0.5);
              histograms.rpcHitTimingTot.fill((Hit->Sector_RPC() - 1) * 7 + Hit->Subsector(), hist_index + 0.5);
            } // End loop: for (auto Hit = HitCollection->begin(); Hit != HitCollection->end(); ++Hit)
          } // End conditional: if (trackHitBX == 0 && ring == 2)
        } // End conditional: if (TrkHit.Is_CSC() == true)

        // Maps RPC station and ring to monitor element index
        const std::map<std::pair<int, int>, int> histIndexRPC = { {{4,3}, 0}, {{4,2}, 1}, {{3,3}, 2}, {{3,2}, 3}, {{2,2}, 4}, {{1,2}, 5}};

        if (TrkHit.Is_RPC() == true && neighbor == false) {
          hist_index = histIndexRPC.at( {station, ring} );
          if (endcap > 0) hist_index = 11 - hist_index;

          histograms.rpcHitTimingInTrack.fill(trackHitBX, hist_index + 0.5);
          histograms.rpcHitTiming[histIndexBX.at(trackHitBX)].fill((TrkHit.Sector_RPC() - 1) * 7 + subsector, hist_index + 0.5);
          histograms.rpcHitTimingTot.fill((TrkHit.Sector_RPC() - 1) * 7 + subsector, hist_index + 0.5);
        } // End conditional: if (TrkHit.Is_RPC() == true && neighbor == false)
        if (TrkHit.Is_RPC() == true && neighbor == true) {
          hist_index = histIndexRPC.at( {station, ring} );
          if (endcap > 0) hist_index = 11 - hist_index;
          histograms.rpcHitTiming[histIndexBX.at(trackHitBX)].fill((TrkHit.Sector_RPC() - 1) * 7, hist_index + 0.5);
        }
      } // End loop: for (int iHit = 0; iHit < numHits; ++iHit)
    } 
    //////////////////////////////////////////////////
    ///  End block for CSC LCT and RPC hit timing  ///
    //////////////////////////////////////////////////
    
  } // End loop: for (auto Track = TrackCollection->begin(); Track != TrackCollection->end(); ++Track)
  
  // Regional Muon Candidates
  edm::Handle<l1t::RegionalMuonCandBxCollection> MuonBxCollection;
  e.getByToken(muonToken, MuonBxCollection);
  
  for (int itBX = MuonBxCollection->getFirstBX(); itBX <= MuonBxCollection->getLastBX(); ++itBX) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator Muon = MuonBxCollection->begin(itBX); Muon != MuonBxCollection->end(itBX); ++Muon) {
      histograms.emtfMuonBX.fill(itBX);
      histograms.emtfMuonhwPt.fill(Muon->hwPt());
      histograms.emtfMuonhwEta.fill(Muon->hwEta());
      histograms.emtfMuonhwPhi.fill(Muon->hwPhi());
      histograms.emtfMuonhwQual.fill(Muon->hwQual());
    }
  }
}
