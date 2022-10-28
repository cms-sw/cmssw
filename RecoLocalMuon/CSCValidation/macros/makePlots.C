// makePlots.C makes plots from root file output of CSCValidation

// Original author: Andy Kubik (NWU)
// - Updated by many people over the past 15 years
// - This version from Sicheng Wang (UCSB) - Jul 2022
// - but with addition of rechit and segment positions from Andy's version
// - Please contact CSC DPG for current status - 12.07.2022

#include "myFunctions.C"

void makePlots(std::string input_file) {
  extern TFile* OpenFiles(std::string path);
  extern void makeCSCOccupancy(std::string histoname, TFile * f1, std::string histotitle, std::string savename);
  extern void Draw2DTempPlot(
      std::string histo, TFile * f1, bool includeME11, std::string savename, bool hasLabels = false);
  extern void make1DPlot(
      std::string histoname, TFile * f1, std::string histotitle, int statoption, std::string savename);
  extern void make1DPlot(std::string histoname,
                         TFile * f1,
                         std::string histotitle,
                         std::string xtitle,
                         std::string ytitle,
                         int statoption,
                         std::string savename);
  extern void printEmptyChambers(std::string histoname, std::string oname, TFile * f);
  extern void GlobalPosfromTree(
      std::string graphname, TFile * f1, int endcap, int station, std::string type, std::string savename);
  extern void NikolaiPlots(TFile * f_in, int flag);
  extern void makeEffGif(std::string histoname, TFile * f1, std::string histotitle, std::string savename);
  extern void Draw2DEfficiency(std::string histo, TFile * f1, std::string title, std::string savename);
  extern void make2DPlot(
      std::string histoname, TFile * f1, std::string histotitle, int statoption, std::string savename);
  extern void make2DPlot(std::string histoname,
                         TFile * f1,
                         std::string histotitle,
                         std::string xtitle,
                         std::string ytitle,
                         int statoption,
                         std::string savename);
  extern void makeProfile(
      std::string histoname, TFile * f1, std::string histotitle, int statoption, std::string savename);
  extern void makeProfile(std::string histoname,
                          TFile * f1,
                          std::string histotitle,
                          std::string xtitle,
                          std::string ytitle,
                          int statoption,
                          std::string savename);

  TFile* f1;
  f1 = OpenFiles(input_file);

  //produce occupancy plots
  makeCSCOccupancy("GeneralHists/hCSCOccupancy", f1, "CSC Occupancy", "CSCOccupancy.png");
  Draw2DTempPlot("Digis/hOWires", f1, true, "hOWires.png");
  Draw2DTempPlot("Digis/hOStrips", f1, true, "hOStrips.png");
  Draw2DTempPlot("recHits/hORecHits", f1, true, "hORecHits.png");
  Draw2DTempPlot("Segments/hOSegments", f1, true, "hOSegments.png");
  make1DPlot("Digis/hOStripSerial", f1, "Strip Occupancy by Chamber Serial", 10, "hOStripSerial.png");
  make1DPlot("Digis/hOWireSerial", f1, "Wire Occupancy by Chamber Serial", 10, "hOWireSerial.png");
  make1DPlot("Segments/hOSegmentsSerial", f1, "Segment Occupancy by Chamber Serial", 10, "hOSegmentsSerial.png");
  make1DPlot("recHits/hORecHitsSerial", f1, "recHit Occupancy by Chamber Serial", 10, "hORecHitsSerial.png");
  make1DPlot("Trigger/hltBits", f1, "HLT Bits Fired", 0, "hltTriggerBits.png");

  //Print dead chamber lists
  printEmptyChambers("Digis/hOWires", "wire digis", f1);
  printEmptyChambers("Digis/hOStrips", "strip digis", f1);
  printEmptyChambers("recHits/hORecHits", "rechits", f1);

  //Make global position graphs from trees
  GlobalPosfromTree("Global recHit positions ME+1", f1, 1, 1, "rechit", "rHglobal_station_+1.png");
  GlobalPosfromTree("Global recHit positions ME+2", f1, 1, 2, "rechit", "rHglobal_station_+2.png");
  GlobalPosfromTree("Global recHit positions ME+3", f1, 1, 3, "rechit", "rHglobal_station_+3.png");
  GlobalPosfromTree("Global recHit positions ME+4", f1, 1, 4, "rechit", "rHglobal_station_+4.png");
  GlobalPosfromTree("Global recHit positions ME-1", f1, 2, 1, "rechit", "rHglobal_station_-1.png");
  GlobalPosfromTree("Global recHit positions ME-2", f1, 2, 2, "rechit", "rHglobal_station_-2.png");
  GlobalPosfromTree("Global recHit positions ME-3", f1, 2, 3, "rechit", "rHglobal_station_-3.png");
  GlobalPosfromTree("Global recHit positions ME-4", f1, 2, 4, "rechit", "rHglobal_station_-4.png");
  GlobalPosfromTree("Global Segment positions ME+1", f1, 1, 1, "segment", "Sglobal_station_+1.png");
  GlobalPosfromTree("Global Segment positions ME+2", f1, 1, 2, "segment", "Sglobal_station_+2.png");
  GlobalPosfromTree("Global Segment positions ME+3", f1, 1, 3, "segment", "Sglobal_station_+3.png");
  GlobalPosfromTree("Global Segment positions ME+4", f1, 1, 4, "segment", "Sglobal_station_+4.png");
  GlobalPosfromTree("Global Segment positions ME-1", f1, 2, 1, "segment", "Sglobal_station_-1.png");
  GlobalPosfromTree("Global Segment positions ME-2", f1, 2, 2, "segment", "Sglobal_station_-2.png");
  GlobalPosfromTree("Global Segment positions ME-3", f1, 2, 3, "segment", "Sglobal_station_-3.png");
  GlobalPosfromTree("Global Segment positions ME-4", f1, 2, 4, "segment", "Sglobal_station_-4.png");

  //Nikolai's plots
  NikolaiPlots(f1, 1);
  NikolaiPlots(f1, 2);
  NikolaiPlots(f1, 3);
  NikolaiPlots(f1, 4);

  //produce number of X per event plots
  make1DPlot("Digis/hStripNFired", f1, "Fired Strips per Event", 111110, "Digis_hStripNFired.png");
  make1DPlot("Digis/hWirenGroupsTotal", f1, "Fired Wires per Event", 111110, "Digis_hWirenGroupsTotal.png");
  make1DPlot("recHits/hRHnrechits", f1, "RecHits per Event", 111110, "recHits_hRHnrechits.png");
  make1DPlot("Segments/hSnSegments", f1, "Segments per Event", 111110, "Segments_hSnSegments.png");

  //produce wire timing plots
  make1DPlot("Digis/hWireTBin+11", f1, "Wire TimeBin Fired ME+1/1", 111110, "Digis_hWireTBin+11.png");
  make1DPlot("Digis/hWireTBin+12", f1, "Wire TimeBin Fired ME+1/2", 111110, "Digis_hWireTBin+12.png");
  make1DPlot("Digis/hWireTBin+13", f1, "Wire TimeBin Fired ME+1/3", 111110, "Digis_hWireTBin+13.png");
  make1DPlot("Digis/hWireTBin+21", f1, "Wire TimeBin Fired ME+2/1", 111110, "Digis_hWireTBin+21.png");
  make1DPlot("Digis/hWireTBin+22", f1, "Wire TimeBin Fired ME+2/2", 111110, "Digis_hWireTBin+22.png");
  make1DPlot("Digis/hWireTBin+31", f1, "Wire TimeBin Fired ME+3/1", 111110, "Digis_hWireTBin+31.png");
  make1DPlot("Digis/hWireTBin+32", f1, "Wire TimeBin Fired ME+3/2", 111110, "Digis_hWireTBin+32.png");
  make1DPlot("Digis/hWireTBin+41", f1, "Wire TimeBin Fired ME+4/1", 111110, "Digis_hWireTBin+41.png");
  make1DPlot("Digis/hWireTBin+42", f1, "Wire TimeBin Fired ME+4/2", 111110, "Digis_hWireTBin+42.png");
  make1DPlot("Digis/hWireTBin-11", f1, "Wire TimeBin Fired ME-1/1", 111110, "Digis_hWireTBin-11.png");
  make1DPlot("Digis/hWireTBin-12", f1, "Wire TimeBin Fired ME-1/2", 111110, "Digis_hWireTBin-12.png");
  make1DPlot("Digis/hWireTBin-13", f1, "Wire TimeBin Fired ME-1/3", 111110, "Digis_hWireTBin-13.png");
  make1DPlot("Digis/hWireTBin-21", f1, "Wire TimeBin Fired ME-2/1", 111110, "Digis_hWireTBin-21.png");
  make1DPlot("Digis/hWireTBin-22", f1, "Wire TimeBin Fired ME-2/2", 111110, "Digis_hWireTBin-22.png");
  make1DPlot("Digis/hWireTBin-31", f1, "Wire TimeBin Fired ME-3/1", 111110, "Digis_hWireTBin-31.png");
  make1DPlot("Digis/hWireTBin-32", f1, "Wire TimeBin Fired ME-3/2", 111110, "Digis_hWireTBin-32.png");
  make1DPlot("Digis/hWireTBin-41", f1, "Wire TimeBin Fired ME-4/1", 111110, "Digis_hWireTBin-41.png");
  make1DPlot("Digis/hWireTBin-42", f1, "Wire TimeBin Fired ME-4/2", 111110, "Digis_hWireTBin-42.png");

  //produce pedestal noise plots
  make1DPlot("PedestalNoise/hStripPedME+11",
             f1,
             "Pedestal Noise Distribution ME+1/1b",
             111110,
             "PedestalNoise_hStripPedME+11.png");
  make1DPlot("PedestalNoise/hStripPedME+14",
             f1,
             "Pedestal Noise Distribution ME+1/1a",
             111110,
             "PedestalNoise_hStripPedME+11a.png");
  make1DPlot("PedestalNoise/hStripPedME+12",
             f1,
             "Pedestal Noise Distribution ME+1/2",
             111110,
             "PedestalNoise_hStripPedME+12.png");
  make1DPlot("PedestalNoise/hStripPedME+13",
             f1,
             "Pedestal Noise Distribution ME+1/3",
             111110,
             "PedestalNoise_hStripPedME+13.png");
  make1DPlot("PedestalNoise/hStripPedME+21",
             f1,
             "Pedestal Noise Distribution ME+2/1",
             111110,
             "PedestalNoise_hStripPedME+21.png");
  make1DPlot("PedestalNoise/hStripPedME+22",
             f1,
             "Pedestal Noise Distribution ME+2/2",
             111110,
             "PedestalNoise_hStripPedME+22.png");
  make1DPlot("PedestalNoise/hStripPedME+31",
             f1,
             "Pedestal Noise Distribution ME+3/1",
             111110,
             "PedestalNoise_hStripPedME+31.png");
  make1DPlot("PedestalNoise/hStripPedME+32",
             f1,
             "Pedestal Noise Distribution ME+3/2",
             111110,
             "PedestalNoise_hStripPedME+32.png");
  make1DPlot("PedestalNoise/hStripPedME+41",
             f1,
             "Pedestal Noise Distribution ME+4/1",
             111110,
             "PedestalNoise_hStripPedME+41.png");
  make1DPlot("PedestalNoise/hStripPedME+42",
             f1,
             "Pedestal Noise Distribution ME+4/2",
             111110,
             "PedestalNoise_hStripPedME+42.png");
  make1DPlot("PedestalNoise/hStripPedME-11",
             f1,
             "Pedestal Noise Distribution ME-1/1b",
             111110,
             "PedestalNoise_hStripPedME-11.png");
  make1DPlot("PedestalNoise/hStripPedME-14",
             f1,
             "Pedestal Noise Distribution ME-1/1a",
             111110,
             "PedestalNoise_hStripPedME-11a.png");
  make1DPlot("PedestalNoise/hStripPedME-12",
             f1,
             "Pedestal Noise Distribution ME-1/2",
             111110,
             "PedestalNoise_hStripPedME-12.png");
  make1DPlot("PedestalNoise/hStripPedME-13",
             f1,
             "Pedestal Noise Distribution ME-1/3",
             111110,
             "PedestalNoise_hStripPedME-13.png");
  make1DPlot("PedestalNoise/hStripPedME-21",
             f1,
             "Pedestal Noise Distribution ME-2/1",
             111110,
             "PedestalNoise_hStripPedME-21.png");
  make1DPlot("PedestalNoise/hStripPedME-22",
             f1,
             "Pedestal Noise Distribution ME-2/2",
             111110,
             "PedestalNoise_hStripPedME-22.png");
  make1DPlot("PedestalNoise/hStripPedME-31",
             f1,
             "Pedestal Noise Distribution ME-3/1",
             111110,
             "PedestalNoise_hStripPedME-31.png");
  make1DPlot("PedestalNoise/hStripPedME-32",
             f1,
             "Pedestal Noise Distribution ME-3/2",
             111110,
             "PedestalNoise_hStripPedME-32.png");
  make1DPlot("PedestalNoise/hStripPedME-41",
             f1,
             "Pedestal Noise Distribution ME-4/1",
             111110,
             "PedestalNoise_hStripPedME-41.png");
  make1DPlot("PedestalNoise/hStripPedME-42",
             f1,
             "Pedestal Noise Distribution ME-4/2",
             111110,
             "PedestalNoise_hStripPedME-42.png");

  // resolution
  make1DPlot("Resolution/hSResid+11",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/1b",
             111110,
             "Resolution_hSResid+11.png");
  make1DPlot("Resolution/hSResid+12",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/2",
             111110,
             "Resolution_hSResid+12.png");
  make1DPlot("Resolution/hSResid+13",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/3",
             111110,
             "Resolution_hSResid+13.png");
  make1DPlot("Resolution/hSResid+14",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/1a",
             111110,
             "Resolution_hSResid+11a.png");
  make1DPlot("Resolution/hSResid+21",
             f1,
             "Expected Position from Fit - Reconstructed, ME+2/1",
             111110,
             "Resolution_hSResid+21.png");
  make1DPlot("Resolution/hSResid+22",
             f1,
             "Expected Position from Fit - Reconstructed, ME+2/2",
             111110,
             "Resolution_hSResid+22.png");
  make1DPlot("Resolution/hSResid+31",
             f1,
             "Expected Position from Fit - Reconstructed, ME+3/1",
             111110,
             "Resolution_hSResid+31.png");
  make1DPlot("Resolution/hSResid+32",
             f1,
             "Expected Position from Fit - Reconstructed, ME+3/2",
             111110,
             "Resolution_hSResid+32.png");
  make1DPlot("Resolution/hSResid+41",
             f1,
             "Expected Position from Fit - Reconstructed, ME+4/1",
             111110,
             "Resolution_hSResid+41.png");
  make1DPlot("Resolution/hSResid+42",
             f1,
             "Expected Position from Fit - Reconstructed, ME+4/2",
             111110,
             "Resolution_hSResid+42.png");
  make1DPlot("Resolution/hSResid-11",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/1b",
             111110,
             "Resolution_hSResid-11.png");
  make1DPlot("Resolution/hSResid-12",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/2",
             111110,
             "Resolution_hSResid-12.png");
  make1DPlot("Resolution/hSResid-13",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/3",
             111110,
             "Resolution_hSResid-13.png");
  make1DPlot("Resolution/hSResid-14",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/1a",
             111110,
             "Resolution_hSResid-11a.png");
  make1DPlot("Resolution/hSResid-21",
             f1,
             "Expected Position from Fit - Reconstructed, ME-2/1",
             111110,
             "Resolution_hSResid-21.png");
  make1DPlot("Resolution/hSResid-22",
             f1,
             "Expected Position from Fit - Reconstructed, ME-2/2",
             111110,
             "Resolution_hSResid-22.png");
  make1DPlot("Resolution/hSResid-31",
             f1,
             "Expected Position from Fit - Reconstructed, ME-3/1",
             111110,
             "Resolution_hSResid-31.png");
  make1DPlot("Resolution/hSResid-32",
             f1,
             "Expected Position from Fit - Reconstructed, ME-3/2",
             111110,
             "Resolution_hSResid-32.png");
  make1DPlot("Resolution/hSResid-41",
             f1,
             "Expected Position from Fit - Reconstructed, ME-4/1",
             111110,
             "Resolution_hSResid-41.png");
  make1DPlot("Resolution/hSResid-42",
             f1,
             "Expected Position from Fit - Reconstructed, ME-4/2",
             111110,
             "Resolution_hSResid-42.png");

  // rechit strip position
  make1DPlot("recHits/hRHstpos+11", f1, "Strip Position (ME+1/1b)", 1110, "recHits_hRHstpos+11.png");
  make1DPlot("recHits/hRHstpos+14", f1, "Strip Position (ME+1/1a)", 1110, "recHits_hRHstpos+11a.png");
  make1DPlot("recHits/hRHstpos+12", f1, "Strip Position (ME+1/2)", 1110, "recHits_hRHstpos+12.png");
  make1DPlot("recHits/hRHstpos+13", f1, "Strip Position (ME+1/3)", 1110, "recHits_hRHstpos+13.png");
  make1DPlot("recHits/hRHstpos+21", f1, "Strip Position (ME+2/1)", 1110, "recHits_hRHstpos+21.png");
  make1DPlot("recHits/hRHstpos+22", f1, "Strip Position (ME+2/2)", 1110, "recHits_hRHstpos+22.png");
  make1DPlot("recHits/hRHstpos+31", f1, "Strip Position (ME+3/1)", 1110, "recHits_hRHstpos+31.png");
  make1DPlot("recHits/hRHstpos+32", f1, "Strip Position (ME+3/2)", 1110, "recHits_hRHstpos+32.png");
  make1DPlot("recHits/hRHstpos+41", f1, "Strip Position (ME+4/1)", 1110, "recHits_hRHstpos+41.png");
  make1DPlot("recHits/hRHstpos+42", f1, "Strip Position (ME+4/2)", 1110, "recHits_hRHstpos+42.png");
  make1DPlot("recHits/hRHstpos-11", f1, "Strip Position (ME-1/1b)", 1110, "recHits_hRHstpos-11.png");
  make1DPlot("recHits/hRHstpos-14", f1, "Strip Position (ME-1/1a)", 1110, "recHits_hRHstpos-11a.png");
  make1DPlot("recHits/hRHstpos-12", f1, "Strip Position (ME-1/2)", 1110, "recHits_hRHstpos-12.png");
  make1DPlot("recHits/hRHstpos-13", f1, "Strip Position (ME-1/3)", 1110, "recHits_hRHstpos-13.png");
  make1DPlot("recHits/hRHstpos-21", f1, "Strip Position (ME-2/1)", 1110, "recHits_hRHstpos-21.png");
  make1DPlot("recHits/hRHstpos-22", f1, "Strip Position (ME-2/2)", 1110, "recHits_hRHstpos-22.png");
  make1DPlot("recHits/hRHstpos-31", f1, "Strip Position (ME-3/1)", 1110, "recHits_hRHstpos-31.png");
  make1DPlot("recHits/hRHstpos-32", f1, "Strip Position (ME-3/2)", 1110, "recHits_hRHstpos-32.png");
  make1DPlot("recHits/hRHstpos-41", f1, "Strip Position (ME-4/1)", 1110, "recHits_hRHstpos-41.png");
  make1DPlot("recHits/hRHstpos-42", f1, "Strip Position (ME-4/2)", 1110, "recHits_hRHstpos-42.png");

  // rechit timing
  make1DPlot("recHits/hRHTiming+11", f1, "RecHit Timing ME+1/1b", 111110, "recHits_hRHTiming+11.png");
  make1DPlot("recHits/hRHTiming+14", f1, "RecHit Timing ME+1/1a", 111110, "recHits_hRHTiming+11a.png");
  make1DPlot("recHits/hRHTiming+12", f1, "RecHit Timing ME+1/2", 111110, "recHits_hRHTiming+12.png");
  make1DPlot("recHits/hRHTiming+13", f1, "RecHit Timing ME+1/3", 111110, "recHits_hRHTiming+13.png");
  make1DPlot("recHits/hRHTiming+21", f1, "RecHit Timing ME+2/1", 111110, "recHits_hRHTiming+21.png");
  make1DPlot("recHits/hRHTiming+22", f1, "RecHit Timing ME+2/2", 111110, "recHits_hRHTiming+22.png");
  make1DPlot("recHits/hRHTiming+31", f1, "RecHit Timing ME+3/1", 111110, "recHits_hRHTiming+31.png");
  make1DPlot("recHits/hRHTiming+32", f1, "RecHit Timing ME+3/2", 111110, "recHits_hRHTiming+32.png");
  make1DPlot("recHits/hRHTiming+41", f1, "RecHit Timing ME+4/1", 111110, "recHits_hRHTiming+41.png");
  make1DPlot("recHits/hRHTiming+42", f1, "RecHit Timing ME+4/2", 111110, "recHits_hRHTiming+42.png");
  make1DPlot("recHits/hRHTiming-11", f1, "RecHit Timing ME-1/1b", 111110, "recHits_hRHTiming-11.png");
  make1DPlot("recHits/hRHTiming-14", f1, "RecHit Timing ME-1/1a", 111110, "recHits_hRHTiming-11a.png");
  make1DPlot("recHits/hRHTiming-12", f1, "RecHit Timing ME-1/2", 111110, "recHits_hRHTiming-12.png");
  make1DPlot("recHits/hRHTiming-13", f1, "RecHit Timing ME-1/3", 111110, "recHits_hRHTiming-13.png");
  make1DPlot("recHits/hRHTiming-21", f1, "RecHit Timing ME-2/1", 111110, "recHits_hRHTiming-21.png");
  make1DPlot("recHits/hRHTiming-22", f1, "RecHit Timing ME-2/2", 111110, "recHits_hRHTiming-22.png");
  make1DPlot("recHits/hRHTiming-31", f1, "RecHit Timing ME-3/1", 111110, "recHits_hRHTiming-31.png");
  make1DPlot("recHits/hRHTiming-32", f1, "RecHit Timing ME-3/2", 111110, "recHits_hRHTiming-32.png");
  make1DPlot("recHits/hRHTiming-41", f1, "RecHit Timing ME-4/1", 111110, "recHits_hRHTiming-41.png");
  make1DPlot("recHits/hRHTiming-42", f1, "RecHit Timing ME-4/2", 111110, "recHits_hRHTiming-42.png");

  // rechit charge
  make1DPlot("recHits/hRHSumQ+11", f1, "Sum 3x3 RecHit Charge ME+1/1b", 111110, "recHits_hRHSumQ+11.png");
  make1DPlot("recHits/hRHSumQ+14", f1, "Sum 3x3 RecHit Charge ME+1/1a", 111110, "recHits_hRHSumQ+11a.png");
  make1DPlot("recHits/hRHSumQ+12", f1, "Sum 3x3 RecHit Charge ME+1/2", 111110, "recHits_hRHSumQ+12.png");
  make1DPlot("recHits/hRHSumQ+13", f1, "Sum 3x3 RecHit Charge ME+1/3", 111110, "recHits_hRHSumQ+13.png");
  make1DPlot("recHits/hRHSumQ+21", f1, "Sum 3x3 RecHit Charge ME+2/1", 111110, "recHits_hRHSumQ+21.png");
  make1DPlot("recHits/hRHSumQ+22", f1, "Sum 3x3 RecHit Charge ME+2/2", 111110, "recHits_hRHSumQ+22.png");
  make1DPlot("recHits/hRHSumQ+31", f1, "Sum 3x3 RecHit Charge ME+3/1", 111110, "recHits_hRHSumQ+31.png");
  make1DPlot("recHits/hRHSumQ+32", f1, "Sum 3x3 RecHit Charge ME+3/2", 111110, "recHits_hRHSumQ+32.png");
  make1DPlot("recHits/hRHSumQ+41", f1, "Sum 3x3 RecHit Charge ME+4/1", 111110, "recHits_hRHSumQ+41.png");
  make1DPlot("recHits/hRHSumQ+42", f1, "Sum 3x3 RecHit Charge ME+4/2", 111110, "recHits_hRHSumQ+42.png");
  make1DPlot("recHits/hRHSumQ-11", f1, "Sum 3x3 RecHit Charge ME-1/1b", 111110, "recHits_hRHSumQ-11.png");
  make1DPlot("recHits/hRHSumQ-14", f1, "Sum 3x3 RecHit Charge ME-1/1a", 111110, "recHits_hRHSumQ-11a.png");
  make1DPlot("recHits/hRHSumQ-12", f1, "Sum 3x3 RecHit Charge ME-1/2", 111110, "recHits_hRHSumQ-12.png");
  make1DPlot("recHits/hRHSumQ-13", f1, "Sum 3x3 RecHit Charge ME-1/3", 111110, "recHits_hRHSumQ-13.png");
  make1DPlot("recHits/hRHSumQ-21", f1, "Sum 3x3 RecHit Charge ME-2/1", 111110, "recHits_hRHSumQ-21.png");
  make1DPlot("recHits/hRHSumQ-22", f1, "Sum 3x3 RecHit Charge ME-2/2", 111110, "recHits_hRHSumQ-22.png");
  make1DPlot("recHits/hRHSumQ-31", f1, "Sum 3x3 RecHit Charge ME-3/1", 111110, "recHits_hRHSumQ-31.png");
  make1DPlot("recHits/hRHSumQ-32", f1, "Sum 3x3 RecHit Charge ME-3/2", 111110, "recHits_hRHSumQ-32.png");
  make1DPlot("recHits/hRHSumQ-41", f1, "Sum 3x3 RecHit Charge ME-4/1", 111110, "recHits_hRHSumQ-41.png");
  make1DPlot("recHits/hRHSumQ-42", f1, "Sum 3x3 RecHit Charge ME-4/2", 111110, "recHits_hRHSumQ-42.png");

  make1DPlot("recHits/hRHRatioQ+11", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/1b", 111110, "recHits_hRHRatioQ+11.png");
  make1DPlot("recHits/hRHRatioQ+14", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/1a", 111110, "recHits_hRHRatioQ+11a.png");
  make1DPlot("recHits/hRHRatioQ+12", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/2", 111110, "recHits_hRHRatioQ+12.png");
  make1DPlot("recHits/hRHRatioQ+13", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/3", 111110, "recHits_hRHRatioQ+13.png");
  make1DPlot("recHits/hRHRatioQ+21", f1, "Charge Ratio (Ql_Qr)/Qt ME+2/1", 111110, "recHits_hRHRatioQ+21.png");
  make1DPlot("recHits/hRHRatioQ+22", f1, "Charge Ratio (Ql_Qr)/Qt ME+2/2", 111110, "recHits_hRHRatioQ+22.png");
  make1DPlot("recHits/hRHRatioQ+31", f1, "Charge Ratio (Ql_Qr)/Qt ME+3/1", 111110, "recHits_hRHRatioQ+31.png");
  make1DPlot("recHits/hRHRatioQ+32", f1, "Charge Ratio (Ql_Qr)/Qt ME+3/2", 111110, "recHits_hRHRatioQ+32.png");
  make1DPlot("recHits/hRHRatioQ+41", f1, "Charge Ratio (Ql_Qr)/Qt ME+4/1", 111110, "recHits_hRHRatioQ+41.png");
  make1DPlot("recHits/hRHRatioQ+42", f1, "Charge Ratio (Ql_Qr)/Qt ME+4/2", 111110, "recHits_hRHRatioQ+42.png");
  make1DPlot("recHits/hRHRatioQ-11", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/1b", 111110, "recHits_hRHRatioQ-11.png");
  make1DPlot("recHits/hRHRatioQ-14", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/1a", 111110, "recHits_hRHRatioQ-11a.png");
  make1DPlot("recHits/hRHRatioQ-12", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/2", 111110, "recHits_hRHRatioQ-12.png");
  make1DPlot("recHits/hRHRatioQ-13", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/3", 111110, "recHits_hRHRatioQ-13.png");
  make1DPlot("recHits/hRHRatioQ-21", f1, "Charge Ratio (Ql_Qr)/Qt ME-2/1", 111110, "recHits_hRHRatioQ-21.png");
  make1DPlot("recHits/hRHRatioQ-22", f1, "Charge Ratio (Ql_Qr)/Qt ME-2/2", 111110, "recHits_hRHRatioQ-22.png");
  make1DPlot("recHits/hRHRatioQ-31", f1, "Charge Ratio (Ql_Qr)/Qt ME-3/1", 111110, "recHits_hRHRatioQ-31.png");
  make1DPlot("recHits/hRHRatioQ-32", f1, "Charge Ratio (Ql_Qr)/Qt ME-3/2", 111110, "recHits_hRHRatioQ-32.png");
  make1DPlot("recHits/hRHRatioQ-41", f1, "Charge Ratio (Ql_Qr)/Qt ME-4/1", 111110, "recHits_hRHRatioQ-41.png");
  make1DPlot("recHits/hRHRatioQ-42", f1, "Charge Ratio (Ql_Qr)/Qt ME-4/2", 111110, "recHits_hRHRatioQ-42.png");

  //hits on a segment
  make1DPlot("Segments/hSnHits+11", f1, "N Hits on Segments ME+1/1b", 1110, "Segments_hSnHits+11.png");
  make1DPlot("Segments/hSnHits+14", f1, "N Hits on Segments ME+1/1a", 1110, "Segments_hSnHits+11a.png");
  make1DPlot("Segments/hSnHits+12", f1, "N Hits on Segments ME+1/2", 1110, "Segments_hSnHits+12.png");
  make1DPlot("Segments/hSnHits+13", f1, "N Hits on Segments ME+1/3", 1110, "Segments_hSnHits+13.png");
  make1DPlot("Segments/hSnHits+21", f1, "N Hits on Segments ME+2/1", 1110, "Segments_hSnHits+21.png");
  make1DPlot("Segments/hSnHits+22", f1, "N Hits on Segments ME+2/2", 1110, "Segments_hSnHits+22.png");
  make1DPlot("Segments/hSnHits+31", f1, "N Hits on Segments ME+3/1", 1110, "Segments_hSnHits+31.png");
  make1DPlot("Segments/hSnHits+32", f1, "N Hits on Segments ME+3/2", 1110, "Segments_hSnHits+32.png");
  make1DPlot("Segments/hSnHits+41", f1, "N Hits on Segments ME+4/1", 1110, "Segments_hSnHits+41.png");
  make1DPlot("Segments/hSnHits+42", f1, "N Hits on Segments ME+4/2", 1110, "Segments_hSnHits+42.png");
  make1DPlot("Segments/hSnHits-11", f1, "N Hits on Segments ME-1/1b", 1110, "Segments_hSnHits-11.png");
  make1DPlot("Segments/hSnHits-14", f1, "N Hits on Segments ME-1/1a", 1110, "Segments_hSnHits-11a.png");
  make1DPlot("Segments/hSnHits-12", f1, "N Hits on Segments ME-1/2", 1110, "Segments_hSnHits-12.png");
  make1DPlot("Segments/hSnHits-13", f1, "N Hits on Segments ME-1/3", 1110, "Segments_hSnHits-13.png");
  make1DPlot("Segments/hSnHits-21", f1, "N Hits on Segments ME-2/1", 1110, "Segments_hSnHits-21.png");
  make1DPlot("Segments/hSnHits-22", f1, "N Hits on Segments ME-2/2", 1110, "Segments_hSnHits-22.png");
  make1DPlot("Segments/hSnHits-31", f1, "N Hits on Segments ME-3/1", 1110, "Segments_hSnHits-31.png");
  make1DPlot("Segments/hSnHits-32", f1, "N Hits on Segments ME-3/2", 1110, "Segments_hSnHits-32.png");
  make1DPlot("Segments/hSnHits-41", f1, "N Hits on Segments ME-4/1", 1110, "Segments_hSnHits-41.png");
  make1DPlot("Segments/hSnHits-42", f1, "N Hits on Segments ME-4/2", 1110, "Segments_hSnHits-42.png");

  // segment chi2
  make1DPlot("Segments/hSChiSq+11", f1, "Segment Chi2/ndof ME+1/1b", 111110, "Segments_hSChiSq+11.png");
  make1DPlot("Segments/hSChiSq+14", f1, "Segment Chi2/ndof ME+1/1a", 111110, "Segments_hSChiSq+11a.png");
  make1DPlot("Segments/hSChiSq+12", f1, "Segment Chi2/ndof ME+1/2", 111110, "Segments_hSChiSq+12.png");
  make1DPlot("Segments/hSChiSq+13", f1, "Segment Chi2/ndof ME+1/3", 111110, "Segments_hSChiSq+13.png");
  make1DPlot("Segments/hSChiSq+21", f1, "Segment Chi2/ndof ME+2/1", 111110, "Segments_hSChiSq+21.png");
  make1DPlot("Segments/hSChiSq+22", f1, "Segment Chi2/ndof ME+2/2", 111110, "Segments_hSChiSq+22.png");
  make1DPlot("Segments/hSChiSq+31", f1, "Segment Chi2/ndof ME+3/1", 111110, "Segments_hSChiSq+31.png");
  make1DPlot("Segments/hSChiSq+32", f1, "Segment Chi2/ndof ME+3/2", 111110, "Segments_hSChiSq+32.png");
  make1DPlot("Segments/hSChiSq+41", f1, "Segment Chi2/ndof ME+4/1", 111110, "Segments_hSChiSq+41.png");
  make1DPlot("Segments/hSChiSq+42", f1, "Segment Chi2/ndof ME+4/2", 111110, "Segments_hSChiSq+42.png");
  make1DPlot("Segments/hSChiSq-11", f1, "Segment Chi2/ndof ME-1/1b", 111110, "Segments_hSChiSq-11.png");
  make1DPlot("Segments/hSChiSq-14", f1, "Segment Chi2/ndof ME-1/1a", 111110, "Segments_hSChiSq-11a.png");
  make1DPlot("Segments/hSChiSq-12", f1, "Segment Chi2/ndof ME-1/2", 111110, "Segments_hSChiSq-12.png");
  make1DPlot("Segments/hSChiSq-13", f1, "Segment Chi2/ndof ME-1/3", 111110, "Segments_hSChiSq-13.png");
  make1DPlot("Segments/hSChiSq-21", f1, "Segment Chi2/ndof ME-2/1", 111110, "Segments_hSChiSq-21.png");
  make1DPlot("Segments/hSChiSq-22", f1, "Segment Chi2/ndof ME-2/2", 111110, "Segments_hSChiSq-22.png");
  make1DPlot("Segments/hSChiSq-31", f1, "Segment Chi2/ndof ME-3/1", 111110, "Segments_hSChiSq-31.png");
  make1DPlot("Segments/hSChiSq-32", f1, "Segment Chi2/ndof ME-3/2", 111110, "Segments_hSChiSq-32.png");
  make1DPlot("Segments/hSChiSq-41", f1, "Segment Chi2/ndof ME-4/1", 111110, "Segments_hSChiSq-41.png");
  make1DPlot("Segments/hSChiSq-42", f1, "Segment Chi2/ndof ME-4/2", 111110, "Segments_hSChiSq-42.png");

  //miscellaneous
  make1DPlot("Segments/hSGlobalPhi", f1, "Segment Global Phi", 1110, "Segments_hSGlobalPhi.png");
  make1DPlot("Segments/hSGlobalTheta", f1, "Segment Global Theta", 1110, "Segments_hSGlobalTheta.png");

  //STA muons
  make1DPlot("STAMuons/trNSAMuons", f1, "Number STA Muons per Event", 111110, "STAMuons_nMuons.png");
  make1DPlot("STAMuons/trLength", f1, "Length along Z of STA Muons", 111110, "STAMuons_zLength.png");
  make1DPlot("STAMuons/trN", f1, "Number hits per STA Muons", 111110, "STAMuons_nHits.png");
  make1DPlot("STAMuons/trNCSC", f1, "Number CSC hits per STA Muons", 111110, "STAMuons_nCSCHits.png");
  make1DPlot("STAMuons/trNp", f1, "Number hits per STA Muons (plus side)", 111110, "STAMuons_nHits_plus.png");
  make1DPlot("STAMuons/trNCSCm", f1, "Number CSC hits per STA Muons (plus side)", 111110, "STAMuons_nCSCHits_plus.png");
  make1DPlot("STAMuons/trNm", f1, "Number hits per STA Muons (minus side)", 111110, "STAMuons_nHits_minus.png");
  make1DPlot(
      "STAMuons/trNCSCp", f1, "Number CSC hits per STA Muons (minus side)", 111110, "STAMuons_nCSCHits_minus.png");
  make1DPlot("STAMuons/trNormChi2", f1, "Normalized Chi^2 of STA Muons", 111110, "STAMuons_chi2.png");
  make1DPlot("STAMuons/trP", f1, "Momentum of STA Muons", 111110, "STAMuons_p.png");
  make1DPlot("STAMuons/trPT", f1, "Transverse Momentum of STA Muons", 111110, "STAMuons_pt.png");

  //Nikolai's plots
  NikolaiPlots(f1, 1);
  NikolaiPlots(f1, 2);
  NikolaiPlots(f1, 3);
  NikolaiPlots(f1, 4);

  //efficiency plots
  //TODO: change how this is done (here rather than in CSCValidation.cc)
  makeEffGif("Efficiency/hRHSTE", f1, "RecHit Efficiency", "Efficiency_hRHEff.png");
  makeEffGif("Efficiency/hSSTE", f1, "Segment Efficiency", "Efficiency_hSEff.png");
  Draw2DEfficiency("Efficiency/hRHSTE2", f1, "RecHit Efficiency 2D", "Efficiency_hRHEff2.png");
  Draw2DEfficiency("Efficiency/hSSTE2", f1, "Segment Efficiency 2D", "Efficiency_hSEff2.png");
  Draw2DEfficiency("Efficiency/hWireSTE2", f1, "Wire Efficiency 2D", "Efficiency_hWireEff2.png");
  Draw2DEfficiency("Efficiency/hStripSTE2", f1, "Strip Efficiency 2D", "Efficiency_hStripEff2.png");
  //Draw2DTempPlot("Efficiency/hSensitiveAreaEvt", f1, false, "Efficiency_hEvts2.png");
  Draw2DEfficiency("Efficiency/hSensitiveAreaEvt", f1, "Events in Sensitive Area", "Efficiency_hEvts2.png");

  makeEffGif("Efficiency/hRHSTETight", f1, "RecHit Efficiency Tight", "Efficiency_hRHEff_tight.png");
  makeEffGif("Efficiency/hSSTETight", f1, "Segment Efficiency Tight", "Efficiency_hSEff_tight.png");
  Draw2DEfficiency("Efficiency/hRHSTE2Tight", f1, "RecHit Efficiency 2D Tight", "Efficiency_hRHEff2_tight.png");
  Draw2DEfficiency("Efficiency/hSSTE2Tight", f1, "Segment Efficiency 2D Tight", "Efficiency_hSEff2_tight.png");
  Draw2DEfficiency("Efficiency/hWireSTE2Tight", f1, "Wire Efficiency 2D Tight", "Efficiency_hWireEff2_tight.png");
  Draw2DEfficiency("Efficiency/hStripSTE2Tight", f1, "Strip Efficiency 2D Tight", "Efficiency_hStripEff2_tight.png");

  //Draw2DTempPlot2("Efficiency/hRHSTE2", f1, true, "Efficiency_hRHEff2.png");
  //Draw2DTempPlot2("Efficiency/hSSTE2", f1, true, "Efficiency_hSEff2.png");
  //Draw2DTempPlot2("Efficiency/hWireSTE2", f1, true, "Efficiency_hWireEff2.png");
  //Draw2DTempPlot2("Efficiency/hWireSTE2", f1, true, "Efficiency_hStripEff2.png");

  //Timing Monitor
  make1DPlot("TimeMonitoring/ALCT_getBX",
             f1,
             "ALCT position in L1A window",
             "CSCALCTDigi.getBX()",
             "",
             111110,
             "ALCT_getBX.png");
  make1DPlot("TimeMonitoring/ALCT_getFullBX",
             f1,
             "BXN l1a_delay clocks ago from L1A",
             "CSCALCTDigi.getFullBX()",
             "",
             111110,
             "ALCT_getFullBX.png");
  make1DPlot("TimeMonitoring/BX_L1CSCCand", f1, "BX of L1GMT CSC Cand", 111110, "BX_L1CSCCand.png");
  make1DPlot("TimeMonitoring/BX_L1CSCCand_w_beamHalo",
             f1,
             "BX of L1GMT CSC Cand (quality=1)",
             111110,
             "BX_L1CSCCand_w_beamHalo.png");
  make1DPlot(
      "TimeMonitoring/CLCT_getBX", f1, "Last two bits of CLCT bxn", "CSCCLCTDigi.getBX()", "", 111110, "CLCT_getBX.png");
  make1DPlot("TimeMonitoring/CLCT_getFullBX",
             f1,
             "CLCT bxn at pretrigger",
             "CSCCLCTDigi.getFullBX()",
             "",
             111110,
             "CLCT_getFullBX.png");
  make1DPlot("TimeMonitoring/CorrelatedLCTS_getBX", f1, "CorrelatedLCT.getBX()", 111110, "CorrelatedLCTS_getBX.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime",
             f1,
             "ALCTMatchTime (all chambers)",
             "ALCT position in CLCT window (bx)",
             "",
             1110,
             "TMB_ALCTMatchTime.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime+1", f1, "TMB_ALCTMatchTime (ME +1)", 1110, "TMB_ALCTMatchTime+1.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime+2", f1, "TMB_ALCTMatchTime (ME +2)", 1110, "TMB_ALCTMatchTime+2.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime+3", f1, "TMB_ALCTMatchTime (ME +3)", 1110, "TMB_ALCTMatchTime+3.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime+4", f1, "TMB_ALCTMatchTime (ME +4)", 1110, "TMB_ALCTMatchTime+4.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime-1", f1, "TMB_ALCTMatchTime (ME -1)", 1110, "TMB_ALCTMatchTime-1.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime-2", f1, "TMB_ALCTMatchTime (ME -2)", 1110, "TMB_ALCTMatchTime-2.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime-3", f1, "TMB_ALCTMatchTime (ME -3)", 1110, "TMB_ALCTMatchTime-3.png");
  make1DPlot("TimeMonitoring/TMB_ALCTMatchTime-4", f1, "TMB_ALCTMatchTime (ME -4)", 1110, "TMB_ALCTMatchTime-4.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+11", f1, "TMB_ALCTMatchTime (ME +1/1b)", 1110, "TMB_ALCTMatchTime+11b.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+12", f1, "TMB_ALCTMatchTime (ME +1/2)", 1110, "TMB_ALCTMatchTime+12.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+13", f1, "TMB_ALCTMatchTime (ME +1/3)", 1110, "TMB_ALCTMatchTime+13.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+14", f1, "TMB_ALCTMatchTime (ME +1/1a)", 1110, "TMB_ALCTMatchTime+11a.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+21", f1, "TMB_ALCTMatchTime (ME +2/1)", 1110, "TMB_ALCTMatchTime+21.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+22", f1, "TMB_ALCTMatchTime (ME +2/2)", 1110, "TMB_ALCTMatchTime+23.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+31", f1, "TMB_ALCTMatchTime (ME +3/1)", 1110, "TMB_ALCTMatchTime+31.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+32", f1, "TMB_ALCTMatchTime (ME +3/2)", 1110, "TMB_ALCTMatchTime+33.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+41", f1, "TMB_ALCTMatchTime (ME +4/1)", 1110, "TMB_ALCTMatchTime+41.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime+42", f1, "TMB_ALCTMatchTime (ME +4/2)", 1110, "TMB_ALCTMatchTime+43.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-11", f1, "TMB_ALCTMatchTime (ME -1/1b)", 1110, "TMB_ALCTMatchTime-11b.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-12", f1, "TMB_ALCTMatchTime (ME -1/2)", 1110, "TMB_ALCTMatchTime-12.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-13", f1, "TMB_ALCTMatchTime (ME -1/3)", 1110, "TMB_ALCTMatchTime-13.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-14", f1, "TMB_ALCTMatchTime (ME -1/1a)", 1110, "TMB_ALCTMatchTime-11a.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-21", f1, "TMB_ALCTMatchTime (ME -2/1)", 1110, "TMB_ALCTMatchTime-21.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-22", f1, "TMB_ALCTMatchTime (ME -2/2)", 1110, "TMB_ALCTMatchTime-22.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-31", f1, "TMB_ALCTMatchTime (ME -3/1)", 1110, "TMB_ALCTMatchTime-31.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-32", f1, "TMB_ALCTMatchTime (ME -3/2)", 1110, "TMB_ALCTMatchTime-32.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-41", f1, "TMB_ALCTMatchTime (ME -4/1)", 1110, "TMB_ALCTMatchTime-41.png");
  make1DPlot(
      "TimeMonitoring/TMB_ALCTMatchTime-42", f1, "TMB_ALCTMatchTime (ME -4/2)", 1110, "TMB_ALCTMatchTime-42.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount",
             f1,
             "TMB BXN count at L1A arrival",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount+1",
             f1,
             "TMB BXN count at L1A arrival  (Station +1)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount+1.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount+2",
             f1,
             "TMB BXN count at L1A arrival  (Station +2)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount+2.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount+3",
             f1,
             "TMB BXN count at L1A arrival  (Station +3)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount+3.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount+4",
             f1,
             "TMB BXN count at L1A arrival  (Station +4)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount+4.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount-1",
             f1,
             "TMB BXN count at L1A arrival  (Station -1)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount-1.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount-2",
             f1,
             "TMB BXN count at L1A arrival  (Station -2)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount-2.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount-3",
             f1,
             "TMB BXN count at L1A arrival  (Station -3)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount-3.png");
  make1DPlot("TimeMonitoring/TMB_BXNCount-4",
             f1,
             "TMB BXN count at L1A arrival  (Station -4)",
             "TMBHeader.BXNCount()",
             "",
             1110,
             "TMB_BXNCount-4.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+11",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +1/1b)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+11b.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+12",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +1/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+12.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+13",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +1/3)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+13.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+14",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +1/1a)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+11a.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+21",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +2/1)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+21.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+22",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +2/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+22.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+31",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +3/1)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+31.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+32",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +3/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+32.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+41",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +4/1)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+41.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType+42",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME +4/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType+42.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-11",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -1/1b)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-11b.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-12",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -1/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-12.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-13",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -1/3)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-13.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-14",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -1/1a)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-11a.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-21",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -2/1)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-21.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-22",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -2/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-22.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-31",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -3/1)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-31.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-32",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -3/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-32.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-41",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -4/1)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-41.png");
  make1DPlot("TimeMonitoring/diff_opposite_endcaps_byType-42",
             f1,
             "#Delta t [ME+]-[ME-] for chambers in the same station and ring (ME -4/2)",
             "ns",
             "",
             1110,
             "diff_opposite_endcaps_byType-42.png");
  make1DPlot("TimeMonitoring/h1D_TMB_ALCT_rel_L1A", f1, "h1D_TMB_ALCT_rel_L1A", 1110, "h1D_TMB_ALCT_rel_L1A.png");
  make2DPlot("TimeMonitoring/h2D_TMB_ALCT_rel_L1A", f1, "h2D_TMB_ALCT_rel_L1A", 1110, "h2D_TMB_ALCT_rel_L1A.png");
  make2DPlot("TimeMonitoring/h2D_TMB_ALCT_rel_L1A_by_ring",
             f1,
             "h2D_TMB_ALCT_rel_L1A_by_ring",
             1110,
             "h2D_TMB_ALCT_rel_L1A_by_ring.png");
  make2DPlot("TimeMonitoring/h2D_TMB_ALCT_rel_L1A_v_ALCTT0KeyWG",
             f1,
             "h2D_TMB_ALCT_rel_L1A_v_ALCTT0KeyWG",
             1110,
             "h2D_TMB_ALCT_rel_L1A_v_ALCTT0KeyWG.png");
  make2DPlot("TimeMonitoring/n_ALCTs_v_BX_L1CSCCand",
             f1,
             "Number of ALCTs vs. BX of L1GMT CSC Cand",
             "bxn",
             "",
             1110,
             "n_ALCTs_v_BX_L1CSCCand.png");
  make2DPlot("TimeMonitoring/n_ALCTs_v_BX_L1CSCCand_w_beamHalo",
             f1,
             "Number of ALCTs vs. BX of L1GMT CSC Cand (quality = 1)",
             "bxn",
             "",
             1110,
             "n_ALCTs_v_BX_L1CSCCand_w_beamHalo.png");
  make2DPlot("TimeMonitoring/n_CLCTs_v_BX_L1CSCCand",
             f1,
             "Number of CLCTs vs. BX of L1GMT CSC Cand",
             "bxn",
             "",
             1110,
             "n_CLCTs_v_BX_L1CSCCand.png");
  make2DPlot("TimeMonitoring/n_CLCTs_v_BX_L1CSCCand_w_beamHalo",
             f1,
             "Number of CLCTs vs. BX of L1GMT CSC Cand (quality = 1)",
             "bxn",
             "",
             1110,
             "n_CLCTs_v_BX_L1CSCCand_w_beamHalo.png");
  make2DPlot("TimeMonitoring/n_CorrelatedLCTs_v_BX_L1CSCCand",
             f1,
             "Number of CorrelatedLCTs vs. BX of L1GMT CSC Cand",
             "bxn",
             "",
             1110,
             "n_CorrelatedLCTs_v_BX_L1CSCCand.png");
  make2DPlot("TimeMonitoring/n_CorrelatedLCTs_v_BX_L1CSCCand_w_beamHalo",
             f1,
             "Number of CorrelatedLCTs vs. BX of L1GMT CSC Cand (quality = 1)",
             "bxn",
             "",
             1110,
             "n_CorrelatedLCTs_v_BX_L1CSCCand_w_beamHalo.png");
  make2DPlot("TimeMonitoring/n_RecHits_v_BX_L1CSCCand",
             f1,
             "Number of RecHits vs. BX of L1GMT CSC Cand",
             "bxn",
             "",
             1110,
             "n_RecHits_v_BX_L1CSCCand.png");
  make2DPlot("TimeMonitoring/n_RecHits_v_BX_L1CSCCand_w_beamHalo",
             f1,
             "Number of RecHits vs. BX of L1GMT CSC Cand (quality = 1)",
             "bxn",
             "",
             1110,
             "n_RecHits_v_BX_L1CSCCand_w_beamHalo.png");
  make2DPlot("TimeMonitoring/n_Segments_v_BX_L1CSCCand",
             f1,
             "Number of Segments vs. BX of L1GMT CSC Cand",
             "bxn",
             "",
             1110,
             "n_Segments_v_BX_L1CSCCand.png");
  make2DPlot("TimeMonitoring/n_Segments_v_BX_L1CSCCand_w_beamHalo",
             f1,
             "Number of Segments vs. BX of L1GMT CSC Cand (quality = 1)",
             "bxn",
             "",
             1110,
             "n_Segments_v_BX_L1CSCCand_w_beamHalo.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+11",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +1/1b)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+11b.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+12",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +1/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+12.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+13",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +1/3)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+13.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+14",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +1/1a)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+11a.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+21",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +2/1)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+21.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+22",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +2/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+22.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+31",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +3/1)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+31.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+32",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +3/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+32.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+41",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +4/1)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+41.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+42",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME +4/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG+42.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-11",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -1/1b)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-11b.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-12",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -1/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-12.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-13",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -1/3)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-13.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-14",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -1/1a)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-11a.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-21",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -2/1)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-21.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-22",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -2/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-23.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-31",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -3/1)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-31.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-32",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -3/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-32.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-41",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -4/1)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-41.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-42",
              f1,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG (ME -4/2)",
              1110,
              "prf_TMB_ALCTMatchTime_v_ALCT0KeyWG-42.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+11",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +1/1b)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+11b.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+12",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +1/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+12.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+13",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +1/3)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+13.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+14",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +1/1a)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+11a.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+21",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +2/1)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+21.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+22",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +2/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+22.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+31",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +3/1)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+31.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+32",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +3/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+32.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+41",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +4/1)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+41.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+42",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME +4/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG+42.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-11",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -1/1b)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-11b.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-12",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -1/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-12.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-13",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -1/3)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-13.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-14",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -1/1a)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-11a.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-21",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -2/1)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-21.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-22",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -2/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-22.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-31",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -3/1)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-31.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-32",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -3/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-32.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-41",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -4/1)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-41.png");
  makeProfile("TimeMonitoring/prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-42",
              f1,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG (ME -4/2)",
              1110,
              "prf_TMB_ALCT_rel_L1A_v_ALCT0KeyWG-42.png");
  makeProfile("TimeMonitoring/prof_TMB_ALCTMatchTime",
              f1,
              "Mean ALCTMatchTime by chamber",
              "serialized chamber number",
              "ALCT position in CLCT window [bx]",
              1110,
              "prof_TMB_ALCTMatchTime.png");
  makeProfile("TimeMonitoring/prof_TMB_ALCTMatchTime_v_ALCT0KeyWG",
              f1,
              "Mean ALCTMatchTime by ALCT0KeyWG",
              "wire group",
              "ALCT position in CLCT window [bx]",
              1110,
              "prof_TMB_ALCTMatchTime_v_ALCT0KeyWG.png");
  makeProfile("TimeMonitoring/prof_TMB_ALCT_rel_L1A",
              f1,
              "Mean (L1A - ALCT) at TMB by chamber",
              "serialized chamber number",
              "L1A - ALCT at TMB [bx]",
              1110,
              "prof_TMB_ALCT_rel_L1A.png");
  makeProfile("TimeMonitoring/prof_TMB_ALCT_rel_L1A_by_ring",
              f1,
              "Mean (L1A - ALCT) at TMB by ring",
              "ring number",
              "L1A - ALCT at TMB [bx]",
              1110,
              "prof_TMB_ALCT_rel_L1A_by_ring.png");
  makeProfile("TimeMonitoring/prof_TMB_ALCT_rel_L1A_v_ALCT0KeyWG",
              f1,
              "Mean (L1A - ALCT) at TMB by ALCT0KeyWG",
              "wire group",
              "L1A - ALCT at TMB [bx]",
              1110,
              "prof_TMB_ALCT_rel_L1A_v_ALCT0KeyWG.png");
  make2DPlot("TimeMonitoring/seg_time_vs_absglobZ",
             f1,
             "Segment time vs. abs(z position)",
             "|z| of segment [cm]",
             "ns",
             1110,
             "seg_time_vs_absglobZ.png");
  make2DPlot("TimeMonitoring/seg_time_vs_distToIP",
             f1,
             "Segment time vs. distance to IP",
             " distance of segment to IP [cm]",
             "ns",
             1110,
             "seg_time_vs_distToIP.png");
  make2DPlot("TimeMonitoring/seg_time_vs_globZ",
             f1,
             "Segment time vs. z position",
             "z of segment [cm]",
             "ns",
             1110,
             "seg_time_vs_globZ.png");
  makeProfile("TimeMonitoring/timeChamber",
              f1,
              "Mean segment time by chamber",
              "serialized chamber number",
              "ns",
              1110,
              "timeChamber.png");
  makeProfile("TimeMonitoring/timeChamberByType+11",
              f1,
              "Segment mean time by chamber (ME +1/1b)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+11b.png");
  makeProfile("TimeMonitoring/timeChamberByType+12",
              f1,
              "Segment mean time by chamber (ME +1/3)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+12.png");
  makeProfile("TimeMonitoring/timeChamberByType+13",
              f1,
              "Segment mean time by chamber (ME +1/3)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+13.png");
  makeProfile("TimeMonitoring/timeChamberByType+14",
              f1,
              "Segment mean time by chamber (ME +1/1a)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+11a.png");
  makeProfile("TimeMonitoring/timeChamberByType+21",
              f1,
              "Segment mean time by chamber (ME +2/1)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+21.png");
  makeProfile("TimeMonitoring/timeChamberByType+22",
              f1,
              "Segment mean time by chamber (ME +2/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+22.png");
  makeProfile("TimeMonitoring/timeChamberByType+31",
              f1,
              "Segment mean time by chamber (ME +3/1)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+31.png");
  makeProfile("TimeMonitoring/timeChamberByType+32",
              f1,
              "Segment mean time by chamber (ME +3/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+32.png");
  makeProfile("TimeMonitoring/timeChamberByType+41",
              f1,
              "Segment mean time by chamber (ME +4/1)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+41.png");
  makeProfile("TimeMonitoring/timeChamberByType+42",
              f1,
              "Segment mean time by chamber (ME +4/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType+42.png");
  makeProfile("TimeMonitoring/timeChamberByType-11",
              f1,
              "Segment mean time by chamber (ME -1/1b)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-11b.png");
  makeProfile("TimeMonitoring/timeChamberByType-12",
              f1,
              "Segment mean time by chamber (ME -1/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-12.png");
  makeProfile("TimeMonitoring/timeChamberByType-13",
              f1,
              "Segment mean time by chamber (ME -1/3)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-13.png");
  makeProfile("TimeMonitoring/timeChamberByType-14",
              f1,
              "Segment mean time by chamber (ME -1/1a)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-11a.png");
  makeProfile("TimeMonitoring/timeChamberByType-21",
              f1,
              "Segment mean time by chamber (ME -2/1)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-21.png");
  makeProfile("TimeMonitoring/timeChamberByType-22",
              f1,
              "Segment mean time by chamber (ME -2/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-22.png");
  makeProfile("TimeMonitoring/timeChamberByType-31",
              f1,
              "Segment mean time by chamber (ME -3/1)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-31.png");
  makeProfile("TimeMonitoring/timeChamberByType-32",
              f1,
              "Segment mean time by chamber (ME -3/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-32.png");
  makeProfile("TimeMonitoring/timeChamberByType-41",
              f1,
              "Segment mean time by chamber (ME -4/1)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-41.png");
  makeProfile("TimeMonitoring/timeChamberByType-42",
              f1,
              "Segment mean time by chamber (ME -4/2)",
              "chamber number",
              "ns",
              1110,
              "timeChamberByType-42.png");
}
