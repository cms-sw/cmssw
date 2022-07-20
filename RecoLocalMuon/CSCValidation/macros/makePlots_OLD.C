// This function makes plots from root file output of CSCValidation, input_file
// Pass it to root e.g. see makePlots.sh script
// It requires local file "myFunctions.C"
// Original author: Andy Kubik (NWU)
// - Please contact CSC DPG for current status - 12.07.2022

// - includes rechit and segment position plots
// - does not include segment time plots - MUST BE ADDED

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

  //  gROOT->Reset();
  //  gROOT->ProcessLine(".L myFunctions.C");

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
  make1DPlot("Digis/hStripNFired", f1, "Fired Strips per Event", 1110, "Digis_hStripNFired.png");
  make1DPlot("Digis/hWirenGroupsTotal", f1, "Fired Wires per Event", 1110, "Digis_hWirenGroupsTotal.png");
  make1DPlot("recHits/hRHnrechits", f1, "RecHits per Event", 1110, "recHits_hRHnrechits.png");
  make1DPlot("Segments/hSnSegments", f1, "Segments per Event", 1110, "Segments_hSnSegments.png");

  //efficiency plots
  makeEffGif("Efficiency/hRHSTE", f1, "RecHit Efficiency", "Efficiency_hRHEff.png");
  makeEffGif("Efficiency/hSSTE", f1, "Segment Efficiency", "Efficiency_hSEff.png");
  Draw2DEfficiency("Efficiency/hRHSTE2", f1, "RecHit Efficiency 2D", "Efficiency_hRHEff2.png");
  Draw2DEfficiency("Efficiency/hSSTE2", f1, "Segment Efficiency 2D", "Efficiency_hSEff2.png");
  Draw2DEfficiency("Efficiency/hWireSTE2", f1, "Wire Efficiency 2D", "Efficiency_hWireEff2.png");
  Draw2DEfficiency("Efficiency/hStripSTE2", f1, "Strip Efficiency 2D", "Efficiency_hStripEff2.png");
  Draw2DTempPlot("Efficiency/hSensitiveAreaEvt", f1, false, "Efficiency_hEvts2.png");

  //produce wire timing plots
  make1DPlot("Digis/hWireTBin+11", f1, "Wire TimeBin Fired ME+1/1", 1110, "Digis_hWireTBin+11.png");
  make1DPlot("Digis/hWireTBin+12", f1, "Wire TimeBin Fired ME+1/2", 1110, "Digis_hWireTBin+12.png");
  make1DPlot("Digis/hWireTBin+13", f1, "Wire TimeBin Fired ME+1/3", 1110, "Digis_hWireTBin+13.png");
  make1DPlot("Digis/hWireTBin+21", f1, "Wire TimeBin Fired ME+2/1", 1110, "Digis_hWireTBin+21.png");
  make1DPlot("Digis/hWireTBin+22", f1, "Wire TimeBin Fired ME+2/2", 1110, "Digis_hWireTBin+22.png");
  make1DPlot("Digis/hWireTBin+31", f1, "Wire TimeBin Fired ME+3/1", 1110, "Digis_hWireTBin+31.png");
  make1DPlot("Digis/hWireTBin+32", f1, "Wire TimeBin Fired ME+3/2", 1110, "Digis_hWireTBin+32.png");
  make1DPlot("Digis/hWireTBin+41", f1, "Wire TimeBin Fired ME+4/1", 1110, "Digis_hWireTBin+41.png");
  make1DPlot("Digis/hWireTBin-11", f1, "Wire TimeBin Fired ME-1/1", 1110, "Digis_hWireTBin-11.png");
  make1DPlot("Digis/hWireTBin-12", f1, "Wire TimeBin Fired ME-1/2", 1110, "Digis_hWireTBin-12.png");
  make1DPlot("Digis/hWireTBin-13", f1, "Wire TimeBin Fired ME-1/3", 1110, "Digis_hWireTBin-13.png");
  make1DPlot("Digis/hWireTBin-21", f1, "Wire TimeBin Fired ME-2/1", 1110, "Digis_hWireTBin-21.png");
  make1DPlot("Digis/hWireTBin-22", f1, "Wire TimeBin Fired ME-2/2", 1110, "Digis_hWireTBin-22.png");
  make1DPlot("Digis/hWireTBin-31", f1, "Wire TimeBin Fired ME-3/1", 1110, "Digis_hWireTBin-31.png");
  make1DPlot("Digis/hWireTBin-32", f1, "Wire TimeBin Fired ME-3/2", 1110, "Digis_hWireTBin-32.png");
  make1DPlot("Digis/hWireTBin-41", f1, "Wire TimeBin Fired ME-4/1", 1110, "Digis_hWireTBin-41.png");

  //produce pedestal noise plots
  make1DPlot("PedestalNoise/hStripPedME+11",
             f1,
             "Pedestal Noise Distribution ME+1/1b",
             1110,
             "PedestalNoise_hStripPedME+11.png");
  make1DPlot("PedestalNoise/hStripPedME+14",
             f1,
             "Pedestal Noise Distribution ME+1/1a",
             1110,
             "PedestalNoise_hStripPedME+11a.png");
  make1DPlot("PedestalNoise/hStripPedME+12",
             f1,
             "Pedestal Noise Distribution ME+1/2",
             1110,
             "PedestalNoise_hStripPedME+12.png");
  make1DPlot("PedestalNoise/hStripPedME+13",
             f1,
             "Pedestal Noise Distribution ME+1/3",
             1110,
             "PedestalNoise_hStripPedME+13.png");
  make1DPlot("PedestalNoise/hStripPedME+21",
             f1,
             "Pedestal Noise Distribution ME+2/1",
             1110,
             "PedestalNoise_hStripPedME+21.png");
  make1DPlot("PedestalNoise/hStripPedME+22",
             f1,
             "Pedestal Noise Distribution ME+2/2",
             1110,
             "PedestalNoise_hStripPedME+22.png");
  make1DPlot("PedestalNoise/hStripPedME+31",
             f1,
             "Pedestal Noise Distribution ME+3/1",
             1110,
             "PedestalNoise_hStripPedME+31.png");
  make1DPlot("PedestalNoise/hStripPedME+32",
             f1,
             "Pedestal Noise Distribution ME+3/2",
             1110,
             "PedestalNoise_hStripPedME+32.png");
  make1DPlot("PedestalNoise/hStripPedME+41",
             f1,
             "Pedestal Noise Distribution ME+4/1",
             1110,
             "PedestalNoise_hStripPedME+41.png");
  make1DPlot("PedestalNoise/hStripPedME-11",
             f1,
             "Pedestal Noise Distribution ME-1/1b",
             1110,
             "PedestalNoise_hStripPedME-11.png");
  make1DPlot("PedestalNoise/hStripPedME-14",
             f1,
             "Pedestal Noise Distribution ME-1/1a",
             1110,
             "PedestalNoise_hStripPedME-11a.png");
  make1DPlot("PedestalNoise/hStripPedME-12",
             f1,
             "Pedestal Noise Distribution ME-1/2",
             1110,
             "PedestalNoise_hStripPedME-12.png");
  make1DPlot("PedestalNoise/hStripPedME-13",
             f1,
             "Pedestal Noise Distribution ME-1/3",
             1110,
             "PedestalNoise_hStripPedME-13.png");
  make1DPlot("PedestalNoise/hStripPedME-21",
             f1,
             "Pedestal Noise Distribution ME-2/1",
             1110,
             "PedestalNoise_hStripPedME-21.png");
  make1DPlot("PedestalNoise/hStripPedME-22",
             f1,
             "Pedestal Noise Distribution ME-2/2",
             1110,
             "PedestalNoise_hStripPedME-22.png");
  make1DPlot("PedestalNoise/hStripPedME-31",
             f1,
             "Pedestal Noise Distribution ME-3/1",
             1110,
             "PedestalNoise_hStripPedME-31.png");
  make1DPlot("PedestalNoise/hStripPedME-32",
             f1,
             "Pedestal Noise Distribution ME-3/2",
             1110,
             "PedestalNoise_hStripPedME-32.png");
  make1DPlot("PedestalNoise/hStripPedME-41",
             f1,
             "Pedestal Noise Distribution ME-4/1",
             1110,
             "PedestalNoise_hStripPedME-41.png");

  // resolution
  make1DPlot("Resolution/hSResid+11",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/1b",
             1110,
             "Resolution_hSResid+11.png");
  make1DPlot("Resolution/hSResid+12",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/2",
             1110,
             "Resolution_hSResid+12.png");
  make1DPlot("Resolution/hSResid+13",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/3",
             1110,
             "Resolution_hSResid+13.png");
  make1DPlot("Resolution/hSResid+14",
             f1,
             "Expected Position from Fit - Reconstructed, ME+1/1a",
             1110,
             "Resolution_hSResid+11a.png");
  make1DPlot("Resolution/hSResid+21",
             f1,
             "Expected Position from Fit - Reconstructed, ME+2/1",
             1110,
             "Resolution_hSResid+21.png");
  make1DPlot("Resolution/hSResid+22",
             f1,
             "Expected Position from Fit - Reconstructed, ME+2/2",
             1110,
             "Resolution_hSResid+22.png");
  make1DPlot("Resolution/hSResid+31",
             f1,
             "Expected Position from Fit - Reconstructed, ME+3/1",
             1110,
             "Resolution_hSResid+31.png");
  make1DPlot("Resolution/hSResid+32",
             f1,
             "Expected Position from Fit - Reconstructed, ME+3/2",
             1110,
             "Resolution_hSResid+32.png");
  make1DPlot("Resolution/hSResid+41",
             f1,
             "Expected Position from Fit - Reconstructed, ME+4/1",
             1110,
             "Resolution_hSResid+41.png");
  make1DPlot("Resolution/hSResid-11",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/1b",
             1110,
             "Resolution_hSResid-11.png");
  make1DPlot("Resolution/hSResid-12",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/2",
             1110,
             "Resolution_hSResid-12.png");
  make1DPlot("Resolution/hSResid-13",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/3",
             1110,
             "Resolution_hSResid-13.png");
  make1DPlot("Resolution/hSResid-14",
             f1,
             "Expected Position from Fit - Reconstructed, ME-1/1a",
             1110,
             "Resolution_hSResid-11a.png");
  make1DPlot("Resolution/hSResid-21",
             f1,
             "Expected Position from Fit - Reconstructed, ME-2/1",
             1110,
             "Resolution_hSResid-21.png");
  make1DPlot("Resolution/hSResid-22",
             f1,
             "Expected Position from Fit - Reconstructed, ME-2/2",
             1110,
             "Resolution_hSResid-22.png");
  make1DPlot("Resolution/hSResid-31",
             f1,
             "Expected Position from Fit - Reconstructed, ME-3/1",
             1110,
             "Resolution_hSResid-31.png");
  make1DPlot("Resolution/hSResid-32",
             f1,
             "Expected Position from Fit - Reconstructed, ME-3/2",
             1110,
             "Resolution_hSResid-32.png");
  make1DPlot("Resolution/hSResid-41",
             f1,
             "Expected Position from Fit - Reconstructed, ME-4/1",
             1110,
             "Resolution_hSResid-41.png");

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
  make1DPlot("recHits/hRHstpos-11", f1, "Strip Position (ME-1/1b)", 1110, "recHits_hRHstpos-11.png");
  make1DPlot("recHits/hRHstpos-14", f1, "Strip Position (ME-1/1a)", 1110, "recHits_hRHstpos-11a.png");
  make1DPlot("recHits/hRHstpos-12", f1, "Strip Position (ME-1/2)", 1110, "recHits_hRHstpos-12.png");
  make1DPlot("recHits/hRHstpos-13", f1, "Strip Position (ME-1/3)", 1110, "recHits_hRHstpos-13.png");
  make1DPlot("recHits/hRHstpos-21", f1, "Strip Position (ME-2/1)", 1110, "recHits_hRHstpos-21.png");
  make1DPlot("recHits/hRHstpos-22", f1, "Strip Position (ME-2/2)", 1110, "recHits_hRHstpos-22.png");
  make1DPlot("recHits/hRHstpos-31", f1, "Strip Position (ME-3/1)", 1110, "recHits_hRHstpos-31.png");
  make1DPlot("recHits/hRHstpos-32", f1, "Strip Position (ME-3/2)", 1110, "recHits_hRHstpos-32.png");
  make1DPlot("recHits/hRHstpos-41", f1, "Strip Position (ME-4/1)", 1110, "recHits_hRHstpos-41.png");

  // rechit timing
  make1DPlot("recHits/hRHTiming+11", f1, "RecHit Timing ME+1/1b", 1110, "recHits_hRHTiming+11.png");
  make1DPlot("recHits/hRHTiming+14", f1, "RecHit Timing ME+1/1a", 1110, "recHits_hRHTiming+11a.png");
  make1DPlot("recHits/hRHTiming+12", f1, "RecHit Timing ME+1/2", 1110, "recHits_hRHTiming+12.png");
  make1DPlot("recHits/hRHTiming+13", f1, "RecHit Timing ME+1/3", 1110, "recHits_hRHTiming+13.png");
  make1DPlot("recHits/hRHTiming+21", f1, "RecHit Timing ME+2/1", 1110, "recHits_hRHTiming+21.png");
  make1DPlot("recHits/hRHTiming+22", f1, "RecHit Timing ME+2/2", 1110, "recHits_hRHTiming+22.png");
  make1DPlot("recHits/hRHTiming+31", f1, "RecHit Timing ME+3/1", 1110, "recHits_hRHTiming+31.png");
  make1DPlot("recHits/hRHTiming+32", f1, "RecHit Timing ME+3/2", 1110, "recHits_hRHTiming+32.png");
  make1DPlot("recHits/hRHTiming+41", f1, "RecHit Timing ME+4/1", 1110, "recHits_hRHTiming+41.png");
  make1DPlot("recHits/hRHTiming-11", f1, "RecHit Timing ME-1/1b", 1110, "recHits_hRHTiming-11.png");
  make1DPlot("recHits/hRHTiming-14", f1, "RecHit Timing ME-1/1a", 1110, "recHits_hRHTiming-11a.png");
  make1DPlot("recHits/hRHTiming-12", f1, "RecHit Timing ME-1/2", 1110, "recHits_hRHTiming-12.png");
  make1DPlot("recHits/hRHTiming-13", f1, "RecHit Timing ME-1/3", 1110, "recHits_hRHTiming-13.png");
  make1DPlot("recHits/hRHTiming-21", f1, "RecHit Timing ME-2/1", 1110, "recHits_hRHTiming-21.png");
  make1DPlot("recHits/hRHTiming-22", f1, "RecHit Timing ME-2/2", 1110, "recHits_hRHTiming-22.png");
  make1DPlot("recHits/hRHTiming-31", f1, "RecHit Timing ME-3/1", 1110, "recHits_hRHTiming-31.png");
  make1DPlot("recHits/hRHTiming-32", f1, "RecHit Timing ME-3/2", 1110, "recHits_hRHTiming-32.png");
  make1DPlot("recHits/hRHTiming-41", f1, "RecHit Timing ME-4/1", 1110, "recHits_hRHTiming-41.png");

  // rechit charge
  make1DPlot("recHits/hRHSumQ+11", f1, "Sum 3x3 RecHit Charge ME+1/1b", 1110, "recHits_hRHSumQ+11.png");
  make1DPlot("recHits/hRHSumQ+14", f1, "Sum 3x3 RecHit Charge ME+1/1a", 1110, "recHits_hRHSumQ+11a.png");
  make1DPlot("recHits/hRHSumQ+12", f1, "Sum 3x3 RecHit Charge ME+1/2", 1110, "recHits_hRHSumQ+12.png");
  make1DPlot("recHits/hRHSumQ+13", f1, "Sum 3x3 RecHit Charge ME+1/3", 1110, "recHits_hRHSumQ+13.png");
  make1DPlot("recHits/hRHSumQ+21", f1, "Sum 3x3 RecHit Charge ME+2/1", 1110, "recHits_hRHSumQ+21.png");
  make1DPlot("recHits/hRHSumQ+22", f1, "Sum 3x3 RecHit Charge ME+2/2", 1110, "recHits_hRHSumQ+22.png");
  make1DPlot("recHits/hRHSumQ+31", f1, "Sum 3x3 RecHit Charge ME+3/1", 1110, "recHits_hRHSumQ+31.png");
  make1DPlot("recHits/hRHSumQ+32", f1, "Sum 3x3 RecHit Charge ME+3/2", 1110, "recHits_hRHSumQ+32.png");
  make1DPlot("recHits/hRHSumQ+41", f1, "Sum 3x3 RecHit Charge ME+4/1", 1110, "recHits_hRHSumQ+41.png");
  make1DPlot("recHits/hRHSumQ-11", f1, "Sum 3x3 RecHit Charge ME-1/1b", 1110, "recHits_hRHSumQ-11.png");
  make1DPlot("recHits/hRHSumQ-14", f1, "Sum 3x3 RecHit Charge ME-1/1a", 1110, "recHits_hRHSumQ-11a.png");
  make1DPlot("recHits/hRHSumQ-12", f1, "Sum 3x3 RecHit Charge ME-1/2", 1110, "recHits_hRHSumQ-12.png");
  make1DPlot("recHits/hRHSumQ-13", f1, "Sum 3x3 RecHit Charge ME-1/3", 1110, "recHits_hRHSumQ-13.png");
  make1DPlot("recHits/hRHSumQ-21", f1, "Sum 3x3 RecHit Charge ME-2/1", 1110, "recHits_hRHSumQ-21.png");
  make1DPlot("recHits/hRHSumQ-22", f1, "Sum 3x3 RecHit Charge ME-2/2", 1110, "recHits_hRHSumQ-22.png");
  make1DPlot("recHits/hRHSumQ-31", f1, "Sum 3x3 RecHit Charge ME-3/1", 1110, "recHits_hRHSumQ-31.png");
  make1DPlot("recHits/hRHSumQ-32", f1, "Sum 3x3 RecHit Charge ME-3/2", 1110, "recHits_hRHSumQ-32.png");
  make1DPlot("recHits/hRHSumQ-41", f1, "Sum 3x3 RecHit Charge ME-4/1", 1110, "recHits_hRHSumQ-41.png");

  make1DPlot("recHits/hRHRatioQ+11", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/1b", 1110, "recHits_hRHRatioQ+11.png");
  make1DPlot("recHits/hRHRatioQ+14", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/1a", 1110, "recHits_hRHRatioQ+11a.png");
  make1DPlot("recHits/hRHRatioQ+12", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/2", 1110, "recHits_hRHRatioQ+12.png");
  make1DPlot("recHits/hRHRatioQ+13", f1, "Charge Ratio (Ql_Qr)/Qt ME+1/3", 1110, "recHits_hRHRatioQ+13.png");
  make1DPlot("recHits/hRHRatioQ+21", f1, "Charge Ratio (Ql_Qr)/Qt ME+2/1", 1110, "recHits_hRHRatioQ+21.png");
  make1DPlot("recHits/hRHRatioQ+22", f1, "Charge Ratio (Ql_Qr)/Qt ME+2/2", 1110, "recHits_hRHRatioQ+22.png");
  make1DPlot("recHits/hRHRatioQ+31", f1, "Charge Ratio (Ql_Qr)/Qt ME+3/1", 1110, "recHits_hRHRatioQ+31.png");
  make1DPlot("recHits/hRHRatioQ+32", f1, "Charge Ratio (Ql_Qr)/Qt ME+3/2", 1110, "recHits_hRHRatioQ+32.png");
  make1DPlot("recHits/hRHRatioQ+41", f1, "Charge Ratio (Ql_Qr)/Qt ME+4/1", 1110, "recHits_hRHRatioQ+41.png");
  make1DPlot("recHits/hRHRatioQ-11", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/1b", 1110, "recHits_hRHRatioQ-11.png");
  make1DPlot("recHits/hRHRatioQ-14", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/1a", 1110, "recHits_hRHRatioQ-11a.png");
  make1DPlot("recHits/hRHRatioQ-12", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/2", 1110, "recHits_hRHRatioQ-12.png");
  make1DPlot("recHits/hRHRatioQ-13", f1, "Charge Ratio (Ql_Qr)/Qt ME-1/3", 1110, "recHits_hRHRatioQ-13.png");
  make1DPlot("recHits/hRHRatioQ-21", f1, "Charge Ratio (Ql_Qr)/Qt ME-2/1", 1110, "recHits_hRHRatioQ-21.png");
  make1DPlot("recHits/hRHRatioQ-22", f1, "Charge Ratio (Ql_Qr)/Qt ME-2/2", 1110, "recHits_hRHRatioQ-22.png");
  make1DPlot("recHits/hRHRatioQ-31", f1, "Charge Ratio (Ql_Qr)/Qt ME-3/1", 1110, "recHits_hRHRatioQ-31.png");
  make1DPlot("recHits/hRHRatioQ-32", f1, "Charge Ratio (Ql_Qr)/Qt ME-3/2", 1110, "recHits_hRHRatioQ-32.png");
  make1DPlot("recHits/hRHRatioQ-41", f1, "Charge Ratio (Ql_Qr)/Qt ME-4/1", 1110, "recHits_hRHRatioQ-41.png");

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
  make1DPlot("Segments/hSnHits-11", f1, "N Hits on Segments ME-1/1b", 1110, "Segments_hSnHits-11.png");
  make1DPlot("Segments/hSnHits-14", f1, "N Hits on Segments ME-1/1a", 1110, "Segments_hSnHits-11a.png");
  make1DPlot("Segments/hSnHits-12", f1, "N Hits on Segments ME-1/2", 1110, "Segments_hSnHits-12.png");
  make1DPlot("Segments/hSnHits-13", f1, "N Hits on Segments ME-1/3", 1110, "Segments_hSnHits-13.png");
  make1DPlot("Segments/hSnHits-21", f1, "N Hits on Segments ME-2/1", 1110, "Segments_hSnHits-21.png");
  make1DPlot("Segments/hSnHits-22", f1, "N Hits on Segments ME-2/2", 1110, "Segments_hSnHits-22.png");
  make1DPlot("Segments/hSnHits-31", f1, "N Hits on Segments ME-3/1", 1110, "Segments_hSnHits-31.png");
  make1DPlot("Segments/hSnHits-32", f1, "N Hits on Segments ME-3/2", 1110, "Segments_hSnHits-32.png");
  make1DPlot("Segments/hSnHits-41", f1, "N Hits on Segments ME-4/1", 1110, "Segments_hSnHits-41.png");

  // segment chi2
  make1DPlot("Segments/hSChiSq+11", f1, "Segment Chi2/ndof ME+1/1b", 1110, "Segments_hSChiSq+11.png");
  make1DPlot("Segments/hSChiSq+14", f1, "Segment Chi2/ndof ME+1/1a", 1110, "Segments_hSChiSq+11a.png");
  make1DPlot("Segments/hSChiSq+12", f1, "Segment Chi2/ndof ME+1/2", 1110, "Segments_hSChiSq+12.png");
  make1DPlot("Segments/hSChiSq+13", f1, "Segment Chi2/ndof ME+1/3", 1110, "Segments_hSChiSq+13.png");
  make1DPlot("Segments/hSChiSq+21", f1, "Segment Chi2/ndof ME+2/1", 1110, "Segments_hSChiSq+21.png");
  make1DPlot("Segments/hSChiSq+22", f1, "Segment Chi2/ndof ME+2/2", 1110, "Segments_hSChiSq+22.png");
  make1DPlot("Segments/hSChiSq+31", f1, "Segment Chi2/ndof ME+3/1", 1110, "Segments_hSChiSq+31.png");
  make1DPlot("Segments/hSChiSq+32", f1, "Segment Chi2/ndof ME+3/2", 1110, "Segments_hSChiSq+32.png");
  make1DPlot("Segments/hSChiSq+41", f1, "Segment Chi2/ndof ME+4/1", 1110, "Segments_hSChiSq+41.png");
  make1DPlot("Segments/hSChiSq-11", f1, "Segment Chi2/ndof ME-1/1b", 1110, "Segments_hSChiSq-11.png");
  make1DPlot("Segments/hSChiSq-14", f1, "Segment Chi2/ndof ME-1/1a", 1110, "Segments_hSChiSq-11a.png");
  make1DPlot("Segments/hSChiSq-12", f1, "Segment Chi2/ndof ME-1/2", 1110, "Segments_hSChiSq-12.png");
  make1DPlot("Segments/hSChiSq-13", f1, "Segment Chi2/ndof ME-1/3", 1110, "Segments_hSChiSq-13.png");
  make1DPlot("Segments/hSChiSq-21", f1, "Segment Chi2/ndof ME-2/1", 1110, "Segments_hSChiSq-21.png");
  make1DPlot("Segments/hSChiSq-22", f1, "Segment Chi2/ndof ME-2/2", 1110, "Segments_hSChiSq-22.png");
  make1DPlot("Segments/hSChiSq-31", f1, "Segment Chi2/ndof ME-3/1", 1110, "Segments_hSChiSq-31.png");
  make1DPlot("Segments/hSChiSq-32", f1, "Segment Chi2/ndof ME-3/2", 1110, "Segments_hSChiSq-32.png");
  make1DPlot("Segments/hSChiSq-41", f1, "Segment Chi2/ndof ME-4/1", 1110, "Segments_hSChiSq-41.png");

  //miscellaneous
  make1DPlot("Segments/hSGlobalPhi", f1, "Segment Global Phi", 1110, "Segments_hSGlobalPhi.png");
  make1DPlot("Segments/hSGlobalTheta", f1, "Segment Global Theta", 1110, "Segments_hSGlobalTheta.png");
}
