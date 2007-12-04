/** \file LaserAlignmentInitHistograms.cc
 *  Histograms for the Laser Alignment System
 *
 *  $Date: 2007/03/18 19:00:20 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h"
#include "TFile.h" 

void LaserAlignment::initHistograms()
{
  // vector to store the Histogram names
  theHistogramNames.reserve(434);

  // --- LaserBeams ---
  TDirectory * BeamDir = theFile->mkdir("LaserBeams");

  TDirectory * TECPosDir = BeamDir->mkdir("TEC+");
  TDirectory * TECNegDir = BeamDir->mkdir("TEC-");
  TDirectory * TOBDir = BeamDir->mkdir("TOB");
  TDirectory * TIBDir = BeamDir->mkdir("TIB");

  TDirectory * Ring4PosDir = TECPosDir->mkdir("Ring 4");
  TDirectory * Ring6PosDir = TECPosDir->mkdir("Ring 6");

  TDirectory * Ring4NegDir = TECNegDir->mkdir("Ring 4");
  TDirectory * Ring6NegDir = TECNegDir->mkdir("Ring 6");

  TDirectory * Ring4Beam0PosDir = Ring4PosDir->mkdir("Beam 0");
  TDirectory * Ring4Beam1PosDir = Ring4PosDir->mkdir("Beam 1");
  TDirectory * Ring4Beam2PosDir = Ring4PosDir->mkdir("Beam 2");
  TDirectory * Ring4Beam3PosDir = Ring4PosDir->mkdir("Beam 3");
  TDirectory * Ring4Beam4PosDir = Ring4PosDir->mkdir("Beam 4");
  TDirectory * Ring4Beam5PosDir = Ring4PosDir->mkdir("Beam 5");
  TDirectory * Ring4Beam6PosDir = Ring4PosDir->mkdir("Beam 6");
  TDirectory * Ring4Beam7PosDir = Ring4PosDir->mkdir("Beam 7");
  TDirectory * Ring6Beam0PosDir = Ring6PosDir->mkdir("Beam 0");
  TDirectory * Ring6Beam1PosDir = Ring6PosDir->mkdir("Beam 1");
  TDirectory * Ring6Beam2PosDir = Ring6PosDir->mkdir("Beam 2");
  TDirectory * Ring6Beam3PosDir = Ring6PosDir->mkdir("Beam 3");
  TDirectory * Ring6Beam4PosDir = Ring6PosDir->mkdir("Beam 4");
  TDirectory * Ring6Beam5PosDir = Ring6PosDir->mkdir("Beam 5");
  TDirectory * Ring6Beam6PosDir = Ring6PosDir->mkdir("Beam 6");
  TDirectory * Ring6Beam7PosDir = Ring6PosDir->mkdir("Beam 7");

  TDirectory * Ring4Beam0NegDir = Ring4NegDir->mkdir("Beam 0");
  TDirectory * Ring4Beam1NegDir = Ring4NegDir->mkdir("Beam 1");
  TDirectory * Ring4Beam2NegDir = Ring4NegDir->mkdir("Beam 2");
  TDirectory * Ring4Beam3NegDir = Ring4NegDir->mkdir("Beam 3");
  TDirectory * Ring4Beam4NegDir = Ring4NegDir->mkdir("Beam 4");
  TDirectory * Ring4Beam5NegDir = Ring4NegDir->mkdir("Beam 5");
  TDirectory * Ring4Beam6NegDir = Ring4NegDir->mkdir("Beam 6");
  TDirectory * Ring4Beam7NegDir = Ring4NegDir->mkdir("Beam 7");
  TDirectory * Ring6Beam0NegDir = Ring6NegDir->mkdir("Beam 0");
  TDirectory * Ring6Beam1NegDir = Ring6NegDir->mkdir("Beam 1");
  TDirectory * Ring6Beam2NegDir = Ring6NegDir->mkdir("Beam 2");
  TDirectory * Ring6Beam3NegDir = Ring6NegDir->mkdir("Beam 3");
  TDirectory * Ring6Beam4NegDir = Ring6NegDir->mkdir("Beam 4");
  TDirectory * Ring6Beam5NegDir = Ring6NegDir->mkdir("Beam 5");
  TDirectory * Ring6Beam6NegDir = Ring6NegDir->mkdir("Beam 6");
  TDirectory * Ring6Beam7NegDir = Ring6NegDir->mkdir("Beam 7");

  TDirectory * Beam0TOBDir = TOBDir->mkdir("Beam 0");
  TDirectory * Beam1TOBDir = TOBDir->mkdir("Beam 1");
  TDirectory * Beam2TOBDir = TOBDir->mkdir("Beam 2");
  TDirectory * Beam3TOBDir = TOBDir->mkdir("Beam 3");
  TDirectory * Beam4TOBDir = TOBDir->mkdir("Beam 4");
  TDirectory * Beam5TOBDir = TOBDir->mkdir("Beam 5");
  TDirectory * Beam6TOBDir = TOBDir->mkdir("Beam 6");
  TDirectory * Beam7TOBDir = TOBDir->mkdir("Beam 7");

  TDirectory * Beam0TIBDir = TIBDir->mkdir("Beam 0");
  TDirectory * Beam1TIBDir = TIBDir->mkdir("Beam 1");
  TDirectory * Beam2TIBDir = TIBDir->mkdir("Beam 2");
  TDirectory * Beam3TIBDir = TIBDir->mkdir("Beam 3");
  TDirectory * Beam4TIBDir = TIBDir->mkdir("Beam 4");
  TDirectory * Beam5TIBDir = TIBDir->mkdir("Beam 5");
  TDirectory * Beam6TIBDir = TIBDir->mkdir("Beam 6");
  TDirectory * Beam7TIBDir = TIBDir->mkdir("Beam 7"); 

  /* LaserBeams in the TEC+ */
  // {{{ ----- Adc counts for Beam 0 in Ring 4
  theBeam0Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc1PosTEC");
  theBeam0Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc2PosTEC");
  theBeam0Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc3PosTEC");
  theBeam0Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc4PosTEC");
  theBeam0Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc5PosTEC");
  theBeam0Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc6PosTEC");
  theBeam0Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc7PosTEC");
  theBeam0Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc8PosTEC");
  theBeam0Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring4Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 1 in Ring 4
  theBeam1Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc1PosTEC");
  theBeam1Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc2PosTEC");
  theBeam1Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc3PosTEC");
  theBeam1Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc4PosTEC");
  theBeam1Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc5PosTEC");
  theBeam1Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc6PosTEC");
  theBeam1Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc7PosTEC");
  theBeam1Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc8PosTEC");
  theBeam1Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc9PosTEC");
  // }}}

  // plots for TEC2TEC beam 1
  theBeam1Ring4Disc1PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc1PosTEC2TECAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc1PosTEC2TEC");
  theBeam1Ring4Disc2PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc2PosTEC2TECAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc2PosTEC2TEC");
  theBeam1Ring4Disc3PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc3PosTEC2TECAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc3PosTEC2TEC");
  theBeam1Ring4Disc4PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc4PosTEC2TECAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc4PosTEC2TEC");
  theBeam1Ring4Disc5PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc5PosTEC2TECAdcCounts->SetDirectory(Ring4Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring4Disc5PosTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 2 in Ring 4
  theBeam2Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc1PosTEC");
  theBeam2Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc2PosTEC");
  theBeam2Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc3PosTEC");
  theBeam2Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc4PosTEC");
  theBeam2Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc5PosTEC");
  theBeam2Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc6PosTEC");
  theBeam2Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc7PosTEC");
  theBeam2Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc8PosTEC");
  theBeam2Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc9PosTEC");
  // }}}

  // plots for TEC2TEC beam 2
  theBeam2Ring4Disc1PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc1PosTEC2TECAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc1PosTEC2TEC");
  theBeam2Ring4Disc2PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc2PosTEC2TECAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc2PosTEC2TEC");
  theBeam2Ring4Disc3PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc3PosTEC2TECAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc3PosTEC2TEC");
  theBeam2Ring4Disc4PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc4PosTEC2TECAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc4PosTEC2TEC");
  theBeam2Ring4Disc5PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc5PosTEC2TECAdcCounts->SetDirectory(Ring4Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring4Disc5PosTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 3 in Ring 4
  theBeam3Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc1PosTEC");
  theBeam3Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc2PosTEC");
  theBeam3Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc3PosTEC");
  theBeam3Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc4PosTEC");
  theBeam3Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc5PosTEC");
  theBeam3Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc6PosTEC");
  theBeam3Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc7PosTEC");
  theBeam3Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc8PosTEC");
  theBeam3Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring4Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 4 in Ring 4
  theBeam4Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc1PosTEC");
  theBeam4Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc2PosTEC");
  theBeam4Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc3PosTEC");
  theBeam4Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc4PosTEC");
  theBeam4Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc5PosTEC");
  theBeam4Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc6PosTEC");
  theBeam4Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc7PosTEC");
  theBeam4Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc8PosTEC");
  theBeam4Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc9PosTEC");
  // }}}

  // plots for TEC2TEC beam 4
  theBeam4Ring4Disc1PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc1PosTEC2TECAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc1PosTEC2TEC");
  theBeam4Ring4Disc2PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc2PosTEC2TECAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc2PosTEC2TEC");
  theBeam4Ring4Disc3PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc3PosTEC2TECAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc3PosTEC2TEC");
  theBeam4Ring4Disc4PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc4PosTEC2TECAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc4PosTEC2TEC");
  theBeam4Ring4Disc5PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc5PosTEC2TECAdcCounts->SetDirectory(Ring4Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring4Disc5PosTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 5 in Ring 4
  theBeam5Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc1PosTEC");
  theBeam5Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc2PosTEC");
  theBeam5Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc3PosTEC");
  theBeam5Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc4PosTEC");
  theBeam5Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc5PosTEC");
  theBeam5Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc6PosTEC");
  theBeam5Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc7PosTEC");
  theBeam5Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc8PosTEC");
  theBeam5Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring4Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 6 in Ring 4
  theBeam6Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc1PosTEC");
  theBeam6Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc2PosTEC");
  theBeam6Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc3PosTEC");
  theBeam6Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc4PosTEC");
  theBeam6Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc5PosTEC");
  theBeam6Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc6PosTEC");
  theBeam6Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc7PosTEC");
  theBeam6Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc8PosTEC");
  theBeam6Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc9PosTEC");
  // }}}

  // plots for TEC2TEC beam 6
  theBeam6Ring4Disc1PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc1PosTEC2TECAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc1PosTEC2TEC");
  theBeam6Ring4Disc2PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc2PosTEC2TECAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc2PosTEC2TEC");
  theBeam6Ring4Disc3PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc3PosTEC2TECAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc3PosTEC2TEC");
  theBeam6Ring4Disc4PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc4PosTEC2TECAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc4PosTEC2TEC");
  theBeam6Ring4Disc5PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc5PosTEC2TECAdcCounts->SetDirectory(Ring4Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring4Disc5PosTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 7 in Ring 4
  theBeam7Ring4Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc1PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc1PosTEC");
  theBeam7Ring4Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc2PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc2PosTEC");
  theBeam7Ring4Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc3PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc3PosTEC");
  theBeam7Ring4Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc4PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc4PosTEC");
  theBeam7Ring4Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc5PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc5PosTEC");
  theBeam7Ring4Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc6PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc6PosTEC");
  theBeam7Ring4Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc7PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc7PosTEC");
  theBeam7Ring4Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc8PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc8PosTEC");
  theBeam7Ring4Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc9PosAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc9PosTEC");
  // }}}

  // plots for TEC2TEC beam 7
  theBeam7Ring4Disc1PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc1PosTEC2TECAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc1PosTEC2TEC");
  theBeam7Ring4Disc2PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc2PosTEC2TECAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc2PosTEC2TEC");
  theBeam7Ring4Disc3PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc3PosTEC2TECAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc3PosTEC2TEC");
  theBeam7Ring4Disc4PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc4PosTEC2TECAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc4PosTEC2TEC");
  theBeam7Ring4Disc5PosTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc5PosTEC2TECAdcCounts->SetDirectory(Ring4Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring4Disc5PosTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 0 in Ring 6
  theBeam0Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc1PosTEC");
  theBeam0Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc2PosTEC");
  theBeam0Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc3PosTEC");
  theBeam0Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc4PosTEC");
  theBeam0Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc5PosTEC");
  theBeam0Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc6PosTEC");
  theBeam0Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc7PosTEC");
  theBeam0Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc8PosTEC");
  theBeam0Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam0PosDir);
  theHistogramNames.push_back("Beam0Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 1 in Ring 6
  theBeam1Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc1PosTEC");
  theBeam1Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc2PosTEC");
  theBeam1Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc3PosTEC");
  theBeam1Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc4PosTEC");
  theBeam1Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc5PosTEC");
  theBeam1Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc6PosTEC");
  theBeam1Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc7PosTEC");
  theBeam1Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc8PosTEC");
  theBeam1Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam1PosDir);
  theHistogramNames.push_back("Beam1Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 2 in Ring 6
  theBeam2Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc1PosTEC");
  theBeam2Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc2PosTEC");
  theBeam2Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc3PosTEC");
  theBeam2Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc4PosTEC");
  theBeam2Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc5PosTEC");
  theBeam2Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc6PosTEC");
  theBeam2Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc7PosTEC");
  theBeam2Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc8PosTEC");
  theBeam2Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam2PosDir);
  theHistogramNames.push_back("Beam2Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 3 in Ring 6
  theBeam3Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc1PosTEC");
  theBeam3Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc2PosTEC");
  theBeam3Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc3PosTEC");
  theBeam3Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc4PosTEC");
  theBeam3Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc5PosTEC");
  theBeam3Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc6PosTEC");
  theBeam3Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc7PosTEC");
  theBeam3Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc8PosTEC");
  theBeam3Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam3PosDir);
  theHistogramNames.push_back("Beam3Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 4 in Ring 6
  theBeam4Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc1PosTEC");
  theBeam4Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc2PosTEC");
  theBeam4Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc3PosTEC");
  theBeam4Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc4PosTEC");
  theBeam4Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc5PosTEC");
  theBeam4Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc6PosTEC");
  theBeam4Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc7PosTEC");
  theBeam4Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc8PosTEC");
  theBeam4Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam4PosDir);
  theHistogramNames.push_back("Beam4Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 5 in Ring 6
  theBeam5Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc1PosTEC");
  theBeam5Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc2PosTEC");
  theBeam5Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc3PosTEC");
  theBeam5Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc4PosTEC");
  theBeam5Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc5PosTEC");
  theBeam5Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc6PosTEC");
  theBeam5Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc7PosTEC");
  theBeam5Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc8PosTEC");
  theBeam5Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam5PosDir);
  theHistogramNames.push_back("Beam5Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 6 in Ring 6
  theBeam6Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc1PosTEC");
  theBeam6Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc2PosTEC");
  theBeam6Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc3PosTEC");
  theBeam6Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc4PosTEC");
  theBeam6Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc5PosTEC");
  theBeam6Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc6PosTEC");
  theBeam6Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc7PosTEC");
  theBeam6Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc8PosTEC");
  theBeam6Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam6PosDir);
  theHistogramNames.push_back("Beam6Ring6Disc9PosTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 7 in Ring 6
  theBeam7Ring6Disc1PosAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc1PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc1PosTEC");
  theBeam7Ring6Disc2PosAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc2PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc2PosTEC");
  theBeam7Ring6Disc3PosAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc3PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc3PosTEC");
  theBeam7Ring6Disc4PosAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc4PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc4PosTEC");
  theBeam7Ring6Disc5PosAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc5PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc5PosTEC");
  theBeam7Ring6Disc6PosAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc6PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc6PosTEC");
  theBeam7Ring6Disc7PosAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc7PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc7PosTEC");
  theBeam7Ring6Disc8PosAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc8PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc8PosTEC");
  theBeam7Ring6Disc9PosAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc9PosAdcCounts->SetDirectory(Ring6Beam7PosDir);
  theHistogramNames.push_back("Beam7Ring6Disc9PosTEC");
  // }}}

  /* LaserBeams in the TEC- */
  // {{{ ----- Adc counts for Beam 0 in Ring 4
  theBeam0Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc1NegTEC");
  theBeam0Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc2NegTEC");
  theBeam0Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc3NegTEC");
  theBeam0Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc4NegTEC");
  theBeam0Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc5NegTEC");
  theBeam0Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc6NegTEC");
  theBeam0Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc7NegTEC");
  theBeam0Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc8NegTEC");
  theBeam0Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 0 in Ring 4", 512, 0, 511);
  theBeam0Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring4Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 1 in Ring 4
  theBeam1Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc1NegTEC");
  theBeam1Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc2NegTEC");
  theBeam1Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc3NegTEC");
  theBeam1Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc4NegTEC");
  theBeam1Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc5NegTEC");
  theBeam1Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc6NegTEC");
  theBeam1Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc7NegTEC");
  theBeam1Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc8NegTEC");
  theBeam1Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc9NegTEC");
  // }}}

  // plots for TEC2TEC beam 1
  theBeam1Ring4Disc1NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc1NegTEC2TECAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc1NegTEC2TEC");
  theBeam1Ring4Disc2NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc2NegTEC2TECAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc2NegTEC2TEC");
  theBeam1Ring4Disc3NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc3NegTEC2TECAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc3NegTEC2TEC");
  theBeam1Ring4Disc4NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc4NegTEC2TECAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc4NegTEC2TEC");
  theBeam1Ring4Disc5NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);
  theBeam1Ring4Disc5NegTEC2TECAdcCounts->SetDirectory(Ring4Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring4Disc5NegTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 2 in Ring 4
  theBeam2Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc1NegTEC");
  theBeam2Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc2NegTEC");
  theBeam2Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc3NegTEC");
  theBeam2Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc4NegTEC");
  theBeam2Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc5NegTEC");
  theBeam2Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc6NegTEC");
  theBeam2Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc7NegTEC");
  theBeam2Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc8NegTEC");
  theBeam2Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc9NegTEC");
  // }}}

  // plots for TEC2TEC beam 2
  theBeam2Ring4Disc1NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc1NegTEC2TECAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc1NegTEC2TEC");
  theBeam2Ring4Disc2NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc2NegTEC2TECAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc2NegTEC2TEC");
  theBeam2Ring4Disc3NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc3NegTEC2TECAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc3NegTEC2TEC");
  theBeam2Ring4Disc4NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc4NegTEC2TECAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc4NegTEC2TEC");
  theBeam2Ring4Disc5NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);
  theBeam2Ring4Disc5NegTEC2TECAdcCounts->SetDirectory(Ring4Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring4Disc5NegTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 3 in Ring 4
  theBeam3Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc1NegTEC");
  theBeam3Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc2NegTEC");
  theBeam3Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc3NegTEC");
  theBeam3Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc4NegTEC");
  theBeam3Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc5NegTEC");
  theBeam3Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc6NegTEC");
  theBeam3Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc7NegTEC");
  theBeam3Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc8NegTEC");
  theBeam3Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 3 in Ring 4", 512, 0, 511);
  theBeam3Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring4Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 4 in Ring 4
  theBeam4Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc1NegTEC");
  theBeam4Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc2NegTEC");
  theBeam4Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc3NegTEC");
  theBeam4Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc4NegTEC");
  theBeam4Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc5NegTEC");
  theBeam4Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc6NegTEC");
  theBeam4Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc7NegTEC");
  theBeam4Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc8NegTEC");
  theBeam4Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc9NegTEC");
  // }}}

  // plots for TEC2TEC beam 4
  theBeam4Ring4Disc1NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc1NegTEC2TECAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc1NegTEC2TEC");
  theBeam4Ring4Disc2NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc2NegTEC2TECAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc2NegTEC2TEC");
  theBeam4Ring4Disc3NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc3NegTEC2TECAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc3NegTEC2TEC");
  theBeam4Ring4Disc4NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc4NegTEC2TECAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc4NegTEC2TEC");
  theBeam4Ring4Disc5NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);
  theBeam4Ring4Disc5NegTEC2TECAdcCounts->SetDirectory(Ring4Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring4Disc5NegTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 5 in Ring 4
  theBeam5Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc1NegTEC");
  theBeam5Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc2NegTEC");
  theBeam5Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc3NegTEC");
  theBeam5Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc4NegTEC");
  theBeam5Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc5NegTEC");
  theBeam5Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc6NegTEC");
  theBeam5Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc7NegTEC");
  theBeam5Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc8NegTEC");
  theBeam5Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 5 in Ring 4", 512, 0, 511);
  theBeam5Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring4Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 6 in Ring 4
  theBeam6Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc1NegTEC");
  theBeam6Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc2NegTEC");
  theBeam6Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc3NegTEC");
  theBeam6Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc4NegTEC");
  theBeam6Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc5NegTEC");
  theBeam6Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc6NegTEC");
  theBeam6Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc7NegTEC");
  theBeam6Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc8NegTEC");
  theBeam6Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc9NegTEC");
  // }}}

  // plots for TEC2TEC beam 6
  theBeam6Ring4Disc1NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc1NegTEC2TECAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc1NegTEC2TEC");
  theBeam6Ring4Disc2NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc2NegTEC2TECAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc2NegTEC2TEC");
  theBeam6Ring4Disc3NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc3NegTEC2TECAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc3NegTEC2TEC");
  theBeam6Ring4Disc4NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc4NegTEC2TECAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc4NegTEC2TEC");
  theBeam6Ring4Disc5NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);
  theBeam6Ring4Disc5NegTEC2TECAdcCounts->SetDirectory(Ring4Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring4Disc5NegTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 7 in Ring 4
  theBeam7Ring4Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc1NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc1NegTEC");
  theBeam7Ring4Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc2NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc2NegTEC");
  theBeam7Ring4Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc3NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc3NegTEC");
  theBeam7Ring4Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc4NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc4NegTEC");
  theBeam7Ring4Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc5NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc5NegTEC");
  theBeam7Ring4Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc6NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc6NegTEC");
  theBeam7Ring4Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc7NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc7NegTEC");
  theBeam7Ring4Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc8NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc8NegTEC");
  theBeam7Ring4Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc9NegAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc9NegTEC");
  // }}}

  // plots for TEC2TEC beam 7
  theBeam7Ring4Disc1NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc1TEC2TEC","Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc1NegTEC2TECAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc1NegTEC2TEC");
  theBeam7Ring4Disc2NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc2TEC2TEC","Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc2NegTEC2TECAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc2NegTEC2TEC");
  theBeam7Ring4Disc3NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc3TEC2TEC","Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc3NegTEC2TECAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc3NegTEC2TEC");
  theBeam7Ring4Disc4NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc4TEC2TEC","Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc4NegTEC2TECAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc4NegTEC2TEC");
  theBeam7Ring4Disc5NegTEC2TECAdcCounts = new TH1D("AdcCountsDisc5TEC2TEC","Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);
  theBeam7Ring4Disc5NegTEC2TECAdcCounts->SetDirectory(Ring4Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring4Disc5NegTEC2TEC");
  // }}}

  // {{{ ----- Adc counts for Beam 0 in Ring 6
  theBeam0Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc1NegTEC");
  theBeam0Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc2NegTEC");
  theBeam0Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc3NegTEC");
  theBeam0Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc4NegTEC");
  theBeam0Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc5NegTEC");
  theBeam0Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc6NegTEC");
  theBeam0Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc7NegTEC");
  theBeam0Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc8NegTEC");
  theBeam0Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 0 in Ring 6", 512, 0, 511);
  theBeam0Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam0NegDir);
  theHistogramNames.push_back("Beam0Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 1 in Ring 6
  theBeam1Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc1NegTEC");
  theBeam1Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc2NegTEC");
  theBeam1Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc3NegTEC");
  theBeam1Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc4NegTEC");
  theBeam1Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc5NegTEC");
  theBeam1Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc6NegTEC");
  theBeam1Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc7NegTEC");
  theBeam1Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc8NegTEC");
  theBeam1Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 1 in Ring 6", 512, 0, 511);
  theBeam1Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam1NegDir);
  theHistogramNames.push_back("Beam1Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 2 in Ring 6
  theBeam2Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc1NegTEC");
  theBeam2Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc2NegTEC");
  theBeam2Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc3NegTEC");
  theBeam2Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc4NegTEC");
  theBeam2Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc5NegTEC");
  theBeam2Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc6NegTEC");
  theBeam2Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc7NegTEC");
  theBeam2Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc8NegTEC");
  theBeam2Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 2 in Ring 6", 512, 0, 511);
  theBeam2Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam2NegDir);
  theHistogramNames.push_back("Beam2Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 3 in Ring 6
  theBeam3Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc1NegTEC");
  theBeam3Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc2NegTEC");
  theBeam3Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc3NegTEC");
  theBeam3Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc4NegTEC");
  theBeam3Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc5NegTEC");
  theBeam3Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc6NegTEC");
  theBeam3Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc7NegTEC");
  theBeam3Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc8NegTEC");
  theBeam3Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 3 in Ring 6", 512, 0, 511);
  theBeam3Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam3NegDir);
  theHistogramNames.push_back("Beam3Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 4 in Ring 6
  theBeam4Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc1NegTEC");
  theBeam4Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc2NegTEC");
  theBeam4Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc3NegTEC");
  theBeam4Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc4NegTEC");
  theBeam4Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc5NegTEC");
  theBeam4Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc6NegTEC");
  theBeam4Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc7NegTEC");
  theBeam4Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc8NegTEC");
  theBeam4Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 4 in Ring 6", 512, 0, 511);
  theBeam4Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam4NegDir);
  theHistogramNames.push_back("Beam4Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 5 in Ring 6
  theBeam5Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc1NegTEC");
  theBeam5Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc2NegTEC");
  theBeam5Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc3NegTEC");
  theBeam5Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc4NegTEC");
  theBeam5Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc5NegTEC");
  theBeam5Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc6NegTEC");
  theBeam5Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc7NegTEC");
  theBeam5Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc8NegTEC");
  theBeam5Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 5 in Ring 6", 512, 0, 511);
  theBeam5Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam5NegDir);
  theHistogramNames.push_back("Beam5Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 6 in Ring 6
  theBeam6Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc1NegTEC");
  theBeam6Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc2NegTEC");
  theBeam6Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc3NegTEC");
  theBeam6Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc4NegTEC");
  theBeam6Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc5NegTEC");
  theBeam6Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc6NegTEC");
  theBeam6Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc7NegTEC");
  theBeam6Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc8NegTEC");
  theBeam6Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 6 in Ring 6", 512, 0, 511);
  theBeam6Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam6NegDir);
  theHistogramNames.push_back("Beam6Ring6Disc9NegTEC");
  // }}}

  // {{{ ----- Adc counts for Beam 7 in Ring 6
  theBeam7Ring6Disc1NegAdcCounts = new TH1D("AdcCountsDisc1","Adc counts on Disc 1 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc1NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc1NegTEC");
  theBeam7Ring6Disc2NegAdcCounts = new TH1D("AdcCountsDisc2","Adc counts on Disc 2 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc2NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc2NegTEC");
  theBeam7Ring6Disc3NegAdcCounts = new TH1D("AdcCountsDisc3","Adc counts on Disc 3 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc3NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc3NegTEC");
  theBeam7Ring6Disc4NegAdcCounts = new TH1D("AdcCountsDisc4","Adc counts on Disc 4 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc4NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc4NegTEC");
  theBeam7Ring6Disc5NegAdcCounts = new TH1D("AdcCountsDisc5","Adc counts on Disc 5 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc5NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc5NegTEC");
  theBeam7Ring6Disc6NegAdcCounts = new TH1D("AdcCountsDisc6","Adc counts on Disc 6 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc6NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc6NegTEC");
  theBeam7Ring6Disc7NegAdcCounts = new TH1D("AdcCountsDisc7","Adc counts on Disc 7 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc7NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc7NegTEC");
  theBeam7Ring6Disc8NegAdcCounts = new TH1D("AdcCountsDisc8","Adc counts on Disc 8 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc8NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc8NegTEC");
  theBeam7Ring6Disc9NegAdcCounts = new TH1D("AdcCountsDisc9","Adc counts on Disc 9 for Beam 7 in Ring 6", 512, 0, 511);
  theBeam7Ring6Disc9NegAdcCounts->SetDirectory(Ring6Beam7NegDir);
  theHistogramNames.push_back("Beam7Ring6Disc9NegTEC");
  // }}}

  /* LaserBeams in the TOB */
  /**************************************** 
   * the different z positions of the beams
   * are numbered in the following way
   *
   * Position1 = +1040 mm
   * Position2 = +580 mm
   * Position3 = +220 mm
   * Position4 = -140 mm
   * Position5 = -500 mm
   * Position6 = -860 mm
   *****************************************/
  // {{{ ----- Adc Counts in Beam 0
  theBeam0TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 0 at z = 1040 mm", 512, 0, 511);
  theBeam0TOBPosition1AdcCounts->SetDirectory(Beam0TOBDir);
  theHistogramNames.push_back("Beam0TOBPosition1");
  theBeam0TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 0 at z = 580 mm", 512, 0, 511);
  theBeam0TOBPosition2AdcCounts->SetDirectory(Beam0TOBDir);
  theHistogramNames.push_back("Beam0TOBPosition2");
  theBeam0TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 0 at z = 220 mm", 512, 0, 511);
  theBeam0TOBPosition3AdcCounts->SetDirectory(Beam0TOBDir);
  theHistogramNames.push_back("Beam0TOBPosition3");
  theBeam0TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 0 at z = -140 mm", 512, 0, 511);
  theBeam0TOBPosition4AdcCounts->SetDirectory(Beam0TOBDir);
  theHistogramNames.push_back("Beam0TOBPosition4");
  theBeam0TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 0 at z = -500 mm", 512, 0, 511);
  theBeam0TOBPosition5AdcCounts->SetDirectory(Beam0TOBDir);
  theHistogramNames.push_back("Beam0TOBPosition5");
  theBeam0TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 0 at z = -860 mm", 512, 0, 511);
  theBeam0TOBPosition6AdcCounts->SetDirectory(Beam0TOBDir);
  theHistogramNames.push_back("Beam0TOBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 1
  theBeam1TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 1 at z = 1040 mm", 512, 0, 511);
  theBeam1TOBPosition1AdcCounts->SetDirectory(Beam1TOBDir);
  theHistogramNames.push_back("Beam1TOBPosition1");
  theBeam1TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 1 at z = 580 mm", 512, 0, 511);
  theBeam1TOBPosition2AdcCounts->SetDirectory(Beam1TOBDir);
  theHistogramNames.push_back("Beam1TOBPosition2");
  theBeam1TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 1 at z = 220 mm", 512, 0, 511);
  theBeam1TOBPosition3AdcCounts->SetDirectory(Beam1TOBDir);
  theHistogramNames.push_back("Beam1TOBPosition3");
  theBeam1TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 1 at z = -140 mm", 512, 0, 511);
  theBeam1TOBPosition4AdcCounts->SetDirectory(Beam1TOBDir);
  theHistogramNames.push_back("Beam1TOBPosition4");
  theBeam1TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 1 at z = -500 mm", 512, 0, 511);
  theBeam1TOBPosition5AdcCounts->SetDirectory(Beam1TOBDir);
  theHistogramNames.push_back("Beam1TOBPosition5");
  theBeam1TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 1 at z = -860 mm", 512, 0, 511);
  theBeam1TOBPosition6AdcCounts->SetDirectory(Beam1TOBDir);
  theHistogramNames.push_back("Beam1TOBPosition6");
  // }}}
  
  // {{{ ----- Adc Counts in Beam 2
  theBeam2TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 2 at z = 1040 mm", 512, 0, 511);
  theBeam2TOBPosition1AdcCounts->SetDirectory(Beam2TOBDir);
  theHistogramNames.push_back("Beam2TOBPosition1");
  theBeam2TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 2 at z = 580 mm", 512, 0, 511);
  theBeam2TOBPosition2AdcCounts->SetDirectory(Beam2TOBDir);
  theHistogramNames.push_back("Beam2TOBPosition2");
  theBeam2TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 2 at z = 220 mm", 512, 0, 511);
  theBeam2TOBPosition3AdcCounts->SetDirectory(Beam2TOBDir);
  theHistogramNames.push_back("Beam2TOBPosition3");
  theBeam2TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 2 at z = -140 mm", 512, 0, 511);
  theBeam2TOBPosition4AdcCounts->SetDirectory(Beam2TOBDir);
  theHistogramNames.push_back("Beam2TOBPosition4");
  theBeam2TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 2 at z = -500 mm", 512, 0, 511);
  theBeam2TOBPosition5AdcCounts->SetDirectory(Beam2TOBDir);
  theHistogramNames.push_back("Beam2TOBPosition5");
  theBeam2TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 2 at z = -860 mm", 512, 0, 511);
  theBeam2TOBPosition6AdcCounts->SetDirectory(Beam2TOBDir);
  theHistogramNames.push_back("Beam2TOBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 3
  theBeam3TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 3 at z = 1040 mm", 512, 0, 511);
  theBeam3TOBPosition1AdcCounts->SetDirectory(Beam3TOBDir);
  theHistogramNames.push_back("Beam3TOBPosition1");
  theBeam3TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 3 at z = 580 mm", 512, 0, 511);
  theBeam3TOBPosition2AdcCounts->SetDirectory(Beam3TOBDir);
  theHistogramNames.push_back("Beam3TOBPosition2");
  theBeam3TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 3 at z = 220 mm", 512, 0, 511);
  theBeam3TOBPosition3AdcCounts->SetDirectory(Beam3TOBDir);
  theHistogramNames.push_back("Beam3TOBPosition3");
  theBeam3TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 3 at z = -140 mm", 512, 0, 511);
  theBeam3TOBPosition4AdcCounts->SetDirectory(Beam3TOBDir);
  theHistogramNames.push_back("Beam3TOBPosition4");
  theBeam3TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 3 at z = -500 mm", 512, 0, 511);
  theBeam3TOBPosition5AdcCounts->SetDirectory(Beam3TOBDir);
  theHistogramNames.push_back("Beam3TOBPosition5");
  theBeam3TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 3 at z = -860 mm", 512, 0, 511);
  theBeam3TOBPosition6AdcCounts->SetDirectory(Beam3TOBDir);
  theHistogramNames.push_back("Beam3TOBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 4
  theBeam4TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 4 at z = 1040 mm", 512, 0, 511);
  theBeam4TOBPosition1AdcCounts->SetDirectory(Beam4TOBDir);
  theHistogramNames.push_back("Beam4TOBPosition1");
  theBeam4TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 4 at z = 580 mm", 512, 0, 511);
  theBeam4TOBPosition2AdcCounts->SetDirectory(Beam4TOBDir);
  theHistogramNames.push_back("Beam4TOBPosition2");
  theBeam4TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 4 at z = 220 mm", 512, 0, 511);
  theBeam4TOBPosition3AdcCounts->SetDirectory(Beam4TOBDir);
  theHistogramNames.push_back("Beam4TOBPosition3");
  theBeam4TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 4 at z = -140 mm", 512, 0, 511);
  theBeam4TOBPosition4AdcCounts->SetDirectory(Beam4TOBDir);
  theHistogramNames.push_back("Beam4TOBPosition4");
  theBeam4TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 4 at z = -500 mm", 512, 0, 511);
  theBeam4TOBPosition5AdcCounts->SetDirectory(Beam4TOBDir);
  theHistogramNames.push_back("Beam4TOBPosition5");
  theBeam4TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 4 at z = -860 mm", 512, 0, 511);
  theBeam4TOBPosition6AdcCounts->SetDirectory(Beam4TOBDir);
  theHistogramNames.push_back("Beam4TOBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 5
  theBeam5TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 5 at z = 1040 mm", 512, 0, 511);
  theBeam5TOBPosition1AdcCounts->SetDirectory(Beam5TOBDir);
  theHistogramNames.push_back("Beam5TOBPosition1");
  theBeam5TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 5 at z = 580 mm", 512, 0, 511);
  theBeam5TOBPosition2AdcCounts->SetDirectory(Beam5TOBDir);
  theHistogramNames.push_back("Beam5TOBPosition2");
  theBeam5TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 5 at z = 220 mm", 512, 0, 511);
  theBeam5TOBPosition3AdcCounts->SetDirectory(Beam5TOBDir);
  theHistogramNames.push_back("Beam5TOBPosition3");
  theBeam5TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 5 at z = -140 mm", 512, 0, 511);
  theBeam5TOBPosition4AdcCounts->SetDirectory(Beam5TOBDir);
  theHistogramNames.push_back("Beam5TOBPosition4");
  theBeam5TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 5 at z = -500 mm", 512, 0, 511);
  theBeam5TOBPosition5AdcCounts->SetDirectory(Beam5TOBDir);
  theHistogramNames.push_back("Beam5TOBPosition5");
  theBeam5TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 5 at z = -860 mm", 512, 0, 511);
  theBeam5TOBPosition6AdcCounts->SetDirectory(Beam5TOBDir);
  theHistogramNames.push_back("Beam5TOBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 6
  theBeam6TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 6 at z = 1040 mm", 512, 0, 511);
  theBeam6TOBPosition1AdcCounts->SetDirectory(Beam6TOBDir);
  theHistogramNames.push_back("Beam6TOBPosition1");
  theBeam6TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 6 at z = 580 mm", 512, 0, 511);
  theBeam6TOBPosition2AdcCounts->SetDirectory(Beam6TOBDir);
  theHistogramNames.push_back("Beam6TOBPosition2");
  theBeam6TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 6 at z = 220 mm", 512, 0, 511);
  theBeam6TOBPosition3AdcCounts->SetDirectory(Beam6TOBDir);
  theHistogramNames.push_back("Beam6TOBPosition3");
  theBeam6TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 6 at z = -140 mm", 512, 0, 511);
  theBeam6TOBPosition4AdcCounts->SetDirectory(Beam6TOBDir);
  theHistogramNames.push_back("Beam6TOBPosition4");
  theBeam6TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 6 at z = -500 mm", 512, 0, 511);
  theBeam6TOBPosition5AdcCounts->SetDirectory(Beam6TOBDir);
  theHistogramNames.push_back("Beam6TOBPosition5");
  theBeam6TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 6 at z = -860 mm", 512, 0, 511);
  theBeam6TOBPosition6AdcCounts->SetDirectory(Beam6TOBDir);
  theHistogramNames.push_back("Beam6TOBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 7
  theBeam7TOBPosition1AdcCounts = new TH1D("AdcCountsZ=1040mm","Adc counts for Beam 7 at z = 1040 mm", 512, 0, 511);
  theBeam7TOBPosition1AdcCounts->SetDirectory(Beam7TOBDir);
  theHistogramNames.push_back("Beam7TOBPosition1");
  theBeam7TOBPosition2AdcCounts = new TH1D("AdcCountsZ=580mm","Adc counts for Beam 7 at z = 580 mm", 512, 0, 511);
  theBeam7TOBPosition2AdcCounts->SetDirectory(Beam7TOBDir);
  theHistogramNames.push_back("Beam7TOBPosition2");
  theBeam7TOBPosition3AdcCounts = new TH1D("AdcCountsZ=220mm","Adc counts for Beam 7 at z = 220 mm", 512, 0, 511);
  theBeam7TOBPosition3AdcCounts->SetDirectory(Beam7TOBDir);
  theHistogramNames.push_back("Beam7TOBPosition3");
  theBeam7TOBPosition4AdcCounts = new TH1D("AdcCountsZ=-140mm","Adc counts for Beam 7 at z = -140 mm", 512, 0, 511);
  theBeam7TOBPosition4AdcCounts->SetDirectory(Beam7TOBDir);
  theHistogramNames.push_back("Beam7TOBPosition4");
  theBeam7TOBPosition5AdcCounts = new TH1D("AdcCountsZ=-500mm","Adc counts for Beam 7 at z = -500 mm", 512, 0, 511);
  theBeam7TOBPosition5AdcCounts->SetDirectory(Beam7TOBDir);
  theHistogramNames.push_back("Beam7TOBPosition5");
  theBeam7TOBPosition6AdcCounts = new TH1D("AdcCountsZ=-860mm","Adc counts for Beam 7 at z = -860 mm", 512, 0, 511);
  theBeam7TOBPosition6AdcCounts->SetDirectory(Beam7TOBDir);
  theHistogramNames.push_back("Beam7TOBPosition6");
  // }}}

  /* LaserBeams in the TIB */
  /**************************************** 
   * the different z positions of the beams
   * are numbered in the following way
   *
   * Position1 = +620 mm
   * Position2 = +380 mm
   * Position3 = +180 mm
   * Position4 = -100 mm
   * Position5 = -340 mm
   * Position6 = -540 mm
   *****************************************/
  // {{{ ----- Adc Counts in Beam 0
  theBeam0TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 0 at z = 620 mm", 512, 0, 511);
  theBeam0TIBPosition1AdcCounts->SetDirectory(Beam0TIBDir);
  theHistogramNames.push_back("Beam0TIBPosition1");
  theBeam0TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 0 at z = 380 mm", 512, 0, 511);
  theBeam0TIBPosition2AdcCounts->SetDirectory(Beam0TIBDir);
  theHistogramNames.push_back("Beam0TIBPosition2");
  theBeam0TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 0 at z = 180 mm", 512, 0, 511);
  theBeam0TIBPosition3AdcCounts->SetDirectory(Beam0TIBDir);
  theHistogramNames.push_back("Beam0TIBPosition3");
  theBeam0TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 0 at z = -100 mm", 512, 0, 511);
  theBeam0TIBPosition4AdcCounts->SetDirectory(Beam0TIBDir);
  theHistogramNames.push_back("Beam0TIBPosition4");
  theBeam0TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 0 at z = -340 mm", 512, 0, 511);
  theBeam0TIBPosition5AdcCounts->SetDirectory(Beam0TIBDir);
  theHistogramNames.push_back("Beam0TIBPosition5");
  theBeam0TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 0 at z = -540 mm", 512, 0, 511);
  theBeam0TIBPosition6AdcCounts->SetDirectory(Beam0TIBDir);
  theHistogramNames.push_back("Beam0TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 1
  theBeam1TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 1 at z = 620 mm", 512, 0, 511);
  theBeam1TIBPosition1AdcCounts->SetDirectory(Beam1TIBDir);
  theHistogramNames.push_back("Beam1TIBPosition1");
  theBeam1TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 1 at z = 380 mm", 512, 0, 511);
  theBeam1TIBPosition2AdcCounts->SetDirectory(Beam1TIBDir);
  theHistogramNames.push_back("Beam1TIBPosition2");
  theBeam1TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 1 at z = 180 mm", 512, 0, 511);
  theBeam1TIBPosition3AdcCounts->SetDirectory(Beam1TIBDir);
  theHistogramNames.push_back("Beam1TIBPosition3");
  theBeam1TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 1 at z = -100 mm", 512, 0, 511);
  theBeam1TIBPosition4AdcCounts->SetDirectory(Beam1TIBDir);
  theHistogramNames.push_back("Beam1TIBPosition4");
  theBeam1TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 1 at z = -340 mm", 512, 0, 511);
  theBeam1TIBPosition5AdcCounts->SetDirectory(Beam1TIBDir);
  theHistogramNames.push_back("Beam1TIBPosition5");
  theBeam1TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 1 at z = -540 mm", 512, 0, 511);
  theBeam1TIBPosition6AdcCounts->SetDirectory(Beam1TIBDir);
  theHistogramNames.push_back("Beam1TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 2
  theBeam2TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 2 at z = 620 mm", 512, 0, 511);
  theBeam2TIBPosition1AdcCounts->SetDirectory(Beam2TIBDir);
  theHistogramNames.push_back("Beam2TIBPosition1");
  theBeam2TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 2 at z = 380 mm", 512, 0, 511);
  theBeam2TIBPosition2AdcCounts->SetDirectory(Beam2TIBDir);
  theHistogramNames.push_back("Beam2TIBPosition2");
  theBeam2TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 2 at z = 180 mm", 512, 0, 511);
  theBeam2TIBPosition3AdcCounts->SetDirectory(Beam2TIBDir);
  theHistogramNames.push_back("Beam2TIBPosition3");
  theBeam2TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 2 at z = -100 mm", 512, 0, 511);
  theBeam2TIBPosition4AdcCounts->SetDirectory(Beam2TIBDir);
  theHistogramNames.push_back("Beam2TIBPosition4");
  theBeam2TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 2 at z = -340 mm", 512, 0, 511);
  theBeam2TIBPosition5AdcCounts->SetDirectory(Beam2TIBDir);
  theHistogramNames.push_back("Beam2TIBPosition5");
  theBeam2TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 2 at z = -540 mm", 512, 0, 511);
  theBeam2TIBPosition6AdcCounts->SetDirectory(Beam2TIBDir);
  theHistogramNames.push_back("Beam2TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 3
  theBeam3TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 3 at z = 620 mm", 512, 0, 511);
  theBeam3TIBPosition1AdcCounts->SetDirectory(Beam3TIBDir);
  theHistogramNames.push_back("Beam3TIBPosition1");
  theBeam3TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 3 at z = 380 mm", 512, 0, 511);
  theBeam3TIBPosition2AdcCounts->SetDirectory(Beam3TIBDir);
  theHistogramNames.push_back("Beam3TIBPosition2");
  theBeam3TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 3 at z = 180 mm", 512, 0, 511);
  theBeam3TIBPosition3AdcCounts->SetDirectory(Beam3TIBDir);
  theHistogramNames.push_back("Beam3TIBPosition3");
  theBeam3TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 3 at z = -100 mm", 512, 0, 511);
  theBeam3TIBPosition4AdcCounts->SetDirectory(Beam3TIBDir);
  theHistogramNames.push_back("Beam3TIBPosition4");
  theBeam3TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 3 at z = -340 mm", 512, 0, 511);
  theBeam3TIBPosition5AdcCounts->SetDirectory(Beam3TIBDir);
  theHistogramNames.push_back("Beam3TIBPosition5");
  theBeam3TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 3 at z = -540 mm", 512, 0, 511);
  theBeam3TIBPosition6AdcCounts->SetDirectory(Beam3TIBDir);
  theHistogramNames.push_back("Beam3TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 4
  theBeam4TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 4 at z = 620 mm", 512, 0, 511);
  theBeam4TIBPosition1AdcCounts->SetDirectory(Beam4TIBDir);
  theHistogramNames.push_back("Beam4TIBPosition1");
  theBeam4TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 4 at z = 380 mm", 512, 0, 511);
  theBeam4TIBPosition2AdcCounts->SetDirectory(Beam4TIBDir);
  theHistogramNames.push_back("Beam4TIBPosition2");
  theBeam4TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 4 at z = 180 mm", 512, 0, 511);
  theBeam4TIBPosition3AdcCounts->SetDirectory(Beam4TIBDir);
  theHistogramNames.push_back("Beam4TIBPosition3");
  theBeam4TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 4 at z = -100 mm", 512, 0, 511);
  theBeam4TIBPosition4AdcCounts->SetDirectory(Beam4TIBDir);
  theHistogramNames.push_back("Beam4TIBPosition4");
  theBeam4TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 4 at z = -340 mm", 512, 0, 511);
  theBeam4TIBPosition5AdcCounts->SetDirectory(Beam4TIBDir);
  theHistogramNames.push_back("Beam4TIBPosition5");
  theBeam4TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 4 at z = -540 mm", 512, 0, 511);
  theBeam4TIBPosition6AdcCounts->SetDirectory(Beam4TIBDir);
  theHistogramNames.push_back("Beam4TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 5
  theBeam5TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 5 at z = 620 mm", 512, 0, 511);
  theBeam5TIBPosition1AdcCounts->SetDirectory(Beam5TIBDir);
  theHistogramNames.push_back("Beam5TIBPosition1");
  theBeam5TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 5 at z = 380 mm", 512, 0, 511);
  theBeam5TIBPosition2AdcCounts->SetDirectory(Beam5TIBDir);
  theHistogramNames.push_back("Beam5TIBPosition2");
  theBeam5TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 5 at z = 180 mm", 512, 0, 511);
  theBeam5TIBPosition3AdcCounts->SetDirectory(Beam5TIBDir);
  theHistogramNames.push_back("Beam5TIBPosition3");
  theBeam5TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 5 at z = -100 mm", 512, 0, 511);
  theBeam5TIBPosition4AdcCounts->SetDirectory(Beam5TIBDir);
  theHistogramNames.push_back("Beam5TIBPosition4");
  theBeam5TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 5 at z = -340 mm", 512, 0, 511);
  theBeam5TIBPosition5AdcCounts->SetDirectory(Beam5TIBDir);
  theHistogramNames.push_back("Beam5TIBPosition5");
  theBeam5TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 5 at z = -540 mm", 512, 0, 511);
  theBeam5TIBPosition6AdcCounts->SetDirectory(Beam5TIBDir);
  theHistogramNames.push_back("Beam5TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 6
  theBeam6TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 6 at z = 620 mm", 512, 0, 511);
  theBeam6TIBPosition1AdcCounts->SetDirectory(Beam6TIBDir);
  theHistogramNames.push_back("Beam6TIBPosition1");
  theBeam6TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 6 at z = 380 mm", 512, 0, 511);
  theBeam6TIBPosition2AdcCounts->SetDirectory(Beam6TIBDir);
  theHistogramNames.push_back("Beam6TIBPosition2");
  theBeam6TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 6 at z = 180 mm", 512, 0, 511);
  theBeam6TIBPosition3AdcCounts->SetDirectory(Beam6TIBDir);
  theHistogramNames.push_back("Beam6TIBPosition3");
  theBeam6TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 6 at z = -100 mm", 512, 0, 511);
  theBeam6TIBPosition4AdcCounts->SetDirectory(Beam6TIBDir);
  theHistogramNames.push_back("Beam6TIBPosition4");
  theBeam6TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 6 at z = -340 mm", 512, 0, 511);
  theBeam6TIBPosition5AdcCounts->SetDirectory(Beam6TIBDir);
  theHistogramNames.push_back("Beam6TIBPosition5");
  theBeam6TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 6 at z = -540 mm", 512, 0, 511);
  theBeam6TIBPosition6AdcCounts->SetDirectory(Beam6TIBDir);
  theHistogramNames.push_back("Beam6TIBPosition6");
  // }}}

  // {{{ ----- Adc Counts in Beam 7
  theBeam7TIBPosition1AdcCounts = new TH1D("AdcCountsZ=620mm","Adc counts for Beam 7 at z = 620 mm", 512, 0, 511);
  theBeam7TIBPosition1AdcCounts->SetDirectory(Beam7TIBDir);
  theHistogramNames.push_back("Beam7TIBPosition1");
  theBeam7TIBPosition2AdcCounts = new TH1D("AdcCountsZ=380mm","Adc counts for Beam 7 at z = 380 mm", 512, 0, 511);
  theBeam7TIBPosition2AdcCounts->SetDirectory(Beam7TIBDir);
  theHistogramNames.push_back("Beam7TIBPosition2");
  theBeam7TIBPosition3AdcCounts = new TH1D("AdcCountsZ=180mm","Adc counts for Beam 7 at z = 180 mm", 512, 0, 511);
  theBeam7TIBPosition3AdcCounts->SetDirectory(Beam7TIBDir);
  theHistogramNames.push_back("Beam7TIBPosition3");
  theBeam7TIBPosition4AdcCounts = new TH1D("AdcCountsZ=-100mm","Adc counts for Beam 7 at z = -100 mm", 512, 0, 511);
  theBeam7TIBPosition4AdcCounts->SetDirectory(Beam7TIBDir);
  theHistogramNames.push_back("Beam7TIBPosition4");
  theBeam7TIBPosition5AdcCounts = new TH1D("AdcCountsZ=-340mm","Adc counts for Beam 7 at z = -340 mm", 512, 0, 511);
  theBeam7TIBPosition5AdcCounts->SetDirectory(Beam7TIBDir);
  theHistogramNames.push_back("Beam7TIBPosition5");
  theBeam7TIBPosition6AdcCounts = new TH1D("AdcCountsZ=-540mm","Adc counts for Beam 7 at z = -540 mm", 512, 0, 511);
  theBeam7TIBPosition6AdcCounts->SetDirectory(Beam7TIBDir);
  theHistogramNames.push_back("Beam7TIBPosition6");
  // }}}

}
