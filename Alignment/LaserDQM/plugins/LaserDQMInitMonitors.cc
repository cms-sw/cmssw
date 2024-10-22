/** \file LaserDQMInitMonitors.cc
 *  Initialisation of the DQM Monitors
 *
 *  $Date: 2007/12/04 23:54:44 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserDQM/plugins/LaserDQM.h"

void LaserDQM::initMonitors() {
  /* LaserBeams in the TEC+ */
  //  ----- Adc counts for Beam 0 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam0");
  theMEBeam0Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 0 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 1 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam1");
  theMEBeam1Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 1 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 1
  theMEBeam1Ring4Disc1PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc2PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc3PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc4PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc5PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 2 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam2");
  theMEBeam2Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 2 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 2
  theMEBeam2Ring4Disc1PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc2PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc3PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc4PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc5PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 3 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam3");
  theMEBeam3Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 3 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 4 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam4");
  theMEBeam4Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 4 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 4
  theMEBeam4Ring4Disc1PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc2PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc3PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc4PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc5PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 5 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam5");
  theMEBeam5Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 5 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 6 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam6");
  theMEBeam6Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 6 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 6
  theMEBeam6Ring4Disc1PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc2PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc3PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc4PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc5PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 7 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring4/Beam7");
  theMEBeam7Ring4Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 7 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 7
  theMEBeam7Ring4Disc1PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc2PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc3PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc4PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc5PosTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 0 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam0");
  theMEBeam0Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 0 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 1 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam1");
  theMEBeam1Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 1 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 2 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam2");
  theMEBeam2Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 2 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 3 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam3");
  theMEBeam3Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 3 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 4 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam4");
  theMEBeam4Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 4 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 5 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam5");
  theMEBeam5Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 5 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 6 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam6");
  theMEBeam6Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 6 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 7 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/PosTEC/Ring6/Beam7");
  theMEBeam7Ring6Disc1PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc2PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc3PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc4PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc5PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc6PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc7PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc8PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc9PosAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 7 in Ring 6", 512, 0, 511);

  /* LaserBeams in the TEC- */
  // ----- Adc counts for Beam 0 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam0");
  theMEBeam0Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 0 in Ring 4", 512, 0, 511);
  theMEBeam0Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 0 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 1 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam1");
  theMEBeam1Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 1 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 1
  theMEBeam1Ring4Disc1NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc2NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc3NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc4NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 1 in Ring 4", 512, 0, 511);
  theMEBeam1Ring4Disc5NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 1 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 2 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam2");
  theMEBeam2Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 2 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 2
  theMEBeam2Ring4Disc1NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc2NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc3NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc4NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 2 in Ring 4", 512, 0, 511);
  theMEBeam2Ring4Disc5NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 2 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 3 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam3");
  theMEBeam3Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 3 in Ring 4", 512, 0, 511);
  theMEBeam3Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 3 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 4 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam4");
  theMEBeam4Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 4 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 4
  theMEBeam4Ring4Disc1NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc2NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc3NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc4NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 4 in Ring 4", 512, 0, 511);
  theMEBeam4Ring4Disc5NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 4 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 5 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam5");
  theMEBeam5Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 5 in Ring 4", 512, 0, 511);
  theMEBeam5Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 5 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 6 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam6");
  theMEBeam6Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 6 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 6
  theMEBeam6Ring4Disc1NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc2NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc3NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc4NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 6 in Ring 4", 512, 0, 511);
  theMEBeam6Ring4Disc5NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 6 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 7 in Ring 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring4/Beam7");
  theMEBeam7Ring4Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 7 in Ring 4", 512, 0, 511);

  // plots for TEC2TEC beam 7
  theMEBeam7Ring4Disc1NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1TEC2TEC", "Adc counts on Disc 1 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc2NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2TEC2TEC", "Adc counts on Disc 2 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc3NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3TEC2TEC", "Adc counts on Disc 3 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc4NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4TEC2TEC", "Adc counts on Disc 4 for Beam 7 in Ring 4", 512, 0, 511);
  theMEBeam7Ring4Disc5NegTEC2TECAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5TEC2TEC", "Adc counts on Disc 5 for Beam 7 in Ring 4", 512, 0, 511);

  // ----- Adc counts for Beam 0 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam0");
  theMEBeam0Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 0 in Ring 6", 512, 0, 511);
  theMEBeam0Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 0 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 1 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam1");
  theMEBeam1Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 1 in Ring 6", 512, 0, 511);
  theMEBeam1Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 1 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 2 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam2");
  theMEBeam2Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 2 in Ring 6", 512, 0, 511);
  theMEBeam2Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 2 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 3 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam3");
  theMEBeam3Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 3 in Ring 6", 512, 0, 511);
  theMEBeam3Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 3 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 4 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam4");
  theMEBeam4Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 4 in Ring 6", 512, 0, 511);
  theMEBeam4Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 4 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 5 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam5");
  theMEBeam5Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 5 in Ring 6", 512, 0, 511);
  theMEBeam5Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 5 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 6 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam6");
  theMEBeam6Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 6 in Ring 6", 512, 0, 511);
  theMEBeam6Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 6 in Ring 6", 512, 0, 511);

  // ----- Adc counts for Beam 7 in Ring 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/NegTEC/Ring6/Beam7");
  theMEBeam7Ring6Disc1NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc1", "Adc counts on Disc 1 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc2NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc2", "Adc counts on Disc 2 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc3NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc3", "Adc counts on Disc 3 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc4NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc4", "Adc counts on Disc 4 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc5NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc5", "Adc counts on Disc 5 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc6NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc6", "Adc counts on Disc 6 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc7NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc7", "Adc counts on Disc 7 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc8NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc8", "Adc counts on Disc 8 for Beam 7 in Ring 6", 512, 0, 511);
  theMEBeam7Ring6Disc9NegAdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsDisc9", "Adc counts on Disc 9 for Beam 7 in Ring 6", 512, 0, 511);

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
  // ----- Adc Counts in Beam 0
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam0");
  theMEBeam0TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 0 at z = 1040 mm", 512, 0, 511);
  theMEBeam0TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 0 at z = 580 mm", 512, 0, 511);
  theMEBeam0TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 0 at z = 220 mm", 512, 0, 511);
  theMEBeam0TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 0 at z = -140 mm", 512, 0, 511);
  theMEBeam0TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 0 at z = -500 mm", 512, 0, 511);
  theMEBeam0TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 0 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 1
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam1");
  theMEBeam1TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 1 at z = 1040 mm", 512, 0, 511);
  theMEBeam1TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 1 at z = 580 mm", 512, 0, 511);
  theMEBeam1TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 1 at z = 220 mm", 512, 0, 511);
  theMEBeam1TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 1 at z = -140 mm", 512, 0, 511);
  theMEBeam1TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 1 at z = -500 mm", 512, 0, 511);
  theMEBeam1TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 1 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 2
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam2");
  theMEBeam2TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 2 at z = 1040 mm", 512, 0, 511);
  theMEBeam2TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 2 at z = 580 mm", 512, 0, 511);
  theMEBeam2TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 2 at z = 220 mm", 512, 0, 511);
  theMEBeam2TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 2 at z = -140 mm", 512, 0, 511);
  theMEBeam2TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 2 at z = -500 mm", 512, 0, 511);
  theMEBeam2TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 2 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 3
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam3");
  theMEBeam3TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 3 at z = 1040 mm", 512, 0, 511);
  theMEBeam3TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 3 at z = 580 mm", 512, 0, 511);
  theMEBeam3TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 3 at z = 220 mm", 512, 0, 511);
  theMEBeam3TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 3 at z = -140 mm", 512, 0, 511);
  theMEBeam3TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 3 at z = -500 mm", 512, 0, 511);
  theMEBeam3TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 3 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam4");
  theMEBeam4TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 4 at z = 1040 mm", 512, 0, 511);
  theMEBeam4TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 4 at z = 580 mm", 512, 0, 511);
  theMEBeam4TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 4 at z = 220 mm", 512, 0, 511);
  theMEBeam4TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 4 at z = -140 mm", 512, 0, 511);
  theMEBeam4TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 4 at z = -500 mm", 512, 0, 511);
  theMEBeam4TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 4 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 5
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam5");
  theMEBeam5TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 5 at z = 1040 mm", 512, 0, 511);
  theMEBeam5TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 5 at z = 580 mm", 512, 0, 511);
  theMEBeam5TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 5 at z = 220 mm", 512, 0, 511);
  theMEBeam5TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 5 at z = -140 mm", 512, 0, 511);
  theMEBeam5TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 5 at z = -500 mm", 512, 0, 511);
  theMEBeam5TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 5 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam6");
  theMEBeam6TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 6 at z = 1040 mm", 512, 0, 511);
  theMEBeam6TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 6 at z = 580 mm", 512, 0, 511);
  theMEBeam6TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 6 at z = 220 mm", 512, 0, 511);
  theMEBeam6TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 6 at z = -140 mm", 512, 0, 511);
  theMEBeam6TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 6 at z = -500 mm", 512, 0, 511);
  theMEBeam6TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 6 at z = -860 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 7
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TOB/Beam7");
  theMEBeam7TOBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=1040mm", "Adc counts for Beam 7 at z = 1040 mm", 512, 0, 511);
  theMEBeam7TOBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=580mm", "Adc counts for Beam 7 at z = 580 mm", 512, 0, 511);
  theMEBeam7TOBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=220mm", "Adc counts for Beam 7 at z = 220 mm", 512, 0, 511);
  theMEBeam7TOBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-140mm", "Adc counts for Beam 7 at z = -140 mm", 512, 0, 511);
  theMEBeam7TOBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-500mm", "Adc counts for Beam 7 at z = -500 mm", 512, 0, 511);
  theMEBeam7TOBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-860mm", "Adc counts for Beam 7 at z = -860 mm", 512, 0, 511);

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
  // ----- Adc Counts in Beam 0
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam0");
  theMEBeam0TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 0 at z = 620 mm", 512, 0, 511);
  theMEBeam0TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 0 at z = 380 mm", 512, 0, 511);
  theMEBeam0TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 0 at z = 180 mm", 512, 0, 511);
  theMEBeam0TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 0 at z = -100 mm", 512, 0, 511);
  theMEBeam0TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 0 at z = -340 mm", 512, 0, 511);
  theMEBeam0TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 0 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 1
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam1");
  theMEBeam1TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 1 at z = 620 mm", 512, 0, 511);
  theMEBeam1TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 1 at z = 380 mm", 512, 0, 511);
  theMEBeam1TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 1 at z = 180 mm", 512, 0, 511);
  theMEBeam1TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 1 at z = -100 mm", 512, 0, 511);
  theMEBeam1TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 1 at z = -340 mm", 512, 0, 511);
  theMEBeam1TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 1 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 2
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam2");
  theMEBeam2TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 2 at z = 620 mm", 512, 0, 511);
  theMEBeam2TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 2 at z = 380 mm", 512, 0, 511);
  theMEBeam2TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 2 at z = 180 mm", 512, 0, 511);
  theMEBeam2TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 2 at z = -100 mm", 512, 0, 511);
  theMEBeam2TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 2 at z = -340 mm", 512, 0, 511);
  theMEBeam2TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 2 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 3
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam3");
  theMEBeam3TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 3 at z = 620 mm", 512, 0, 511);
  theMEBeam3TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 3 at z = 380 mm", 512, 0, 511);
  theMEBeam3TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 3 at z = 180 mm", 512, 0, 511);
  theMEBeam3TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 3 at z = -100 mm", 512, 0, 511);
  theMEBeam3TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 3 at z = -340 mm", 512, 0, 511);
  theMEBeam3TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 3 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 4
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam4");
  theMEBeam4TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 4 at z = 620 mm", 512, 0, 511);
  theMEBeam4TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 4 at z = 380 mm", 512, 0, 511);
  theMEBeam4TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 4 at z = 180 mm", 512, 0, 511);
  theMEBeam4TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 4 at z = -100 mm", 512, 0, 511);
  theMEBeam4TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 4 at z = -340 mm", 512, 0, 511);
  theMEBeam4TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 4 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 5
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam5");
  theMEBeam5TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 5 at z = 620 mm", 512, 0, 511);
  theMEBeam5TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 5 at z = 380 mm", 512, 0, 511);
  theMEBeam5TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 5 at z = 180 mm", 512, 0, 511);
  theMEBeam5TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 5 at z = -100 mm", 512, 0, 511);
  theMEBeam5TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 5 at z = -340 mm", 512, 0, 511);
  theMEBeam5TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 5 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 6
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam6");
  theMEBeam6TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 6 at z = 620 mm", 512, 0, 511);
  theMEBeam6TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 6 at z = 380 mm", 512, 0, 511);
  theMEBeam6TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 6 at z = 180 mm", 512, 0, 511);
  theMEBeam6TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 6 at z = -100 mm", 512, 0, 511);
  theMEBeam6TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 6 at z = -340 mm", 512, 0, 511);
  theMEBeam6TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 6 at z = -540 mm", 512, 0, 511);

  // ----- Adc Counts in Beam 7
  theDaqMonitorBEI->setCurrentFolder("LaserAlignment/TIB/Beam7");
  theMEBeam7TIBPosition1AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=620mm", "Adc counts for Beam 7 at z = 620 mm", 512, 0, 511);
  theMEBeam7TIBPosition2AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=380mm", "Adc counts for Beam 7 at z = 380 mm", 512, 0, 511);
  theMEBeam7TIBPosition3AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=180mm", "Adc counts for Beam 7 at z = 180 mm", 512, 0, 511);
  theMEBeam7TIBPosition4AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-100mm", "Adc counts for Beam 7 at z = -100 mm", 512, 0, 511);
  theMEBeam7TIBPosition5AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-340mm", "Adc counts for Beam 7 at z = -340 mm", 512, 0, 511);
  theMEBeam7TIBPosition6AdcCounts =
      theDaqMonitorBEI->book1D("AdcCountsZ=-540mm", "Adc counts for Beam 7 at z = -540 mm", 512, 0, 511);
}
