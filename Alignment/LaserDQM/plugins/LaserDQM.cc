/** \file LaserDQM.cc
 *  DQM Monitors for Laser Alignment System
 *
 *  $Date: 2009/12/14 22:21:46 $
 *  $Revision: 1.7 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserDQM/plugins/LaserDQM.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

LaserDQM::LaserDQM(edm::ParameterSet const &theConf)
    : tTopoToken(esConsumes()),
      tGeoToken(esConsumes()),
      theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel", 0)),
      theSearchPhiTIB(theConf.getUntrackedParameter<double>("SearchWindowPhiTIB", 0.05)),
      theSearchPhiTOB(theConf.getUntrackedParameter<double>("SearchWindowPhiTOB", 0.05)),
      theSearchPhiTEC(theConf.getUntrackedParameter<double>("SearchWindowPhiTEC", 0.05)),
      theSearchZTIB(theConf.getUntrackedParameter<double>("SearchWindowZTIB", 1.0)),
      theSearchZTOB(theConf.getUntrackedParameter<double>("SearchWindowZTOB", 1.0)),
      theDigiProducersList(theConf.getParameter<Parameters>("DigiProducersList")),
      theDQMFileName(theConf.getUntrackedParameter<std::string>("DQMFileName", "testDQM.root")),
      theDaqMonitorBEI(),
      theMEBeam0Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam0Ring4Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 1 in Ring 4
      theMEBeam1Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam1Ring4Disc9PosAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam1Ring4Disc1PosTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc2PosTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc3PosTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc4PosTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc5PosTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 2 in Ring 4
      theMEBeam2Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam2Ring4Disc9PosAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam2Ring4Disc1PosTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc2PosTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc3PosTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc4PosTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc5PosTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 3 in Ring 4
      theMEBeam3Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam3Ring4Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 4 in Ring 4
      theMEBeam4Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam4Ring4Disc9PosAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam4Ring4Disc1PosTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc2PosTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc3PosTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc4PosTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc5PosTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 5 in Ring 4
      theMEBeam5Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam5Ring4Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 6 in Ring 4
      theMEBeam6Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam6Ring4Disc9PosAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam6Ring4Disc1PosTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc2PosTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc3PosTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc4PosTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc5PosTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 7 in Ring 4
      theMEBeam7Ring4Disc1PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc2PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc3PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc4PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc5PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc6PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc7PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc8PosAdcCounts(nullptr),
      theMEBeam7Ring4Disc9PosAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam7Ring4Disc1PosTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc2PosTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc3PosTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc4PosTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc5PosTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 0 in Ring 6
      theMEBeam0Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam0Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 1 in Ring 6
      theMEBeam1Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam1Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 2 in Ring 6
      theMEBeam2Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam2Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 3 in Ring 6
      theMEBeam3Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam3Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 4 in Ring 6
      theMEBeam4Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam4Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 5 in Ring 6
      theMEBeam5Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam5Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 6 in Ring 6
      theMEBeam6Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam6Ring6Disc9PosAdcCounts(nullptr),
      // Adc counts for Beam 7 in Ring 6
      theMEBeam7Ring6Disc1PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc2PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc3PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc4PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc5PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc6PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc7PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc8PosAdcCounts(nullptr),
      theMEBeam7Ring6Disc9PosAdcCounts(nullptr),
      /* Laser Beams in TEC- */
      // Adc counts for Beam 0 in Ring 4
      theMEBeam0Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam0Ring4Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 1 in Ring 4
      theMEBeam1Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam1Ring4Disc9NegAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam1Ring4Disc1NegTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc2NegTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc3NegTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc4NegTEC2TECAdcCounts(nullptr),
      theMEBeam1Ring4Disc5NegTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 2 in Ring 4
      theMEBeam2Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam2Ring4Disc9NegAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam2Ring4Disc1NegTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc2NegTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc3NegTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc4NegTEC2TECAdcCounts(nullptr),
      theMEBeam2Ring4Disc5NegTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 3 in Ring 4
      theMEBeam3Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam3Ring4Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 4 in Ring 4
      theMEBeam4Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam4Ring4Disc9NegAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam4Ring4Disc1NegTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc2NegTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc3NegTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc4NegTEC2TECAdcCounts(nullptr),
      theMEBeam4Ring4Disc5NegTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 5 in Ring 4
      theMEBeam5Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam5Ring4Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 6 in Ring 4
      theMEBeam6Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam6Ring4Disc9NegAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam6Ring4Disc1NegTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc2NegTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc3NegTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc4NegTEC2TECAdcCounts(nullptr),
      theMEBeam6Ring4Disc5NegTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 7 in Ring 4
      theMEBeam7Ring4Disc1NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc2NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc3NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc4NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc5NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc6NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc7NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc8NegAdcCounts(nullptr),
      theMEBeam7Ring4Disc9NegAdcCounts(nullptr),
      // plots for TEC2TEC
      theMEBeam7Ring4Disc1NegTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc2NegTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc3NegTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc4NegTEC2TECAdcCounts(nullptr),
      theMEBeam7Ring4Disc5NegTEC2TECAdcCounts(nullptr),
      // Adc counts for Beam 0 in Ring 6
      theMEBeam0Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam0Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 1 in Ring 6
      theMEBeam1Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam1Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 2 in Ring 6
      theMEBeam2Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam2Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 3 in Ring 6
      theMEBeam3Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam3Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 4 in Ring 6
      theMEBeam4Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam4Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 5 in Ring 6
      theMEBeam5Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam5Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 6 in Ring 6
      theMEBeam6Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam6Ring6Disc9NegAdcCounts(nullptr),
      // Adc counts for Beam 7 in Ring 6
      theMEBeam7Ring6Disc1NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc2NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc3NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc4NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc5NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc6NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc7NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc8NegAdcCounts(nullptr),
      theMEBeam7Ring6Disc9NegAdcCounts(nullptr),
      // TOB Beams
      // Adc counts for Beam 0
      theMEBeam0TOBPosition1AdcCounts(nullptr),
      theMEBeam0TOBPosition2AdcCounts(nullptr),
      theMEBeam0TOBPosition3AdcCounts(nullptr),
      theMEBeam0TOBPosition4AdcCounts(nullptr),
      theMEBeam0TOBPosition5AdcCounts(nullptr),
      theMEBeam0TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 1
      theMEBeam1TOBPosition1AdcCounts(nullptr),
      theMEBeam1TOBPosition2AdcCounts(nullptr),
      theMEBeam1TOBPosition3AdcCounts(nullptr),
      theMEBeam1TOBPosition4AdcCounts(nullptr),
      theMEBeam1TOBPosition5AdcCounts(nullptr),
      theMEBeam1TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 2
      theMEBeam2TOBPosition1AdcCounts(nullptr),
      theMEBeam2TOBPosition2AdcCounts(nullptr),
      theMEBeam2TOBPosition3AdcCounts(nullptr),
      theMEBeam2TOBPosition4AdcCounts(nullptr),
      theMEBeam2TOBPosition5AdcCounts(nullptr),
      theMEBeam2TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 3
      theMEBeam3TOBPosition1AdcCounts(nullptr),
      theMEBeam3TOBPosition2AdcCounts(nullptr),
      theMEBeam3TOBPosition3AdcCounts(nullptr),
      theMEBeam3TOBPosition4AdcCounts(nullptr),
      theMEBeam3TOBPosition5AdcCounts(nullptr),
      theMEBeam3TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 4
      theMEBeam4TOBPosition1AdcCounts(nullptr),
      theMEBeam4TOBPosition2AdcCounts(nullptr),
      theMEBeam4TOBPosition3AdcCounts(nullptr),
      theMEBeam4TOBPosition4AdcCounts(nullptr),
      theMEBeam4TOBPosition5AdcCounts(nullptr),
      theMEBeam4TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 5
      theMEBeam5TOBPosition1AdcCounts(nullptr),
      theMEBeam5TOBPosition2AdcCounts(nullptr),
      theMEBeam5TOBPosition3AdcCounts(nullptr),
      theMEBeam5TOBPosition4AdcCounts(nullptr),
      theMEBeam5TOBPosition5AdcCounts(nullptr),
      theMEBeam5TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 6
      theMEBeam6TOBPosition1AdcCounts(nullptr),
      theMEBeam6TOBPosition2AdcCounts(nullptr),
      theMEBeam6TOBPosition3AdcCounts(nullptr),
      theMEBeam6TOBPosition4AdcCounts(nullptr),
      theMEBeam6TOBPosition5AdcCounts(nullptr),
      theMEBeam6TOBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 7
      theMEBeam7TOBPosition1AdcCounts(nullptr),
      theMEBeam7TOBPosition2AdcCounts(nullptr),
      theMEBeam7TOBPosition3AdcCounts(nullptr),
      theMEBeam7TOBPosition4AdcCounts(nullptr),
      theMEBeam7TOBPosition5AdcCounts(nullptr),
      theMEBeam7TOBPosition6AdcCounts(nullptr),
      // TIB Beams
      // Adc counts for Beam 0
      theMEBeam0TIBPosition1AdcCounts(nullptr),
      theMEBeam0TIBPosition2AdcCounts(nullptr),
      theMEBeam0TIBPosition3AdcCounts(nullptr),
      theMEBeam0TIBPosition4AdcCounts(nullptr),
      theMEBeam0TIBPosition5AdcCounts(nullptr),
      theMEBeam0TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 1
      theMEBeam1TIBPosition1AdcCounts(nullptr),
      theMEBeam1TIBPosition2AdcCounts(nullptr),
      theMEBeam1TIBPosition3AdcCounts(nullptr),
      theMEBeam1TIBPosition4AdcCounts(nullptr),
      theMEBeam1TIBPosition5AdcCounts(nullptr),
      theMEBeam1TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 2
      theMEBeam2TIBPosition1AdcCounts(nullptr),
      theMEBeam2TIBPosition2AdcCounts(nullptr),
      theMEBeam2TIBPosition3AdcCounts(nullptr),
      theMEBeam2TIBPosition4AdcCounts(nullptr),
      theMEBeam2TIBPosition5AdcCounts(nullptr),
      theMEBeam2TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 3
      theMEBeam3TIBPosition1AdcCounts(nullptr),
      theMEBeam3TIBPosition2AdcCounts(nullptr),
      theMEBeam3TIBPosition3AdcCounts(nullptr),
      theMEBeam3TIBPosition4AdcCounts(nullptr),
      theMEBeam3TIBPosition5AdcCounts(nullptr),
      theMEBeam3TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 4
      theMEBeam4TIBPosition1AdcCounts(nullptr),
      theMEBeam4TIBPosition2AdcCounts(nullptr),
      theMEBeam4TIBPosition3AdcCounts(nullptr),
      theMEBeam4TIBPosition4AdcCounts(nullptr),
      theMEBeam4TIBPosition5AdcCounts(nullptr),
      theMEBeam4TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 5
      theMEBeam5TIBPosition1AdcCounts(nullptr),
      theMEBeam5TIBPosition2AdcCounts(nullptr),
      theMEBeam5TIBPosition3AdcCounts(nullptr),
      theMEBeam5TIBPosition4AdcCounts(nullptr),
      theMEBeam5TIBPosition5AdcCounts(nullptr),
      theMEBeam5TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 6
      theMEBeam6TIBPosition1AdcCounts(nullptr),
      theMEBeam6TIBPosition2AdcCounts(nullptr),
      theMEBeam6TIBPosition3AdcCounts(nullptr),
      theMEBeam6TIBPosition4AdcCounts(nullptr),
      theMEBeam6TIBPosition5AdcCounts(nullptr),
      theMEBeam6TIBPosition6AdcCounts(nullptr),
      // Adc counts for Beam 7
      theMEBeam7TIBPosition1AdcCounts(nullptr),
      theMEBeam7TIBPosition2AdcCounts(nullptr),
      theMEBeam7TIBPosition3AdcCounts(nullptr),
      theMEBeam7TIBPosition4AdcCounts(nullptr),
      theMEBeam7TIBPosition5AdcCounts(nullptr),
      theMEBeam7TIBPosition6AdcCounts(nullptr) {
  // load the configuration from the ParameterSet
  edm::LogInfo("LaserDQM") << "==========================================================="
                           << "\n===                Start configuration                  ==="
                           << "\n    theDebugLevel              = " << theDebugLevel
                           << "\n    theSearchPhiTIB            = " << theSearchPhiTIB
                           << "\n    theSearchPhiTOB            = " << theSearchPhiTOB
                           << "\n    theSearchPhiTEC            = " << theSearchPhiTEC
                           << "\n    theSearchZTIB              = " << theSearchZTIB
                           << "\n    theSearchZTOB              = " << theSearchZTOB
                           << "\n    DQM filename               = " << theDQMFileName
                           << "\n===========================================================";
}

LaserDQM::~LaserDQM() {}

void LaserDQM::analyze(edm::Event const &theEvent, edm::EventSetup const &theSetup) {
  // do the Tracker Statistics
  trackerStatistics(theEvent, theSetup);
}

void LaserDQM::beginJob() {
  // get hold of DQM Backend interface
  theDaqMonitorBEI = edm::Service<DQMStore>().operator->();

  // initialize the Monitor Elements
  initMonitors();
}

void LaserDQM::endJob(void) { theDaqMonitorBEI->save(theDQMFileName); }

void LaserDQM::fillAdcCounts(MonitorElement *theMonitor,
                             edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator,
                             edm::DetSet<SiStripDigi>::const_iterator digiRangeIteratorEnd) {
  // get the ROOT object from the MonitorElement
  TH1F *theMEHistogram = theMonitor->getTH1F();

  // loop over all the digis in this det
  for (; digiRangeIterator != digiRangeIteratorEnd; ++digiRangeIterator) {
    const SiStripDigi *digi = &*digiRangeIterator;

    if (theDebugLevel > 4) {
      std::cout << " Channel " << digi->channel() << " has " << digi->adc() << " adc counts " << std::endl;
    }

    // fill the number of adc counts in the histogram
    if (digi->channel() < 512) {
      Double_t theBinContent = theMEHistogram->GetBinContent(digi->channel()) + digi->adc();
      theMEHistogram->SetBinContent(digi->channel(), theBinContent);
    }
  }
}

// define the SEAL module

DEFINE_FWK_MODULE(LaserDQM);
