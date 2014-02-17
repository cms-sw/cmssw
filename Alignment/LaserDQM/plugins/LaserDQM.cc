/** \file LaserDQM.cc
 *  DQM Monitors for Laser Alignment System
 *
 *  $Date: 2011/09/15 13:00:21 $
 *  $Revision: 1.8 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserDQM/plugins/LaserDQM.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" 

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



LaserDQM::LaserDQM(edm::ParameterSet const& theConf) 
  : theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theSearchPhiTIB(theConf.getUntrackedParameter<double>("SearchWindowPhiTIB",0.05)),
    theSearchPhiTOB(theConf.getUntrackedParameter<double>("SearchWindowPhiTOB",0.05)),
    theSearchPhiTEC(theConf.getUntrackedParameter<double>("SearchWindowPhiTEC",0.05)),
    theSearchZTIB(theConf.getUntrackedParameter<double>("SearchWindowZTIB",1.0)),
    theSearchZTOB(theConf.getUntrackedParameter<double>("SearchWindowZTOB",1.0)),
    theDigiProducersList(theConf.getParameter<Parameters>("DigiProducersList")),
    theDQMFileName(theConf.getUntrackedParameter<std::string>("DQMFileName","testDQM.root")),
    theDaqMonitorBEI(),
    theMEBeam0Ring4Disc1PosAdcCounts(0),
    theMEBeam0Ring4Disc2PosAdcCounts(0),
    theMEBeam0Ring4Disc3PosAdcCounts(0),
    theMEBeam0Ring4Disc4PosAdcCounts(0),
    theMEBeam0Ring4Disc5PosAdcCounts(0),
    theMEBeam0Ring4Disc6PosAdcCounts(0),
    theMEBeam0Ring4Disc7PosAdcCounts(0),
    theMEBeam0Ring4Disc8PosAdcCounts(0),
    theMEBeam0Ring4Disc9PosAdcCounts(0),
    // Adc counts for Beam 1 in Ring 4
    theMEBeam1Ring4Disc1PosAdcCounts(0),
    theMEBeam1Ring4Disc2PosAdcCounts(0),
    theMEBeam1Ring4Disc3PosAdcCounts(0),
    theMEBeam1Ring4Disc4PosAdcCounts(0),
    theMEBeam1Ring4Disc5PosAdcCounts(0),
    theMEBeam1Ring4Disc6PosAdcCounts(0),
    theMEBeam1Ring4Disc7PosAdcCounts(0),
    theMEBeam1Ring4Disc8PosAdcCounts(0),
    theMEBeam1Ring4Disc9PosAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam1Ring4Disc1PosTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc2PosTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc3PosTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc4PosTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc5PosTEC2TECAdcCounts(0),
    // Adc counts for Beam 2 in Ring 4
    theMEBeam2Ring4Disc1PosAdcCounts(0),
    theMEBeam2Ring4Disc2PosAdcCounts(0),
    theMEBeam2Ring4Disc3PosAdcCounts(0),
    theMEBeam2Ring4Disc4PosAdcCounts(0),
    theMEBeam2Ring4Disc5PosAdcCounts(0),
    theMEBeam2Ring4Disc6PosAdcCounts(0),
    theMEBeam2Ring4Disc7PosAdcCounts(0),
    theMEBeam2Ring4Disc8PosAdcCounts(0),
    theMEBeam2Ring4Disc9PosAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam2Ring4Disc1PosTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc2PosTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc3PosTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc4PosTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc5PosTEC2TECAdcCounts(0),
    // Adc counts for Beam 3 in Ring 4
    theMEBeam3Ring4Disc1PosAdcCounts(0),
    theMEBeam3Ring4Disc2PosAdcCounts(0),
    theMEBeam3Ring4Disc3PosAdcCounts(0),
    theMEBeam3Ring4Disc4PosAdcCounts(0),
    theMEBeam3Ring4Disc5PosAdcCounts(0),
    theMEBeam3Ring4Disc6PosAdcCounts(0),
    theMEBeam3Ring4Disc7PosAdcCounts(0),
    theMEBeam3Ring4Disc8PosAdcCounts(0),
    theMEBeam3Ring4Disc9PosAdcCounts(0),
    // Adc counts for Beam 4 in Ring 4
    theMEBeam4Ring4Disc1PosAdcCounts(0),
    theMEBeam4Ring4Disc2PosAdcCounts(0),
    theMEBeam4Ring4Disc3PosAdcCounts(0),
    theMEBeam4Ring4Disc4PosAdcCounts(0),
    theMEBeam4Ring4Disc5PosAdcCounts(0),
    theMEBeam4Ring4Disc6PosAdcCounts(0),
    theMEBeam4Ring4Disc7PosAdcCounts(0),
    theMEBeam4Ring4Disc8PosAdcCounts(0),
    theMEBeam4Ring4Disc9PosAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam4Ring4Disc1PosTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc2PosTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc3PosTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc4PosTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc5PosTEC2TECAdcCounts(0),
    // Adc counts for Beam 5 in Ring 4
    theMEBeam5Ring4Disc1PosAdcCounts(0),
    theMEBeam5Ring4Disc2PosAdcCounts(0),
    theMEBeam5Ring4Disc3PosAdcCounts(0),
    theMEBeam5Ring4Disc4PosAdcCounts(0),
    theMEBeam5Ring4Disc5PosAdcCounts(0),
    theMEBeam5Ring4Disc6PosAdcCounts(0),
    theMEBeam5Ring4Disc7PosAdcCounts(0),
    theMEBeam5Ring4Disc8PosAdcCounts(0),
    theMEBeam5Ring4Disc9PosAdcCounts(0),
    // Adc counts for Beam 6 in Ring 4
    theMEBeam6Ring4Disc1PosAdcCounts(0),
    theMEBeam6Ring4Disc2PosAdcCounts(0),
    theMEBeam6Ring4Disc3PosAdcCounts(0),
    theMEBeam6Ring4Disc4PosAdcCounts(0),
    theMEBeam6Ring4Disc5PosAdcCounts(0),
    theMEBeam6Ring4Disc6PosAdcCounts(0),
    theMEBeam6Ring4Disc7PosAdcCounts(0),
    theMEBeam6Ring4Disc8PosAdcCounts(0),
    theMEBeam6Ring4Disc9PosAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam6Ring4Disc1PosTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc2PosTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc3PosTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc4PosTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc5PosTEC2TECAdcCounts(0),
    // Adc counts for Beam 7 in Ring 4
    theMEBeam7Ring4Disc1PosAdcCounts(0),
    theMEBeam7Ring4Disc2PosAdcCounts(0),
    theMEBeam7Ring4Disc3PosAdcCounts(0),
    theMEBeam7Ring4Disc4PosAdcCounts(0),
    theMEBeam7Ring4Disc5PosAdcCounts(0),
    theMEBeam7Ring4Disc6PosAdcCounts(0),
    theMEBeam7Ring4Disc7PosAdcCounts(0),
    theMEBeam7Ring4Disc8PosAdcCounts(0),
    theMEBeam7Ring4Disc9PosAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam7Ring4Disc1PosTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc2PosTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc3PosTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc4PosTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc5PosTEC2TECAdcCounts(0),
    // Adc counts for Beam 0 in Ring 6
    theMEBeam0Ring6Disc1PosAdcCounts(0),
    theMEBeam0Ring6Disc2PosAdcCounts(0),
    theMEBeam0Ring6Disc3PosAdcCounts(0),
    theMEBeam0Ring6Disc4PosAdcCounts(0),
    theMEBeam0Ring6Disc5PosAdcCounts(0),
    theMEBeam0Ring6Disc6PosAdcCounts(0),
    theMEBeam0Ring6Disc7PosAdcCounts(0),
    theMEBeam0Ring6Disc8PosAdcCounts(0),
    theMEBeam0Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 1 in Ring 6
    theMEBeam1Ring6Disc1PosAdcCounts(0),
    theMEBeam1Ring6Disc2PosAdcCounts(0),
    theMEBeam1Ring6Disc3PosAdcCounts(0),
    theMEBeam1Ring6Disc4PosAdcCounts(0),
    theMEBeam1Ring6Disc5PosAdcCounts(0),
    theMEBeam1Ring6Disc6PosAdcCounts(0),
    theMEBeam1Ring6Disc7PosAdcCounts(0),
    theMEBeam1Ring6Disc8PosAdcCounts(0),
    theMEBeam1Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 2 in Ring 6
    theMEBeam2Ring6Disc1PosAdcCounts(0),
    theMEBeam2Ring6Disc2PosAdcCounts(0),
    theMEBeam2Ring6Disc3PosAdcCounts(0),
    theMEBeam2Ring6Disc4PosAdcCounts(0),
    theMEBeam2Ring6Disc5PosAdcCounts(0),
    theMEBeam2Ring6Disc6PosAdcCounts(0),
    theMEBeam2Ring6Disc7PosAdcCounts(0),
    theMEBeam2Ring6Disc8PosAdcCounts(0),
    theMEBeam2Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 3 in Ring 6
    theMEBeam3Ring6Disc1PosAdcCounts(0),
    theMEBeam3Ring6Disc2PosAdcCounts(0),
    theMEBeam3Ring6Disc3PosAdcCounts(0),
    theMEBeam3Ring6Disc4PosAdcCounts(0),
    theMEBeam3Ring6Disc5PosAdcCounts(0),
    theMEBeam3Ring6Disc6PosAdcCounts(0),
    theMEBeam3Ring6Disc7PosAdcCounts(0),
    theMEBeam3Ring6Disc8PosAdcCounts(0),
    theMEBeam3Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 4 in Ring 6
    theMEBeam4Ring6Disc1PosAdcCounts(0),
    theMEBeam4Ring6Disc2PosAdcCounts(0),
    theMEBeam4Ring6Disc3PosAdcCounts(0),
    theMEBeam4Ring6Disc4PosAdcCounts(0),
    theMEBeam4Ring6Disc5PosAdcCounts(0),
    theMEBeam4Ring6Disc6PosAdcCounts(0),
    theMEBeam4Ring6Disc7PosAdcCounts(0),
    theMEBeam4Ring6Disc8PosAdcCounts(0),
    theMEBeam4Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 5 in Ring 6
    theMEBeam5Ring6Disc1PosAdcCounts(0),
    theMEBeam5Ring6Disc2PosAdcCounts(0),
    theMEBeam5Ring6Disc3PosAdcCounts(0),
    theMEBeam5Ring6Disc4PosAdcCounts(0),
    theMEBeam5Ring6Disc5PosAdcCounts(0),
    theMEBeam5Ring6Disc6PosAdcCounts(0),
    theMEBeam5Ring6Disc7PosAdcCounts(0),
    theMEBeam5Ring6Disc8PosAdcCounts(0),
    theMEBeam5Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 6 in Ring 6
    theMEBeam6Ring6Disc1PosAdcCounts(0),
    theMEBeam6Ring6Disc2PosAdcCounts(0),
    theMEBeam6Ring6Disc3PosAdcCounts(0),
    theMEBeam6Ring6Disc4PosAdcCounts(0),
    theMEBeam6Ring6Disc5PosAdcCounts(0),
    theMEBeam6Ring6Disc6PosAdcCounts(0),
    theMEBeam6Ring6Disc7PosAdcCounts(0),
    theMEBeam6Ring6Disc8PosAdcCounts(0),
    theMEBeam6Ring6Disc9PosAdcCounts(0),
    // Adc counts for Beam 7 in Ring 6
    theMEBeam7Ring6Disc1PosAdcCounts(0),
    theMEBeam7Ring6Disc2PosAdcCounts(0),
    theMEBeam7Ring6Disc3PosAdcCounts(0),
    theMEBeam7Ring6Disc4PosAdcCounts(0),
    theMEBeam7Ring6Disc5PosAdcCounts(0),
    theMEBeam7Ring6Disc6PosAdcCounts(0),
    theMEBeam7Ring6Disc7PosAdcCounts(0),
    theMEBeam7Ring6Disc8PosAdcCounts(0),
    theMEBeam7Ring6Disc9PosAdcCounts(0),
    /* Laser Beams in TEC- */
    // Adc counts for Beam 0 in Ring 4
    theMEBeam0Ring4Disc1NegAdcCounts(0),
    theMEBeam0Ring4Disc2NegAdcCounts(0),
    theMEBeam0Ring4Disc3NegAdcCounts(0),
    theMEBeam0Ring4Disc4NegAdcCounts(0),
    theMEBeam0Ring4Disc5NegAdcCounts(0),
    theMEBeam0Ring4Disc6NegAdcCounts(0),
    theMEBeam0Ring4Disc7NegAdcCounts(0),
    theMEBeam0Ring4Disc8NegAdcCounts(0),
    theMEBeam0Ring4Disc9NegAdcCounts(0),
    // Adc counts for Beam 1 in Ring 4
    theMEBeam1Ring4Disc1NegAdcCounts(0),
    theMEBeam1Ring4Disc2NegAdcCounts(0),
    theMEBeam1Ring4Disc3NegAdcCounts(0),
    theMEBeam1Ring4Disc4NegAdcCounts(0),
    theMEBeam1Ring4Disc5NegAdcCounts(0),
    theMEBeam1Ring4Disc6NegAdcCounts(0),
    theMEBeam1Ring4Disc7NegAdcCounts(0),
    theMEBeam1Ring4Disc8NegAdcCounts(0),
    theMEBeam1Ring4Disc9NegAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam1Ring4Disc1NegTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc2NegTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc3NegTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc4NegTEC2TECAdcCounts(0),
    theMEBeam1Ring4Disc5NegTEC2TECAdcCounts(0),
    // Adc counts for Beam 2 in Ring 4
    theMEBeam2Ring4Disc1NegAdcCounts(0),
    theMEBeam2Ring4Disc2NegAdcCounts(0),
    theMEBeam2Ring4Disc3NegAdcCounts(0),
    theMEBeam2Ring4Disc4NegAdcCounts(0),
    theMEBeam2Ring4Disc5NegAdcCounts(0),
    theMEBeam2Ring4Disc6NegAdcCounts(0),
    theMEBeam2Ring4Disc7NegAdcCounts(0),
    theMEBeam2Ring4Disc8NegAdcCounts(0),
    theMEBeam2Ring4Disc9NegAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam2Ring4Disc1NegTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc2NegTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc3NegTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc4NegTEC2TECAdcCounts(0),
    theMEBeam2Ring4Disc5NegTEC2TECAdcCounts(0),
    // Adc counts for Beam 3 in Ring 4
    theMEBeam3Ring4Disc1NegAdcCounts(0),
    theMEBeam3Ring4Disc2NegAdcCounts(0),
    theMEBeam3Ring4Disc3NegAdcCounts(0),
    theMEBeam3Ring4Disc4NegAdcCounts(0),
    theMEBeam3Ring4Disc5NegAdcCounts(0),
    theMEBeam3Ring4Disc6NegAdcCounts(0),
    theMEBeam3Ring4Disc7NegAdcCounts(0),
    theMEBeam3Ring4Disc8NegAdcCounts(0),
    theMEBeam3Ring4Disc9NegAdcCounts(0),
    // Adc counts for Beam 4 in Ring 4
    theMEBeam4Ring4Disc1NegAdcCounts(0),
    theMEBeam4Ring4Disc2NegAdcCounts(0),
    theMEBeam4Ring4Disc3NegAdcCounts(0),
    theMEBeam4Ring4Disc4NegAdcCounts(0),
    theMEBeam4Ring4Disc5NegAdcCounts(0),
    theMEBeam4Ring4Disc6NegAdcCounts(0),
    theMEBeam4Ring4Disc7NegAdcCounts(0),
    theMEBeam4Ring4Disc8NegAdcCounts(0),
    theMEBeam4Ring4Disc9NegAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam4Ring4Disc1NegTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc2NegTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc3NegTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc4NegTEC2TECAdcCounts(0),
    theMEBeam4Ring4Disc5NegTEC2TECAdcCounts(0),
    // Adc counts for Beam 5 in Ring 4
    theMEBeam5Ring4Disc1NegAdcCounts(0),
    theMEBeam5Ring4Disc2NegAdcCounts(0),
    theMEBeam5Ring4Disc3NegAdcCounts(0),
    theMEBeam5Ring4Disc4NegAdcCounts(0),
    theMEBeam5Ring4Disc5NegAdcCounts(0),
    theMEBeam5Ring4Disc6NegAdcCounts(0),
    theMEBeam5Ring4Disc7NegAdcCounts(0),
    theMEBeam5Ring4Disc8NegAdcCounts(0),
    theMEBeam5Ring4Disc9NegAdcCounts(0),
    // Adc counts for Beam 6 in Ring 4
    theMEBeam6Ring4Disc1NegAdcCounts(0),
    theMEBeam6Ring4Disc2NegAdcCounts(0),
    theMEBeam6Ring4Disc3NegAdcCounts(0),
    theMEBeam6Ring4Disc4NegAdcCounts(0),
    theMEBeam6Ring4Disc5NegAdcCounts(0),
    theMEBeam6Ring4Disc6NegAdcCounts(0),
    theMEBeam6Ring4Disc7NegAdcCounts(0),
    theMEBeam6Ring4Disc8NegAdcCounts(0),
    theMEBeam6Ring4Disc9NegAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam6Ring4Disc1NegTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc2NegTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc3NegTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc4NegTEC2TECAdcCounts(0),
    theMEBeam6Ring4Disc5NegTEC2TECAdcCounts(0),
    // Adc counts for Beam 7 in Ring 4
    theMEBeam7Ring4Disc1NegAdcCounts(0),
    theMEBeam7Ring4Disc2NegAdcCounts(0),
    theMEBeam7Ring4Disc3NegAdcCounts(0),
    theMEBeam7Ring4Disc4NegAdcCounts(0),
    theMEBeam7Ring4Disc5NegAdcCounts(0),
    theMEBeam7Ring4Disc6NegAdcCounts(0),
    theMEBeam7Ring4Disc7NegAdcCounts(0),
    theMEBeam7Ring4Disc8NegAdcCounts(0),
    theMEBeam7Ring4Disc9NegAdcCounts(0),
    // plots for TEC2TEC
    theMEBeam7Ring4Disc1NegTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc2NegTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc3NegTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc4NegTEC2TECAdcCounts(0),
    theMEBeam7Ring4Disc5NegTEC2TECAdcCounts(0),
    // Adc counts for Beam 0 in Ring 6
    theMEBeam0Ring6Disc1NegAdcCounts(0),
    theMEBeam0Ring6Disc2NegAdcCounts(0),
    theMEBeam0Ring6Disc3NegAdcCounts(0),
    theMEBeam0Ring6Disc4NegAdcCounts(0),
    theMEBeam0Ring6Disc5NegAdcCounts(0),
    theMEBeam0Ring6Disc6NegAdcCounts(0),
    theMEBeam0Ring6Disc7NegAdcCounts(0),
    theMEBeam0Ring6Disc8NegAdcCounts(0),
    theMEBeam0Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 1 in Ring 6
    theMEBeam1Ring6Disc1NegAdcCounts(0),
    theMEBeam1Ring6Disc2NegAdcCounts(0),
    theMEBeam1Ring6Disc3NegAdcCounts(0),
    theMEBeam1Ring6Disc4NegAdcCounts(0),
    theMEBeam1Ring6Disc5NegAdcCounts(0),
    theMEBeam1Ring6Disc6NegAdcCounts(0),
    theMEBeam1Ring6Disc7NegAdcCounts(0),
    theMEBeam1Ring6Disc8NegAdcCounts(0),
    theMEBeam1Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 2 in Ring 6
    theMEBeam2Ring6Disc1NegAdcCounts(0),
    theMEBeam2Ring6Disc2NegAdcCounts(0),
    theMEBeam2Ring6Disc3NegAdcCounts(0),
    theMEBeam2Ring6Disc4NegAdcCounts(0),
    theMEBeam2Ring6Disc5NegAdcCounts(0),
    theMEBeam2Ring6Disc6NegAdcCounts(0),
    theMEBeam2Ring6Disc7NegAdcCounts(0),
    theMEBeam2Ring6Disc8NegAdcCounts(0),
    theMEBeam2Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 3 in Ring 6
    theMEBeam3Ring6Disc1NegAdcCounts(0),
    theMEBeam3Ring6Disc2NegAdcCounts(0),
    theMEBeam3Ring6Disc3NegAdcCounts(0),
    theMEBeam3Ring6Disc4NegAdcCounts(0),
    theMEBeam3Ring6Disc5NegAdcCounts(0),
    theMEBeam3Ring6Disc6NegAdcCounts(0),
    theMEBeam3Ring6Disc7NegAdcCounts(0),
    theMEBeam3Ring6Disc8NegAdcCounts(0),
    theMEBeam3Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 4 in Ring 6
    theMEBeam4Ring6Disc1NegAdcCounts(0),
    theMEBeam4Ring6Disc2NegAdcCounts(0),
    theMEBeam4Ring6Disc3NegAdcCounts(0),
    theMEBeam4Ring6Disc4NegAdcCounts(0),
    theMEBeam4Ring6Disc5NegAdcCounts(0),
    theMEBeam4Ring6Disc6NegAdcCounts(0),
    theMEBeam4Ring6Disc7NegAdcCounts(0),
    theMEBeam4Ring6Disc8NegAdcCounts(0),
    theMEBeam4Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 5 in Ring 6
    theMEBeam5Ring6Disc1NegAdcCounts(0),
    theMEBeam5Ring6Disc2NegAdcCounts(0),
    theMEBeam5Ring6Disc3NegAdcCounts(0),
    theMEBeam5Ring6Disc4NegAdcCounts(0),
    theMEBeam5Ring6Disc5NegAdcCounts(0),
    theMEBeam5Ring6Disc6NegAdcCounts(0),
    theMEBeam5Ring6Disc7NegAdcCounts(0),
    theMEBeam5Ring6Disc8NegAdcCounts(0),
    theMEBeam5Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 6 in Ring 6
    theMEBeam6Ring6Disc1NegAdcCounts(0),
    theMEBeam6Ring6Disc2NegAdcCounts(0),
    theMEBeam6Ring6Disc3NegAdcCounts(0),
    theMEBeam6Ring6Disc4NegAdcCounts(0),
    theMEBeam6Ring6Disc5NegAdcCounts(0),
    theMEBeam6Ring6Disc6NegAdcCounts(0),
    theMEBeam6Ring6Disc7NegAdcCounts(0),
    theMEBeam6Ring6Disc8NegAdcCounts(0),
    theMEBeam6Ring6Disc9NegAdcCounts(0),
    // Adc counts for Beam 7 in Ring 6
    theMEBeam7Ring6Disc1NegAdcCounts(0),
    theMEBeam7Ring6Disc2NegAdcCounts(0),
    theMEBeam7Ring6Disc3NegAdcCounts(0),
    theMEBeam7Ring6Disc4NegAdcCounts(0),
    theMEBeam7Ring6Disc5NegAdcCounts(0),
    theMEBeam7Ring6Disc6NegAdcCounts(0),
    theMEBeam7Ring6Disc7NegAdcCounts(0),
    theMEBeam7Ring6Disc8NegAdcCounts(0),
    theMEBeam7Ring6Disc9NegAdcCounts(0),
    // TOB Beams
    // Adc counts for Beam 0
    theMEBeam0TOBPosition1AdcCounts(0),
    theMEBeam0TOBPosition2AdcCounts(0),
    theMEBeam0TOBPosition3AdcCounts(0),
    theMEBeam0TOBPosition4AdcCounts(0),
    theMEBeam0TOBPosition5AdcCounts(0),
    theMEBeam0TOBPosition6AdcCounts(0),
    // Adc counts for Beam 1
    theMEBeam1TOBPosition1AdcCounts(0),
    theMEBeam1TOBPosition2AdcCounts(0),
    theMEBeam1TOBPosition3AdcCounts(0),
    theMEBeam1TOBPosition4AdcCounts(0),
    theMEBeam1TOBPosition5AdcCounts(0),
    theMEBeam1TOBPosition6AdcCounts(0),
    // Adc counts for Beam 2
    theMEBeam2TOBPosition1AdcCounts(0),
    theMEBeam2TOBPosition2AdcCounts(0),
    theMEBeam2TOBPosition3AdcCounts(0),
    theMEBeam2TOBPosition4AdcCounts(0),
    theMEBeam2TOBPosition5AdcCounts(0),
    theMEBeam2TOBPosition6AdcCounts(0),
    // Adc counts for Beam 3
    theMEBeam3TOBPosition1AdcCounts(0),
    theMEBeam3TOBPosition2AdcCounts(0),
    theMEBeam3TOBPosition3AdcCounts(0),
    theMEBeam3TOBPosition4AdcCounts(0),
    theMEBeam3TOBPosition5AdcCounts(0),
    theMEBeam3TOBPosition6AdcCounts(0),
    // Adc counts for Beam 4
    theMEBeam4TOBPosition1AdcCounts(0),
    theMEBeam4TOBPosition2AdcCounts(0),
    theMEBeam4TOBPosition3AdcCounts(0),
    theMEBeam4TOBPosition4AdcCounts(0),
    theMEBeam4TOBPosition5AdcCounts(0),
    theMEBeam4TOBPosition6AdcCounts(0),
    // Adc counts for Beam 5
    theMEBeam5TOBPosition1AdcCounts(0),
    theMEBeam5TOBPosition2AdcCounts(0),
    theMEBeam5TOBPosition3AdcCounts(0),
    theMEBeam5TOBPosition4AdcCounts(0),
    theMEBeam5TOBPosition5AdcCounts(0),
    theMEBeam5TOBPosition6AdcCounts(0),
    // Adc counts for Beam 6
    theMEBeam6TOBPosition1AdcCounts(0),
    theMEBeam6TOBPosition2AdcCounts(0),
    theMEBeam6TOBPosition3AdcCounts(0),
    theMEBeam6TOBPosition4AdcCounts(0),
    theMEBeam6TOBPosition5AdcCounts(0),
    theMEBeam6TOBPosition6AdcCounts(0),
    // Adc counts for Beam 7
    theMEBeam7TOBPosition1AdcCounts(0),
    theMEBeam7TOBPosition2AdcCounts(0),
    theMEBeam7TOBPosition3AdcCounts(0),
    theMEBeam7TOBPosition4AdcCounts(0),
    theMEBeam7TOBPosition5AdcCounts(0),
    theMEBeam7TOBPosition6AdcCounts(0),
    // TIB Beams
    // Adc counts for Beam 0
    theMEBeam0TIBPosition1AdcCounts(0),
    theMEBeam0TIBPosition2AdcCounts(0),
    theMEBeam0TIBPosition3AdcCounts(0),
    theMEBeam0TIBPosition4AdcCounts(0),
    theMEBeam0TIBPosition5AdcCounts(0),
    theMEBeam0TIBPosition6AdcCounts(0),
    // Adc counts for Beam 1
    theMEBeam1TIBPosition1AdcCounts(0),
    theMEBeam1TIBPosition2AdcCounts(0),
    theMEBeam1TIBPosition3AdcCounts(0),
    theMEBeam1TIBPosition4AdcCounts(0),
    theMEBeam1TIBPosition5AdcCounts(0),
    theMEBeam1TIBPosition6AdcCounts(0),
    // Adc counts for Beam 2
    theMEBeam2TIBPosition1AdcCounts(0),
    theMEBeam2TIBPosition2AdcCounts(0),
    theMEBeam2TIBPosition3AdcCounts(0),
    theMEBeam2TIBPosition4AdcCounts(0),
    theMEBeam2TIBPosition5AdcCounts(0),
    theMEBeam2TIBPosition6AdcCounts(0),
    // Adc counts for Beam 3
    theMEBeam3TIBPosition1AdcCounts(0),
    theMEBeam3TIBPosition2AdcCounts(0),
    theMEBeam3TIBPosition3AdcCounts(0),
    theMEBeam3TIBPosition4AdcCounts(0),
    theMEBeam3TIBPosition5AdcCounts(0),
    theMEBeam3TIBPosition6AdcCounts(0),
    // Adc counts for Beam 4
    theMEBeam4TIBPosition1AdcCounts(0),
    theMEBeam4TIBPosition2AdcCounts(0),
    theMEBeam4TIBPosition3AdcCounts(0),
    theMEBeam4TIBPosition4AdcCounts(0),
    theMEBeam4TIBPosition5AdcCounts(0),
    theMEBeam4TIBPosition6AdcCounts(0),
    // Adc counts for Beam 5
    theMEBeam5TIBPosition1AdcCounts(0),
    theMEBeam5TIBPosition2AdcCounts(0),
    theMEBeam5TIBPosition3AdcCounts(0),
    theMEBeam5TIBPosition4AdcCounts(0),
    theMEBeam5TIBPosition5AdcCounts(0),
    theMEBeam5TIBPosition6AdcCounts(0),
    // Adc counts for Beam 6
    theMEBeam6TIBPosition1AdcCounts(0),
    theMEBeam6TIBPosition2AdcCounts(0),
    theMEBeam6TIBPosition3AdcCounts(0),
    theMEBeam6TIBPosition4AdcCounts(0),
    theMEBeam6TIBPosition5AdcCounts(0),
    theMEBeam6TIBPosition6AdcCounts(0),
    // Adc counts for Beam 7
    theMEBeam7TIBPosition1AdcCounts(0),
    theMEBeam7TIBPosition2AdcCounts(0),
    theMEBeam7TIBPosition3AdcCounts(0),
    theMEBeam7TIBPosition4AdcCounts(0),
    theMEBeam7TIBPosition5AdcCounts(0),
    theMEBeam7TIBPosition6AdcCounts(0)
{
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

void LaserDQM::analyze(edm::Event const& theEvent, edm::EventSetup const& theSetup) 
{
  // do the Tracker Statistics
  trackerStatistics(theEvent, theSetup);
}

void LaserDQM::beginJob()
{
  // get hold of DQM Backend interface
  theDaqMonitorBEI = edm::Service<DQMStore>().operator->();
      
  // initialize the Monitor Elements
  initMonitors();
}

void LaserDQM::endJob(void)
{
  theDaqMonitorBEI->save(theDQMFileName.c_str());
}

void LaserDQM::fillAdcCounts(MonitorElement * theMonitor, 
			     edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator,
			     edm::DetSet<SiStripDigi>::const_iterator digiRangeIteratorEnd)
{
  // get the ROOT object from the MonitorElement
  TH1F * theMEHistogram = theMonitor->getTH1F();

  // loop over all the digis in this det
  for (; digiRangeIterator != digiRangeIteratorEnd; ++digiRangeIterator) 
    {
      const SiStripDigi *digi = &*digiRangeIterator;
      
      if ( theDebugLevel > 4 ) 
	{ std::cout << " Channel " << digi->channel() << " has " << digi->adc() << " adc counts " << std::endl; }

      // fill the number of adc counts in the histogram
      if (digi->channel() < 512)
	{
	  Double_t theBinContent = theMEHistogram->GetBinContent(digi->channel()) + digi->adc();
	  theMEHistogram->SetBinContent(digi->channel(), theBinContent);
	}
    }
}

// define the SEAL module

DEFINE_FWK_MODULE(LaserDQM);
