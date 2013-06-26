#ifndef LaserDQM_LaserDQM_H
#define LaserDQM_LaserDQM_H

/** \class LaserDQM
 *  DQM Monitor Elements for the Laser Alignment System
 *
 *  $Date: 2009/12/14 22:21:46 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

// DQM
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>

class LaserDQM : public edm::EDAnalyzer
{
 public:
  typedef std::vector<edm::ParameterSet> Parameters;

	/// constructor
  explicit LaserDQM(edm::ParameterSet const& theConf);
	/// destructor
  ~LaserDQM();
  
  /// this method will do the user analysis 
  virtual void analyze(edm::Event const& theEvent, edm::EventSetup const& theSetup);
  /// begin job
  virtual void beginJob();
	/// end job
  virtual void endJob(void);
    
 private:
	/// fill adc counts from the laser beam into a monitor histogram
  void fillAdcCounts(MonitorElement * theMonitor,
		     edm::DetSet<SiStripDigi>::const_iterator digiRangeIterator,
		     edm::DetSet<SiStripDigi>::const_iterator digiRangeIteratorEnd);
	/// initialize monitors
  void initMonitors();
	/// find dets which are hit by a laser beam and fill the monitors
  void trackerStatistics(edm::Event const& theEvent, edm::EventSetup const& theSetup);
  
 private:
  int theDebugLevel;
  double theSearchPhiTIB;
  double theSearchPhiTOB;
  double theSearchPhiTEC;
  double theSearchZTIB;
  double theSearchZTOB;

  // digi producer
  Parameters theDigiProducersList;

  // output file for DQM MonitorElements
  std::string theDQMFileName;

  // DQM Backend Interface
  DQMStore * theDaqMonitorBEI;

  // DQM Monitor Elements

  /* Laser Beams in TEC+ */
  // Adc counts for Beam 0 in Ring 4
  MonitorElement * theMEBeam0Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc9PosAdcCounts;

  // Adc counts for Beam 1 in Ring 4
  MonitorElement * theMEBeam1Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc9PosAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam1Ring4Disc1PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc2PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc3PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc4PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc5PosTEC2TECAdcCounts;

  // Adc counts for Beam 2 in Ring 4
  MonitorElement * theMEBeam2Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc9PosAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam2Ring4Disc1PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc2PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc3PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc4PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc5PosTEC2TECAdcCounts;

  // Adc counts for Beam 3 in Ring 4
  MonitorElement * theMEBeam3Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc9PosAdcCounts;

  // Adc counts for Beam 4 in Ring 4
  MonitorElement * theMEBeam4Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc9PosAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam4Ring4Disc1PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc2PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc3PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc4PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc5PosTEC2TECAdcCounts;

  // Adc counts for Beam 5 in Ring 4
  MonitorElement * theMEBeam5Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc9PosAdcCounts;

  // Adc counts for Beam 6 in Ring 4
  MonitorElement * theMEBeam6Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc9PosAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam6Ring4Disc1PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc2PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc3PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc4PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc5PosTEC2TECAdcCounts;

  // Adc counts for Beam 7 in Ring 4
  MonitorElement * theMEBeam7Ring4Disc1PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc2PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc3PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc4PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc5PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc6PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc7PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc8PosAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc9PosAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam7Ring4Disc1PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc2PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc3PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc4PosTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc5PosTEC2TECAdcCounts;

  // Adc counts for Beam 0 in Ring 6
  MonitorElement * theMEBeam0Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 1 in Ring 6
  MonitorElement * theMEBeam1Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 2 in Ring 6
  MonitorElement * theMEBeam2Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 3 in Ring 6
  MonitorElement * theMEBeam3Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 4 in Ring 6
  MonitorElement * theMEBeam4Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 5 in Ring 6
  MonitorElement * theMEBeam5Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 6 in Ring 6
  MonitorElement * theMEBeam6Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc9PosAdcCounts;

  // Adc counts for Beam 7 in Ring 6
  MonitorElement * theMEBeam7Ring6Disc1PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc2PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc3PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc4PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc5PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc6PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc7PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc8PosAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc9PosAdcCounts;

  /* Laser Beams in TEC- */
  // Adc counts for Beam 0 in Ring 4
  MonitorElement * theMEBeam0Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam0Ring4Disc9NegAdcCounts;

  // Adc counts for Beam 1 in Ring 4
  MonitorElement * theMEBeam1Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc9NegAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam1Ring4Disc1NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc2NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc3NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc4NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam1Ring4Disc5NegTEC2TECAdcCounts;

  // Adc counts for Beam 2 in Ring 4
  MonitorElement * theMEBeam2Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc9NegAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam2Ring4Disc1NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc2NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc3NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc4NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam2Ring4Disc5NegTEC2TECAdcCounts;

  // Adc counts for Beam 3 in Ring 4
  MonitorElement * theMEBeam3Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam3Ring4Disc9NegAdcCounts;

  // Adc counts for Beam 4 in Ring 4
  MonitorElement * theMEBeam4Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc9NegAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam4Ring4Disc1NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc2NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc3NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc4NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam4Ring4Disc5NegTEC2TECAdcCounts;

  // Adc counts for Beam 5 in Ring 4
  MonitorElement * theMEBeam5Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam5Ring4Disc9NegAdcCounts;

  // Adc counts for Beam 6 in Ring 4
  MonitorElement * theMEBeam6Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc9NegAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam6Ring4Disc1NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc2NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc3NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc4NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam6Ring4Disc5NegTEC2TECAdcCounts;

  // Adc counts for Beam 7 in Ring 4
  MonitorElement * theMEBeam7Ring4Disc1NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc2NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc3NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc4NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc5NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc6NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc7NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc8NegAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc9NegAdcCounts;

  // plots for TEC2TEC
  MonitorElement * theMEBeam7Ring4Disc1NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc2NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc3NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc4NegTEC2TECAdcCounts;
  MonitorElement * theMEBeam7Ring4Disc5NegTEC2TECAdcCounts;

  // Adc counts for Beam 0 in Ring 6
  MonitorElement * theMEBeam0Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam0Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 1 in Ring 6
  MonitorElement * theMEBeam1Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam1Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 2 in Ring 6
  MonitorElement * theMEBeam2Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam2Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 3 in Ring 6
  MonitorElement * theMEBeam3Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam3Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 4 in Ring 6
  MonitorElement * theMEBeam4Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam4Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 5 in Ring 6
  MonitorElement * theMEBeam5Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam5Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 6 in Ring 6
  MonitorElement * theMEBeam6Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam6Ring6Disc9NegAdcCounts;

  // Adc counts for Beam 7 in Ring 6
  MonitorElement * theMEBeam7Ring6Disc1NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc2NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc3NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc4NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc5NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc6NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc7NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc8NegAdcCounts;
  MonitorElement * theMEBeam7Ring6Disc9NegAdcCounts;

  // TOB Beams
  // Adc counts for Beam 0
  MonitorElement * theMEBeam0TOBPosition1AdcCounts;
  MonitorElement * theMEBeam0TOBPosition2AdcCounts;
  MonitorElement * theMEBeam0TOBPosition3AdcCounts;
  MonitorElement * theMEBeam0TOBPosition4AdcCounts;
  MonitorElement * theMEBeam0TOBPosition5AdcCounts;
  MonitorElement * theMEBeam0TOBPosition6AdcCounts;

  // Adc counts for Beam 1
  MonitorElement * theMEBeam1TOBPosition1AdcCounts;
  MonitorElement * theMEBeam1TOBPosition2AdcCounts;
  MonitorElement * theMEBeam1TOBPosition3AdcCounts;
  MonitorElement * theMEBeam1TOBPosition4AdcCounts;
  MonitorElement * theMEBeam1TOBPosition5AdcCounts;
  MonitorElement * theMEBeam1TOBPosition6AdcCounts;

  // Adc counts for Beam 2
  MonitorElement * theMEBeam2TOBPosition1AdcCounts;
  MonitorElement * theMEBeam2TOBPosition2AdcCounts;
  MonitorElement * theMEBeam2TOBPosition3AdcCounts;
  MonitorElement * theMEBeam2TOBPosition4AdcCounts;
  MonitorElement * theMEBeam2TOBPosition5AdcCounts;
  MonitorElement * theMEBeam2TOBPosition6AdcCounts;

  // Adc counts for Beam 3
  MonitorElement * theMEBeam3TOBPosition1AdcCounts;
  MonitorElement * theMEBeam3TOBPosition2AdcCounts;
  MonitorElement * theMEBeam3TOBPosition3AdcCounts;
  MonitorElement * theMEBeam3TOBPosition4AdcCounts;
  MonitorElement * theMEBeam3TOBPosition5AdcCounts;
  MonitorElement * theMEBeam3TOBPosition6AdcCounts;

  // Adc counts for Beam 4
  MonitorElement * theMEBeam4TOBPosition1AdcCounts;
  MonitorElement * theMEBeam4TOBPosition2AdcCounts;
  MonitorElement * theMEBeam4TOBPosition3AdcCounts;
  MonitorElement * theMEBeam4TOBPosition4AdcCounts;
  MonitorElement * theMEBeam4TOBPosition5AdcCounts;
  MonitorElement * theMEBeam4TOBPosition6AdcCounts;

  // Adc counts for Beam 5
  MonitorElement * theMEBeam5TOBPosition1AdcCounts;
  MonitorElement * theMEBeam5TOBPosition2AdcCounts;
  MonitorElement * theMEBeam5TOBPosition3AdcCounts;
  MonitorElement * theMEBeam5TOBPosition4AdcCounts;
  MonitorElement * theMEBeam5TOBPosition5AdcCounts;
  MonitorElement * theMEBeam5TOBPosition6AdcCounts;

  // Adc counts for Beam 6
  MonitorElement * theMEBeam6TOBPosition1AdcCounts;
  MonitorElement * theMEBeam6TOBPosition2AdcCounts;
  MonitorElement * theMEBeam6TOBPosition3AdcCounts;
  MonitorElement * theMEBeam6TOBPosition4AdcCounts;
  MonitorElement * theMEBeam6TOBPosition5AdcCounts;
  MonitorElement * theMEBeam6TOBPosition6AdcCounts;

  // Adc counts for Beam 7
  MonitorElement * theMEBeam7TOBPosition1AdcCounts;
  MonitorElement * theMEBeam7TOBPosition2AdcCounts;
  MonitorElement * theMEBeam7TOBPosition3AdcCounts;
  MonitorElement * theMEBeam7TOBPosition4AdcCounts;
  MonitorElement * theMEBeam7TOBPosition5AdcCounts;
  MonitorElement * theMEBeam7TOBPosition6AdcCounts;

  // TIB Beams
  // Adc counts for Beam 0
  MonitorElement * theMEBeam0TIBPosition1AdcCounts;
  MonitorElement * theMEBeam0TIBPosition2AdcCounts;
  MonitorElement * theMEBeam0TIBPosition3AdcCounts;
  MonitorElement * theMEBeam0TIBPosition4AdcCounts;
  MonitorElement * theMEBeam0TIBPosition5AdcCounts;
  MonitorElement * theMEBeam0TIBPosition6AdcCounts;

  // Adc counts for Beam 1
  MonitorElement * theMEBeam1TIBPosition1AdcCounts;
  MonitorElement * theMEBeam1TIBPosition2AdcCounts;
  MonitorElement * theMEBeam1TIBPosition3AdcCounts;
  MonitorElement * theMEBeam1TIBPosition4AdcCounts;
  MonitorElement * theMEBeam1TIBPosition5AdcCounts;
  MonitorElement * theMEBeam1TIBPosition6AdcCounts;

  // Adc counts for Beam 2
  MonitorElement * theMEBeam2TIBPosition1AdcCounts;
  MonitorElement * theMEBeam2TIBPosition2AdcCounts;
  MonitorElement * theMEBeam2TIBPosition3AdcCounts;
  MonitorElement * theMEBeam2TIBPosition4AdcCounts;
  MonitorElement * theMEBeam2TIBPosition5AdcCounts;
  MonitorElement * theMEBeam2TIBPosition6AdcCounts;

  // Adc counts for Beam 3
  MonitorElement * theMEBeam3TIBPosition1AdcCounts;
  MonitorElement * theMEBeam3TIBPosition2AdcCounts;
  MonitorElement * theMEBeam3TIBPosition3AdcCounts;
  MonitorElement * theMEBeam3TIBPosition4AdcCounts;
  MonitorElement * theMEBeam3TIBPosition5AdcCounts;
  MonitorElement * theMEBeam3TIBPosition6AdcCounts;

  // Adc counts for Beam 4
  MonitorElement * theMEBeam4TIBPosition1AdcCounts;
  MonitorElement * theMEBeam4TIBPosition2AdcCounts;
  MonitorElement * theMEBeam4TIBPosition3AdcCounts;
  MonitorElement * theMEBeam4TIBPosition4AdcCounts;
  MonitorElement * theMEBeam4TIBPosition5AdcCounts;
  MonitorElement * theMEBeam4TIBPosition6AdcCounts;

  // Adc counts for Beam 5
  MonitorElement * theMEBeam5TIBPosition1AdcCounts;
  MonitorElement * theMEBeam5TIBPosition2AdcCounts;
  MonitorElement * theMEBeam5TIBPosition3AdcCounts;
  MonitorElement * theMEBeam5TIBPosition4AdcCounts;
  MonitorElement * theMEBeam5TIBPosition5AdcCounts;
  MonitorElement * theMEBeam5TIBPosition6AdcCounts;

  // Adc counts for Beam 6
  MonitorElement * theMEBeam6TIBPosition1AdcCounts;
  MonitorElement * theMEBeam6TIBPosition2AdcCounts;
  MonitorElement * theMEBeam6TIBPosition3AdcCounts;
  MonitorElement * theMEBeam6TIBPosition4AdcCounts;
  MonitorElement * theMEBeam6TIBPosition5AdcCounts;
  MonitorElement * theMEBeam6TIBPosition6AdcCounts;

  // Adc counts for Beam 7
  MonitorElement * theMEBeam7TIBPosition1AdcCounts;
  MonitorElement * theMEBeam7TIBPosition2AdcCounts;
  MonitorElement * theMEBeam7TIBPosition3AdcCounts;
  MonitorElement * theMEBeam7TIBPosition4AdcCounts;
  MonitorElement * theMEBeam7TIBPosition5AdcCounts;
  MonitorElement * theMEBeam7TIBPosition6AdcCounts;

};
#endif
