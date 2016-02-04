/*
 *  Offline DQM module for CSC local reconstruction.
 *
 *  Andy Kubik
 *  Northwestern University
 */

#include "DQMOffline/Muon/src/CSCOfflineMonitor.h"

using namespace std;
using namespace edm;


///////////////////
//  CONSTRUCTOR  //
///////////////////
CSCOfflineMonitor::CSCOfflineMonitor(const ParameterSet& pset){

  param = pset;

  stripDigiTag_  = pset.getParameter<edm::InputTag>("stripDigiTag");
  wireDigiTag_   = pset.getParameter<edm::InputTag>("wireDigiTag"); 
  cscRecHitTag_  = pset.getParameter<edm::InputTag>("cscRecHitTag");
  cscSegTag_     = pset.getParameter<edm::InputTag>("cscSegTag");

}

void CSCOfflineMonitor::beginJob(void){
      dbe = Service<DQMStore>().operator->();

      // Occupancies
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/Occupancy");
      hCSCOccupancy = dbe->book1D("hCSCOccupancy","overall CSC occupancy",13,-0.5,12.5);
      hCSCOccupancy->setBinLabel(2,"Total Events");
      hCSCOccupancy->setBinLabel(4,"# Events with Wires");
      hCSCOccupancy->setBinLabel(6,"# Events with Strips");
      hCSCOccupancy->setBinLabel(8,"# Events with Wires&Strips");
      hCSCOccupancy->setBinLabel(10,"# Events with Rechits");
      hCSCOccupancy->setBinLabel(12,"# Events with Segments");
      hOWires = dbe->book2D("hOWires","Wire Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOWires->setAxisTitle("Chamber #");
      hOWires->setBinLabel(1,"ME -4/2",2);
      hOWires->setBinLabel(2,"ME -4/1",2);
      hOWires->setBinLabel(3,"ME -3/2",2);
      hOWires->setBinLabel(4,"ME -2/1",2);
      hOWires->setBinLabel(5,"ME -2/2",2);
      hOWires->setBinLabel(6,"ME -2/1",2);
      hOWires->setBinLabel(7,"ME -1/3",2);
      hOWires->setBinLabel(8,"ME -1/2",2);
      hOWires->setBinLabel(9,"ME -1/1b",2);
      hOWires->setBinLabel(10,"ME -1/1a",2);
      hOWires->setBinLabel(11,"ME +1/1a",2);
      hOWires->setBinLabel(12,"ME +1/1b",2);
      hOWires->setBinLabel(13,"ME +1/2",2);
      hOWires->setBinLabel(14,"ME +1/3",2);
      hOWires->setBinLabel(15,"ME +2/1",2);
      hOWires->setBinLabel(16,"ME +2/2",2);
      hOWires->setBinLabel(17,"ME +3/1",2);
      hOWires->setBinLabel(18,"ME +3/2",2);
      hOWires->setBinLabel(19,"ME +4/1",2);
      hOWires->setBinLabel(20,"ME +4/2",2);
      hOWireSerial = dbe->book1D("hOWireSerial","Wire Occupancy by Chamber Serial",601,-0.5,600.5);
      hOWireSerial->setAxisTitle("Chamber Serial Number");
      hOStrips = dbe->book2D("hOStrips","Strip Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOStrips->setAxisTitle("Chamber #");
      hOStrips->setBinLabel(1,"ME -4/2",2);
      hOStrips->setBinLabel(2,"ME -4/1",2);
      hOStrips->setBinLabel(3,"ME -3/2",2);
      hOStrips->setBinLabel(4,"ME -2/1",2);
      hOStrips->setBinLabel(5,"ME -2/2",2);
      hOStrips->setBinLabel(6,"ME -2/1",2);
      hOStrips->setBinLabel(7,"ME -1/3",2);
      hOStrips->setBinLabel(8,"ME -1/2",2);
      hOStrips->setBinLabel(9,"ME -1/1b",2);
      hOStrips->setBinLabel(10,"ME -1/1a",2);
      hOStrips->setBinLabel(11,"ME +1/1a",2);
      hOStrips->setBinLabel(12,"ME +1/1b",2);
      hOStrips->setBinLabel(13,"ME +1/2",2);
      hOStrips->setBinLabel(14,"ME +1/3",2);
      hOStrips->setBinLabel(15,"ME +2/1",2);
      hOStrips->setBinLabel(16,"ME +2/2",2);
      hOStrips->setBinLabel(17,"ME +3/1",2);
      hOStrips->setBinLabel(18,"ME +3/2",2);
      hOStrips->setBinLabel(19,"ME +4/1",2);
      hOStrips->setBinLabel(20,"ME +4/2",2);
      hOStripSerial = dbe->book1D("hOStripSerial","Strip Occupancy by Chamber Serial",601,-0.5,600.5);
      hOStripSerial->setAxisTitle("Chamber Serial Number");
      hORecHits = dbe->book2D("hORecHits","RecHit Occupancy",36,0.5,36.5,20,0.5,20.5);
      hORecHits->setAxisTitle("Chamber #");
      hORecHits->setBinLabel(1,"ME -4/2",2);
      hORecHits->setBinLabel(2,"ME -4/1",2);
      hORecHits->setBinLabel(3,"ME -3/2",2);
      hORecHits->setBinLabel(4,"ME -2/1",2);
      hORecHits->setBinLabel(5,"ME -2/2",2);
      hORecHits->setBinLabel(6,"ME -2/1",2);
      hORecHits->setBinLabel(7,"ME -1/3",2);
      hORecHits->setBinLabel(8,"ME -1/2",2);
      hORecHits->setBinLabel(9,"ME -1/1b",2);
      hORecHits->setBinLabel(10,"ME -1/1a",2);
      hORecHits->setBinLabel(11,"ME +1/1a",2);
      hORecHits->setBinLabel(12,"ME +1/1b",2);
      hORecHits->setBinLabel(13,"ME +1/2",2);
      hORecHits->setBinLabel(14,"ME +1/3",2);
      hORecHits->setBinLabel(15,"ME +2/1",2);
      hORecHits->setBinLabel(16,"ME +2/2",2);
      hORecHits->setBinLabel(17,"ME +3/1",2);
      hORecHits->setBinLabel(18,"ME +3/2",2);
      hORecHits->setBinLabel(19,"ME +4/1",2);
      hORecHits->setBinLabel(20,"ME +4/2",2);
      hORecHitsSerial = dbe->book1D("hORecHitSerial","RecHit Occupancy by Chamber Serial",601,-0.5,600.5);
      hORecHitsSerial->setAxisTitle("Chamber Serial Number");
      hOSegments = dbe->book2D("hOSegments","Segment Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOSegments->setAxisTitle("Chamber #");
      hOSegments->setBinLabel(1,"ME -4/2",2);
      hOSegments->setBinLabel(2,"ME -4/1",2);
      hOSegments->setBinLabel(3,"ME -3/2",2);
      hOSegments->setBinLabel(4,"ME -2/1",2);
      hOSegments->setBinLabel(5,"ME -2/2",2);
      hOSegments->setBinLabel(6,"ME -2/1",2);
      hOSegments->setBinLabel(7,"ME -1/3",2);
      hOSegments->setBinLabel(8,"ME -1/2",2);
      hOSegments->setBinLabel(9,"ME -1/1b",2);
      hOSegments->setBinLabel(10,"ME -1/1a",2);
      hOSegments->setBinLabel(11,"ME +1/1a",2);
      hOSegments->setBinLabel(12,"ME +1/1b",2);
      hOSegments->setBinLabel(13,"ME +1/2",2);
      hOSegments->setBinLabel(14,"ME +1/3",2);
      hOSegments->setBinLabel(15,"ME +2/1",2);
      hOSegments->setBinLabel(16,"ME +2/2",2);
      hOSegments->setBinLabel(17,"ME +3/1",2);
      hOSegments->setBinLabel(18,"ME +3/2",2);
      hOSegments->setBinLabel(19,"ME +4/1",2);
      hOSegments->setBinLabel(20,"ME +4/2",2);
      hOSegmentsSerial = dbe->book1D("hOSegmentSerial","Segment Occupancy by Chamber Serial",601,-0.5,600.5);
      hOSegmentsSerial->setAxisTitle("Chamber Serial Number");

      // wire digis
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/Digis");
      hWirenGroupsTotal = dbe->book1D("hWirenGroupsTotal","Fired Wires per Event; # Wiregroups Fired",61,-0.5,60.5);
      hWireTBin.push_back(dbe->book1D("hWireTBin_m42","Wire TBin Fired (ME -4/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m41","Wire TBin Fired (ME -4/1); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m32","Wire TBin Fired (ME -3/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m31","Wire TBin Fired (ME -3/1); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m22","Wire TBin Fired (ME -2/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m21","Wire TBin Fired (ME -2/1); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m11a","Wire TBin Fired (ME -1/1a); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m13","Wire TBin Fired (ME -1/3); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m12","Wire TBin Fired (ME -1/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_m11b","Wire TBin Fired (ME -1/1b); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p11b","Wire TBin Fired (ME +1/1b); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p12","Wire TBin Fired (ME +1/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p13","Wire TBin Fired (ME +1/3); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p11a","Wire TBin Fired (ME +1/1a); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p21","Wire TBin Fired (ME +2/1); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p22","Wire TBin Fired (ME +2/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p31","Wire TBin Fired (ME +3/1); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p32","Wire TBin Fired (ME +3/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p41","Wire TBin Fired (ME +4/1); Time Bin (25ns)",17,-0.5,16.5));
      hWireTBin.push_back(dbe->book1D("hWireTBin_p42","Wire TBin Fired (ME +4/2); Time Bin (25ns)",17,-0.5,16.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m42","Wiregroup Number Fired (ME -4/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m41","Wiregroup Number Fired (ME -4/1); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m32","Wiregroup Number Fired (ME -3/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m31","Wiregroup Number Fired (ME -3/1); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m22","Wiregroup Number Fired (ME -2/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m21","Wiregroup Number Fired (ME -2/1); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m11a","Wiregroup Number Fired (ME -1/1a); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m13","Wiregroup Number Fired (ME -1/3); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m12","Wiregroup Number Fired (ME -1/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_m11b","Wiregroup Number Fired (ME -1/1b); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p11b","Wiregroup Number Fired (ME +1/1b); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p12","Wiregroup Number Fired (ME +1/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p13","Wiregroup Number Fired (ME +1/3); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p11a","Wiregroup Number Fired (ME +1/1a); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p21","Wiregroup Number Fired (ME +2/1); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p22","Wiregroup Number Fired (ME +2/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p31","Wiregroup Number Fired (ME +3/1); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p32","Wiregroup Number Fired (ME +3/2); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p41","Wiregroup Number Fired (ME +4/1); Wiregroup #",113,-0.5,112.5));
      hWireNumber.push_back(dbe->book1D("hWireNumber_p42","Wiregroup Number Fired (ME +4/2); Wiregroup #",113,-0.5,112.5));

      // strip digis
      hStripNFired = dbe->book1D("hStripNFired","Fired Strips per Event; # Strips Fired (above 13 ADC)",101,-0.5,100.5);
      hStripNumber.push_back(dbe->book1D("hStripNumber_m42","Strip Number Fired (ME -4/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m41","Strip Number Fired (ME -4/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m32","Strip Number Fired (ME -3/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m31","Strip Number Fired (ME -3/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m22","Strip Number Fired (ME -2/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m21","Strip Number Fired (ME -2/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m11a","Strip Number Fired (ME -1/1a); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m13","Strip Number Fired (ME -1/3); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m12","Strip Number Fired (ME -1/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_m11b","Strip Number Fired (ME -1/1b); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p11b","Strip Number Fired (ME +1/1b); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p12","Strip Number Fired (ME +1/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p13","Strip Number Fired (ME +1/3); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p11a","Strip Number Fired (ME +1/1a); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p21","Strip Number Fired (ME +2/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p22","Strip Number Fired (ME +2/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p31","Strip Number Fired (ME +3/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p32","Strip Number Fired (ME +3/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p41","Strip Number Fired (ME +4/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
      hStripNumber.push_back(dbe->book1D("hStripNumber_p42","Stripgroup Number Fired (ME +4/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));

      //Pedestal Noise Plots
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/PedestalNoise");
      hStripPed.push_back(dbe->book1D("hStripPedMEm42","Pedestal Noise Distribution Chamber ME -4/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm41","Pedestal Noise Distribution Chamber ME -4/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm32","Pedestal Noise Distribution Chamber ME -3/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm31","Pedestal Noise Distribution Chamber ME -3/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm22","Pedestal Noise Distribution Chamber ME -2/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm21","Pedestal Noise Distribution Chamber ME -2/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm11a","Pedestal Noise Distribution Chamber ME -1/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm13","Pedestal Noise Distribution Chamber ME -1/3; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm12","Pedestal Noise Distribution Chamber ME -1/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm11b","Pedestal Noise Distribution Chamber ME -1/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp11b","Pedestal Noise Distribution Chamber ME +1/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp12","Pedestal Noise Distribution Chamber ME +1/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp13","Pedestal Noise Distribution Chamber ME +1/3; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp11a","Pedestal Noise Distribution Chamber ME +1/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp21","Pedestal Noise Distribution Chamber ME +2/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp22","Pedestal Noise Distribution Chamber ME +2/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp31","Pedestal Noise Distribution Chamber ME +3/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp32","Pedestal Noise Distribution Chamber ME +3/2; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp41","Pedestal Noise Distribution Chamber ME +4/1; ADC Counts",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp42","Pedestal Noise Distribution Chamber ME +4/2; ADC Counts",50,-25.,25.));

      // recHits
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/recHits");
      hRHnrechits = dbe->book1D("hRHnrechits","recHits per Event (all chambers); # of RecHits",50,0,50);
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp1","recHit global X,Y station +1; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp2","recHit global X,Y station +2; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp3","recHit global X,Y station +3; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp4","recHit global X,Y station +4; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm1","recHit global X,Y station -1; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm2","recHit global X,Y station -2; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm3","recHit global X,Y station -3; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm4","recHit global X,Y station -4; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm42","Sum 3x3 recHit Charge (ME -4/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm41","Sum 3x3 recHit Charge (ME -4/1); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm32","Sum 3x3 recHit Charge (ME -3/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm31","Sum 3x3 recHit Charge (ME -3/1); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm22","Sum 3x3 recHit Charge (ME -2/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm21","Sum 3x3 recHit Charge (ME -2/1); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm11a","Sum 3x3 recHit Charge (ME -1/1a); ADC counts",100,0,4000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm13","Sum 3x3 recHit Charge (ME -1/3); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm12","Sum 3x3 recHit Charge (ME -1/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm11b","Sum 3x3 recHit Charge (ME -1/1b); ADC counts",100,0,4000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp11b","Sum 3x3 recHit Charge (ME +1/1b); ADC counts",100,0,4000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp12","Sum 3x3 recHit Charge (ME +1/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp13","Sum 3x3 recHit Charge (ME +1/3); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp11a","Sum 3x3 recHit Charge (ME +1/1a); ADC counts",100,0,4000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp21","Sum 3x3 recHit Charge (ME +2/1); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp22","Sum 3x3 recHit Charge (ME +2/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp31","Sum 3x3 recHit Charge (ME +3/1); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp32","Sum 3x3 recHit Charge (ME +3/2); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp41","Sum 3x3 recHit Charge (ME +4/1); ADC counts",100,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp42","Sum 3x3 recHit Charge (ME +4/2); ADC counts",100,0,2000));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm42","Charge Ratio (Ql+Qr)/Qt (ME -4/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm41","Charge Ratio (Ql+Qr)/Qt (ME -4/1); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm32","Charge Ratio (Ql+Qr)/Qt (ME -3/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm31","Charge Ratio (Ql+Qr)/Qt (ME -3/1); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm22","Charge Ratio (Ql+Qr)/Qt (ME -2/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm21","Charge Ratio (Ql+Qr)/Qt (ME -2/1); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm11a","Charge Ratio (Ql+Qr)/Qt (ME -1/1a); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm13","Charge Ratio (Ql+Qr)/Qt (ME -1/3); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm12","Charge Ratio (Ql+Qr)/Qt (ME -1/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm11b","Charge Ratio (Ql+Qr)/Qt (ME -1/1b); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp11b","Charge Ratio (Ql+Qr)/Qt (ME +1/1b); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp12","Charge Ratio (Ql+Qr)/Qt (ME +1/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp13","Charge Ratio (Ql+Qr)/Qt (ME +1/3); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp11a","Charge Ratio (Ql+Qr)/Qt (ME +1/1a); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp21","Charge Ratio (Ql+Qr)/Qt (ME +2/1); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp22","Charge Ratio (Ql+Qr)/Qt (ME +2/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp31","Charge Ratio (Ql+Qr)/Qt (ME +3/1); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp32","Charge Ratio (Ql+Qr)/Qt (ME +3/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp41","Charge Ratio (Ql+Qr)/Qt (ME +4/1); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp42","Charge Ratio (Ql+Qr)/Qt (ME +4/2); (Ql+Qr)/Qt",100,-0.1,1.1));
      hRHTiming.push_back(dbe->book1D("hRHTimingm42","recHit Time (ME -4/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm41","recHit Time (ME -4/1); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm32","recHit Time (ME -3/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm31","recHit Time (ME -3/1); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm22","recHit Time (ME -2/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm21","recHit Time (ME -2/1); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm11a","recHit Time (ME -1/1a); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm13","recHit Time (ME -1/3); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm12","recHit Time (ME -1/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingm11b","recHit Time (ME -1/1b); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp11b","recHit Time (ME +1/1b); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp12","recHit Time (ME +1/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp13","recHit Time (ME +1/3); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp11a","recHit Time (ME +1/1a); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp21","recHit Time (ME +2/1); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp22","recHit Time (ME +2/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp31","recHit Time (ME +3/1); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp32","recHit Time (ME +3/2); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp41","recHit Time (ME +4/1); ns",200,-500.,500.));
      hRHTiming.push_back(dbe->book1D("hRHTimingp42","recHit Time (ME +4/2); ns",200,-500.,500.));
      hRHstpos.push_back(dbe->book1D("hRHstposm42","Reconstructed Position on Strip (ME -4/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm41","Reconstructed Position on Strip (ME -4/1); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm32","Reconstructed Position on Strip (ME -3/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm31","Reconstructed Position on Strip (ME -3/1); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm22","Reconstructed Position on Strip (ME -2/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm21","Reconstructed Position on Strip (ME -2/1); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm11a","Reconstructed Position on Strip (ME -1/1a); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm13","Reconstructed Position on Strip (ME -1/3); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm12","Reconstructed Position on Strip (ME -1/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposm11b","Reconstructed Position on Strip (ME -1/1b); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp11b","Reconstructed Position on Strip (ME +1/1b); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp12","Reconstructed Position on Strip (ME +1/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp13","Reconstructed Position on Strip (ME +1/3); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp11a","Reconstructed Position on Strip (ME +1/1a); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp21","Reconstructed Position on Strip (ME +2/1); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp22","Reconstructed Position on Strip (ME +2/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp31","Reconstructed Position on Strip (ME +3/1); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp32","Reconstructed Position on Strip (ME +3/2); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp41","Reconstructed Position on Strip (ME +4/1); Strip Widths",120,-0.6,0.6));
      hRHstpos.push_back(dbe->book1D("hRHstposp42","Reconstructed Position on Strip (ME +4/2); Strip Widths",120,-0.6,0.6));
      hRHsterr.push_back(dbe->book1D("hRHsterrm42","Estimated Error on Strip Measurement (ME -4/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm41","Estimated Error on Strip Measurement (ME -4/1); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm32","Estimated Error on Strip Measurement (ME -3/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm31","Estimated Error on Strip Measurement (ME -3/1); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm22","Estimated Error on Strip Measurement (ME -2/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm21","Estimated Error on Strip Measurement (ME -2/1); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm11a","Estimated Error on Strip Measurement (ME -1/1a); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm13","Estimated Error on Strip Measurement (ME -1/3); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm12","Estimated Error on Strip Measurement (ME -1/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrm11b","Estimated Error on Strip Measurement (ME -1/1b); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp11b","Estimated Error on Strip Measurement (ME +1/1b); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp12","Estimated Error on Strip Measurement (ME +1/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp13","Estimated Error on Strip Measurement (ME +1/3); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp11a","Estimated Error on Strip Measurement (ME +1/1a); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp21","Estimated Error on Strip Measurement (ME +2/1); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp22","Estimated Error on Strip Measurement (ME +2/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp31","Estimated Error on Strip Measurement (ME +3/1); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp32","Estimated Error on Strip Measurement (ME +3/2); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp41","Estimated Error on Strip Measurement (ME +4/1); Strip Widths",75,-0.01,0.24));
      hRHsterr.push_back(dbe->book1D("hRHsterrp42","Estimated Error on Strip Measurement (ME +4/2); Strip Widths",75,-0.01,0.24));


      // segments
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/Segments");
      hSnSegments   = dbe->book1D("hSnSegments","Number of Segments per Event; # of Segments",11,-0.5,10.5);
      hSnhitsAll = dbe->book1D("hSnhits","N hits on Segments; # of hits",8,-0.5,7.5);
      hSnhits.push_back(dbe->book1D("hSnhitsm42","# of hits on Segments (ME -4/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm41","# of hits on Segments (ME -4/1); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm32","# of hits on Segments (ME -3/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm31","# of hits on Segments (ME -3/1); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm22","# of hits on Segments (ME -2/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm21","# of hits on Segments (ME -2/1); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm11a","# of hits on Segments (ME -1/1a); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm13","# of hits on Segments (ME -1/3); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm12","# of hits on Segments (ME -1/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsm11b","# of hits on Segments (ME -1/1b); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp11b","# of hits on Segments (ME +1/1b); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp12","# of hits on Segments (ME +1/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp13","# of hits on Segments (ME +1/3); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp11a","# of hits on Segments (ME +1/1a); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp21","# of hits on Segments (ME +2/1); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp22","# of hits on Segments (ME +2/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp31","# of hits on Segments (ME +3/1); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp32","# of hits on Segments (ME +3/2); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp41","# of hits on Segments (ME +4/1); # of hits",8,-0.5,7.5));
      hSnhits.push_back(dbe->book1D("hSnhitsp42","# of hits on Segments (ME +4/2); # of hits",8,-0.5,7.5));
      hSChiSqAll = dbe->book1D("hSChiSq","Segment Normalized Chi2; Chi2/ndof",110,-0.05,10.5);
      hSChiSq.push_back(dbe->book1D("hSChiSqm42","Segment Normalized Chi2 (ME -4/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm41","Segment Normalized Chi2 (ME -4/1); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm32","Segment Normalized Chi2 (ME -3/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm31","Segment Normalized Chi2 (ME -3/1); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm22","Segment Normalized Chi2 (ME -2/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm21","Segment Normalized Chi2 (ME -2/1); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm11a","Segment Normalized Chi2 (ME -1/1a); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm13","Segment Normalized Chi2 (ME -1/3); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm12","Segment Normalized Chi2 (ME -1/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqm11b","Segment Normalized Chi2 (ME -1/1b); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp11b","Segment Normalized Chi2 (ME +1/1b); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp12","Segment Normalized Chi2 (ME +1/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp13","Segment Normalized Chi2 (ME +1/3); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp11a","Segment Normalized Chi2 (ME +1/1a); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp21","Segment Normalized Chi2 (ME +2/1); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp22","Segment Normalized Chi2 (ME +2/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp31","Segment Normalized Chi2 (ME +3/1); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp32","Segment Normalized Chi2 (ME +3/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp41","Segment Normalized Chi2 (ME +4/1); Chi2/ndof",110,-0.05,10.5));
      hSChiSq.push_back(dbe->book1D("hSChiSqp42","Segment Normalized Chi2 (ME +4/2); Chi2/ndof",110,-0.05,10.5));
      hSChiSqProbAll = dbe->book1D("hSChiSqProb","Segment chi2 Probability; Probability",110,-0.05,1.05);
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm42","Segment chi2 Probability (ME -4/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm41","Segment chi2 Probability (ME -4/1); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm32","Segment chi2 Probability (ME -3/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm31","Segment chi2 Probability (ME -3/1); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm22","Segment chi2 Probability (ME -2/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm21","Segment chi2 Probability (ME -2/1); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm11a","Segment chi2 Probability (ME -1/1a); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm13","Segment chi2 Probability (ME -1/3); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm12","Segment chi2 Probability (ME -1/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbm11b","Segment chi2 Probability (ME -1/1b); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp11b","Segment chi2 Probability (ME +1/1b); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp12","Segment chi2 Probability (ME +1/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp13","Segment chi2 Probability (ME +1/3); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp11a","Segment chi2 Probability (ME +1/1a); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp21","Segment chi2 Probability (ME +2/1); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp22","Segment chi2 Probability (ME +2/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp31","Segment chi2 Probability (ME +3/1); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp32","Segment chi2 Probability (ME +3/2); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp41","Segment chi2 Probability (ME +4/1); Probability",110,-0.05,1.05));
      hSChiSqProb.push_back(dbe->book1D("hSChiSqProbp42","Segment chi2 Probability (ME +4/2); Probability",110,-0.05,1.05));
      hSGlobalTheta = dbe->book1D("hSGlobalTheta","Segment Direction (Global Theta); Global Theta (radians)",136,-0.1,3.3);
      hSGlobalPhi   = dbe->book1D("hSGlobalPhi","Segment Direction (Global Phi); Global Phi (radians)",  128,-3.2,3.2);

      // Resolution
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/Resolution");
      hSResid.push_back(dbe->book1D("hSResidm42","Fitted Position on Strip - Reconstructed for Layer 3 (ME -4/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm41","Fitted Position on Strip - Reconstructed for Layer 3 (ME -4/1); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm32","Fitted Position on Strip - Reconstructed for Layer 3 (ME -3/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm31","Fitted Position on Strip - Reconstructed for Layer 3 (ME -3/1); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm22","Fitted Position on Strip - Reconstructed for Layer 3 (ME -2/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm21","Fitted Position on Strip - Reconstructed for Layer 3 (ME -2/1); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/1a); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm13","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/3); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm12","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/1b); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/1b); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp12","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp13","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/3); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/1a); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp21","Fitted Position on Strip - Reconstructed for Layer 3 (ME +2/1); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp22","Fitted Position on Strip - Reconstructed for Layer 3 (ME +2/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp31","Fitted Position on Strip - Reconstructed for Layer 3 (ME +3/1); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp32","Fitted Position on Strip - Reconstructed for Layer 3 (ME +3/2); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp41","Fitted Position on Strip - Reconstructed for Layer 3 (ME +4/1); Strip Widths",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp42","Fitted Position on Strip - Reconstructed for Layer 3 (ME +4/2); Strip Widths",100,-0.5,0.5));

      // Efficiency
      dbe->setCurrentFolder("CSC/CSCOfflineMonitor/Efficiency");
      hSSTE = new TH1F("hSSTE","hSSTE",40,0.5,40.5);
      hRHSTE = new TH1F("hRHSTE","hRHSTE",40,0.5,40.5);
      hSEff = dbe->book1D("hSEff","Segment Efficiency",20,0.5,20.5);
      hSEff->setBinLabel(1,"ME +1/1b");
      hSEff->setBinLabel(2,"ME +1/2");
      hSEff->setBinLabel(3,"ME +1/3");
      hSEff->setBinLabel(4,"ME +1/1a");
      hSEff->setBinLabel(5,"ME +2/1");
      hSEff->setBinLabel(6,"ME +2/2");
      hSEff->setBinLabel(7,"ME +3/1");
      hSEff->setBinLabel(8,"ME +3/2");
      hSEff->setBinLabel(9,"ME +4/1");
      hSEff->setBinLabel(10,"ME +4/2");
      hSEff->setBinLabel(11,"ME -1/1b");
      hSEff->setBinLabel(12,"ME -1/2");
      hSEff->setBinLabel(13,"ME -1/3");
      hSEff->setBinLabel(14,"ME -1/1a");
      hSEff->setBinLabel(15,"ME -2/1");
      hSEff->setBinLabel(16,"ME -2/2");
      hSEff->setBinLabel(17,"ME -3/1");
      hSEff->setBinLabel(18,"ME -3/2");
      hSEff->setBinLabel(19,"ME -4/1");
      hSEff->setBinLabel(20,"ME -4/2");
      hRHEff = dbe->book1D("hRHEff","recHit Efficiency",20,0.5,20.5);
      hRHEff->setBinLabel(1,"ME +1/1b");
      hRHEff->setBinLabel(2,"ME +1/2");
      hRHEff->setBinLabel(3,"ME +1/3");
      hRHEff->setBinLabel(4,"ME +1/1a");
      hRHEff->setBinLabel(5,"ME +2/1");
      hRHEff->setBinLabel(6,"ME +2/2");
      hRHEff->setBinLabel(7,"ME +3/1");
      hRHEff->setBinLabel(8,"ME +3/2");
      hRHEff->setBinLabel(9,"ME +4/1");
      hRHEff->setBinLabel(10,"ME +4/2");
      hRHEff->setBinLabel(11,"ME -1/1b");
      hRHEff->setBinLabel(12,"ME -1/2");
      hRHEff->setBinLabel(13,"ME -1/3");
      hRHEff->setBinLabel(14,"ME -1/1a");
      hRHEff->setBinLabel(15,"ME -2/1");
      hRHEff->setBinLabel(16,"ME -2/2");
      hRHEff->setBinLabel(17,"ME -3/1");
      hRHEff->setBinLabel(18,"ME -3/2");
      hRHEff->setBinLabel(19,"ME -4/1");
      hRHEff->setBinLabel(20,"ME -4/2");
      hSSTE2 = new TH2F("hSSTE2","hSSTE2",36,0.5,36.5, 18, 0.5, 18.5);
      hRHSTE2 = new TH2F("hRHSTE2","hRHSTE2",36,0.5,36.5, 18, 0.5, 18.5);
      hStripSTE2 = new TH2F("hStripSTE2","hStripSTE2",36,0.5,36.5, 18, 0.5, 18.5);
      hWireSTE2 = new TH2F("hWireSTE2","hWireSTE2",36,0.5,36.5, 18, 0.5, 18.5);
      hEffDenominator = new TH2F("hEffDenominator","hEffDenominator",36,0.5,36.5, 18, 0.5, 18.5);
      hSEff2 = dbe->book2D("hSEff2","Segment Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
      hSEff2->setAxisTitle("Chamber #");
      hSEff2->setBinLabel(1,"ME -4/1",2);
      hSEff2->setBinLabel(2,"ME -3/2",2);
      hSEff2->setBinLabel(3,"ME -2/1",2);
      hSEff2->setBinLabel(4,"ME -2/2",2);
      hSEff2->setBinLabel(5,"ME -2/1",2);
      hSEff2->setBinLabel(6,"ME -1/3",2);
      hSEff2->setBinLabel(7,"ME -1/2",2);
      hSEff2->setBinLabel(8,"ME -1/1b",2);
      hSEff2->setBinLabel(9,"ME -1/1a",2);
      hSEff2->setBinLabel(10,"ME +1/1a",2);
      hSEff2->setBinLabel(11,"ME +1/1b",2);
      hSEff2->setBinLabel(12,"ME +1/2",2);
      hSEff2->setBinLabel(13,"ME +1/3",2);
      hSEff2->setBinLabel(14,"ME +2/1",2);
      hSEff2->setBinLabel(15,"ME +2/2",2);
      hSEff2->setBinLabel(16,"ME +3/1",2);
      hSEff2->setBinLabel(17,"ME +3/2",2);
      hSEff2->setBinLabel(18,"ME +4/1",2);
      hRHEff2 = dbe->book2D("hRHEff2","recHit Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
      hRHEff2->setAxisTitle("Chamber #");
      hRHEff2->setBinLabel(1,"ME -4/1",2);
      hRHEff2->setBinLabel(2,"ME -3/2",2);
      hRHEff2->setBinLabel(3,"ME -2/1",2);
      hRHEff2->setBinLabel(4,"ME -2/2",2);
      hRHEff2->setBinLabel(5,"ME -2/1",2);
      hRHEff2->setBinLabel(6,"ME -1/3",2);
      hRHEff2->setBinLabel(7,"ME -1/2",2);
      hRHEff2->setBinLabel(8,"ME -1/1b",2);
      hRHEff2->setBinLabel(9,"ME -1/1a",2);
      hRHEff2->setBinLabel(10,"ME +1/1a",2);
      hRHEff2->setBinLabel(11,"ME +1/1b",2);
      hRHEff2->setBinLabel(12,"ME +1/2",2);
      hRHEff2->setBinLabel(13,"ME +1/3",2);
      hRHEff2->setBinLabel(14,"ME +2/1",2);
      hRHEff2->setBinLabel(15,"ME +2/2",2);
      hRHEff2->setBinLabel(16,"ME +3/1",2);
      hRHEff2->setBinLabel(17,"ME +3/2",2);
      hRHEff2->setBinLabel(18,"ME +4/1",2);
      hStripEff2 = dbe->book2D("hStripEff2","strip Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
      hStripEff2->setAxisTitle("Chamber #");
      hStripEff2->setBinLabel(1,"ME -4/1",2);
      hStripEff2->setBinLabel(2,"ME -3/2",2);
      hStripEff2->setBinLabel(3,"ME -2/1",2);
      hStripEff2->setBinLabel(4,"ME -2/2",2);
      hStripEff2->setBinLabel(5,"ME -2/1",2);
      hStripEff2->setBinLabel(6,"ME -1/3",2);
      hStripEff2->setBinLabel(7,"ME -1/2",2);
      hStripEff2->setBinLabel(8,"ME -1/1b",2);
      hStripEff2->setBinLabel(9,"ME -1/1a",2);
      hStripEff2->setBinLabel(10,"ME +1/1a",2);
      hStripEff2->setBinLabel(11,"ME +1/1b",2);
      hStripEff2->setBinLabel(12,"ME +1/2",2);
      hStripEff2->setBinLabel(13,"ME +1/3",2);
      hStripEff2->setBinLabel(14,"ME +2/1",2);
      hStripEff2->setBinLabel(15,"ME +2/2",2);
      hStripEff2->setBinLabel(16,"ME +3/1",2);
      hStripEff2->setBinLabel(17,"ME +3/2",2);
      hStripEff2->setBinLabel(18,"ME +4/1",2);
      hWireEff2 = dbe->book2D("hWireEff2","wire Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
      hWireEff2->setAxisTitle("Chamber #");
      hWireEff2->setBinLabel(1,"ME -4/1",2);
      hWireEff2->setBinLabel(2,"ME -3/2",2);
      hWireEff2->setBinLabel(3,"ME -2/1",2);
      hWireEff2->setBinLabel(4,"ME -2/2",2);
      hWireEff2->setBinLabel(5,"ME -2/1",2);
      hWireEff2->setBinLabel(6,"ME -1/3",2);
      hWireEff2->setBinLabel(7,"ME -1/2",2);
      hWireEff2->setBinLabel(8,"ME -1/1b",2);
      hWireEff2->setBinLabel(9,"ME -1/1a",2);
      hWireEff2->setBinLabel(10,"ME +1/1a",2);
      hWireEff2->setBinLabel(11,"ME +1/1b",2);
      hWireEff2->setBinLabel(12,"ME +1/2",2);
      hWireEff2->setBinLabel(13,"ME +1/3",2);
      hWireEff2->setBinLabel(14,"ME +2/1",2);
      hWireEff2->setBinLabel(15,"ME +2/2",2);
      hWireEff2->setBinLabel(16,"ME +3/1",2);
      hWireEff2->setBinLabel(17,"ME +3/2",2);
      hWireEff2->setBinLabel(18,"ME +4/1",2);
      hSensitiveAreaEvt = dbe->book2D("hSensitiveAreaEvt","Events Passing Selection for Efficiency",36,0.5,36.5, 18, 0.5, 18.5);
      hSensitiveAreaEvt->setAxisTitle("Chamber #");
      hSensitiveAreaEvt->setBinLabel(1,"ME -4/1",2);
      hSensitiveAreaEvt->setBinLabel(2,"ME -3/2",2);
      hSensitiveAreaEvt->setBinLabel(3,"ME -2/1",2);
      hSensitiveAreaEvt->setBinLabel(4,"ME -2/2",2);
      hSensitiveAreaEvt->setBinLabel(5,"ME -2/1",2);
      hSensitiveAreaEvt->setBinLabel(6,"ME -1/3",2);
      hSensitiveAreaEvt->setBinLabel(7,"ME -1/2",2);
      hSensitiveAreaEvt->setBinLabel(8,"ME -1/1b",2);
      hSensitiveAreaEvt->setBinLabel(9,"ME -1/1a",2);
      hSensitiveAreaEvt->setBinLabel(10,"ME +1/1a",2);
      hSensitiveAreaEvt->setBinLabel(11,"ME +1/1b",2);
      hSensitiveAreaEvt->setBinLabel(12,"ME +1/2",2);
      hSensitiveAreaEvt->setBinLabel(13,"ME +1/3",2);
      hSensitiveAreaEvt->setBinLabel(14,"ME +2/1",2);
      hSensitiveAreaEvt->setBinLabel(15,"ME +2/2",2);
      hSensitiveAreaEvt->setBinLabel(16,"ME +3/1",2);
      hSensitiveAreaEvt->setBinLabel(17,"ME +3/2",2);
      hSensitiveAreaEvt->setBinLabel(18,"ME +4/1",2);

}

//////////////////
//  DESTRUCTOR  //
//////////////////
CSCOfflineMonitor::~CSCOfflineMonitor(){

}

void CSCOfflineMonitor::endJob(void) {

  hRHEff = dbe->get("CSC/CSCOfflineMonitor/Efficiency/hRHEff");
  hSEff = dbe->get("CSC/CSCOfflineMonitor/Efficiency/hSEff");
  hSEff2 = dbe->get("CSC/CSCOfflineMonitor/Efficiency/hSEff2");
  hRHEff2 = dbe->get("CSC/CSCOfflineMonitor/Efficiency/hRHEff2");
  hStripEff2 = dbe->get("CSC/CSCOfflineMonitor/Efficiency/hStripEff2");
  hWireEff2 = dbe->get("CSC/CSCOfflineMonitor/Efficiency/hWireEff2");

  if (hRHEff) histoEfficiency(hRHSTE,hRHEff);
  if (hSEff) histoEfficiency(hSSTE,hSEff);
  if (hSEff2) hSEff2->getTH2F()->Divide(hSSTE2,hEffDenominator,1.,1.,"B");
  if (hRHEff2) hRHEff2->getTH2F()->Divide(hRHSTE2,hEffDenominator,1.,1.,"B");
  if (hStripEff2) hStripEff2->getTH2F()->Divide(hStripSTE2,hEffDenominator,1.,1.,"B");
  if (hWireEff2) hWireEff2->getTH2F()->Divide(hWireSTE2,hEffDenominator,1.,1.,"B");

  bool saveHistos = param.getParameter<bool>("saveHistos");
  string outputFileName = param.getParameter<string>("outputFileName");
  if(saveHistos){
    dbe->save(outputFileName);
  }

}

////////////////
//  Analysis  //
////////////////
void CSCOfflineMonitor::analyze(const Event & event, const EventSetup& eventSetup){

  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  event.getByLabel(stripDigiTag_,strips);
  event.getByLabel(wireDigiTag_,wires);

  // Get the CSC Geometry :
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits;
  event.getByLabel(cscRecHitTag_,recHits);

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel(cscSegTag_, cscSegments);


  doOccupancies(strips,wires,recHits,cscSegments);
  doStripDigis(strips);
  doWireDigis(wires);
  doRecHits(recHits,strips,cscGeom);
  doSegments(cscSegments,cscGeom);
  doResolution(cscSegments,cscGeom);
  doPedestalNoise(strips);
  doEfficiencies(wires,strips, recHits, cscSegments,cscGeom);

}

// ==============================================
//
// look at Occupancies
//
// ==============================================

void CSCOfflineMonitor::doOccupancies(edm::Handle<CSCStripDigiCollection> strips, edm::Handle<CSCWireDigiCollection> wires,
                                      edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments){

  bool wireo[2][4][4][36];
  bool stripo[2][4][4][36];
  bool rechito[2][4][4][36];
  bool segmento[2][4][4][36];

  bool hasWires = false;
  bool hasStrips = false;
  bool hasRecHits = false;
  bool hasSegments = false;

  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          wireo[e][s][r][c] = false;
          stripo[e][s][r][c] = false;
          rechito[e][s][r][c] = false;
          segmento[e][s][r][c] = false;
        }
      }
    }
  }

  //wires
  for (CSCWireDigiCollection::DigiRangeIterator wi=wires->begin(); wi!=wires->end(); wi++) {
    CSCDetId id = (CSCDetId)(*wi).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    std::vector<CSCWireDigi>::const_iterator wireIt = (*wi).second.first;
    std::vector<CSCWireDigi>::const_iterator lastWire = (*wi).second.second;
    for( ; wireIt != lastWire; ++wireIt){
      if (!wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
        wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
        hOWires->Fill(kChamber,typeIndex(id,2));
        hOWireSerial->Fill(chamberSerial(id));
        hasWires = true;
      }
    }
  }
  
  //strips
  for (CSCStripDigiCollection::DigiRangeIterator si=strips->begin(); si!=strips->end(); si++) {
    CSCDetId id = (CSCDetId)(*si).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    std::vector<CSCStripDigi>::const_iterator stripIt = (*si).second.first;
    std::vector<CSCStripDigi>::const_iterator lastStrip = (*si).second.second;
    for( ; stripIt != lastStrip; ++stripIt) {
      std::vector<int> myADCVals = stripIt->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
        diff = (float)myADCVals[iCount]-thisPedestal;
        if (diff > threshold) { thisStripFired = true; }
      }
      if (thisStripFired) {
        if (!stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
          stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
          hOStrips->Fill(kChamber,typeIndex(id,2));
          hOStripSerial->Fill(chamberSerial(id));
          hasStrips = true;
        }
      }
    }
  }

  //rechits
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    if (!rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
      rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      hORecHitsSerial->Fill(chamberSerial(idrec));
      hORecHits->Fill(kChamber,typeIndex(idrec,2));
      hasRecHits = true;
    }
  }

  //segments
  for(CSCSegmentCollection::const_iterator segIt=cscSegments->begin(); segIt != cscSegments->end(); segIt++) {
    CSCDetId id  = (CSCDetId)(*segIt).cscDetId();
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    if (!segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
      segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      hOSegmentsSerial->Fill(chamberSerial(id));
      hOSegments->Fill(kChamber,typeIndex(id,2));
      hasSegments = true;
    }
  }

  //Overall CSC Occupancy
  hCSCOccupancy->Fill(1);
  if (hasWires) hCSCOccupancy->Fill(3);
  if (hasStrips) hCSCOccupancy->Fill(5);
  if (hasWires && hasStrips) hCSCOccupancy->Fill(7);
  if (hasRecHits) hCSCOccupancy->Fill(9);
  if (hasSegments) hCSCOccupancy->Fill(11);


}


// ==============================================
//
// look at WIRE DIGIs
//
// ==============================================

void CSCOfflineMonitor::doWireDigis(edm::Handle<CSCWireDigiCollection> wires){

  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator dWDiter=wires->begin(); dWDiter!=wires->end(); dWDiter++) {
    CSCDetId id = (CSCDetId)(*dWDiter).first;
    std::vector<CSCWireDigi>::const_iterator wireIter = (*dWDiter).second.first;
    std::vector<CSCWireDigi>::const_iterator lWire = (*dWDiter).second.second;
    for( ; wireIter != lWire; ++wireIter) {
      int myWire = wireIter->getWireGroup();
      int myTBin = wireIter->getTimeBin();
      nWireGroupsTotal++;
      hWireTBin[typeIndex(id)-1]->Fill(myTBin);
      hWireNumber[typeIndex(id)-1]->Fill(myWire);
    }
  } // end wire loop

  // this way you can zero suppress but still store info on # events with no digis
  if (nWireGroupsTotal == 0) nWireGroupsTotal = -1;
  hWirenGroupsTotal->Fill(nWireGroupsTotal);
  
}


// ==============================================
//
// look at STRIP DIGIs
//
// ==============================================

void CSCOfflineMonitor::doStripDigis(edm::Handle<CSCStripDigiCollection> strips){

  int nStripsFired = 0;
  for (CSCStripDigiCollection::DigiRangeIterator dSDiter=strips->begin(); dSDiter!=strips->end(); dSDiter++) {
    CSCDetId id = (CSCDetId)(*dSDiter).first;
    std::vector<CSCStripDigi>::const_iterator stripIter = (*dSDiter).second.first;
    std::vector<CSCStripDigi>::const_iterator lStrip = (*dSDiter).second.second;
    for( ; stripIter != lStrip; ++stripIter) {
      int myStrip = stripIter->getStrip();
      std::vector<int> myADCVals = stripIter->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
	diff = (float)myADCVals[iCount]-thisPedestal;
	if (diff > threshold) { thisStripFired = true; }
      } 
      if (thisStripFired) {
        nStripsFired++;
        hStripNumber[typeIndex(id)-1]->Fill(myStrip);
      }
    }
  } // end strip loop

  // this way you can zero suppress but still store info on # events with no digis
  if (nStripsFired == 0) nStripsFired = -1;
  hStripNFired->Fill(nStripsFired);
  // fill n per event

}


//=======================================================
//
// Look at the Pedestal Noise Distributions
//
//=======================================================

void CSCOfflineMonitor::doPedestalNoise(edm::Handle<CSCStripDigiCollection> strips){

  for (CSCStripDigiCollection::DigiRangeIterator dPNiter=strips->begin(); dPNiter!=strips->end(); dPNiter++) {
    CSCDetId id = (CSCDetId)(*dPNiter).first;
    int kStation = id.station();
    int kRing = id.ring();
    std::vector<CSCStripDigi>::const_iterator pedIt = (*dPNiter).second.first;
    std::vector<CSCStripDigi>::const_iterator lStrip = (*dPNiter).second.second;
    for( ; pedIt != lStrip; ++pedIt) {
      int myStrip = pedIt->getStrip();
      std::vector<int> myADCVals = pedIt->getADCCounts();
      float TotalADC = getSignal(*strips, id, myStrip);
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float thisSignal = (1./6)*(myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
      float threshold = 13.3;
      if(kStation == 1 && kRing == 4)
	{
	  kRing = 1;
	  if(myStrip <= 16) myStrip += 64; // no trapping for any bizarreness
	}
      if (TotalADC > threshold) { thisStripFired = true;}
      if (!thisStripFired){
	float ADC = thisSignal - thisPedestal;
        hStripPed[typeIndex(id)-1]->Fill(ADC);
      }
    }
  }

}


// ==============================================
//
// look at RECHITs
//
// ==============================================

void CSCOfflineMonitor::doRecHits(edm::Handle<CSCRecHit2DCollection> recHits,
                                  edm::Handle<CSCStripDigiCollection> strips,
                                  edm::ESHandle<CSCGeometry> cscGeom){

  // Get the RecHits collection :
  int nRecHits = recHits->size();
 
  // ---------------------
  // Loop over rechits 
  // ---------------------
  // Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator dRHIter;
  for (dRHIter = recHits->begin(); dRHIter != recHits->end(); dRHIter++) {

    // Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*dRHIter).cscDetId();

    // Store rechit as a Local Point:
    LocalPoint rhitlocal = (*dRHIter).localPosition();  
    //float xreco = rhitlocal.x();
    //float yreco = rhitlocal.y();

    // Get the reconstucted strip position and error
    float stpos = (*dRHIter).positionWithinStrip();
    float sterr = (*dRHIter).errorWithinStrip();

    // Find the charge associated with this hit
    CSCRecHit2D::ADCContainer adcs = (*dRHIter).adcs();
    int adcsize = adcs.size();
    float rHSumQ = 0;
    float sumsides = 0;
    for (int i = 0; i < adcsize; i++){
      if (i != 3 && i != 7 && i != 11){
        rHSumQ = rHSumQ + adcs[i]; 
      }
      if (adcsize == 12 && (i < 3 || i > 7) && i < 12){
        sumsides = sumsides + adcs[i];
      }
    }
    float rHratioQ = sumsides/rHSumQ;
    if (adcsize != 12) rHratioQ = -99;

    // Get the signal timing of this hit
    float rHtime = (*dRHIter).tpeak();
 
    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx   =  rhitglobal.x();
    float grecy   =  rhitglobal.y();

    // Fill some histograms
    int sIndex = idrec.station() + ((idrec.endcap()-1) * 4);
    int tIndex = typeIndex(idrec);
    hRHSumQ[tIndex-1]->Fill(rHSumQ);
    hRHRatioQ[tIndex-1]->Fill(rHratioQ);
    hRHstpos[tIndex-1]->Fill(stpos);
    hRHsterr[tIndex-1]->Fill(sterr);
    hRHTiming[tIndex-1]->Fill(rHtime);
    hRHGlobal[sIndex-1]->Fill(grecx,grecy);

  } //end rechit loop

  if (nRecHits == 0) nRecHits = -1;
  hRHnrechits->Fill(nRecHits);

}


// ==============================================
//
// look at SEGMENTs
//
// ===============================================

void CSCOfflineMonitor::doSegments(edm::Handle<CSCSegmentCollection> cscSegments,
                                   edm::ESHandle<CSCGeometry> cscGeom){

  // get CSC segment collection
  int nSegments = cscSegments->size();

  for(CSCSegmentCollection::const_iterator dSiter=cscSegments->begin(); dSiter != cscSegments->end(); dSiter++) {
    CSCDetId id  = (CSCDetId)(*dSiter).cscDetId();
    float chisq    = (*dSiter).chi2();
    int nhits      = (*dSiter).nRecHits();
    int nDOF       = 2*nhits-4;
    float nChi2    = chisq/nDOF;
    double chisqProb = ChiSquaredProbability( (double)chisq, nDOF );
    //LocalPoint localPos = (*dSiter).localPosition();
    LocalVector segDir = (*dSiter).localDirection();

    // global transformation
    //float globX = 0.;
    //float globY = 0.;
    float globTheta = 0.;
    float globPhi   = 0.;
    const CSCChamber* cscchamber = cscGeom->chamber(id);
    if (cscchamber) {
      //GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      //globX = globalPosition.x();
      //globY = globalPosition.y();
      GlobalVector globalDirection = cscchamber->toGlobal(segDir);
      globTheta = globalDirection.theta();
      globPhi   = globalDirection.phi();
    }

    // Fill histos
    int tIndex = typeIndex(id);
    hSnhitsAll->Fill(nhits);
    hSnhits[tIndex-1]->Fill(nhits);
    hSChiSqAll->Fill(nChi2);
    hSChiSq[tIndex-1]->Fill(nChi2);
    hSChiSqProbAll->Fill(chisqProb);
    hSChiSqProb[tIndex-1]->Fill(chisqProb);
    hSGlobalTheta->Fill(globTheta);
    hSGlobalPhi->Fill(globPhi);


  } // end segment loop

  if (nSegments == 0) nSegments = -1;
  hSnSegments->Fill(nSegments);

}

// ==============================================
//
// look at hit Resolution
//
// ==============================================
void CSCOfflineMonitor::doResolution(edm::Handle<CSCSegmentCollection> cscSegments,
                                     edm::ESHandle<CSCGeometry> cscGeom){

  for(CSCSegmentCollection::const_iterator dSiter=cscSegments->begin(); dSiter != cscSegments->end(); dSiter++) {
    CSCDetId id  = (CSCDetId)(*dSiter).cscDetId();
    //
    // try to get the CSC recHits that contribute to this segment.
    std::vector<CSCRecHit2D> theseRecHits = (*dSiter).specificRecHits();
    int nRH = (*dSiter).nRecHits();
    int jRH = 0;
    CLHEP::HepMatrix sp(6,1);
    CLHEP::HepMatrix se(6,1);
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      //int kEndcap  = idRH.endcap();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
      //int kChamber = idRH.chamber();
      int kLayer   = idRH.layer();

      // Find the strip containing this hit
      CSCRecHit2D::ChannelContainer hitstrips = (*iRH).channels();
      int nStrips     =  hitstrips.size();
      int centerid    =  nStrips/2 + 1;
      int centerStrip =  hitstrips[centerid - 1];

      // If this segment has 6 hits, find the position of each hit on the strip in units of stripwidth and store values
      if (nRH == 6){
        float stpos = (*iRH).positionWithinStrip();
        se(kLayer,1) = (*iRH).errorWithinStrip();
        // Take into account half-strip staggering of layers (ME1/1 has no staggering)
        if (kStation == 1 && (kRing == 1 || kRing == 4)) sp(kLayer,1) = stpos + centerStrip;
        else{
          if (kLayer == 1 || kLayer == 3 || kLayer == 5) sp(kLayer,1) = stpos + centerStrip;
          if (kLayer == 2 || kLayer == 4 || kLayer == 6) sp(kLayer,1) = stpos - 0.5 + centerStrip;
        }
      }

    }

    float residual = -99;
    // Fit all points except layer 3, then compare expected value for layer 3 to reconstructed value
    if (nRH == 6){
      float expected = fitX(sp,se);
      residual = expected - sp(3,1);
    }

    hSResid[typeIndex(id)-1]->Fill(residual);

  } // end segment loop



}


//-------------------------------------------------------------------------------------
// Fits a straight line to a set of 5 points with errors.  Functions assumes 6 points
// and removes hit in layer 3.  It then returns the expected position value in layer 3
// based on the fit.
//-------------------------------------------------------------------------------------
float CSCOfflineMonitor::fitX(CLHEP::HepMatrix points, CLHEP::HepMatrix errors){

  float S   = 0;
  float Sx  = 0;
  float Sy  = 0;
  float Sxx = 0;
  float Sxy = 0;
  float sigma2 = 0;

  for (int i=1;i<7;i++){
    if (i != 3){
      sigma2 = errors(i,1)*errors(i,1);
      S = S + (1/sigma2);
      Sy = Sy + (points(i,1)/sigma2);
      Sx = Sx + ((i)/sigma2);
      Sxx = Sxx + (i*i)/sigma2;
      Sxy = Sxy + (((i)*points(i,1))/sigma2);
    }
  }

  float delta = S*Sxx - Sx*Sx;
  float intercept = (Sxx*Sy - Sx*Sxy)/delta;
  float slope = (S*Sxy - Sx*Sy)/delta;

  return (intercept + slope*3);

}

//---------------------------------------------------------------------------------------
// Given a set of digis, the CSCDetId, and the central strip of your choosing, returns
// the avg. Signal-Pedestal for 6 time bin x 5 strip .
//
// Author: P. Jindal
//---------------------------------------------------------------------------------------

float CSCOfflineMonitor::getSignal(const CSCStripDigiCollection& stripdigis, 
                                   CSCDetId idCS, int centerStrip){

  float SigADC[5];
  float TotalADC = 0;
  SigADC[0] = 0;
  SigADC[1] = 0;
  SigADC[2] = 0;
  SigADC[3] = 0;
  SigADC[4] = 0;

 
  // Loop over strip digis 
  CSCStripDigiCollection::DigiRangeIterator sIt;
  
  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idCS){

      // First, find the Signal-Pedestal for center strip
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[0] = thisSignal - 6*thisPedestal;
	}
     // Now,find the Signal-Pedestal for neighbouring 4 strips
        if (thisStrip == (centerStrip+1)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[1] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip+2)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[2] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip-1)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[3] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip-2)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[4] = thisSignal - 6*thisPedestal;
	}
      }
      TotalADC = 0.2*(SigADC[0]+SigADC[1]+SigADC[2]+SigADC[3]+SigADC[4]);
    }
  }
  return TotalADC;
}



//----------------------------------------------------------------------------
// Calculate basic efficiencies for recHits and Segments
// Author: S. Stoynev
//----------------------------------------------------------------------------

void CSCOfflineMonitor::doEfficiencies(edm::Handle<CSCWireDigiCollection> wires, edm::Handle<CSCStripDigiCollection> strips,
                                       edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments,
                                       edm::ESHandle<CSCGeometry> cscGeom){

  bool allWires[2][4][4][36][6];
  bool allStrips[2][4][4][36][6];
  bool AllRecHits[2][4][4][36][6];
  bool AllSegments[2][4][4][36];
  
  //bool MultiSegments[2][4][4][36];
  for(int iE = 0;iE<2;iE++){
    for(int iS = 0;iS<4;iS++){
      for(int iR = 0; iR<4;iR++){
        for(int iC =0;iC<36;iC++){
          AllSegments[iE][iS][iR][iC] = false;
          //MultiSegments[iE][iS][iR][iC] = false;
          for(int iL=0;iL<6;iL++){
	    allWires[iE][iS][iR][iC][iL] = false;
	    allStrips[iE][iS][iR][iC][iL] = false;
            AllRecHits[iE][iS][iR][iC][iL] = false;
          }
        }
      }
    }
  }
  
  
  // Wires
  for (CSCWireDigiCollection::DigiRangeIterator dWDiter=wires->begin(); dWDiter!=wires->end(); dWDiter++) {
    CSCDetId idrec = (CSCDetId)(*dWDiter).first;
    std::vector<CSCWireDigi>::const_iterator wireIter = (*dWDiter).second.first;
    std::vector<CSCWireDigi>::const_iterator lWire = (*dWDiter).second.second;
    for( ; wireIter != lWire; ++wireIter) {
      allWires[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;
      break;
    }
  }

  //---- STRIPS
  for (CSCStripDigiCollection::DigiRangeIterator dSDiter=strips->begin(); dSDiter!=strips->end(); dSDiter++) {
    CSCDetId idrec = (CSCDetId)(*dSDiter).first;
    std::vector<CSCStripDigi>::const_iterator stripIter = (*dSDiter).second.first;
    std::vector<CSCStripDigi>::const_iterator lStrip = (*dSDiter).second.second;
    for( ; stripIter != lStrip; ++stripIter) {
      std::vector<int> myADCVals = stripIter->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
        diff = (float)myADCVals[iCount]-thisPedestal;
        if (diff > threshold) {
          thisStripFired = true;
	  break;
        }
      }
      if(thisStripFired){
	allStrips[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;
	break;
      }
    }
  }

  // Rechits
  for (CSCRecHit2DCollection::const_iterator recEffIt = recHits->begin(); recEffIt != recHits->end(); recEffIt++) {
    //CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    CSCDetId  idrec = (CSCDetId)(*recEffIt).cscDetId();
    AllRecHits[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber() -1][idrec.layer() -1] = true;

  }

  std::vector <uint> seg_ME2(2,0) ;
  std::vector <uint> seg_ME3(2,0) ;
  std::vector < pair <CSCDetId, CSCSegment> > theSegments(4);
  // Segments
  for(CSCSegmentCollection::const_iterator segEffIt=cscSegments->begin(); segEffIt != cscSegments->end(); segEffIt++) {
    CSCDetId idseg  = (CSCDetId)(*segEffIt).cscDetId();
    //if(AllSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()]){
    //MultiSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()] = true;
    //}
    AllSegments[idseg.endcap() -1][idseg.station() -1][idseg.ring() -1][idseg.chamber() -1] = true;
    // "Intrinsic" efficiency measurement relies on "good" segment extrapolation - we need the pre-selection below
    // station 2 "good" segment will be used for testing efficiencies in ME1 and ME3
    // station 3 "good" segment will be used for testing efficiencies in ME2 and ME4
    if(2==idseg.station() || 3==idseg.station()){
      uint seg_tmp ; 
      if(2==idseg.station()){
	++seg_ME2[idseg.endcap() -1];
	seg_tmp = seg_ME2[idseg.endcap() -1];
      }
      else{
	++seg_ME3[idseg.endcap() -1];
	seg_tmp = seg_ME3[idseg.endcap() -1];
      }
      // is the segment good
      if(1== seg_tmp&& 6==(*segEffIt).nRecHits() && (*segEffIt).chi2()/(*segEffIt).degreesOfFreedom()<3.){
	pair <CSCDetId, CSCSegment> specSeg = make_pair( (CSCDetId)(*segEffIt).cscDetId(),*segEffIt);
	theSegments[2*(idseg.endcap()-1)+(idseg.station() -2)] = specSeg;
      }
    }
    /*
    if(2==idseg.station()){
	++seg_ME2[idseg.endcap() -1];
       if(1==seg_ME2[idseg.endcap() -1] && 6==(*segEffIt).nRecHits() && (*segEffIt).chi2()/(*segEffIt).degreesOfFreedom()<3.){
           pair <CSCDetId, CSCSegment> specSeg = make_pair( (CSCDetId)(*segEffIt).cscDetId(),*segEffIt);
           theSegments[2*(idseg.endcap()-1)+(idseg.station() -2)] = specSeg;
       }
    }
    else if(3==idseg.station()){
	++seg_ME3[idseg.endcap() -1];
	if(1==seg_ME3[idseg.endcap() -1] && 6==(*segEffIt).nRecHits() && (*segEffIt).chi2()/(*segEffIt).degreesOfFreedom()<3.){
         pair <CSCDetId, CSCSegment> specSeg = make_pair( (CSCDetId)(*segEffIt).cscDetId(),*segEffIt);
	 theSegments[2*(idseg.endcap()-1)+(idseg.station() -2)] = specSeg;
       }
    }
    */
    
  }
  // Simple efficiency calculations
  for(int iE = 0;iE<2;iE++){
    for(int iS = 0;iS<4;iS++){
      for(int iR = 0; iR<4;iR++){
        for(int iC =0;iC<36;iC++){
          int NumberOfLayers = 0;
          for(int iL=0;iL<6;iL++){
            if(AllRecHits[iE][iS][iR][iC][iL]){
              NumberOfLayers++;
            }
          }
          int bin = 0;
          if (iS==0) bin = iR+1+(iE*10);
          else bin = (iS+1)*2 + (iR+1) + (iE*10);
          if(NumberOfLayers>1){
            //if(!(MultiSegments[iE][iS][iR][iC])){
            if(AllSegments[iE][iS][iR][iC]){
              //---- Efficient segment evenents
              hSSTE->Fill(bin);
            }
            //---- All segment events (normalization)
            hSSTE->Fill(20+bin);
            //}
          }
          if(AllSegments[iE][iS][iR][iC]){
            if(NumberOfLayers==6){
              //---- Efficient rechit events
              hRHSTE->Fill(bin);;
            }
            //---- All rechit events (normalization)
            hRHSTE->Fill(20+bin);;
          }
        }
      }
    }
  }

  // pick a segment only if there are no others in the station
  std::vector < pair <CSCDetId, CSCSegment> * > theSeg;
  if(1==seg_ME2[0]) theSeg.push_back(&theSegments[0]);
  if(1==seg_ME3[0]) theSeg.push_back(&theSegments[1]);
  if(1==seg_ME2[1]) theSeg.push_back(&theSegments[2]);
  if(1==seg_ME3[1]) theSeg.push_back(&theSegments[3]);

  // Needed for plots
  // at the end the chamber types will be numbered as 1 to 18 
  // (ME-4/1, -ME3/2, -ME3/1, ..., +ME3/1, +ME3/2, ME+4/1 ) 
  std::map <std::string, float> chamberTypes;
  chamberTypes["ME1/a"] = 0.5;
  chamberTypes["ME1/b"] = 1.5;
  chamberTypes["ME1/2"] = 2.5;
  chamberTypes["ME1/3"] = 3.5;
  chamberTypes["ME2/1"] = 4.5;
  chamberTypes["ME2/2"] = 5.5;
  chamberTypes["ME3/1"] = 6.5;
  chamberTypes["ME3/2"] = 7.5;
  chamberTypes["ME4/1"] = 8.5;

  if(theSeg.size()){
    std::map <int , GlobalPoint> extrapolatedPoint;
    std::map <int , GlobalPoint>::iterator it;
    const std::vector<CSCChamber*> ChamberContainer = cscGeom->chambers();
    // Pick which chamber with which segment to test
    for(unsigned int nCh=0;nCh<ChamberContainer.size();nCh++){
      const CSCChamber *cscchamber = ChamberContainer[nCh];
      pair <CSCDetId, CSCSegment> * thisSegment = 0;
      for(uint iSeg =0;iSeg<theSeg.size();++iSeg ){
        if(cscchamber->id().endcap() == theSeg[iSeg]->first.endcap()){ 
          if(1==cscchamber->id().station() || 3==cscchamber->id().station() ){
	    if(2==theSeg[iSeg]->first.station()){
	      thisSegment = theSeg[iSeg];
	    }
	  }
	  else if (2==cscchamber->id().station() || 4==cscchamber->id().station()){
	    if(3==theSeg[iSeg]->first.station()){
	      thisSegment = theSeg[iSeg];
	    }
	  }
	}
      }
      // this chamber is to be tested with thisSegment
      if(thisSegment){
	CSCSegment * seg = &(thisSegment->second);
	const CSCChamber *segChamber = cscGeom->chamber(thisSegment->first);
	LocalPoint localCenter(0.,0.,0);
	GlobalPoint cscchamberCenter =  cscchamber->toGlobal(localCenter);
	// try to save some time (extrapolate a segment to a certain position only once)
	it = extrapolatedPoint.find(int(cscchamberCenter.z()));
	if(it==extrapolatedPoint.end()){
	  GlobalPoint segPos = segChamber->toGlobal(seg->localPosition());
	  GlobalVector segDir = segChamber->toGlobal(seg->localDirection());
	  double paramaterLine = lineParametrization(segPos.z(),cscchamberCenter.z() , segDir.z());
	  double xExtrapolated = extrapolate1D(segPos.x(),segDir.x(), paramaterLine);
	  double yExtrapolated = extrapolate1D(segPos.y(),segDir.y(), paramaterLine);
	  GlobalPoint globP (xExtrapolated, yExtrapolated, cscchamberCenter.z());
	  extrapolatedPoint[int(cscchamberCenter.z())] = globP;
	}
	// Where does the extrapolated point lie in the (tested) chamber local frame? Here: 
	LocalPoint extrapolatedPointLocal = cscchamber->toLocal(extrapolatedPoint[int(cscchamberCenter.z())]);
	const CSCLayer *layer_p = cscchamber->layer(1);//layer 1
	const CSCLayerGeometry *layerGeom = layer_p->geometry ();
	const std::vector<float> layerBounds = layerGeom->parameters ();
	float shiftFromEdge = 15.;//cm
	float shiftFromDeadZone = 10.;
	// is the extrapolated point within a sensitive region
	bool pass = withinSensitiveRegion(extrapolatedPointLocal, layerBounds, 
					  cscchamber->id().station(), cscchamber->id().ring(), 
					  shiftFromEdge, shiftFromDeadZone);
	if(pass){// the extrapolation point of the segment lies within sensitive region of that chamber
	  // how many rechit layers are there in the chamber?
	  // 0 - maybe the muon died or is deflected at large angle? do not use that case
	  // 1 - could be noise...
	  // 2 or more - this is promissing; this is our definition of a reliable signal; use it below
	  // is other definition better? 
	  int nRHLayers = 0;
	  for(int iL =0;iL<6;++iL){
	    if(AllRecHits[cscchamber->id().endcap()-1]
	       [cscchamber->id().station()-1]
	       [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
	      ++nRHLayers;
	    }
	  }
	  //std::cout<<" nRHLayers = "<<nRHLayers<<std::endl;
	  float verticalScale = chamberTypes[cscchamber->specs()->chamberTypeName()];
	  if(cscchamberCenter.z()<0){
	    verticalScale = - verticalScale;
	  } 
	  verticalScale +=9.5;
	  hSensitiveAreaEvt->Fill(float(cscchamber->id().chamber()),verticalScale);
	  if(nRHLayers>1){// this chamber contains a reliable signal
	    //chamberTypes[cscchamber->specs()->chamberTypeName()];
	    // "intrinsic" efficiencies
	    //std::cout<<" verticalScale = "<<verticalScale<<" chType = "<<cscchamber->specs()->chamberTypeName()<<std::endl;
	    // this is the denominator forr all efficiencies
	    hEffDenominator->Fill(float(cscchamber->id().chamber()),verticalScale);
	    // Segment efficiency
	    if(AllSegments[cscchamber->id().endcap()-1]
	       [cscchamber->id().station()-1]
	       [cscchamber->id().ring()-1][cscchamber->id().chamber()-1]){
	      hSSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale));
	    }
	  
	    for(int iL =0;iL<6;++iL){
	      float weight = 1./6.;
	      // one shold account for the weight in the efficiency...
	      // Rechit efficiency
	      if(AllRecHits[cscchamber->id().endcap()-1]
		 [cscchamber->id().station()-1]
		 [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
		hRHSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale),weight);
	      }
	      // Wire efficiency
	      if(allWires[cscchamber->id().endcap()-1]
		 [cscchamber->id().station()-1]
		 [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
		// one shold account for the weight in the efficiency...
		hWireSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale),weight);
	      }
	      // Strip efficiency
	      if(allStrips[cscchamber->id().endcap()-1]
		 [cscchamber->id().station()-1]
		 [cscchamber->id().ring()-1][cscchamber->id().chamber()-1][iL]){
		// one shold account for the weight in the efficiency...
		hStripSTE2->Fill(float(cscchamber->id().chamber()),float(verticalScale),weight);
	      }
	    }
	  }
	}
      }
    }
  }
  //


}

void CSCOfflineMonitor::getEfficiency(float bin, float Norm, std::vector<float> &eff){
  //---- Efficiency with binomial error
  float Efficiency = 0.;
  float EffError = 0.;
  if(fabs(Norm)>0.000000001){
    Efficiency = bin/Norm;
    if(bin<Norm){
      EffError = sqrt( (1.-Efficiency)*Efficiency/Norm );
    }
  }
  eff[0] = Efficiency;
  eff[1] = EffError;
}

void CSCOfflineMonitor::histoEfficiency(TH1F *readHisto, MonitorElement *writeHisto){
  std::vector<float> eff(2);
  int Nbins =  readHisto->GetSize()-2;//without underflows and overflows
  std::vector<float> bins(Nbins);
  std::vector<float> Efficiency(Nbins);
  std::vector<float> EffError(Nbins);
  float Num = 1;
  float Den = 1;
  for (int i=0;i<20;i++){
    Num = readHisto->GetBinContent(i+1);
    Den = readHisto->GetBinContent(i+21);
    getEfficiency(Num, Den, eff);
    Efficiency[i] = eff[0];
    EffError[i] = eff[1];
    writeHisto->setBinContent(i+1, Efficiency[i]);
    writeHisto->setBinError(i+1, EffError[i]);
  }
}

bool CSCOfflineMonitor::withinSensitiveRegion(LocalPoint localPos, const std::vector<float> layerBounds,
                                              int station, int ring, float shiftFromEdge, float shiftFromDeadZone){
//---- check if it is in a good local region (sensitive area - geometrical and HV boundaries excluded) 
  bool pass = false;

  float y_center = 0.;
  double yUp = layerBounds[3] + y_center;
  double yDown = - layerBounds[3] + y_center;
  double xBound1Shifted = layerBounds[0] - shiftFromEdge;//
  double xBound2Shifted = layerBounds[1] - shiftFromEdge;//
  double lineSlope = (yUp - yDown)/(xBound2Shifted-xBound1Shifted);
  double lineConst = yUp - lineSlope*xBound2Shifted;
  double yBorder =  lineSlope*abs(localPos.x()) + lineConst;
      
  //bool withinChamberOnly = false;// false = "good region"; true - boundaries only
  std::vector <float> deadZoneCenter(6);
  float cutZone = shiftFromDeadZone;//cm
  //---- hardcoded... not good
  if(station>1 && station<5){
    if(2==ring){
      deadZoneCenter[0]= -162.48 ;
      deadZoneCenter[1] = -81.8744;
      deadZoneCenter[2] = -21.18165;
      deadZoneCenter[3] = 39.51105;
      deadZoneCenter[4] = 100.2939;
      deadZoneCenter[5] = 160.58;
      
      if(localPos.y() >yBorder &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone) ||
	  (localPos.y()> deadZoneCenter[3] + cutZone && localPos.y()< deadZoneCenter[4] - cutZone) ||
	  (localPos.y()> deadZoneCenter[4] + cutZone && localPos.y()< deadZoneCenter[5] - cutZone))){
	pass = true;
      }
    }
    else if(1==ring){
      if(2==station){
	deadZoneCenter[0]= -95.80 ;
	deadZoneCenter[1] = -27.47;
	deadZoneCenter[2] = 33.67;
	deadZoneCenter[3] = 90.85;
        }
      else if(3==station){
	deadZoneCenter[0]= -89.305 ;
	deadZoneCenter[1] = -39.705;
	deadZoneCenter[2] = 20.195;
	deadZoneCenter[3] = 77.395;
      }
      else if(4==station){
	deadZoneCenter[0]= -75.645;
	deadZoneCenter[1] = -26.055;
	deadZoneCenter[2] = 23.855;
	deadZoneCenter[3] = 70.575;
      }
      if(localPos.y() >yBorder &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone))){
	pass = true;
      }
    }
  }
  else if(1==station){
    if(3==ring){
      deadZoneCenter[0]= -83.155 ;
      deadZoneCenter[1] = -22.7401;
      deadZoneCenter[2] = 27.86665;
      deadZoneCenter[3] = 81.005;
      if(localPos.y() > yBorder &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone))){
	pass = true;
      }
    }
    else if(2==ring){
      deadZoneCenter[0]= -86.285 ;
      deadZoneCenter[1] = -32.88305;
      deadZoneCenter[2] = 32.867423;
      deadZoneCenter[3] = 88.205;
      if(localPos.y() > (yBorder) &&
	 ((localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone) ||
	  (localPos.y()> deadZoneCenter[1] + cutZone && localPos.y()< deadZoneCenter[2] - cutZone) ||
	  (localPos.y()> deadZoneCenter[2] + cutZone && localPos.y()< deadZoneCenter[3] - cutZone))){
	pass = true;
      }
    }
    else{
      deadZoneCenter[0]= -81.0;
      deadZoneCenter[1] = 81.0;
      if(localPos.y() > (yBorder) &&
	 (localPos.y()> deadZoneCenter[0] + cutZone && localPos.y()< deadZoneCenter[1] - cutZone )){
	pass = true;
      }
    }
  }
  return pass;
}


int CSCOfflineMonitor::typeIndex(CSCDetId id, int flag){
  
  if (flag == 1){
    // linearlized index bases on endcap, station, and ring
    int index = 0;
    if (id.station() == 1) index = id.ring();
    else index = id.station()*2 + id.ring();
    if (id.endcap() == 1) index = index + 10;
    if (id.endcap() == 2) index = 11 - index;
    return index;
  }

  else if (flag == 2){
    int index = 0;
    if (id.station() == 1 && id.ring() != 4) index = id.ring()+1;
    if (id.station() == 1 && id.ring() == 4) index = 1; 
    if (id.station() != 1) index = id.station()*2 + id.ring();
    if (id.endcap() == 1) index = index + 10;
    if (id.endcap() == 2) index = 11 - index;
    return index;
  }

  else return 0;

}

int CSCOfflineMonitor::chamberSerial( CSCDetId id ) {
  int st = id.station();
  int ri = id.ring();
  int ch = id.chamber();
  int ec = id.endcap();
  int kSerial = ch;
  if (st == 1 && ri == 1) kSerial = ch;
  if (st == 1 && ri == 2) kSerial = ch + 36;
  if (st == 1 && ri == 3) kSerial = ch + 72;
  if (st == 1 && ri == 4) kSerial = ch;
  if (st == 2 && ri == 1) kSerial = ch + 108;
  if (st == 2 && ri == 2) kSerial = ch + 126;
  if (st == 3 && ri == 1) kSerial = ch + 162;
  if (st == 3 && ri == 2) kSerial = ch + 180;
  if (st == 4 && ri == 1) kSerial = ch + 216;
  if (st == 4 && ri == 2) kSerial = ch + 234;  // one day...
  if (ec == 2) kSerial = kSerial + 300;
  return kSerial;
}



DEFINE_FWK_MODULE(CSCOfflineMonitor);

