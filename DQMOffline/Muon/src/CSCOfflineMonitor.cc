/*
 *  Offline DQM module for CSC local reconstruction - based on CSCValidation
 *
 */

#include "DQMOffline/Muon/interface/CSCOfflineMonitor.h"

using namespace std;

///////////////////
//  CONSTRUCTOR  //
///////////////////
CSCOfflineMonitor::CSCOfflineMonitor(const edm::ParameterSet& pset)
{

  rd_token = consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("FEDRawDataCollectionTag") );
  sd_token = consumes<CSCStripDigiCollection>( pset.getParameter<edm::InputTag>("stripDigiTag") );
  wd_token = consumes<CSCWireDigiCollection>( pset.getParameter<edm::InputTag>("wireDigiTag") );
  al_token = consumes<CSCALCTDigiCollection>( pset.getParameter<edm::InputTag>("alctDigiTag") );
  cl_token = consumes<CSCCLCTDigiCollection>( pset.getParameter<edm::InputTag>("clctDigiTag") );
  rh_token = consumes<CSCRecHit2DCollection>( pset.getParameter<edm::InputTag>("cscRecHitTag") );
  se_token = consumes<CSCSegmentCollection>( pset.getParameter<edm::InputTag>("cscSegTag") );

}

void CSCOfflineMonitor::bookHistograms(DQMStore::IBooker & ibooker,
								  edm::Run const & iRun,
								  edm::EventSetup const & /* iSetup */)
{
	  // occupancies
	  ibooker.cd();
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/Occupancy");

	  hCSCOccupancy = ibooker.book1D("hCSCOccupancy","overall CSC occupancy",13,-0.5,12.5);
	  hCSCOccupancy->setBinLabel(2,"Total Events");
	  hCSCOccupancy->setBinLabel(4,"# Events with Wires");
	  hCSCOccupancy->setBinLabel(6,"# Events with Strips");
	  hCSCOccupancy->setBinLabel(8,"# Events with Wires&Strips");
	  hCSCOccupancy->setBinLabel(10,"# Events with Rechits");
	  hCSCOccupancy->setBinLabel(12,"# Events with Segments");
	  hOWiresAndCLCT = ibooker.book2D("hOWiresAndCLCT","Wire and CLCT Digi Occupancy ",36,0.5,36.5,20,0.5,20.5);
	  hOWires = ibooker.book2D("hOWires","Wire Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
	  hOWireSerial = ibooker.book1D("hOWireSerial","Wire Occupancy by Chamber Serial",601,-0.5,600.5);
	  hOWireSerial->setAxisTitle("Chamber Serial Number");
	  hOStrips = ibooker.book2D("hOStrips","Strip Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
	  hOStripSerial = ibooker.book1D("hOStripSerial","Strip Occupancy by Chamber Serial",601,-0.5,600.5);
	  hOStripSerial->setAxisTitle("Chamber Serial Number");
	  hOStripsAndWiresAndCLCT = ibooker.book2D("hOStripsAndWiresAndCLCT","Strip And Wire And CLCT Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
	  hOStripsAndWiresAndCLCT->setAxisTitle("Chamber #");
	  hORecHits = ibooker.book2D("hORecHits","RecHit Occupancy",36,0.5,36.5,20,0.5,20.5);
	  hORecHitsSerial = ibooker.book1D("hORecHitSerial","RecHit Occupancy by Chamber Serial",601,-0.5,600.5);
	  hORecHitsSerial->setAxisTitle("Chamber Serial Number");
	  hOSegments = ibooker.book2D("hOSegments","Segment Occupancy",36,0.5,36.5,20,0.5,20.5);
	  hOSegmentsSerial = ibooker.book1D("hOSegmentSerial","Segment Occupancy by Chamber Serial",601,-0.5,600.5);
	  hOSegmentsSerial->setAxisTitle("Chamber Serial Number");

	  // wire digis
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/Digis");

	  hWirenGroupsTotal = ibooker.book1D("hWirenGroupsTotal","Fired Wires per Event; # Wiregroups Fired",200,-0.5,199.5);
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m42","Wire TBin Fired (ME -4/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m41","Wire TBin Fired (ME -4/1); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m32","Wire TBin Fired (ME -3/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m31","Wire TBin Fired (ME -3/1); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m22","Wire TBin Fired (ME -2/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m21","Wire TBin Fired (ME -2/1); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m11a","Wire TBin Fired (ME -1/1a); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m13","Wire TBin Fired (ME -1/3); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m12","Wire TBin Fired (ME -1/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_m11b","Wire TBin Fired (ME -1/1b); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p11b","Wire TBin Fired (ME +1/1b); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p12","Wire TBin Fired (ME +1/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p13","Wire TBin Fired (ME +1/3); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p11a","Wire TBin Fired (ME +1/1a); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p21","Wire TBin Fired (ME +2/1); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p22","Wire TBin Fired (ME +2/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p31","Wire TBin Fired (ME +3/1); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p32","Wire TBin Fired (ME +3/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p41","Wire TBin Fired (ME +4/1); Time Bin (25ns)",17,-0.5,16.5));
	  hWireTBin.push_back(ibooker.book1D("hWireTBin_p42","Wire TBin Fired (ME +4/2); Time Bin (25ns)",17,-0.5,16.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m42","Wiregroup Number Fired (ME -4/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m41","Wiregroup Number Fired (ME -4/1); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m32","Wiregroup Number Fired (ME -3/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m31","Wiregroup Number Fired (ME -3/1); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m22","Wiregroup Number Fired (ME -2/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m21","Wiregroup Number Fired (ME -2/1); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m11a","Wiregroup Number Fired (ME -1/1a); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m13","Wiregroup Number Fired (ME -1/3); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m12","Wiregroup Number Fired (ME -1/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_m11b","Wiregroup Number Fired (ME -1/1b); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p11b","Wiregroup Number Fired (ME +1/1b); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p12","Wiregroup Number Fired (ME +1/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p13","Wiregroup Number Fired (ME +1/3); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p11a","Wiregroup Number Fired (ME +1/1a); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p21","Wiregroup Number Fired (ME +2/1); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p22","Wiregroup Number Fired (ME +2/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p31","Wiregroup Number Fired (ME +3/1); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p32","Wiregroup Number Fired (ME +3/2); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p41","Wiregroup Number Fired (ME +4/1); Wiregroup #",113,-0.5,112.5));
	  hWireNumber.push_back(ibooker.book1D("hWireNumber_p42","Wiregroup Number Fired (ME +4/2); Wiregroup #",113,-0.5,112.5));

	  // strip digis
	  hStripNFired = ibooker.book1D("hStripNFired","Fired Strips per Event; # Strips Fired (above 13 ADC)",300,-0.5,299.5);
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m42","Strip Number Fired (ME -4/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m41","Strip Number Fired (ME -4/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m32","Strip Number Fired (ME -3/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m31","Strip Number Fired (ME -3/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m22","Strip Number Fired (ME -2/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m21","Strip Number Fired (ME -2/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m11a","Strip Number Fired (ME -1/1a); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m13","Strip Number Fired (ME -1/3); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m12","Strip Number Fired (ME -1/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_m11b","Strip Number Fired (ME -1/1b); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p11b","Strip Number Fired (ME +1/1b); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p12","Strip Number Fired (ME +1/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p13","Strip Number Fired (ME +1/3); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p11a","Strip Number Fired (ME +1/1a); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p21","Strip Number Fired (ME +2/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p22","Strip Number Fired (ME +2/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p31","Strip Number Fired (ME +3/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p32","Strip Number Fired (ME +3/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p41","Strip Number Fired (ME +4/1); Strip # Fired (above 13 ADC)",81,-0.5,80.5));
	  hStripNumber.push_back(ibooker.book1D("hStripNumber_p42","Stripgroup Number Fired (ME +4/2); Strip # Fired (above 13 ADC)",81,-0.5,80.5));

	  // pedestal noise
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/PedestalNoise");

	  hStripPed.push_back(ibooker.book1D("hStripPedMEm42","Pedestal Noise Distribution Chamber ME -4/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm41","Pedestal Noise Distribution Chamber ME -4/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm32","Pedestal Noise Distribution Chamber ME -3/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm31","Pedestal Noise Distribution Chamber ME -3/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm22","Pedestal Noise Distribution Chamber ME -2/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm21","Pedestal Noise Distribution Chamber ME -2/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm11a","Pedestal Noise Distribution Chamber ME -1/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm13","Pedestal Noise Distribution Chamber ME -1/3; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm12","Pedestal Noise Distribution Chamber ME -1/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEm11b","Pedestal Noise Distribution Chamber ME -1/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp11b","Pedestal Noise Distribution Chamber ME +1/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp12","Pedestal Noise Distribution Chamber ME +1/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp13","Pedestal Noise Distribution Chamber ME +1/3; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp11a","Pedestal Noise Distribution Chamber ME +1/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp21","Pedestal Noise Distribution Chamber ME +2/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp22","Pedestal Noise Distribution Chamber ME +2/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp31","Pedestal Noise Distribution Chamber ME +3/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp32","Pedestal Noise Distribution Chamber ME +3/2; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp41","Pedestal Noise Distribution Chamber ME +4/1; ADC Counts",50,-25.,25.));
	  hStripPed.push_back(ibooker.book1D("hStripPedMEp42","Pedestal Noise Distribution Chamber ME +4/2; ADC Counts",50,-25.,25.));

	  // rechits
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/recHits");

	  hRHnrechits = ibooker.book1D("hRHnrechits","recHits per Event (all chambers); # of RecHits",200,-0.50,199.5);
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalp1","recHit global X,Y station +1; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalp2","recHit global X,Y station +2; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalp3","recHit global X,Y station +3; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalp4","recHit global X,Y station +4; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalm1","recHit global X,Y station -1; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalm2","recHit global X,Y station -2; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalm3","recHit global X,Y station -3; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHGlobal.push_back(ibooker.book2D("hRHGlobalm4","recHit global X,Y station -4; Global X (cm); Global Y (cm)",100,-800.,800.,100,-800.,800.));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm42","Sum 3x3 recHit Charge (ME -4/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm41","Sum 3x3 recHit Charge (ME -4/1); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm32","Sum 3x3 recHit Charge (ME -3/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm31","Sum 3x3 recHit Charge (ME -3/1); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm22","Sum 3x3 recHit Charge (ME -2/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm21","Sum 3x3 recHit Charge (ME -2/1); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm11a","Sum 3x3 recHit Charge (ME -1/1a); ADC counts",100,0,4000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm13","Sum 3x3 recHit Charge (ME -1/3); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm12","Sum 3x3 recHit Charge (ME -1/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQm11b","Sum 3x3 recHit Charge (ME -1/1b); ADC counts",100,0,4000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp11b","Sum 3x3 recHit Charge (ME +1/1b); ADC counts",100,0,4000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp12","Sum 3x3 recHit Charge (ME +1/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp13","Sum 3x3 recHit Charge (ME +1/3); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp11a","Sum 3x3 recHit Charge (ME +1/1a); ADC counts",100,0,4000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp21","Sum 3x3 recHit Charge (ME +2/1); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp22","Sum 3x3 recHit Charge (ME +2/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp31","Sum 3x3 recHit Charge (ME +3/1); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp32","Sum 3x3 recHit Charge (ME +3/2); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp41","Sum 3x3 recHit Charge (ME +4/1); ADC counts",100,0,2000));
	  hRHSumQ.push_back(ibooker.book1D("hRHSumQp42","Sum 3x3 recHit Charge (ME +4/2); ADC counts",100,0,2000));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm42","Charge Ratio (Ql+Qr)/Qt (ME -4/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm41","Charge Ratio (Ql+Qr)/Qt (ME -4/1); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm32","Charge Ratio (Ql+Qr)/Qt (ME -3/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm31","Charge Ratio (Ql+Qr)/Qt (ME -3/1); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm22","Charge Ratio (Ql+Qr)/Qt (ME -2/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm21","Charge Ratio (Ql+Qr)/Qt (ME -2/1); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm11a","Charge Ratio (Ql+Qr)/Qt (ME -1/1a); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm13","Charge Ratio (Ql+Qr)/Qt (ME -1/3); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm12","Charge Ratio (Ql+Qr)/Qt (ME -1/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQm11b","Charge Ratio (Ql+Qr)/Qt (ME -1/1b); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp11b","Charge Ratio (Ql+Qr)/Qt (ME +1/1b); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp12","Charge Ratio (Ql+Qr)/Qt (ME +1/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp13","Charge Ratio (Ql+Qr)/Qt (ME +1/3); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp11a","Charge Ratio (Ql+Qr)/Qt (ME +1/1a); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp21","Charge Ratio (Ql+Qr)/Qt (ME +2/1); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp22","Charge Ratio (Ql+Qr)/Qt (ME +2/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp31","Charge Ratio (Ql+Qr)/Qt (ME +3/1); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp32","Charge Ratio (Ql+Qr)/Qt (ME +3/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp41","Charge Ratio (Ql+Qr)/Qt (ME +4/1); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHRatioQ.push_back(ibooker.book1D("hRHRatioQp42","Charge Ratio (Ql+Qr)/Qt (ME +4/2); (Ql+Qr)/Qt",100,-0.1,1.1));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm42","recHit Time (ME -4/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm41","recHit Time (ME -4/1); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm32","recHit Time (ME -3/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm31","recHit Time (ME -3/1); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm22","recHit Time (ME -2/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm21","recHit Time (ME -2/1); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm11a","recHit Time (ME -1/1a); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm13","recHit Time (ME -1/3); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm12","recHit Time (ME -1/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingm11b","recHit Time (ME -1/1b); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp11b","recHit Time (ME +1/1b); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp12","recHit Time (ME +1/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp13","recHit Time (ME +1/3); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp11a","recHit Time (ME +1/1a); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp21","recHit Time (ME +2/1); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp22","recHit Time (ME +2/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp31","recHit Time (ME +3/1); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp32","recHit Time (ME +3/2); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp41","recHit Time (ME +4/1); ns",200,-500.,500.));
	  hRHTiming.push_back(ibooker.book1D("hRHTimingp42","recHit Time (ME +4/2); ns",200,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem42","Anode recHit Time (ME -4/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem41","Anode recHit Time (ME -4/1); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem32","Anode recHit Time (ME -3/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem31","Anode recHit Time (ME -3/1); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem22","Anode recHit Time (ME -2/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem21","Anode recHit Time (ME -2/1); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem11a","Anode recHit Time (ME -1/1a); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem13","Anode recHit Time (ME -1/3); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem12","Anode recHit Time (ME -1/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodem11b","Anode recHit Time (ME -1/1b); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep11b","Anode recHit Time (ME +1/1b); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep12","Anode recHit Time (ME +1/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep13","Anode recHit Time (ME +1/3); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep11a","Anode recHit Time (ME +1/1a); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep21","Anode recHit Time (ME +2/1); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep22","Anode recHit Time (ME +2/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep31","Anode recHit Time (ME +3/1); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep32","Anode recHit Time (ME +3/2); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep41","Anode recHit Time (ME +4/1); ns",80,-500.,500.));
	  hRHTimingAnode.push_back(ibooker.book1D("hRHTimingAnodep42","Anode recHit Time (ME +4/2); ns",80,-500.,500.));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm42","Reconstructed Position on Strip (ME -4/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm41","Reconstructed Position on Strip (ME -4/1); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm32","Reconstructed Position on Strip (ME -3/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm31","Reconstructed Position on Strip (ME -3/1); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm22","Reconstructed Position on Strip (ME -2/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm21","Reconstructed Position on Strip (ME -2/1); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm11a","Reconstructed Position on Strip (ME -1/1a); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm13","Reconstructed Position on Strip (ME -1/3); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm12","Reconstructed Position on Strip (ME -1/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposm11b","Reconstructed Position on Strip (ME -1/1b); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp11b","Reconstructed Position on Strip (ME +1/1b); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp12","Reconstructed Position on Strip (ME +1/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp13","Reconstructed Position on Strip (ME +1/3); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp11a","Reconstructed Position on Strip (ME +1/1a); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp21","Reconstructed Position on Strip (ME +2/1); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp22","Reconstructed Position on Strip (ME +2/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp31","Reconstructed Position on Strip (ME +3/1); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp32","Reconstructed Position on Strip (ME +3/2); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp41","Reconstructed Position on Strip (ME +4/1); Strip Widths",120,-0.6,0.6));
	  hRHstpos.push_back(ibooker.book1D("hRHstposp42","Reconstructed Position on Strip (ME +4/2); Strip Widths",120,-0.6,0.6));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm42","Estimated Error on Strip Measurement (ME -4/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm41","Estimated Error on Strip Measurement (ME -4/1); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm32","Estimated Error on Strip Measurement (ME -3/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm31","Estimated Error on Strip Measurement (ME -3/1); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm22","Estimated Error on Strip Measurement (ME -2/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm21","Estimated Error on Strip Measurement (ME -2/1); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm11a","Estimated Error on Strip Measurement (ME -1/1a); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm13","Estimated Error on Strip Measurement (ME -1/3); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm12","Estimated Error on Strip Measurement (ME -1/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrm11b","Estimated Error on Strip Measurement (ME -1/1b); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp11b","Estimated Error on Strip Measurement (ME +1/1b); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp12","Estimated Error on Strip Measurement (ME +1/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp13","Estimated Error on Strip Measurement (ME +1/3); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp11a","Estimated Error on Strip Measurement (ME +1/1a); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp21","Estimated Error on Strip Measurement (ME +2/1); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp22","Estimated Error on Strip Measurement (ME +2/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp31","Estimated Error on Strip Measurement (ME +3/1); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp32","Estimated Error on Strip Measurement (ME +3/2); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp41","Estimated Error on Strip Measurement (ME +4/1); Strip Widths",75,-0.01,0.24));
	  hRHsterr.push_back(ibooker.book1D("hRHsterrp42","Estimated Error on Strip Measurement (ME +4/2); Strip Widths",75,-0.01,0.24));

	  // segments
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/Segments");

	  hSnSegments   = ibooker.book1D("hSnSegments","Number of Segments per Event; # of Segments",26,-0.5,25.5);
	  hSnhitsAll = ibooker.book1D("hSnhits","N hits on Segments; # of hits",8,-0.5,7.5);
	  hSnhits.push_back(ibooker.book1D("hSnhitsm42","# of hits on Segments (ME -4/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm41","# of hits on Segments (ME -4/1); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm32","# of hits on Segments (ME -3/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm31","# of hits on Segments (ME -3/1); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm22","# of hits on Segments (ME -2/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm21","# of hits on Segments (ME -2/1); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm11a","# of hits on Segments (ME -1/1a); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm13","# of hits on Segments (ME -1/3); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm12","# of hits on Segments (ME -1/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsm11b","# of hits on Segments (ME -1/1b); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp11b","# of hits on Segments (ME +1/1b); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp12","# of hits on Segments (ME +1/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp13","# of hits on Segments (ME +1/3); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp11a","# of hits on Segments (ME +1/1a); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp21","# of hits on Segments (ME +2/1); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp22","# of hits on Segments (ME +2/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp31","# of hits on Segments (ME +3/1); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp32","# of hits on Segments (ME +3/2); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp41","# of hits on Segments (ME +4/1); # of hits",8,-0.5,7.5));
	  hSnhits.push_back(ibooker.book1D("hSnhitsp42","# of hits on Segments (ME +4/2); # of hits",8,-0.5,7.5));
	  hSChiSqAll = ibooker.book1D("hSChiSq","Segment Normalized Chi2; Chi2/ndof",110,-0.05,10.5);
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm42","Segment Normalized Chi2 (ME -4/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm41","Segment Normalized Chi2 (ME -4/1); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm32","Segment Normalized Chi2 (ME -3/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm31","Segment Normalized Chi2 (ME -3/1); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm22","Segment Normalized Chi2 (ME -2/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm21","Segment Normalized Chi2 (ME -2/1); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm11a","Segment Normalized Chi2 (ME -1/1a); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm13","Segment Normalized Chi2 (ME -1/3); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm12","Segment Normalized Chi2 (ME -1/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqm11b","Segment Normalized Chi2 (ME -1/1b); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp11b","Segment Normalized Chi2 (ME +1/1b); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp12","Segment Normalized Chi2 (ME +1/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp13","Segment Normalized Chi2 (ME +1/3); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp11a","Segment Normalized Chi2 (ME +1/1a); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp21","Segment Normalized Chi2 (ME +2/1); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp22","Segment Normalized Chi2 (ME +2/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp31","Segment Normalized Chi2 (ME +3/1); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp32","Segment Normalized Chi2 (ME +3/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp41","Segment Normalized Chi2 (ME +4/1); Chi2/ndof",110,-0.05,10.5));
	  hSChiSq.push_back(ibooker.book1D("hSChiSqp42","Segment Normalized Chi2 (ME +4/2); Chi2/ndof",110,-0.05,10.5));
	  hSChiSqProbAll = ibooker.book1D("hSChiSqProb","Segment chi2 Probability; Probability",110,-0.05,1.05);
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm42","Segment chi2 Probability (ME -4/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm41","Segment chi2 Probability (ME -4/1); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm32","Segment chi2 Probability (ME -3/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm31","Segment chi2 Probability (ME -3/1); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm22","Segment chi2 Probability (ME -2/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm21","Segment chi2 Probability (ME -2/1); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm11a","Segment chi2 Probability (ME -1/1a); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm13","Segment chi2 Probability (ME -1/3); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm12","Segment chi2 Probability (ME -1/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbm11b","Segment chi2 Probability (ME -1/1b); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp11b","Segment chi2 Probability (ME +1/1b); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp12","Segment chi2 Probability (ME +1/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp13","Segment chi2 Probability (ME +1/3); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp11a","Segment chi2 Probability (ME +1/1a); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp21","Segment chi2 Probability (ME +2/1); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp22","Segment chi2 Probability (ME +2/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp31","Segment chi2 Probability (ME +3/1); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp32","Segment chi2 Probability (ME +3/2); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp41","Segment chi2 Probability (ME +4/1); Probability",110,-0.05,1.05));
	  hSChiSqProb.push_back(ibooker.book1D("hSChiSqProbp42","Segment chi2 Probability (ME +4/2); Probability",110,-0.05,1.05));
	  hSGlobalTheta = ibooker.book1D("hSGlobalTheta","Segment Direction (Global Theta); Global Theta (radians)",136,-0.1,3.3);
	  hSGlobalPhi   = ibooker.book1D("hSGlobalPhi","Segment Direction (Global Phi); Global Phi (radians)",  128,-3.2,3.2);
          hSTimeDiff  = ibooker.book1D("hSTimeDiff", "Anode Minus Cathode Segment Time  [ns]",50,-50,50);
          hSTimeAnode  = ibooker.book1D("hSTimeAnode", "Anode Only Segment Time  [ns]",200,-200,200);
	  hSTimeCathode  = ibooker.book1D("hSTimeCathode", "Cathode Only Segment Time  [ns]",200,-200,200);
	  hSTimeCombined = ibooker.book1D("hSTimeCombined", "Segment Time (anode+cathode times) [ns]",200,-200,200);
          hSTimeDiffSerial  = ibooker.book2D("hSTimeDiffSerial", "Anode Minus Cathode Segment Time  [ns]",601,-0.5,600.5,200,-50,50);
          hSTimeAnodeSerial  = ibooker.book2D("hSTimeAnodeSerial", "Anode Only Segment Time  [ns]",601,-0.5,600.5,200,-200,200);
          hSTimeCathodeSerial  = ibooker.book2D("hSTimeCathodeSerial", "Cathode Only Segment Time  [ns]",601,-0.5,600.5,200,-200,200);
          hSTimeCombinedSerial = ibooker.book2D("hSTimeCombinedSerial", "Segment Time (anode+cathode times) [ns]",601,-0.5,600.5,200,-200,200);


	  hSTimeVsZ	  = ibooker.book2D("hSTimeVsZ","Segment Time vs. Z; [ns] vs. [cm]",200,-1200,1200,200,-200,200);
	  hSTimeVsTOF = ibooker.book2D("hSTimeVsTOF","Segment Time vs. Distance from IP; [ns] vs. [cm]",180,500,1400, 200,-200,200);


	  // resolution
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/Resolution");

	  hSResid.push_back(ibooker.book1D("hSResidm42","Fitted Position on Strip - Reconstructed for Layer 3 (ME -4/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm41","Fitted Position on Strip - Reconstructed for Layer 3 (ME -4/1); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm32","Fitted Position on Strip - Reconstructed for Layer 3 (ME -3/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm31","Fitted Position on Strip - Reconstructed for Layer 3 (ME -3/1); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm22","Fitted Position on Strip - Reconstructed for Layer 3 (ME -2/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm21","Fitted Position on Strip - Reconstructed for Layer 3 (ME -2/1); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/1a); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm13","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/3); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm12","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidm11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/1b); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/1b); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp12","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp13","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/3); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/1a); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp21","Fitted Position on Strip - Reconstructed for Layer 3 (ME +2/1); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp22","Fitted Position on Strip - Reconstructed for Layer 3 (ME +2/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp31","Fitted Position on Strip - Reconstructed for Layer 3 (ME +3/1); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp32","Fitted Position on Strip - Reconstructed for Layer 3 (ME +3/2); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp41","Fitted Position on Strip - Reconstructed for Layer 3 (ME +4/1); Strip Widths",100,-0.5,0.5));
	  hSResid.push_back(ibooker.book1D("hSResidp42","Fitted Position on Strip - Reconstructed for Layer 3 (ME +4/2); Strip Widths",100,-0.5,0.5));

	  // efficiency
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/Efficiency");

	  //      hSSTE = ibooker.book1D("hSSTE","hSSTE",40,0.5,40.5);
	  //      hRHSTE = ibooker.book1D("hRHSTE","hRHSTE",40,0.5,40.5);
	  hSnum = ibooker.book1D("hSnum", "CSC w rechits in 2+ layers && segment(s)", 20, 0.5, 20.5);
	  hSden = ibooker.book1D("hSden", "CSC w rechits in 2+ layers", 20, 0.5, 20.5);
	  hRHnum = ibooker.book1D("hRHnum", "CSC w segment(s) && rechits in 6 layers", 20, 0.5, 20.5);
	  hRHden = ibooker.book1D("hRHden", "CSC w segment(s)", 20, 0.5, 20.5);
	  applyCSClabels(hSnum, EXTENDED, X);
	  applyCSClabels(hSden, EXTENDED, X);
	  applyCSClabels(hRHnum, EXTENDED, X);
	  applyCSClabels(hRHden, EXTENDED, X);

	  //      hSEff = ibooker.book1D("hSEff","Segment Efficiency",20,0.5,20.5);
	  //      hRHEff = ibooker.book1D("hRHEff","recHit Efficiency",20,0.5,20.5);
	  hSSTE2 = ibooker.book2D("hSSTE2","hSSTE2",36,0.5,36.5, 20, 0.5, 20.5);
	  hRHSTE2 = ibooker.book2D("hRHSTE2","hRHSTE2",36,0.5,36.5, 20, 0.5, 20.5);
	  hStripSTE2 = ibooker.book2D("hStripSTE2","hStripSTE2",36,0.5,36.5, 20, 0.5, 20.5);
	  hWireSTE2 = ibooker.book2D("hWireSTE2","hWireSTE2",36,0.5,36.5, 20, 0.5, 20.5);
	  hEffDenominator = ibooker.book2D("hEffDenominator","hEffDenominator",36,0.5,36.5, 20, 0.5, 20.5);
	  //      hSEff2 = ibooker.book2D("hSEff2","Segment Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
	  //      hRHEff2 = ibooker.book2D("hRHEff2","recHit Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
	  //      hStripReadoutEff2 = ibooker.book2D("hStripReadoutEff2","strip readout ratio [(strip+clct+wires)/(clct+wires)] 2D",36,0.5,36.5, 20, 0.5, 20.5);
	  //      hStripReadoutEff2->setAxisTitle("Chamber #");
	  //      hStripEff2 = ibooker.book2D("hStripEff2","strip Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
	  //      hWireEff2 = ibooker.book2D("hWireEff2","wire Efficiency 2D",36,0.5,36.5, 18, 0.5, 18.5);
	  hSensitiveAreaEvt = ibooker.book2D("hSensitiveAreaEvt","Events Passing Selection for Efficiency",36,0.5,36.5, 20, 0.5, 20.5);

	  // bx monitor for trigger synchronization
	  ibooker.setCurrentFolder("CSC/CSCOfflineMonitor/BXMonitor");

	  hALCTgetBX = ibooker.book1D("hALCTgetBX","ALCT position in ALCT-L1A match window [BX]",7,-0.5,6.5);
	  //      hALCTgetBXChamberMeans = ibooker.book1D("hALCTgetBXChamberMeans","Chamber Mean ALCT position in ALCT-L1A match window [BX]",60,0,6);
	  hALCTgetBXSerial = ibooker.book2D("hALCTgetBXSerial","ALCT position in ALCT-L1A match window [BX]",601,-0.5,600.5,7,-0.5,6.5);
	  hALCTgetBXSerial->setAxisTitle("Chamber Serial Number");
	  hALCTgetBX2DNumerator = ibooker.book2D("hALCTgetBX2DNumerator","ALCT position in ALCT-L1A match window [BX] (sum)",36,0.5,36.5,20,0.5,20.5);
	  //      hALCTgetBX2DMeans = ibooker.book2D("hALCTgetBX2DMeans","ALCT position in ALCT-L1A match window [BX]",36,0.5,36.5,20,0.5,20.5);
	  hALCTgetBX2Denominator = ibooker.book2D("hALCTgetBX2Denominator","Number of ALCT Digis checked",36,0.5,36.5,20,0.5,20.5);
	  //      hALCTgetBX2DMeans->setAxisTitle("Chamber #");
	  hALCTgetBX2Denominator->setAxisTitle("Chamber #");
	  hALCTMatch = ibooker.book1D("hALCTMatch","ALCT position in ALCT-CLCT match window [BX]",7,-0.5,6.5);
	  //      hALCTMatchChamberMeans = ibooker.book1D("hALCTMatchChamberMeans","Chamber Mean ALCT position in ALCT-CLCT match window [BX]",60,0,6);
	  hALCTMatchSerial = ibooker.book2D("hALCTMatchSerial","ALCT position in ALCT-CLCT match window [BX]",601,-0.5,600.5,7,-0.5,6.5);
	  hALCTMatchSerial->setAxisTitle("Chamber Serial Number");
	  hALCTMatch2DNumerator = ibooker.book2D("hALCTMatch2DNumerator","ALCT position in ALCT-CLCT match window [BX] (sum)",36,0.5,36.5,20,0.5,20.5);
	  //      hALCTMatch2DMeans = ibooker.book2D("hALCTMatch2DMeans","ALCT position in ALCT-CLCT match window [BX]",36,0.5,36.5,20,0.5,20.5);
	  hALCTMatch2Denominator = ibooker.book2D("hALCTMatch2Denominator","Number of ALCT-CLCT matches checked",36,0.5,36.5,20,0.5,20.5);
	  //      hALCTMatch2DMeans->setAxisTitle("Chamber #");
	  hALCTMatch2Denominator->setAxisTitle("Chamber #");
	  hCLCTL1A = ibooker.book1D("hCLCTL1A","L1A - CLCTpreTrigger at TMB [BX]",40,149.5,189.5);
	  //      hCLCTL1AChamberMeans = ibooker.book1D("hCLCTL1AChamberMeans","Chamber Mean L1A - CLCTpreTrigger at TMB [BX]",90,150,159);
	  hCLCTL1ASerial = ibooker.book2D("hCLCTL1ASerial","L1A - CLCTpreTrigger at TMB [BX]",601,-0.5,600.5,40,149.5,189.5);
	  hCLCTL1ASerial->setAxisTitle("Chamber Serial Number");
	  hCLCTL1A2DNumerator = ibooker.book2D("hCLCTL1A2DNumerator","L1A - CLCTpreTrigger at TMB [BX] (sum)",36,0.5,36.5,20,0.5,20.5);
	  //      hCLCTL1A2DMeans = ibooker.book2D("hCLCTL1A2DMeans","L1A - CLCTpreTrigger at TMB [BX]",36,0.5,36.5,20,0.5,20.5);
	  hCLCTL1A2Denominator = ibooker.book2D("hCLCTL1A2Denominator","Number of TMB CLCTs checked",36,0.5,36.5,20,0.5,20.5);


	  // labels
	  applyCSClabels(hOWiresAndCLCT, EXTENDED, Y);
	  applyCSClabels(hOWires, EXTENDED, Y);
	  applyCSClabels(hOStrips, EXTENDED, Y);
	  applyCSClabels(hOStripsAndWiresAndCLCT, EXTENDED, Y);
	  applyCSClabels(hORecHits, EXTENDED, Y);
	  applyCSClabels(hOSegments, EXTENDED, Y);
	  //      applyCSClabels(hSEff, EXTENDED, X);
	  //      applyCSClabels(hRHEff, EXTENDED, X);
	  //      applyCSClabels(hSEff2, SMALL, Y);
	  applyCSClabels(hSSTE2, EXTENDED, Y);
	  applyCSClabels(hEffDenominator, EXTENDED, Y);
	  //      applyCSClabels(hRHEff2, SMALL, Y);
	  applyCSClabels(hRHSTE2, EXTENDED, Y);
	  //      applyCSClabels(hStripReadoutEff2, EXTENDED, Y);
	  //      applyCSClabels(hStripEff2, SMALL, Y);
	  applyCSClabels(hStripSTE2, EXTENDED, Y);
	  //      applyCSClabels(hWireEff2, SMALL, Y);
	  applyCSClabels(hWireSTE2, EXTENDED, Y);
	  applyCSClabels(hSensitiveAreaEvt, EXTENDED, Y);
	  //      applyCSClabels(hALCTgetBX2DMeans, EXTENDED, Y);
	  applyCSClabels(hALCTgetBX2Denominator, EXTENDED, Y);
	  //      applyCSClabels(hALCTMatch2DMeans, EXTENDED, Y);
	  applyCSClabels(hALCTMatch2Denominator, EXTENDED, Y);
	  //      applyCSClabels(hCLCTL1A2DMeans, EXTENDED, Y);
	  applyCSClabels(hCLCTL1A2Denominator,EXTENDED, Y);
}

////////////////
//  Analysis  //
////////////////
void CSCOfflineMonitor::analyze(const edm::Event & event, const edm::EventSetup& eventSetup){

  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  event.getByToken( sd_token, strips );
  event.getByToken( wd_token, wires );
  event.getByToken( al_token, alcts );
  event.getByToken( cl_token, clcts );

  // Get the CSC Geometry :
  edm::ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  // Get the RecHits collection :
  edm::Handle<CSCRecHit2DCollection> recHits;
  event.getByToken( rh_token, recHits );

  // get CSC segment collection
  edm::Handle<CSCSegmentCollection> cscSegments;
  event.getByToken( se_token, cscSegments );

  doOccupancies(strips,wires,recHits,cscSegments,clcts);
  doStripDigis(strips);
  doWireDigis(wires);
  doRecHits(recHits,strips,cscGeom);
  doSegments(cscSegments,cscGeom);
  doResolution(cscSegments,cscGeom);
  doPedestalNoise(strips, cscGeom);
  doEfficiencies(wires,strips, recHits, cscSegments,cscGeom);
  doBXMonitor(alcts, clcts, event, eventSetup);
}

// ==============================================
//
// look at Occupancies
//
// ==============================================

void CSCOfflineMonitor::doOccupancies(edm::Handle<CSCStripDigiCollection> strips, edm::Handle<CSCWireDigiCollection> wires,
									  edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments,
					  edm::Handle<CSCCLCTDigiCollection> clcts){

  bool clcto[2][4][4][36];
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
		  clcto[e][s][r][c] = false;
		  wireo[e][s][r][c] = false;
		  stripo[e][s][r][c] = false;
		  rechito[e][s][r][c] = false;
		  segmento[e][s][r][c] = false;
		}
	  }
	}
  }

  //clcts
  for (CSCCLCTDigiCollection::DigiRangeIterator j=clcts->begin(); j!=clcts->end(); j++) {
	CSCDetId id = (CSCDetId)(*j).first;
	int kEndcap  = id.endcap();
	int kRing    = id.ring();
	int kStation = id.station();
	int kChamber = id.chamber();
	const CSCCLCTDigiCollection::Range& range =(*j).second;
	for (CSCCLCTDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){
	  // Valid digi in the chamber (or in neighbouring chamber) 
	  if((*digiIt).isValid()){
	  //Check whether this CLCT came from ME11a
	if( kStation ==1 && kRing==1 && (*digiIt).getKeyStrip()>128)
	  kRing = 4;
	clcto[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
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
	if( clcto[kEndcap-1][kStation-1][kRing-1][kChamber-1])
	  hOWiresAndCLCT->Fill(kChamber,typeIndex(id,2));
	//Also check for a CLCT in ME11a if you're in ME11 already
	if (kStation==1 && kRing==1 && clcto[kEndcap-1][kStation-1][3][kChamber-1]){
	  CSCDetId idME11a = CSCDetId(kEndcap, kStation, 4, kChamber);
	  hOWiresAndCLCT->Fill(kChamber,typeIndex(idME11a,2));
	} 
	  }
	}//end for loop
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
	  if (clcto[kEndcap-1][kStation-1][kRing-1][kChamber-1]){
		// check if there is a wire digi in this chamber too
		// for ME 1/4 check for a wire in ME 1/1
		if(wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1] || (kRing==4 && wireo[kEndcap-1][kStation-1][0][kChamber-1]) ){
		  hOStripsAndWiresAndCLCT->Fill(kChamber,typeIndex(id,2));
		}
	  }//end clct and wire digi check
		}
	  }//end if (thisStripFired)
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
  // Tim: I'm unhappy with that since it breaks hist statistics
  //  if (nWireGroupsTotal == 0) nWireGroupsTotal = -1;
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
  // Tim: I guess the above comment means 'zero suppress' because the hist range is from -0.5.
  // But doing this means the hist statistics are broken. If the zero bin is high, just apply log scale?
  //  if (nStripsFired == 0) nStripsFired = -1;
  hStripNFired->Fill(nStripsFired);
  // fill n per event

}


//=======================================================
//
// Look at the Pedestal Noise Distributions
//
//=======================================================

void CSCOfflineMonitor::doPedestalNoise(edm::Handle<CSCStripDigiCollection> strips,
					edm::ESHandle<CSCGeometry> cscGeom ) {

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

	  // Why is this code block even here? Doesn't use myStrip after setting it (converts channel to strip
	  // for ganged ME11A
	  if( (kStation == 1 && kRing == 4) && cscGeom->gangedStrips() ) 
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

	/// Find the charge associated with this hit

	int adcsize = dRHIter->nStrips()*dRHIter->nTimeBins();
	float rHSumQ = 0;
	float sumsides = 0;
	for (unsigned int i = 0; i < dRHIter->nStrips(); i++){
	  for (unsigned int j=0; j<dRHIter->nTimeBins()-1; j++) {
	rHSumQ+=dRHIter->adcs(i,j);
		if ( i != 1 ) sumsides += dRHIter->adcs(i,j); // skip central strip
	  }
	}	

	float rHratioQ = sumsides/rHSumQ;
	if (adcsize != 12) rHratioQ = -99;

	// Get the signal timing of this hit
	float rHtime = (*dRHIter).tpeak(); //calculated from cathode SCA bins
	float rHtimeAnode = (*dRHIter).wireTime(); // calculated from anode wire bx

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
	hRHTimingAnode[tIndex-1]->Fill(rHtimeAnode);
	hRHGlobal[sIndex-1]->Fill(grecx,grecy);

  } //end rechit loop

  //  if (nRecHits == 0) nRecHits = -1; // I see no point in doing this
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
	LocalPoint localPos = (*dSiter).localPosition();
	LocalVector segDir = (*dSiter).localDirection();

	// prepare to calculate segment times 
	float timeCathode = 0;  //average from cathode information alone
	float timeAnode = 0;  //average from pruned anode information alone
	float timeCombined = 0; //average from cathode hits and pruned anode list
	std::vector<float> cathodeTimes;
	std::vector<float> anodeTimes;
	// Get the CSC recHits that contribute to this segment.
	std::vector<CSCRecHit2D> theseRecHits = (*dSiter).specificRecHits();
	for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
	  if ( !((*iRH).isValid()) ) continue;  // only interested in valid hits
	  cathodeTimes.push_back((*iRH).tpeak());
	  anodeTimes.push_back((*iRH).wireTime());
	}//end rechit loop

	// Calculate cathode average
	for (unsigned int i=0; i<cathodeTimes.size(); i++) 
	timeCathode+=cathodeTimes[i]/cathodeTimes.size();

	// Prune the anode list to deal with the late tail 
	float anodeMaxDiff;
	bool modified = false;
	std::vector<float>::iterator anodeMaxHit;
	do {
	  if (anodeTimes.size()==0) continue;
	  timeAnode=0;
	  anodeMaxDiff=0;
	  modified=false;

	  // Find the average
	  for (unsigned int j=0; j<anodeTimes.size(); j++) timeAnode+=anodeTimes[j]/anodeTimes.size();

	  // Find the maximum outlier hit
	  for (unsigned int j=0; j<anodeTimes.size(); j++) {
	if (fabs(anodeTimes[j]-timeAnode)>anodeMaxDiff) {
	  anodeMaxHit=anodeTimes.begin()+j;
	  anodeMaxDiff=fabs(anodeTimes[j]-timeAnode);
	}
	  }

	  // Cut hit if its greater than some time away
	  if (anodeMaxDiff>26) {
	modified=true;
	anodeTimes.erase(anodeMaxHit);
	  }
	} while (modified);

	// Calculate combined anode and cathode time average
	if(cathodeTimes.size()+anodeTimes.size() >0 )
	  timeCombined = (timeCathode*cathodeTimes.size() + timeAnode*anodeTimes.size())/(cathodeTimes.size()+anodeTimes.size());

	// global transformation
	float globX = 0.;
	float globY = 0.;
	float globZ = 0.;
	float globTOF = 0.;
	float globTheta = 0.;
	float globPhi   = 0.;
	const CSCChamber* cscchamber = cscGeom->chamber(id);
	if (cscchamber) {
	  GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
	  globX = globalPosition.x();
	  globY = globalPosition.y();
	  globZ = globalPosition.z();
	  globTOF = sqrt(globX*globX+globY*globY+globZ*globZ);
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
	hSTimeDiff->Fill(timeAnode-timeCathode);
        hSTimeAnode->Fill(timeAnode);
	hSTimeCathode->Fill(timeCathode);
	hSTimeCombined->Fill(timeCombined);
        hSTimeDiffSerial->Fill(chamberSerial(id),timeAnode-timeCathode);
        hSTimeAnodeSerial->Fill(chamberSerial(id),timeAnode);
        hSTimeCathodeSerial->Fill(chamberSerial(id),timeCathode);
        hSTimeCombinedSerial->Fill(chamberSerial(id),timeCombined);
	hSTimeVsZ->Fill(globZ, timeCombined);
	hSTimeVsTOF->Fill(globTOF, timeCombined);

  } // end segment loop

  //  if (nSegments == 0) nSegments = -1; // I see no point in doing this
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

	  /// Find the strip containing this hit
	  int centerid    =  iRH->nStrips()/2 + 1;
	  int centerStrip =  iRH->channels(centerid - 1);   

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



	// Fit all points except layer 3, then compare expected value for layer 3 to reconstructed value
	//    float residual = -99.; // used to fill always
	if (nRH == 6){
	  float expected = fitX(sp,se);
	  float residual = expected - sp(3,1);

	  hSResid[typeIndex(id)-1]->Fill(residual); // fill here so stats make sense
	}

	//  hSResid[typeIndex(id)-1]->Fill(residual); // used to fill here but then stats distorted by underflows

  } // end segment loop



}


//-------------------------------------------------------------------------------------
// Fits a straight line to a set of 5 points with errors.  Functions assumes 6 points
// and removes hit in layer 3.  It then returns the expected position value in layer 3
// based on the fit.
//-------------------------------------------------------------------------------------
float CSCOfflineMonitor::fitX(const CLHEP::HepMatrix& points, const CLHEP::HepMatrix& errors){

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
			  hSnum->Fill(bin);
			}
			//---- All segment events (normalization)
			hSden->Fill(bin);
			//}
		  }
		  if(AllSegments[iE][iS][iR][iC]){
			if(NumberOfLayers==6){
			  //---- Efficient rechit events
			  hRHnum->Fill(bin);;
			}
			//---- All rechit events (normalization)
			hRHden->Fill(bin);;
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
  // at the end the chamber types will be numbered as 1 to 20
  // (ME-4./2, ME-4/1, -ME3/2, -ME3/1, ..., +ME3/1, +ME3/2, ME+4/1, ME+4/2) 
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
  chamberTypes["ME4/2"] = 9.5;

  if(theSeg.size()){
	std::map <int , GlobalPoint> extrapolatedPoint;
	std::map <int , GlobalPoint>::iterator it;
	const CSCGeometry::ChamberContainer& ChamberContainer = cscGeom->chambers();
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
	const std::array<const float, 4> & layerBounds = layerGeom->parameters ();
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
	  verticalScale += 10.5;
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

// ==============================================
//
// Look at BX level trigger synchronization
//
// ==============================================

void CSCOfflineMonitor::doBXMonitor(edm::Handle<CSCALCTDigiCollection> alcts, edm::Handle<CSCCLCTDigiCollection> clcts,
   const edm::Event & event, const edm::EventSetup& eventSetup){

  // Loop over ALCTDigis

  for (CSCALCTDigiCollection::DigiRangeIterator j=alcts->begin(); j!=alcts->end(); j++) {
	const CSCDetId& idALCT = (*j).first;
	const CSCALCTDigiCollection::Range& range =(*j).second;
	for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt){
	  // Valid digi in the chamber (or in neighbouring chamber)  
	  if((*digiIt).isValid()){
	hALCTgetBX->Fill((*digiIt).getBX());
	hALCTgetBXSerial->Fill(chamberSerial(idALCT),(*digiIt).getBX());
	hALCTgetBX2DNumerator->Fill(idALCT.chamber(),typeIndex(idALCT,2),(*digiIt).getBX());
	hALCTgetBX2Denominator->Fill(idALCT.chamber(),typeIndex(idALCT,2));
	  }
	}
  }// end ALCT Digi loop


  // Loop over raw data to get TMBHeader information
  // Taking code from EventFilter/CSCRawToDigis/CSCDCCUnpacker.cc
  edm::ESHandle<CSCCrateMap> hcrate;
  eventSetup.get<CSCCrateMapRcd>().get(hcrate); 
  const CSCCrateMap* pcrate = hcrate.product();


  // Try to get raw data
  edm::Handle<FEDRawDataCollection> rawdata;
  if ( !( event.getByToken( rd_token, rawdata ) ) ){
	edm::LogWarning("CSCOfflineMonitor") << " FEDRawDataCollection not available";
	return;
  }



  bool goodEvent = false;
  unsigned long dccBinCheckMask = 0x06080016;
  unsigned int examinerMask = 0x1FEBF3F6;
  unsigned int errorMask = 0x0;

  // For new CSC readout layout, which doesn't include DCCs need to loop over DDU FED IDs. 
  // DCC IDs are included for backward compatibility with old data
  std::vector<unsigned int> cscFEDids;

  for (unsigned int id=FEDNumbering::MINCSCFEDID;
	   id<=FEDNumbering::MAXCSCFEDID; ++id)   // loop over DCCs
	{
	  cscFEDids.push_back(id);
	}

  for (unsigned int id=FEDNumbering::MINCSCDDUFEDID;
	   id<=FEDNumbering::MAXCSCDDUFEDID; ++id)   // loop over DDUs
	{
	  cscFEDids.push_back(id);
	}

  for (unsigned int i=0; i<cscFEDids.size(); i++)   // loop over all CSC FEDs (DCCs and DDUs)
	{
	  unsigned int id = cscFEDids[i];
	  bool isDDU_FED = ((id >= FEDNumbering::MINCSCDDUFEDID) && (id <= FEDNumbering::MAXCSCDDUFEDID))?true:false;

	/// uncomment this for regional unpacking
	/// if (id!=SOME_ID) continue;

	/// Take a reference to this FED's data
	const FEDRawData& fedData = rawdata->FEDData(id);
	unsigned long length =  fedData.size();

	if (length>=32){ ///if fed has data then unpack it
	  CSCDCCExaminer* examiner = NULL;
	  std::stringstream examiner_out, examiner_err;
	  goodEvent = true;
	  examiner = new CSCDCCExaminer();
	  if( examinerMask&0x40000 ) examiner->crcCFEB(1);
	  if( examinerMask&0x8000  ) examiner->crcTMB (1);
	  if( examinerMask&0x0400  ) examiner->crcALCT(1);
	  examiner->setMask(examinerMask);
	  const short unsigned int *data = (short unsigned int *)fedData.data();

	  int res = examiner->check(data,long(fedData.size()/2));
	  if( res < 0 )   {
	goodEvent=false;
	  } 
	  else {    
	goodEvent=!(examiner->errors()&dccBinCheckMask);
	  }


	  if (goodEvent) {
	///get a pointer to data and pass it to constructor for unpacking

	CSCDCCExaminer * ptrExaminer = examiner;

		std::vector<CSCDDUEventData> fed_Data;
		std::vector<CSCDDUEventData>* ptr_fedData = &fed_Data;


		if (isDDU_FED) // Use new DDU FED readout mode
		  {

			CSCDDUEventData single_dduData((short unsigned int *) fedData.data(), ptrExaminer);
			fed_Data.push_back(single_dduData);

		  }
		else  // Use old DCC FED readout mode
		  {
			CSCDCCEventData dccData((short unsigned int *) fedData.data(), ptrExaminer);
			fed_Data = dccData.dduData();
		  }

		///get a reference to dduData
		const std::vector<CSCDDUEventData> & dduData = *ptr_fedData;

		/// set default detid to that for E=+z, S=1, R=1, C=1, L=1
		CSCDetId layer(1, 1, 1, 1, 1);
		for (unsigned int iDDU=0; iDDU<dduData.size(); ++iDDU) {  // loop over DDUs
		   /// skip the DDU if its data has serious errors
		   /// define a mask for serious errors 
	  if (dduData[iDDU].trailer().errorstat()&errorMask) {
		LogTrace("CSCDCCUnpacker|CSCRawToDigi") << "DDU# " << iDDU << " has serious error - no digis unpacked! " <<
		  std::hex << dduData[iDDU].trailer().errorstat();
		continue; // to next iteration of DDU loop
	  }

	  ///get a reference to chamber data
	  const std::vector<CSCEventData> & cscData = dduData[iDDU].cscData();


	  for (unsigned int iCSC=0; iCSC<cscData.size(); ++iCSC) { // loop over CSCs

		///first process chamber-wide digis such as LCT
		int vmecrate = cscData[iCSC].dmbHeader()->crateID();
		int dmb = cscData[iCSC].dmbHeader()->dmbID();

		int icfeb = 0;  /// default value for all digis not related to cfebs
		int ilayer = 0; /// layer=0 flags entire chamber

			 if ((vmecrate>=1)&&(vmecrate<=60) && (dmb>=1)&&(dmb<=10)&&(dmb!=6)) {
			   layer = pcrate->detId(vmecrate, dmb,icfeb,ilayer );
			 } 
			 else{
			   LogTrace ("CSCOfflineMonitor") << " detID input out of range!!! ";
			   LogTrace ("CSCOfflineMonitor")
				 << " skipping chamber vme= " << vmecrate << " dmb= " << dmb;
			   continue; // to next iteration of iCSC loop
			 }


   		/// check alct data integrity 
  		int nalct = cscData[iCSC].dmbHeader()->nalct();
  		bool goodALCT=false;
  		//if (nalct&&(cscData[iCSC].dataPresent>>6&0x1)==1) {
  		if (nalct&&cscData[iCSC].alctHeader()) {  
  		  if (cscData[iCSC].alctHeader()->check()){
  		goodALCT=true;
  		  }
  		}

		///check tmb data integrity
		int nclct = cscData[iCSC].dmbHeader()->nclct();
		bool goodTMB=false;
		if (nclct&&cscData[iCSC].tmbData()) {
		  if (cscData[iCSC].tmbHeader()->check()){
		if (cscData[iCSC].clctData()->check()) goodTMB=true; 
		  }
		}

 		if (goodTMB && goodALCT) { 
		  const CSCTMBHeader *tmbHead = cscData[iCSC].tmbHeader();
		  std::vector<CSCCLCTDigi> clcts = cscData[iCSC].tmbHeader()->CLCTDigis(layer.rawId());
		  if (clcts.size()==0 || !(clcts[0].isValid()))
		continue;
		  // Check if the CLCT was in ME11a (ring 4)
		  if(layer.station()==1 && layer.ring() ==1 && clcts[0].getKeyStrip()>128){
		layer = CSCDetId(layer.endcap(),layer.station(),4, layer.chamber()); 
		  }
		  hALCTMatch->Fill(tmbHead->ALCTMatchTime());
		  hALCTMatchSerial->Fill(chamberSerial(layer),tmbHead->ALCTMatchTime());
		  // Only fill big 2D display if ALCTMatchTime !=6, since this bin is polluted by the CLCT only triggers
		  // One will have to look at the serial plots to see if the are a lot of entries here
		  if(tmbHead->ALCTMatchTime()!=6){
		hALCTMatch2DNumerator->Fill(layer.chamber(),typeIndex(layer,2),tmbHead->ALCTMatchTime());
		hALCTMatch2Denominator->Fill(layer.chamber(),typeIndex(layer,2));	      
		  }

		  int TMB_CLCTpre_rel_L1A = tmbHead->BXNCount()-clcts[0].getFullBX();
		  if (TMB_CLCTpre_rel_L1A > 3563)
		TMB_CLCTpre_rel_L1A = TMB_CLCTpre_rel_L1A - 3564;
		  if (TMB_CLCTpre_rel_L1A < 0)
		TMB_CLCTpre_rel_L1A = TMB_CLCTpre_rel_L1A + 3564;

		  hCLCTL1A->Fill(TMB_CLCTpre_rel_L1A);
		  hCLCTL1ASerial->Fill(chamberSerial(layer),TMB_CLCTpre_rel_L1A);
		  hCLCTL1A2DNumerator->Fill(layer.chamber(),typeIndex(layer,2),TMB_CLCTpre_rel_L1A);
		  hCLCTL1A2Denominator->Fill(layer.chamber(),typeIndex(layer,2));	      

		}// end if goodTMB and goodALCT
	  }// end loop CSCData
	}// end loop DDU
	  }// end if good event
	  if (examiner!=NULL) delete examiner;
	}// end if non-zero fed data
  }// end DCC loop for NON-REFERENCE

  return;

}

bool CSCOfflineMonitor::withinSensitiveRegion(LocalPoint localPos, const std::array<const float, 4> & layerBounds,
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

	// linearized index based on endcap, station, and ring

  if (flag == 1){
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
  if (st == 4 && ri == 2) kSerial = ch + 234;  // from 2014
  if (ec == 2) kSerial = kSerial + 300;
  return kSerial;
}

void CSCOfflineMonitor::applyCSClabels( MonitorElement* me, LabelType t, AxisType a ) {
  if (me != NULL)
  {
	me->setAxisTitle("Chamber #");
	if (t == EXTENDED)
	{
	  me->setBinLabel(1,"ME -4/2",a);
	  me->setBinLabel(2,"ME -4/1",a);
	  me->setBinLabel(3,"ME -3/2",a);
	  me->setBinLabel(4,"ME -3/1",a);
	  me->setBinLabel(5,"ME -2/2",a);
	  me->setBinLabel(6,"ME -2/1",a);
	  me->setBinLabel(7,"ME -1/3",a);
	  me->setBinLabel(8,"ME -1/2",a);
	  me->setBinLabel(9,"ME -1/1b",a);
	  me->setBinLabel(10,"ME -1/1a",a);
	  me->setBinLabel(11,"ME +1/1a",a);
	  me->setBinLabel(12,"ME +1/1b",a);
	  me->setBinLabel(13,"ME +1/2",a);
	  me->setBinLabel(14,"ME +1/3",a);
	  me->setBinLabel(15,"ME +2/1",a);
	  me->setBinLabel(16,"ME +2/2",a);
	  me->setBinLabel(17,"ME +3/1",a);
	  me->setBinLabel(18,"ME +3/2",a);
	  me->setBinLabel(19,"ME +4/1",a);
	  me->setBinLabel(20,"ME +4/2",a);
	}
	else if (t == SMALL)
	{
	  me->setBinLabel(1,"ME -4/1",a);
	  me->setBinLabel(2,"ME -3/2",a);
	  me->setBinLabel(3,"ME -3/1",a);
	  me->setBinLabel(4,"ME -2/2",a);
	  me->setBinLabel(5,"ME -2/1",a);
	  me->setBinLabel(6,"ME -1/3",a);
	  me->setBinLabel(7,"ME -1/2",a);
	  me->setBinLabel(8,"ME -1/1b",a);
	  me->setBinLabel(9,"ME -1/1a",a);
	  me->setBinLabel(10,"ME +1/1a",a);
	  me->setBinLabel(11,"ME +1/1b",a);
	  me->setBinLabel(12,"ME +1/2",a);
	  me->setBinLabel(13,"ME +1/3",a);
	  me->setBinLabel(14,"ME +2/1",a);
	  me->setBinLabel(15,"ME +2/2",a);
	  me->setBinLabel(16,"ME +3/1",a);
	  me->setBinLabel(17,"ME +3/2",a);
	  me->setBinLabel(18,"ME +4/1",a);
	}
  }
}

DEFINE_FWK_MODULE(CSCOfflineMonitor);
