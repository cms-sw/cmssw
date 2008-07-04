/*
 *  simple validation package for CSC DIGIs, RECHITs and SEGMENTs.
 *
 *  Michael Schmitt
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

}

void CSCOfflineMonitor::beginJob(edm::EventSetup const& iSetup){
      dbe = Service<DQMStore>().operator->();

      // wire digis
      dbe->setCurrentFolder("Muons/CSCOfflineMonitor/Digis");
      hWireAll  = dbe->book1D("hWireAll","all wire group numbers",121,-0.5,120.5);
      hWireTBinAll  = dbe->book1D("hWireTBinAll","time bins all wires",21,-0.5,20.5);
      hWirenGroupsTotal = dbe->book1D("hWirenGroupsTotal","total number of wire groups",101,-0.5,100.5);
      hWireCodeBroad = dbe->book1D("hWireCodeBroad","broad scope code for wires",33,-16.5,16.5);
      hWireCodeNarrow.push_back(dbe->book1D("hWireCodeNarrow1","narrow scope wire code station 1",801,-400.5,400.5));
      hWireCodeNarrow.push_back(dbe->book1D("hWireCodeNarrow2","narrow scope wire code station 2",801,-400.5,400.5));
      hWireCodeNarrow.push_back(dbe->book1D("hWireCodeNarrow3","narrow scope wire code station 3",801,-400.5,400.5));
      hWireCodeNarrow.push_back(dbe->book1D("hWireCodeNarrow4","narrow scope wire code station 4",801,-400.5,400.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m42","wire number ME -4/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m41","wire number ME -4/1",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m32","wire number ME -3/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m31","wire number ME -3/1",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m22","wire number ME -2/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m21","wire number ME -2/1",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m11a","wire number ME -1/1a",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m13","wire number ME -1/3",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m12","wire number ME -1/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_m11b","wire number ME -1/1b",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p11b","wire number ME +1/1b",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p12","wire number ME +1/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p13","wire number ME +1/3",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p11a","wire number ME +1/1a",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p21","wire number ME +2/1",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p22","wire number ME +2/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p31","wire number ME +3/1",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p32","wire number ME +3/2",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p41","wire number ME +4/1",113,-0.5,112.5));
      hWireWire.push_back(dbe->book1D("hWireWire_p42","wire number ME +4/2",113,-0.5,112.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m42","layer wire ME -4/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m41","layer wire ME -4/1",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m32","layer wire ME -3/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m31","layer wire ME -3/1",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m22","layer wire ME -2/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m21","layer wire ME -2/1",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m11a","layer wire ME -1/1a",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m13","layer wire ME -1/3",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m12","layer wire ME -1/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_m11b","layer wire ME -1/1b",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p11b","layer wire ME +1/1b",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p12","layer wire ME +1/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p13","layer wire ME +1/3",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p11a","layer wire ME +1/1a",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p21","layer wire ME +2/1",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p22","layer wire ME +2/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p31","layer wire ME +3/1",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p32","layer wire ME +3/2",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p41","layer wire ME +4/1",7,-0.5,6.5));
      hWireLayer.push_back(dbe->book1D("hWireLayer_p42","layer wire ME +4/2",7,-0.5,6.5));

      // strip digis
      hStripAll = dbe->book1D("hStripAll","all strip numbers",81,-0.5,80.5);
      hStripNFired = dbe->book1D("hStripNFired","total number of fired strips",601,-0.5,600.5);
      hStripCodeBroad = dbe->book1D("hStripCodeBroad","broad scope code for strips",33,-16.5,16.5);
      hStripCodeNarrow.push_back(dbe->book1D("hStripCodeNarrow1","narrow scope strip code station 1",801,-400.5,400.5));
      hStripCodeNarrow.push_back(dbe->book1D("hStripCodeNarrow2","narrow scope strip code station 2",801,-400.5,400.5));
      hStripCodeNarrow.push_back(dbe->book1D("hStripCodeNarrow3","narrow scope strip code station 3",801,-400.5,400.5));
      hStripCodeNarrow.push_back(dbe->book1D("hStripCodeNarrow4","narrow scope strip code station 4",801,-400.5,400.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m42","layer strip ME -4/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m41","layer strip ME -4/1",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m32","layer strip ME -3/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m31","layer strip ME -3/1",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m22","layer strip ME -2/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m21","layer strip ME -2/1",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m11a","layer strip ME -1/1a",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m13","layer strip ME -1/3",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m12","layer strip ME -1/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_m11b","layer strip ME -1/1b",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p11b","layer strip ME +1/1b",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p12","layer strip ME +1/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p13","layer strip ME +1/3",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p11a","layer strip ME +1/1a",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p21","layer strip ME +2/1",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p22","layer strip ME +2/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p31","layer strip ME +3/1",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p32","layer strip ME +3/2",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p41","layer strip ME +4/1",7,-0.5,6.5));
      hStripLayer.push_back(dbe->book1D("hStripLayer_p42","layer strip ME +4/2",7,-0.5,6.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m42","strip number ME -4/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m41","strip number ME -4/1",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m32","strip number ME -3/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m31","strip number ME -3/1",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m22","strip number ME -2/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m21","strip number ME -2/1",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m11a","strip number ME -1/1a",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m13","strip number ME -1/3",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m12","strip number ME -1/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_m11b","strip number ME -1/1b",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p11b","strip number ME +1/1b",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p12","strip number ME +1/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p13","strip number ME +1/3",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p11a","strip number ME +1/1a",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p21","strip number ME +2/1",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p22","strip number ME +2/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p31","strip number ME +3/1",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p32","strip number ME +3/2",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p41","strip number ME +4/1",81,-0.5,80.5));
      hStripStrip.push_back(dbe->book1D("hStripStrip_p42","strip number ME +4/2",81,-0.5,80.5));


      //Pedestal Noise Plots
      dbe->setCurrentFolder("Muons/CSCOfflineMonitor/PedestalNoise");

      hStripPedAll = dbe->book1D("hStripPed","Pedestal Noise Distribution",50,-25.,25.);

      hStripPed.push_back(dbe->book1D("hStripPedMEm42","Pedestal Noise Distribution Chamber ME -4/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm41","Pedestal Noise Distribution Chamber ME -4/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm32","Pedestal Noise Distribution Chamber ME -3/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm31","Pedestal Noise Distribution Chamber ME -3/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm22","Pedestal Noise Distribution Chamber ME -2/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm21","Pedestal Noise Distribution Chamber ME -2/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm11a","Pedestal Noise Distribution Chamber ME -1/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm13","Pedestal Noise Distribution Chamber ME -1/3 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm12","Pedestal Noise Distribution Chamber ME -1/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEm11b","Pedestal Noise Distribution Chamber ME -1/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp11b","Pedestal Noise Distribution Chamber ME +1/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp12","Pedestal Noise Distribution Chamber ME +1/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp13","Pedestal Noise Distribution Chamber ME +1/3 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp11a","Pedestal Noise Distribution Chamber ME +1/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp21","Pedestal Noise Distribution Chamber ME +2/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp22","Pedestal Noise Distribution Chamber ME +2/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp31","Pedestal Noise Distribution Chamber ME +3/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp32","Pedestal Noise Distribution Chamber ME +3/2 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp41","Pedestal Noise Distribution Chamber ME +4/1 ",50,-25.,25.));
      hStripPed.push_back(dbe->book1D("hStripPedMEp42","Pedestal Noise Distribution Chamber ME +4/2 ",50,-25.,25.));

      //hPedvsStrip = dbe->book2D("hPedvsStrip","Pedestal Noise Distribution",4000000,1000000.,5000000.,50,-25.,25.);

      // recHits
      dbe->setCurrentFolder("Muons/CSCOfflineMonitor/recHits");
      hRHCodeBroad = dbe->book1D("hRHCodeBroad","broad scope code for recHits",33,-16.5,16.5);
      hRHCodeNarrow.push_back(dbe->book1D("hRHCodeNarrow1","narrow scope recHit code station 1",801,-400.5,400.5));
      hRHCodeNarrow.push_back(dbe->book1D("hRHCodeNarrow2","narrow scope recHit code station 2",801,-400.5,400.5));
      hRHCodeNarrow.push_back(dbe->book1D("hRHCodeNarrow3","narrow scope recHit code station 3",801,-400.5,400.5));
      hRHCodeNarrow.push_back(dbe->book1D("hRHCodeNarrow4","narrow scope recHit code station 4",801,-400.5,400.5));

      hRHLayer.push_back(dbe->book1D("hRHLayerm42","layer recHit ME -4/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm41","layer recHit ME -4/1",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm32","layer recHit ME -3/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm31","layer recHit ME -3/1",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm22","layer recHit ME -2/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm21","layer recHit ME -2/1",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm11a","layer recHit ME -1/1a",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm13","layer recHit ME -1/3",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm12","layer recHit ME -1/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerm11b","layer recHit ME -1/1b",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp11b","layer recHit ME +1/1b",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp12","layer recHit ME +1/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp13","layer recHit ME +1/3",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp11a","layer recHit ME +1/1a",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp21","layer recHit ME +2/1",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp22","layer recHit ME +2/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp31","layer recHit ME +3/1",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp32","layer recHit ME +3/2",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp41","layer recHit ME +4/1",7,-0.5,6.5));
      hRHLayer.push_back(dbe->book1D("hRHLayerp42","layer recHit ME +4/2",7,-0.5,6.5));


      hRHX.push_back(dbe->book1D("hRHXm42","local X recHit ME -4/2",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXm41","local X recHit ME -4/1",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXm32","local X recHit ME -3/2",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXm31","local X recHit ME -3/1",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXm22","local X recHit ME -2/2",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXm21","local X recHit ME -2/1",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXm11a","local X recHit ME -1/1a",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXm13","local X recHit ME -1/3",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXm12","local X recHit ME -1/2",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXm11b","local X recHit ME -1/1b",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXp11b","local X recHit ME +1/1b",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXp12","local X recHit ME +1/2",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXp13","local X recHit ME +1/3",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXp11a","local X recHit ME +1/1a",120,-60.,60.));
      hRHX.push_back(dbe->book1D("hRHXp21","local X recHit ME +2/1",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXp22","local X recHit ME +2/2",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXp31","local X recHit ME +3/1",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXp32","local X recHit ME +3/2",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXp41","local X recHit ME +4/1",160,-80.,80.));
      hRHX.push_back(dbe->book1D("hRHXp42","local X recHit ME +4/2",160,-80.,80.));


      hRHY.push_back(dbe->book1D("hRHYm42","local Y recHit ME -4/2",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYm41","local Y recHit ME -4/1",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYm32","local Y recHit ME -3/2",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYm31","local Y recHit ME -3/1",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYm22","local Y recHit ME -2/2",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYm21","local Y recHit ME -2/1",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYm11a","local Y recHit ME -1/1a",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYm13","local Y recHit ME -1/3",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYm12","local Y recHit ME -1/2",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYm11b","local Y recHit ME -1/1b",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYp11b","local Y recHit ME +1/1b",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYp12","local Y recHit ME +1/2",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYp13","local Y recHit ME +1/3",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYp11a","local Y recHit ME +1/1a",50,-100.,100.));
      hRHY.push_back(dbe->book1D("hRHYp21","local Y recHit ME +2/1",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYp22","local Y recHit ME +2/2",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYp31","local Y recHit ME +3/1",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYp32","local Y recHit ME +3/2",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYp41","local Y recHit ME +4/1",60,-180.,180.));
      hRHY.push_back(dbe->book1D("hRHYp42","local Y recHit ME +4/2",60,-180.,180.));


      hRHGlobal.push_back(dbe->book2D("hRHGlobalp1","recHit global X,Y station +1",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp2","recHit global X,Y station +2",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp3","recHit global X,Y station +3",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalp4","recHit global X,Y station +4",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm1","recHit global X,Y station -1",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm2","recHit global X,Y station -2",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm3","recHit global X,Y station -3",400,-800.,800.,400,-800.,800.));
      hRHGlobal.push_back(dbe->book2D("hRHGlobalm4","recHit global X,Y station -4",400,-800.,800.,400,-800.,800.));


      hRHSumQ.push_back(dbe->book1D("hRHSumQm42","Sum 3x3 recHit Charge (ME -4/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm41","Sum 3x3 recHit Charge (ME -4/1)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm32","Sum 3x3 recHit Charge (ME -3/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm31","Sum 3x3 recHit Charge (ME -3/1)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm22","Sum 3x3 recHit Charge (ME -2/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm21","Sum 3x3 recHit Charge (ME -2/1)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm11a","Sum 3x3 recHit Charge (ME -1/1a)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm13","Sum 3x3 recHit Charge (ME -1/3)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm12","Sum 3x3 recHit Charge (ME -1/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQm11b","Sum 3x3 recHit Charge (ME -1/1b)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp11b","Sum 3x3 recHit Charge (ME +1/1b)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp12","Sum 3x3 recHit Charge (ME +1/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp13","Sum 3x3 recHit Charge (ME +1/3)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp11a","Sum 3x3 recHit Charge (ME +1/1a)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp21","Sum 3x3 recHit Charge (ME +2/1)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp22","Sum 3x3 recHit Charge (ME +2/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp31","Sum 3x3 recHit Charge (ME +3/1)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp32","Sum 3x3 recHit Charge (ME +3/2)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp41","Sum 3x3 recHit Charge (ME +4/1)",250,0,2000));
      hRHSumQ.push_back(dbe->book1D("hRHSumQp42","Sum 3x3 recHit Charge (ME +4/2)",250,0,2000));


      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm42","Ratio (Ql+Qr)/Qt (ME -4/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm41","Ratio (Ql+Qr)/Qt (ME -4/1)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm32","Ratio (Ql+Qr)/Qt (ME -3/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm31","Ratio (Ql+Qr)/Qt (ME -3/1)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm22","Ratio (Ql+Qr)/Qt (ME -2/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm21","Ratio (Ql+Qr)/Qt (ME -2/1)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm11a","Ratio (Ql+Qr)/Qt (ME -1/1a)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm13","Ratio (Ql+Qr)/Qt (ME -1/3)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm12","Ratio (Ql+Qr)/Qt (ME -1/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQm11b","Ratio (Ql+Qr)/Qt (ME -1/1b)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp11b","Ratio (Ql+Qr)/Qt (ME +1/1b)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp12","Ratio (Ql+Qr)/Qt (ME +1/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp13","Ratio (Ql+Qr)/Qt (ME +1/3)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp11a","Ratio (Ql+Qr)/Qt (ME +1/1a)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp21","Ratio (Ql+Qr)/Qt (ME +2/1)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp22","Ratio (Ql+Qr)/Qt (ME +2/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp31","Ratio (Ql+Qr)/Qt (ME +3/1)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp32","Ratio (Ql+Qr)/Qt (ME +3/2)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp41","Ratio (Ql+Qr)/Qt (ME +4/1)",120,-0.1,1.1));
      hRHRatioQ.push_back(dbe->book1D("hRHRatioQp42","Ratio (Ql+Qr)/Qt (ME +4/2)",120,-0.1,1.1));
 

      hRHTiming.push_back(dbe->book1D("hRHTimingm42","recHit Timing (ME -4/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm41","recHit Timing (ME -4/1)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm32","recHit Timing (ME -3/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm31","recHit Timing (ME -3/1)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm22","recHit Timing (ME -2/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm21","recHit Timing (ME -2/1)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm11a","recHit Timing (ME -1/1a)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm13","recHit Timing (ME -1/3)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm12","recHit Timing (ME -1/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingm11b","recHit Timing (ME -1/1b)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp11b","recHit Timing (ME +1/1b)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp12","recHit Timing (ME +1/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp13","recHit Timing (ME +1/3)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp11a","recHit Timing (ME +1/1a)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp21","recHit Timing (ME +2/1)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp22","recHit Timing (ME +2/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp31","recHit Timing (ME +3/1)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp32","recHit Timing (ME +3/2)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp41","recHit Timing (ME +4/1)",100,0,10));
      hRHTiming.push_back(dbe->book1D("hRHTimingp42","recHit Timing (ME +4/2)",100,0,10));

      hRHnrechits = dbe->book1D("hRHnrechits","recHits per Event (all chambers)",50,0,50);

      // segments
      dbe->setCurrentFolder("Muons/CSCOfflineMonitor/Segments");
      hSCodeBroad = dbe->book1D("hSCodeBroad","broad scope code for recHits",33,-16.5,16.5);
      hSCodeNarrow.push_back(dbe->book1D("hSCodeNarrow1","narrow scope Segment code station 1",801,-400.5,400.5));
      hSCodeNarrow.push_back(dbe->book1D("hSCodeNarrow2","narrow scope Segment code station 2",801,-400.5,400.5));
      hSCodeNarrow.push_back(dbe->book1D("hSCodeNarrow3","narrow scope Segment code station 3",801,-400.5,400.5));
      hSCodeNarrow.push_back(dbe->book1D("hSCodeNarrow4","narrow scope Segment code station 4",801,-400.5,400.5));

      hSnHits.push_back(dbe->book1D("hSnHitsm42","N hits on Segments ME -4/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm41","N hits on Segments ME -4/1",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm32","N hits on Segments ME -3/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm31","N hits on Segments ME -3/1",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm22","N hits on Segments ME -2/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm21","N hits on Segments ME -2/1",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm11a","N hits on Segments ME -1/1a",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm13","N hits on Segments ME -1/3",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm12","N hits on Segments ME -1/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsm11b","N hits on Segments ME -1/1b",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp11b","N hits on Segments ME +1/1b",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp12","N hits on Segments ME +1/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp13","N hits on Segments ME +1/3",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp11a","N hits on Segments ME +1/1a",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp21","N hits on Segments ME +2/1",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp22","N hits on Segments ME +2/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp31","N hits on Segments ME +3/1",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp32","N hits on Segments ME +3/2",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp41","N hits on Segments ME +4/1",7,-0.5,6.5));
      hSnHits.push_back(dbe->book1D("hSnHitsp42","N hits on Segments ME +4/2",7,-0.5,6.5));

      hSTheta.push_back(dbe->book1D("hSThetam42","local theta segments in ME -4/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam41","local theta segments in ME -4/1",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam32","local theta segments in ME -3/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam31","local theta segments in ME -3/1",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam22","local theta segments in ME -2/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam21","local theta segments in ME -2/1",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam11a","local theta segments ME -1/1a",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam13","local theta segments ME -1/3",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam12","local theta segments ME -1/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetam11b","local theta segments ME -1/1b",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap11b","local theta segments ME +1/1b",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap12","local theta segments ME +1/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap13","local theta segments ME +1/3",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap11a","local theta segments ME +1/1a",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap21","local theta segments in ME +2/1",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap22","local theta segments in ME +2/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap31","local theta segments in ME +3/1",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap32","local theta segments in ME +3/2",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap41","local theta segments in ME +4/1",128,-3.2,3.2));
      hSTheta.push_back(dbe->book1D("hSThetap42","local theta segments in ME +4/2",128,-3.2,3.2));

      hSGlobal.push_back(dbe->book2D("hSGlobalp1","segment global X,Y station +1",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalp2","segment global X,Y station +2",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalp3","segment global X,Y station +3",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalp4","segment global X,Y station +4",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalm1","segment global X,Y station -1",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalm2","segment global X,Y station -2",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalm3","segment global X,Y station -3",400,-800.,800.,400,-800.,800.));
      hSGlobal.push_back(dbe->book2D("hSGlobalm4","segment global X,Y station -4",400,-800.,800.,400,-800.,800.));

      hSnhitsAll = dbe->book1D("hSnhits","N hits on Segments",7,-0.5,6.5);
      hSChiSqProb = dbe->book1D("hSChiSqProb","segments chi-squared probability",100,0.,1.);
      hSGlobalTheta = dbe->book1D("hSGlobalTheta","segment global theta",64,0,1.6);
      hSGlobalPhi   = dbe->book1D("hSGlobalPhi",  "segment global phi",  128,-3.2,3.2);
      hSnSegments   = dbe->book1D("hSnSegments","number of segments per event",11,-0.5,10.5);

      hSResid.push_back(dbe->book1D("hSResidm42","Fitted Position on Strip - Reconstructed for Layer 3 (ME -4/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm41","Fitted Position on Strip - Reconstructed for Layer 3 (ME -4/1)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm32","Fitted Position on Strip - Reconstructed for Layer 3 (ME -3/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm31","Fitted Position on Strip - Reconstructed for Layer 3 (ME -3/1)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm22","Fitted Position on Strip - Reconstructed for Layer 3 (ME -2/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm21","Fitted Position on Strip - Reconstructed for Layer 3 (ME -2/1)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/1a)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm13","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/3)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm12","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidm11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME -1/1b)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp11b","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/1b)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp12","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp13","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/3)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp11a","Fitted Position on Strip - Reconstructed for Layer 3 (ME +1/1a)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp21","Fitted Position on Strip - Reconstructed for Layer 3 (ME +2/1)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp22","Fitted Position on Strip - Reconstructed for Layer 3 (ME +2/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp31","Fitted Position on Strip - Reconstructed for Layer 3 (ME +3/1)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp32","Fitted Position on Strip - Reconstructed for Layer 3 (ME +3/2)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp41","Fitted Position on Strip - Reconstructed for Layer 3 (ME +4/1)",100,-0.5,0.5));
      hSResid.push_back(dbe->book1D("hSResidp42","Fitted Position on Strip - Reconstructed for Layer 3 (ME +4/2)",100,-0.5,0.5));

      //occupancy plots
      hOWires = dbe->book2D("hOWires","Wire Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOStrips = dbe->book2D("hOStrips","Strip Digi Occupancy",36,0.5,36.5,20,0.5,20.5);
      hORecHits = dbe->book2D("hORecHits","RecHit Occupancy",36,0.5,36.5,20,0.5,20.5);
      hOSegments = dbe->book2D("hOSegments","Segment Occupancy",36,0.5,36.5,20,0.5,20.5);



}

//////////////////
//  DESTRUCTOR  //
//////////////////
CSCOfflineMonitor::~CSCOfflineMonitor(){

}

void CSCOfflineMonitor::endJob(void) {
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
  
  // Variables for occupancy plots
  bool wireo[2][4][4][36];
  bool stripo[2][4][4][36];
  bool rechito[2][4][4][36];
  bool segmento[2][4][4][36];

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

  // ==============================================
  //
  // look at DIGIs
  //
  // ==============================================


  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  event.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
  event.getByLabel("muonCSCDigis","MuonCSCStripDigi",strips);

  //
  // WIRE GROUPS
  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int index    = typeIndex(id);
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myWire = digiItr->getWireGroup();
      int myTBin = digiItr->getTimeBin();
      nWireGroupsTotal++;
      int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
      int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;

      // fill wire histos
      hWireTBinAll->Fill(myTBin);
      hWireAll->Fill(myWire);
      hWireCodeBroad->Fill(kCodeBroad);
      hWireCodeNarrow[kStation-1]->Fill(kCodeNarrow);
      hWireLayer[index]->Fill(kLayer);
      hWireWire[index]->Fill(myWire);

      wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;      
    }
  } // end wire loop
  

  //
  // STRIPS
  //
  int nStripsFired = 0;
  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int index    = typeIndex(id);
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
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
        int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
        int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;
        // fill strip histos
        hStripAll->Fill(myStrip);
        hStripCodeBroad->Fill(kCodeBroad);
        hStripCodeNarrow[kStation-1]->Fill(kCodeNarrow);
        hStripLayer[index]->Fill(kLayer);
        hStripStrip[index]->Fill(myStrip);

        stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      }
    }
  } // end strip loop

  //=======================================================
  //
  // Look at the Pedestal Noise Distributions
  //
  //=======================================================

  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int index    = typeIndex(id);
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    //int kChamber = id.chamber();
    //int kLayer   = id.layer();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
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
      //int globalStrip = cEndcap*( kStation*1000000 + kRing*100000 + kChamber*1000 + kLayer*100 + myStrip);
      if (TotalADC > threshold) { thisStripFired = true;}
      if (!thisStripFired){
	float ADC = thisSignal - thisPedestal;
	hStripPedAll->Fill(ADC);
	//hPedvsStrip->Fill(globalStrip,ADC);
        hStripPed[index]->Fill(ADC);
      }
    }
  }




  // ==============================================
  //
  // look at RECHITs
  //
  // ==============================================

  // Get the CSC Geometry :
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);
  
  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits; 
  event.getByLabel("csc2DRecHits",recHits);  
  int nRecHits = recHits->size();

 
  // ---------------------
  // Loop over rechits 
  // ---------------------
  int iHit = 0;

  // Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    iHit++;

    // Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int index    = typeIndex(idrec);
    int kEndcap  = idrec.endcap();
    int cEndcap  = idrec.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;

    // Store rechit as a Local Point:
    LocalPoint rhitlocal = (*recIt).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();

    // Find the strip containing this hit
    CSCRecHit2D::ChannelContainer hitstrips = (*recIt).channels();
    int nStrips     =  hitstrips.size();
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];

    // Find the charge associated with this hit

    CSCRecHit2D::ADCContainer adcs = (*recIt).adcs();
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
    //float rHtime = (*recIt).tpeak();
    float rHtime = getTiming(*strips, idrec, centerStrip);

    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx   =  rhitglobal.x();
    float grecy   =  rhitglobal.y();

    
    // Simple occupancy variables
    int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;

    // Fill some histograms
    hRHCodeBroad->Fill(kCodeBroad);
    hRHCodeNarrow[kStation-1]->Fill(kCodeNarrow);
    hRHGlobal[(kStation-1) + ((kEndcap - 1)*4)]->Fill(grecx,grecy);
    hRHLayer[index]->Fill(kLayer);
    hRHX[index]->Fill(xreco);
    hRHY[index]->Fill(yreco);
    hRHSumQ[index]->Fill(rHSumQ);
    hRHRatioQ[index]->Fill(rHratioQ);
    hRHTiming[index]->Fill(rHtime);

  } //end rechit loop

  // ==============================================
  //
  // look at SEGMENTs
  //
  // ===============================================

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);
  int nSegments = cscSegments->size();

  // -----------------------
  // loop over segments
  // -----------------------
  int iSegment = 0;
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    iSegment++;
    //
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int index    = typeIndex(id);
    int kEndcap  = id.endcap();
    int cEndcap  = id.endcap();
    if (kEndcap == 2) cEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;


    //
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();
    int nDOF       = 2*nhits-4;
    double chisqProb = ChiSquaredProbability( (double)chisq, nDOF );
    LocalPoint localPos = (*it).localPosition();
    LocalVector segDir = (*it).localDirection();
    double theta   = segDir.theta();

    //
    // try to get the CSC recHits that contribute to this segment.
    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    int nRH = (*it).nRecHits();
    int jRH = 0;
    HepMatrix sp(6,1);
    HepMatrix se(6,1);
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
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

    // global transformation
    float globX = 0.;
    float globY = 0.;
    float globZ = 0.;
    float globpPhi = 0.;
    float globR = 0.;
    float globTheta = 0.;
    float globPhi   = 0.;
    const CSCChamber* cscchamber = cscGeom->chamber(id);
    if (cscchamber) {
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globX = globalPosition.x();
      globY = globalPosition.y();
      globZ = globalPosition.z();
      globpPhi =  globalPosition.phi();
      globR   =  sqrt(globX*globX + globY*globY);
      GlobalVector globalDirection = cscchamber->toGlobal(segDir);
      globTheta = globalDirection.theta();
      globPhi   = globalDirection.phi();
    }

    // Simple occupancy variables
    int kCodeBroad  = cEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = cEndcap * ( 100*(kRing-1) + kChamber) ;


    // Fill histos
    hSCodeBroad->Fill(kCodeBroad);
    hSChiSqProb->Fill(chisqProb);
    hSGlobalTheta->Fill(globTheta);
    hSGlobalPhi->Fill(globPhi);
    hSnhitsAll->Fill(nhits);
    hSCodeNarrow[kStation-1]->Fill(kCodeNarrow);
    hSGlobal[(kStation-1) + ((kEndcap - 1)*4)]->Fill(globX,globY);
    hSnHits[index]->Fill(nhits);
    hSTheta[index]->Fill(theta);
    hSResid[index]->Fill(residual);

  } // end segment loop

  // Fill # per even histos (how many stips/wires/rechits/segments per event)
  hWirenGroupsTotal->Fill(nWireGroupsTotal);
  hStripNFired->Fill(nStripsFired);
  hSnSegments->Fill(nSegments);
  hRHnrechits->Fill(nRecHits);

  // Fill occupancy plots
  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          int type = 0;
          if ((s+1) == 1) type = (r+1);
          else type = (s+1)*2 + (r+1);
          if ((e+1) == 1) type = type + 10;
          if ((e+1) == 2) type = 11 - type;
          //int bin = hOWires->GetBin(chamber,type);
          //hOWires->AddBinContent(bin);
          if (wireo[e][s][r][c]) hOWires->Fill((c+1),type);
          if (stripo[e][s][r][c]) hOStrips->Fill((c+1),type);
          if (rechito[e][s][r][c]) hORecHits->Fill((c+1),type);
          if (segmento[e][s][r][c]) hOSegments->Fill((c+1),type);
        }
      }
    }
  }


}

//-------------------------------------------------------------------------------------
// Fits a straight line to a set of 5 points with errors.  Functions assumes 6 points
// and removes hit in layer 3.  It then returns the expected position value in layer 3
// based on the fit.
//-------------------------------------------------------------------------------------
float CSCOfflineMonitor::fitX(HepMatrix points, HepMatrix errors){

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

  float chi = 0;
  float chi2 = 0;

  // calculate chi2 (not currently used)
  for (int i=1;i<7;i++){
    chi = (points(i,1) - intercept - slope*i)/(errors(i,1));
    chi2 = chi2 + chi*chi;
  }

  return (intercept + slope*3);

}

//---------------------------------------------------------------------------------------
// Find the signal timing based on a weighted mean of the pulse.
// Function is meant to take the DetId and center strip number of a recHit and return
// the timing in units of time buckets (50ns)
//---------------------------------------------------------------------------------------

float CSCOfflineMonitor::getTiming(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){

  float ADC[8];
  float timing = 0;

  // Loop over strip digis responsible for this recHit and sum charge
  CSCStripDigiCollection::DigiRangeIterator sIt;

  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idRH){
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
          float diff = 0;
          vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
          for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
            diff = (float)myADCVals[iCount]-thisPedestal;
            ADC[iCount] = diff;
          }
        }
      }

    }

  }

  timing = (ADC[2]*2 + ADC[3]*3 + ADC[4]*4 + ADC[5]*5 + ADC[6]*6)/(ADC[2] + ADC[3] + ADC[4] + ADC[5] + ADC[6]);
  return timing;


}


//---------------------------------------------------------------------------------------
// Given a set of digis, the CSCDetId, and the central strip of your choosing, returns
// the avg. Signal-Pedestal for 6 time bin x 5 strip .
//
// Author: P. Jindal
//---------------------------------------------------------------------------------------

float CSCOfflineMonitor::getSignal(const CSCStripDigiCollection&
 stripdigis, CSCDetId idCS, int centerStrip){

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

int CSCOfflineMonitor::typeIndex(CSCDetId id){

  // linearlized index bases on endcap, station, and ring
  int index = 0;
  if (id.station() == 1) index = id.ring();
  else index = id.station()*2 + id.ring();
  if (id.endcap() == 1) index = index + 9;
  if (id.endcap() == 2) index = 10 - index;
  return index;

}


DEFINE_FWK_MODULE(CSCOfflineMonitor);

