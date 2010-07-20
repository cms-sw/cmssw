#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"

#include <algorithm>
#include <iostream>

namespace spr{

  std::vector<DetId> matrixECALIds(const DetId& det,int ieta,int iphi, 
				   const CaloGeometry* geo, 
				   const CaloTopology* caloTopology,
				   bool debug) {


    const CaloSubdetectorTopology *barrelTopo = (caloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel));
    const CaloSubdetectorTopology *endcapTopo = (caloTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap));
    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));

    if (debug) {
      std::cout << "matrixECALIds::Add " << ieta << " rows and " << iphi 
		<< " columns of cells for 1 cell" << std::endl;
      if (det.subdetId() == EcalBarrel) {
	EBDetId id = det;
	std::cout << "matrixECALIds::Cell 0x" << std::hex << det() << std::dec 
		  << " " << id << std::endl;
      } else if (det.subdetId() == EcalEndcap) {
	EEDetId id = det;
	std::cout << "matrixECALIds::Cell 0x" << std::hex << det() << std::dec
		  << " " << id << std::endl;
      } else {
	std::cout << "matrixECALIds::Cell 0x" << std::hex << det() << std::dec
		  << " Unknown Type" << std::endl;
      }
    }

    std::vector<DetId> dets(1,det);
    std::vector<CaloDirection> dirs(1,NORTH);
    std::vector<DetId> vdetN = spr::newECALIdNS(dets, 0, ieta, iphi, dirs,
						*barrelTopo,*endcapTopo, 
						*barrelGeom,*endcapGeom,debug);
    dirs[0] = SOUTH;
    std::vector<DetId> vdetS = spr::newECALIdNS(dets, 0, ieta, iphi, dirs,
						*barrelTopo,*endcapTopo, 
						*barrelGeom,*endcapGeom,debug);
    for (unsigned int i1=0; i1<vdetS.size(); i1++) {
      if (std::count(vdetN.begin(),vdetN.end(),vdetS[i1]) == 0)
	vdetN.push_back(vdetS[i1]);
    }
    unsigned int ndet = (2*ieta+1)*(2*iphi+1);
    if (vdetN.size() != ndet) {
      vdetS = spr::extraIds(det, vdetN, ieta, ieta, iphi, iphi, 
			    *barrelGeom, *endcapGeom, debug);
      if (vdetS.size() > 0) 
	vdetN.insert(vdetN.end(), vdetS.begin(), vdetS.end());
    }

    if (debug) {
      std::cout << "matrixECALIds::Total number of cells found is " 
		<< vdetN.size() << std::endl;
      for (unsigned int i1=0; i1<vdetN.size(); i1++) {
	if (vdetN[i1].subdetId() == EcalBarrel) {
	  EBDetId id = vdetN[i1];
	  std::cout << "matrixECALIds::Cell " << i1 << " 0x" << std::hex 
		    << vdetN[i1]() << std::dec << " " << id << std::endl;
	} else if (vdetN[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdetN[i1];
	  std::cout << "matrixECALIds::Cell " << i1 << " 0x" << std::hex 
		    << vdetN[i1]() << std::dec << " " << id << std::endl;
	} else {
	  std::cout << "matrixECALIds::Cell " << i1 << " 0x" << std::hex 
		    << vdetN[i1]() << std::dec << " Unknown Type" << std::endl;
	}
      }
    }
    return vdetN;
  }

  std::vector<DetId> matrixECALIds(const DetId& det, int ietaE, int ietaW,
				   int iphiN, int iphiS,
				   const CaloGeometry* geo, 
				   const CaloTopology* caloTopology,
				   bool debug) {

    const CaloSubdetectorTopology *barrelTopo = (caloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel));
    const CaloSubdetectorTopology *endcapTopo = (caloTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap));
    const EcalBarrelGeometry *barrelGeom = (dynamic_cast< const EcalBarrelGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalBarrel)));
    const EcalEndcapGeometry *endcapGeom = (dynamic_cast< const EcalEndcapGeometry *> (geo->getSubdetectorGeometry(DetId::Ecal,EcalEndcap)));

    if (debug) {
      std::cout << "matrixECALIds::Add " << ietaE << "|" << ietaW
		<< " rows and " << iphiN << "|" << iphiS
		<< " columns of cells for 1 cell" << std::endl;
      if (det.subdetId() == EcalBarrel) {
	EBDetId id = det;
	std::cout << "matrixECALIds::Cell 0x" << std::hex << det() << std::dec 
		  << " " << id << std::endl;
      } else if (det.subdetId() == EcalEndcap) {
	EEDetId id = det;
	std::cout << "matrixECALIds::Cell 0x" << std::hex << det() << std::dec
		  << " " << id << std::endl;
      } else {
	std::cout << "matrixECALIds::Cell 0x" << std::hex << det() << std::dec
		  << " Unknown Type" << std::endl;
      }
    }

    std::vector<DetId> dets(1,det);
    std::vector<CaloDirection> dirs(1,NORTH);
    std::vector<int> jetaE(1,ietaE), jetaW(1,ietaW);
    std::vector<int> jphiN(1,iphiN), jphiS(1,iphiS);
    std::vector<DetId> vdetN = spr::newECALIdNS(dets, 0, jetaE, jetaW, jphiN,
						jphiS, dirs, *barrelTopo,
						*endcapTopo, *barrelGeom,
						*endcapGeom, debug);
    dirs[0] = SOUTH;
    std::vector<DetId> vdetS = spr::newECALIdNS(dets, 0, jetaE, jetaW, jphiN,
						jphiS, dirs, *barrelTopo,
						*endcapTopo, *barrelGeom,
						*endcapGeom, debug);
    for (unsigned int i1=0; i1<vdetS.size(); i1++) {
      if (std::count(vdetN.begin(),vdetN.end(),vdetS[i1]) == 0)
	vdetN.push_back(vdetS[i1]);
    }

    unsigned int ndet = (ietaE+ietaW+1)*(iphiN+iphiS+1);
    if (vdetN.size() != ndet) {
      vdetS = spr::extraIds(det, vdetN, ietaE, ietaW, iphiN, iphiS, 
			    *barrelGeom, *endcapGeom, debug);
      if (vdetS.size() > 0) 
	vdetN.insert(vdetN.end(), vdetS.begin(), vdetS.end());
    }

    if (debug) {
      std::cout << "matrixECALIds::Total number of cells found is " 
		<< vdetN.size() << std::endl;
      for (unsigned int i1=0; i1<vdetN.size(); i1++) {
	if (vdetN[i1].subdetId() == EcalBarrel) {
	  EBDetId id = vdetN[i1];
	  std::cout << "matrixECALIds::Cell " << i1 << " 0x" << std::hex 
		    << vdetN[i1]() << std::dec << " " << id << std::endl;
	} else if (vdetN[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdetN[i1];
	  std::cout << "matrixECALIds::Cell " << i1 << " 0x" << std::hex 
		    << vdetN[i1]() << std::dec << " " << id << std::endl;
	} else {
	  std::cout << "matrixECALIds::Cell " << i1 << " 0x" << std::hex 
		    << vdetN[i1]() << std::dec << " Unknown Type" << std::endl;
	}
      }
    }
    return vdetN;
  }

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last,
				 int ieta, int iphi, 
				 std::vector<CaloDirection>& dir,
				 const CaloSubdetectorTopology& barrelTopo,
				 const CaloSubdetectorTopology& endcapTopo,
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdNS::Add " << iphi << " columns of cells for " 
		<< (dets.size()-last) << " cells (last " << last << ")"
		<< std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " "  << id << " along "
		    << dir[i1] << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " " << id << " along "
		    << dir[i1] << std::endl;
	} else {
	  std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << std::dec << " Unknown Type along " 
		    << dir[i1] << std::endl;
	}
      }
    }
    
    std::vector<DetId> vdets;
    std::vector<CaloDirection> dirs;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    dirs.insert(dirs.end(), dir.begin(), dir.end());

    std::vector<DetId> vdetE, vdetW;
    if (last == 0) {
      unsigned int ndet = vdets.size();
      std::vector<CaloDirection> dirE(ndet,EAST), dirW(ndet,WEST);
      vdetE = spr::newECALIdEW(dets, last, ieta, dirE, barrelTopo, endcapTopo, 
			       barrelGeom, endcapGeom, debug);
      vdetW = spr::newECALIdEW(dets, last, ieta, dirW, barrelTopo, endcapTopo,
			       barrelGeom, endcapGeom, debug);
      for (unsigned int i1=0; i1<vdetW.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i1]) == 0) {
	  vdets.push_back(vdetW[i1]);
	  dirs.push_back(dir[0]);
	}
      }
      for (unsigned int i1=0; i1<vdetE.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i1]) == 0) {
	  vdets.push_back(vdetE[i1]);
	  dirs.push_back(dir[0]);
	}
      }
      if (debug) {
	std::cout <<"newECALIdNS::With Added cells along E/W results a set of "
		  << (vdets.size()-dets.size()) << " new  cells" << std::endl;
	for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << " along "
		      << dirs[i1] << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << " along "
		      << dirs[i1] << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << std::dec << " Unknown Type along " 
		      << dirs[i1] << std::endl;
	  }
	}
      }
    }

    unsigned int last0 = vdets.size();
    std::vector<DetId> vdetnew;
    std::vector<CaloDirection> dirnew;
    if (iphi > 0) {
      std::vector<DetId> vdetn(1);
      std::vector<CaloDirection> dirn(1);
      std::vector<CaloDirection> dirnE(1,EAST), dirnW(1,WEST);
      int flag=0;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::vector<DetId> cells = spr::simpleMove(dets[i1], dir[i1], 
						   barrelTopo, endcapTopo,
						   barrelGeom, endcapGeom,
						   flag, debug);
	/*
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "Moved Cell " << i1 << " "  << id << " along "
		    << dir[i1] << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "Moved Cell " << i1 << " " << id << " along "
		    << dir[i1] << std::endl;
	} else {
	  std::cout << "Moved Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << std::dec << " Unknown Type along " 
		    << dir[i1] << std::endl;
	}
	for (unsigned int kk=0; kk<cells.size(); kk++) {
	  if (cells[kk].subdetId() == EcalBarrel) {
	    EBDetId id = cells[kk];
	    std::cout << "Moved to " << id << " flag " << flag << "\n";
	  } else if (cells[kk].subdetId() == EcalEndcap) {
	    EEDetId id = cells[kk];
	    std::cout << "Moved to " << id << " flag " << flag << "\n";
	  } else {
	    std::cout << "Moved Cell " << i1 << " 0x" << std::hex 
		      << cells[kk]() << std::dec << " Unknown Type flag " 
		      << flag << std::endl;
	  }
	}
	*/
	if (flag != 0) {
	  if (std::count(vdets.begin(),vdets.end(),cells[0]) == 0) {
	    vdetn[0] = cells[0];
	    vdetnew.push_back(vdetn[0]);
	    dirn[0] = dir[i1];
	    if (flag < 0) {
	      if (dirn[0] == NORTH) dirn[0] = SOUTH;
	      else                  dirn[0] = NORTH;
	    }
	    dirnew.push_back(dirn[0]);
	    vdetE = spr::newECALIdEW(vdetn, 0, ieta, dirnE, barrelTopo, 
				     endcapTopo, barrelGeom, endcapGeom,debug);
	    vdetW = spr::newECALIdEW(vdetn, 0, ieta, dirnW, barrelTopo, 
				     endcapTopo, barrelGeom, endcapGeom,debug);
	    for (unsigned int i2=0; i2<vdetW.size(); i2++) {
	      if (std::count(vdets.begin(),vdets.end(),vdetW[i2]) == 0 &&
		  std::count(vdetnew.begin(),vdetnew.end(),vdetW[i2]) == 0) {
		vdets.push_back(vdetW[i2]);
		dirs.push_back(dirn[0]);
	      }
	    }
	    for (unsigned int i2=0; i2<vdetE.size(); i2++) {
	      if (std::count(vdets.begin(),vdets.end(),vdetE[i2]) == 0 &&
		  std::count(vdetnew.begin(),vdetnew.end(),vdetE[i2]) == 0) {
		vdets.push_back(vdetE[i2]);
		dirs.push_back(dirn[0]);
	      }
	    }
	  }
	}
      }
      iphi--;
      last = vdets.size();
      for (unsigned int i2=0; i2<vdetnew.size(); i2++) {
	if (std::count(vdets.begin(),vdets.end(),vdetnew[i2]) == 0) {
	  vdets.push_back(vdetnew[i2]);
	  dirs.push_back(dirnew[i2]);
	}
      }
      if (debug) {
	std::cout << "newECALIdNS::Addition results a set of " 
		  << (vdets.size()-last0)  << " new  cells (last " << last0
		  << ", iphi " << iphi << ")" << std::endl;
	for (unsigned int i1=last0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << " along "
		      << dirs[i1] << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << " along "
		      << dirs[i1] << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << std::dec << " Unknown Type along " 
		      << dirs[i1] << std::endl;
	  }
	}
      }
      last0 = last;
    }

    if (iphi > 0) {
      last = last0;
      return spr::newECALIdNS(vdets,last,ieta,iphi,dirs,barrelTopo,endcapTopo,barrelGeom,endcapGeom,debug);
    } else {
      if (debug) {
	std::cout << "newECALIdNS::Final list consists of " << vdets.size()
		  << " cells" << std::endl;
	for (unsigned int i1=0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << std::dec << " Unknown Type" 
		      << std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last,
				 std::vector<int>& ietaE, 
				 std::vector<int>& ietaW, 
				 std::vector<int>& iphiN, 
				 std::vector<int>& iphiS,
				 std::vector<CaloDirection>& dir,
				 const CaloSubdetectorTopology& barrelTopo,
				 const CaloSubdetectorTopology& endcapTopo,
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdNS::Add columns of cells for " 
		<< (dets.size()-last) << " cells (last) " << last << std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " "  << id << " along "
		    << dir[i1] << " # " << iphiN[i1] << "|" << iphiS[i1]
		    << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " " << id << " along "
		    << dir[i1] << " # " << iphiN[i1] << "|" << iphiS[i1]
		    << std::endl;
	} else {
	  std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << std::dec << " Unknown Type along " 
		    << dir[i1] << " # " << iphiN[i1] << "|" << iphiS[i1] 
		    << std::endl;
	}
      }
    }

    std::vector<DetId> vdets;
    std::vector<CaloDirection> dirs;
    std::vector<int> jetaE, jetaW, jphiN, jphiS;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    dirs.insert(dirs.end(), dir.begin(), dir.end());
    jetaE.insert(jetaE.end(), ietaE.begin(), ietaE.end());
    jetaW.insert(jetaW.end(), ietaW.begin(), ietaW.end());
    jphiN.insert(jphiN.end(), iphiN.begin(), iphiN.end());
    jphiS.insert(jphiS.end(), iphiS.begin(), iphiS.end());
    std::vector<DetId> vdetE, vdetW;
    if (last == 0) {
      unsigned int ndet = vdets.size();
      std::vector<CaloDirection> dirE(ndet,EAST), dirW(ndet,WEST);
      vdetE = spr::newECALIdEW(dets, last, ietaE, ietaW, dirE, barrelTopo, 
			       endcapTopo, barrelGeom, endcapGeom, debug);
      vdetW = spr::newECALIdEW(dets, last, ietaE, ietaW, dirW, barrelTopo,
			       endcapTopo, barrelGeom, endcapGeom, debug);
      for (unsigned int i1=0; i1<vdetW.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i1]) == 0) {
	  vdets.push_back(vdetW[i1]);
	  dirs.push_back(dir[0]);
	  jetaE.push_back(0);
	  jetaW.push_back(0);
	  jphiN.push_back(iphiN[0]);
	  jphiS.push_back(iphiS[0]);
	}
      }
      for (unsigned int i1=0; i1<vdetE.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i1]) == 0) {
	  vdets.push_back(vdetE[i1]);
	  dirs.push_back(dir[0]);
	  jetaE.push_back(0);
	  jetaW.push_back(0);
	  jphiN.push_back(iphiN[0]);
	  jphiS.push_back(iphiS[0]);
	}
      }
      if (debug) {
	std::cout <<"newECALIdNS::With Added cells along E/W results a set of "
		  << (vdets.size()-dets.size()) << " new  cells" << std::endl;
	for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << " along "
		      << dirs[i1] << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << " along "
		      << dirs[i1] << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << std::dec << " Unknown Type along " 
		      << dirs[i1] << std::endl;
	  }
	}
      }
    }

    unsigned int last0 = vdets.size();
    std::vector<DetId> vdetnew;
    std::vector<CaloDirection> dirnew;
    std::vector<int> kphiN, kphiS, ketaE, ketaW;
    int kphi = 0;
    for (unsigned int i1=last; i1<dets.size(); i1++) {
      int iphi = iphiS[i1];
      if (dir[i1] == NORTH) iphi = iphiN[i1];
      if (iphi > 0) {
	std::vector<DetId>         vdetn(1);
	std::vector<CaloDirection> dirn(1);
	std::vector<CaloDirection> dirnE(1,EAST), dirnW(1,WEST);
	int flag=0;
	std::vector<DetId> cells = spr::simpleMove(dets[i1], dir[i1], 
						   barrelTopo, endcapTopo,
						   barrelGeom, endcapGeom,
						   flag, debug);
	iphi--;
	if (iphi > kphi) kphi = iphi;
	if (dir[i1] == NORTH) jphiN[i1] = iphi;
	else                  jphiS[i1] = iphi;
	if (flag != 0) {
	  if (std::count(vdets.begin(),vdets.end(),cells[0]) == 0) {
	    int kfiN = iphiN[i1];
	    int kfiS = iphiS[i1];
	    vdetn[0] = cells[0];
	    vdetnew.push_back(vdetn[0]);
	    dirn[0] = dir[i1];
	    if (dir[i1] == NORTH) kfiN = iphi;
	    else                  kfiS = iphi;
	    if (flag < 0) {
	      int ktmp = kfiS; kfiS = kfiN; kfiN = ktmp;
	      if (dirn[0] == NORTH) dirn[0] = SOUTH;
	      else                  dirn[0] = NORTH;
	    }
	    dirnew.push_back(dirn[0]);
	    kphiN.push_back(kfiN); ketaE.push_back(ietaE[i1]);
	    kphiS.push_back(kfiS); ketaW.push_back(ietaW[i1]);
	    std::vector<int>       ietE(1,ietaE[i1]), ietW(1,ietaW[i1]);
	    vdetE = spr::newECALIdEW(vdetn, 0, ietE, ietW, dirnE, barrelTopo,
				     endcapTopo, barrelGeom, endcapGeom,debug);
	    vdetW = spr::newECALIdEW(vdetn, 0, ietE, ietW, dirnW, barrelTopo,
				     endcapTopo, barrelGeom, endcapGeom,debug);
	    for (unsigned int i2=0; i2<vdetW.size(); i2++) {
	      if (std::count(vdets.begin(),vdets.end(),vdetW[i2]) == 0 &&
		  std::count(vdetnew.begin(),vdetnew.end(),vdetW[i2]) == 0) {
		vdets.push_back(vdetW[i2]);
		dirs.push_back(dirn[0]);
		jetaE.push_back(0); jphiN.push_back(kfiN);
		jetaW.push_back(0); jphiS.push_back(kfiS);
	      }
	    }
	    for (unsigned int i2=0; i2<vdetE.size(); i2++) {
	      if (std::count(vdets.begin(),vdets.end(),vdetE[i2]) == 0 &&
		  std::count(vdetnew.begin(),vdetnew.end(),vdetE[i2]) == 0) {
		vdets.push_back(vdetE[i2]);
		dirs.push_back(dirn[0]);
		jetaE.push_back(0); jphiN.push_back(kfiN);
		jetaW.push_back(0); jphiS.push_back(kfiS);
	      }
	    }
	  } 
	}
      }
    }
    last = vdets.size();
    for (unsigned int i2=0; i2<vdetnew.size(); i2++) {
      if (std::count(vdets.begin(),vdets.end(),vdetnew[i2]) == 0) {
	vdets.push_back(vdetnew[i2]);
	dirs.push_back(dirnew[i2]);
	jetaE.push_back(ketaE[i2]);
	jetaW.push_back(ketaW[i2]);
	jphiN.push_back(kphiN[i2]);
	jphiS.push_back(kphiS[i2]);
      }
    }
    if (debug) {
      std::cout << "newECALIdNS::Addition results a set of " 
		<< (vdets.size()-last0)  << " new  cells (last " << last0
		<< ", iphi " << kphi << ")" << std::endl;
      for (unsigned int i1=last0; i1<vdets.size(); i1++) {
	if (vdets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = vdets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " "  << id << " along "
		    << dirs[i1] << " iphi " << jphiN[i1] << "|" << jphiS[i1]
		    << " ieta " << jetaE[i1] << "|" << jetaW[i1] << std::endl;
	} else if (vdets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " " << id << " along "
		    << dirs[i1] << " iphi " << jphiN[i1] << "|" << jphiS[i1]
		    << " ieta " << jetaE[i1] << "|" << jetaW[i1] << std::endl;
	} else {
	  std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		    << vdets[i1]() << std::dec << " Unknown Type along "
		    << dirs[i1] << " iphi " << jphiN[i1] << "|" << jphiS[i1]
		    << " ieta " << jetaE[i1] << "|" << jetaW[i1] << std::endl;
	}
      }
    }
    last0 = last;
      
    if (kphi > 0) {
      last = last0;
      return spr::newECALIdNS(vdets, last, jetaE, jetaW, jphiN, jphiS, dirs,
			      barrelTopo, endcapTopo, barrelGeom, endcapGeom,
			      debug);
    } else {
      if (debug) {
	std::cout << "newECALIdNS::Final list consists of " << vdets.size()
		  << " cells" << std::endl;
	for (unsigned int i1=0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << std::dec << " Unknown Type" 
		      << std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last,
				 int ieta, std::vector<CaloDirection>& dir, 
				 const CaloSubdetectorTopology& barrelTopo, 
				 const CaloSubdetectorTopology& endcapTopo, 
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdEW::Add " << ieta << " rows of cells for " 
		<< (dets.size()-last) << " cells" << std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << " along "
		    << dir[i1] << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << " along "
		    << dir[i1] << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << std::dec << " Unknown Type along " 
		    << dir[i1] << std::endl;
	}
      }
    }

    std::vector<DetId> vdets; vdets.clear();
    std::vector<CaloDirection> dirs; dirs.clear();
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    dirs.insert(dirs.end(), dir.begin(), dir.end());

    if (ieta > 0) {
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	int flag = 0;
	std::vector<DetId> cells = spr::simpleMove(dets[i1], dir[i1], 
						   barrelTopo, endcapTopo,
						   barrelGeom, endcapGeom,
						   flag, debug);
	/*
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "Moved Cell " << i1 << " "  << id << " along "
		    << dir[i1] << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "Moved Cell " << i1 << " " << id << " along "
		    << dir[i1] << std::endl;
	} else {
	  std::cout << "Moved Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << std::dec << " Unknown Type along " 
		    << dir[i1] << std::endl;
	}
	for (unsigned int kk=0; kk<cells.size(); kk++) {
	  if (cells[kk].subdetId() == EcalBarrel) {
	    EBDetId id = cells[kk];
	    std::cout << "Moved to " << id << " flag " << flag << "\n";
	  } else if (cells[kk].subdetId() == EcalEndcap) {
	    EEDetId id = cells[kk];
	    std::cout << "Moved to " << id << " flag " << flag << "\n";
	  } else {
	    std::cout << "Moved Cell " << i1 << " 0x" << std::hex 
		      << cells[kk]() << std::dec << " Unknown Type flag " 
		      << flag << std::endl;
	  }
	}
	*/
	if (flag != 0) {
	  if (std::count(vdets.begin(),vdets.end(),cells[0]) == 0) {
	    CaloDirection dirn = dir[i1];
	    if (flag < 0) {
	      if (dirn == EAST) dirn = WEST;
	      else              dirn = EAST;
	    }
	    vdets.push_back(cells[0]);
	    dirs.push_back(dirn);
	  }
	}
      }
      ieta--;
    }
    
    if (debug) {
      std::cout << "newECALIdEW::Addition results a set of " 
		<< (vdets.size()-dets.size()) << " new  cells" << std::endl;
      for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	if (vdets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << " along " 
		    << dirs[i1] << std::endl;
	} else if (vdets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << " along " 
		    << dirs[i1] << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << vdets[i1]() << std::dec << " Unknown Type along " 
		    << dirs[i1] << std::endl;
	}
      }
    }

    if (ieta > 0) {
      last = dets.size();
      return spr::newECALIdEW(vdets,last,ieta,dirs,barrelTopo,endcapTopo,barrelGeom,endcapGeom,debug);
    } else {
      if (debug) {
	std::cout << "newECALIdEW::Final list (EW) consists of " <<vdets.size()
		  << " cells" << std::endl;
	for (unsigned int i1=0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() <<std::dec << " Unknown Type" <<std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last,
				 std::vector<int>& ietaE, 
				 std::vector<int>& ietaW,
				 std::vector<CaloDirection>& dir, 
				 const CaloSubdetectorTopology& barrelTopo, 
				 const CaloSubdetectorTopology& endcapTopo, 
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdEW::Add " << ietaE[0] << "|" << ietaW[0]
		<< " rows of cells for " << (dets.size()-last) 
		<< " cells (last " << last << ")" << std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << " along "
		    << dir[i1] << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << " along "
		    << dir[i1] << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << std::dec << " Unknown Type along " 
		    << dir[i1] << std::endl;
	}
      }
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    std::vector<CaloDirection> dirs;
    dirs.insert(dirs.end(), dir.begin(), dir.end());
    std::vector<int> jetaE, jetaW;
    jetaE.insert(jetaE.end(), ietaE.begin(), ietaE.end());
    jetaW.insert(jetaW.end(), ietaW.begin(), ietaW.end());
    int keta = 0;
    for (unsigned int i1=last; i1<dets.size(); i1++) {
      int ieta = ietaW[i1];
      if (dir[i1] == EAST) ieta = ietaE[i1]; 
      if (ieta > 0) {
	int flag=0;
	std::vector<DetId> cells = spr::simpleMove(dets[i1], dir[i1], 
						   barrelTopo, endcapTopo,
						   barrelGeom, endcapGeom,
						   flag, debug);
	ieta--;
	if (ieta > keta) keta = ieta;
	if (dir[i1] == EAST) jetaE[i1] = ieta;
	else                 jetaW[i1] = ieta;
	if (flag != 0) {
	  if (std::count(vdets.begin(),vdets.end(),cells[0]) == 0) {
	    vdets.push_back(cells[0]);
	    CaloDirection dirn = dir[i1];
	    int ketaE = ietaE[i1];
	    int ketaW = ietaW[i1];
	    if (dirn == EAST) ketaE = ieta;
	    else              ketaW = ieta;
	    if (flag < 0) {
	      int ktmp = ketaW; ketaW    = ketaE; ketaE    = ktmp;
	      if (dirn == EAST) dirn = WEST;
	      else              dirn = EAST;
	    }
	    dirs.push_back(dirn);
	    jetaE.push_back(ketaE);
	    jetaW.push_back(ketaW);
	  }
	}
      }
    }
    
    if (debug) {
      std::cout << "newECALIdEW::Addition results a set of " 
		<< (vdets.size()-dets.size()) << " new  cells (last " 
		<< dets.size() << ", ieta " << keta << ")" << std::endl;
      for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	if (vdets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	} else if (vdets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << vdets[i1]() << std::dec << " Unknown Type" << std::endl;
	}
      }
    }

    if (keta > 0) {
      last = dets.size();
      return spr::newECALIdEW(vdets, last, jetaE, jetaW, dirs, barrelTopo,
			      endcapTopo, barrelGeom, endcapGeom, debug);
    } else {
      if (debug) {
	std::cout << "newECALIdEW::Final list (EW) consists of " <<vdets.size()
		  << " cells" << std::endl;
	for (unsigned int i1=0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() <<std::dec << " Unknown Type" <<std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> simpleMove(DetId& det, const CaloDirection& dir, 
				const CaloSubdetectorTopology& barrelTopo, 
				const CaloSubdetectorTopology& endcapTopo, 
				const EcalBarrelGeometry& barrelGeom, 
				const EcalEndcapGeometry& endcapGeom, 
				int& ok, bool debug) {
    std::vector<DetId> cells;
    DetId cell;
    ok = 0;
    if (det.subdetId() == EcalBarrel) {
      EBDetId detId = det;
      std::vector<DetId> neighbours = barrelTopo.getNeighbours(detId,dir);
      if (neighbours.size()>0 && !neighbours[0].null()) {
	cells.push_back(neighbours[0]);
	cell = neighbours[0];
	ok   = 1;
      } else {
	const int ietaAbs ( detId.ietaAbs() ) ; // abs value of ieta
	if (EBDetId::MAX_IETA == ietaAbs ) {
	  // get ee nbrs for for end of barrel crystals
	  const EcalBarrelGeometry::OrderedListOfEEDetId&
	    ol( * barrelGeom.getClosestEndcapCells(detId) ) ;
	  // take closest neighbour on the other side, that is in the endcap
	  cell = *(ol.begin() );
	  neighbours = endcapTopo.getNeighbours(cell,dir);
	  if (neighbours.size()>0 && !neighbours[0].null()) ok = 1;
	  else                                              ok =-1;
	  for (EcalBarrelGeometry::OrderedListOfEEDetId::const_iterator iptr=ol.begin(); iptr != ol.end(); ++iptr)
	    cells.push_back(*iptr);
	}
      }
    } else if (det.subdetId() == EcalEndcap) {
      EEDetId detId = det;
      std::vector<DetId> neighbours = endcapTopo.getNeighbours(detId,dir);
      if (neighbours.size()>0 && !neighbours[0].null()) {
	cells.push_back(neighbours[0]);
	cell = neighbours[0];
	ok   = 1;
      } else {
	// are we on the outer ring ?
	const int iphi ( detId.iPhiOuterRing() ) ;
	if (iphi!= 0) {
	  // get eb nbrs for for end of endcap crystals
	  const EcalEndcapGeometry::OrderedListOfEBDetId&
	    ol( * endcapGeom.getClosestBarrelCells(detId) ) ;
	  // take closest neighbour on the other side, that is in the barrel.
	  cell = *(ol.begin() );
	  neighbours = barrelTopo.getNeighbours(cell,dir);
	  if (neighbours.size()>0 && !neighbours[0].null()) ok = 1;
	  else                                              ok =-1;
	  for (EcalEndcapGeometry::OrderedListOfEBDetId::const_iterator iptr=ol.begin(); iptr != ol.end(); ++iptr)
	    cells.push_back(*iptr);
	}
      }
    }  
    if (debug) {
      std::cout << "simpleMove:: Move DetId 0x" << std::hex << det() 
		<< std::dec << " along " << dir << " to get 0x" << std::hex
		<< cell() << std::dec << " with flag " << ok << " # "
		<< cells.size() << " " << std::hex << cells[0]() << std::dec
		<< std::endl;
    }
    return cells;
  }

  std::vector<DetId> extraIds(const DetId& det, std::vector<DetId>& dets, int ietaE, int ietaW, int iphiN, int iphiS, const EcalBarrelGeometry& barrelGeom, const EcalEndcapGeometry& endcapGeom, bool debug) {

    std::vector<DetId> cells;
    if (det.subdetId() == EcalBarrel) {
      EBDetId id = det;
      if (debug) std::cout << "extraIds::Cell " << id << " rows "  << ietaW
			   << "|" << ietaE << " columns " << iphiS << "|"
			   << iphiN << std::endl;
      int etaC = id.ietaAbs();
      int phiC = id.iphi();
      int zsid = id.zside();
      for (int eta = -ietaW; eta <= ietaE; ++eta) {
	for (int phi = -iphiS; phi <= iphiN; ++phi) {
	  int iphi = phiC+phi;
	  if (iphi < 0)        iphi += 360;
	  else if (iphi > 360) iphi -= 360;    
	  int ieta = zsid*(etaC+eta);
	  if (EBDetId::validDetId(ieta,iphi)) {
	    id = EBDetId(ieta,iphi);
	    if (barrelGeom.present(id)) {
	      if (std::count(dets.begin(),dets.end(),(DetId)id) == 0) {
		cells.push_back((DetId)id);
	      }
	    }
	  }
	}
      }
    } else if (det.subdetId() == EcalEndcap) {
      EEDetId id = det;
      if (debug) std::cout << "extraIds::Cell " << id << " rows "  << ietaW
			   << "|" << ietaE << " columns " << iphiS << "|"
			   << iphiN << std::endl;
      int ixC  = id.ix();
      int iyC  = id.iy();
      int zsid = id.zside();
      for (int kx = -ietaW; kx <= ietaE; ++kx) {
	for (int ky = -iphiS; ky <= iphiN; ++ky) {
	  int ix = ixC+kx;
	  int iy = iyC+ky;
	  if (EEDetId::validDetId(ix,iy,zsid)) {
	    id = EEDetId(ix,iy,zsid);
	    if (endcapGeom.present(id)) {
	      if (std::count(dets.begin(),dets.end(),(DetId)id) == 0) {
		cells.push_back((DetId)id);
	      }
	    }
	  }
	}
      }
    } 

    if (debug) {
      std::cout << "extraIds:: finds " << cells.size() << " new cells" 
		<< std::endl;
      for (unsigned int i1=0; i1<cells.size(); ++i1) {
	if (cells[i1].subdetId() == EcalBarrel) {
	  EBDetId id = cells[i1];
	  std::cout << "extraIds::Cell " << i1 << " "  << id << std::endl;
	} else if (cells[i1].subdetId() == EcalEndcap) {
	  EEDetId id = cells[i1];
	  std::cout << "ectraIds::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "extraIds::Cell " << i1 << " 0x" << std::hex 
		    << cells[i1]() <<std::dec << " Unknown Type" <<std::endl;
	}
      }
    }
    return cells;
  }
}
