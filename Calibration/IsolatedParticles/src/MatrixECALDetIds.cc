#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"

#include <iostream>

namespace spr{

  std::vector<DetId> matrixECALIds(const DetId& det,int ieta,int iphi, 
				   const CaloSubdetectorTopology& barrelTopo,
				   const CaloSubdetectorTopology& endcapTopo,
				   const EcalBarrelGeometry& barrelGeom,
				   const EcalEndcapGeometry& endcapGeom, 
				   bool debug) {

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
    std::vector<DetId> vdetN = spr::newECALIdNS(dets, 0, ieta, iphi, NORTH,
						barrelTopo,  endcapTopo, 
						barrelGeom, endcapGeom, debug);
    std::vector<DetId> vdetS = spr::newECALIdNS(dets, 0, ieta, iphi, SOUTH,
						barrelTopo,  endcapTopo, 
						barrelGeom, endcapGeom, debug);
    for (unsigned int i1=0; i1<vdetS.size(); i1++) {
      if (std::count(vdetN.begin(),vdetN.end(),vdetS[i1]) == 0)
	vdetN.push_back(vdetS[i1]);
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
				   const CaloSubdetectorTopology& barrelTopo,
				   const CaloSubdetectorTopology& endcapTopo,
				   const EcalBarrelGeometry& barrelGeom,
				   const EcalEndcapGeometry& endcapGeom, 
				   bool debug) {

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
    std::vector<DetId> vdetN = spr::newECALIdNS(dets, 0, ietaE, ietaW, iphiN,
						iphiS, NORTH, barrelTopo,
						endcapTopo, barrelGeom,
						endcapGeom, debug);
    std::vector<DetId> vdetS = spr::newECALIdNS(dets, 0, ietaE, ietaW, iphiN,
						iphiS, SOUTH, barrelTopo,
						endcapTopo, barrelGeom,
						endcapGeom, debug);
    for (unsigned int i1=0; i1<vdetS.size(); i1++) {
      if (std::count(vdetN.begin(),vdetN.end(),vdetS[i1]) == 0)
	vdetN.push_back(vdetS[i1]);
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
				 int ieta, int iphi, const CaloDirection& dir,
				 const CaloSubdetectorTopology& barrelTopo,
				 const CaloSubdetectorTopology& endcapTopo,
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdNS::Add " << iphi << " columns of cells along " 
		<< dir << " for " << (dets.size()-last) << " cells" 
		<< std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << " Unknown Type" << std::endl;
	}
      }
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    std::vector<DetId> vdetE, vdetW;
    if (last == 0) {
      vdetE = spr::newECALIdEW(dets, last, ieta, EAST, barrelTopo, endcapTopo, 
			       barrelGeom, endcapGeom, debug);
      vdetW = spr::newECALIdEW(dets, last, ieta, WEST, barrelTopo, endcapTopo,
			       barrelGeom, endcapGeom, debug);
      for (unsigned int i1=0; i1<vdetW.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i1]) == 0)
	  vdets.push_back(vdetW[i1]);
      }
      for (unsigned int i1=0; i1<vdetE.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i1]) == 0)
	  vdets.push_back(vdetE[i1]);
      }
      if (debug) {
	std::cout <<"newECALIdNS::With Added cells along E/W results a set of "
		  << (vdets.size()-dets.size()) << " new  cells" << std::endl;
	for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
    }
    unsigned int last0 = vdets.size();
    if (iphi > 0) {
      std::vector<DetId> vdetnew;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::pair<DetId,bool> result = spr::simpleMove(dets[i1], dir, 
						       barrelTopo,endcapTopo,
						       barrelGeom,endcapGeom,
						       debug);
	if (result.second == true) {
	  if (std::count(vdets.begin(),vdets.end(),result.first) == 0)
	    vdetnew.push_back(result.first);
	}
      }
      iphi--;
      vdetE = spr::newECALIdEW(vdetnew, 0, ieta, EAST, barrelTopo, endcapTopo, 
			       barrelGeom, endcapGeom, debug);
      vdetW = spr::newECALIdEW(vdetnew, 0, ieta, WEST, barrelTopo, endcapTopo,
			       barrelGeom, endcapGeom, debug);
      for (unsigned int i2=0; i2<vdetW.size(); i2++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i2]) == 0 &&
	    std::count(vdetnew.begin(),vdetnew.end(),vdetW[i2]) == 0)
	  vdets.push_back(vdetW[i2]);
      }
      for (unsigned int i2=0; i2<vdetE.size(); i2++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i2]) == 0 &&
	    std::count(vdetnew.begin(),vdetnew.end(),vdetE[i2]) == 0)
	  vdets.push_back(vdetE[i2]);
      }
      last = vdets.size();
      vdets.insert(vdets.end(), vdetnew.begin(), vdetnew.end());
      if (debug) {
	std::cout << "newECALIdNS::Addition results a set of " 
		  << (vdets.size()-last0)  << " new  cells" << std::endl;
	for (unsigned int i1=last0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
      last0 = last;
    }

    if (iphi > 0) {
      last = last0;
      return spr::newECALIdNS(vdets,last,ieta,iphi,dir,barrelTopo,endcapTopo,barrelGeom,endcapGeom,debug);
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
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> newECALIdNS(std::vector<DetId>& dets, unsigned int last,
				 int ietaE, int ietaW, int iphiN, int iphiS,
				 const CaloDirection& dir,
				 const CaloSubdetectorTopology& barrelTopo,
				 const CaloSubdetectorTopology& endcapTopo,
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdNS::Add " << iphiN << "|" << iphiS 
		<< " columns of cells along " << dir << " for " 
		<< (dets.size()-last) << " cells" << std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << " Unknown Type" << std::endl;
	}
      }
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    std::vector<DetId> vdetE, vdetW;
    if (last == 0) {
      vdetE = spr::newECALIdEW(dets, last, ietaE, ietaW, EAST, barrelTopo, 
			       endcapTopo, barrelGeom, endcapGeom, debug);
      vdetW = spr::newECALIdEW(dets, last, ietaE, ietaW, WEST, barrelTopo,
			       endcapTopo, barrelGeom, endcapGeom, debug);
      for (unsigned int i1=0; i1<vdetW.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i1]) == 0)
	  vdets.push_back(vdetW[i1]);
      }
      for (unsigned int i1=0; i1<vdetE.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i1]) == 0)
	  vdets.push_back(vdetE[i1]);
      }
      if (debug) {
	std::cout <<"newECALIdNS::With Added cells along E/W results a set of "
		  << (vdets.size()-dets.size()) << " new  cells" << std::endl;
	for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
    }
    unsigned int last0 = vdets.size();
    int iphi = iphiS;
    if (dir == NORTH) iphi = iphiN;
    if (iphi > 0) {
      std::vector<DetId> vdetnew;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::pair<DetId,bool> result = spr::simpleMove(dets[i1], dir, 
						       barrelTopo,endcapTopo,
						       barrelGeom,endcapGeom,
						       debug);
	if (result.second == true) {
	  if (std::count(vdets.begin(),vdets.end(),result.first) == 0)
	    vdetnew.push_back(result.first);
	}
      }
      iphi--;
      vdetE = spr::newECALIdEW(vdetnew, 0, ietaE, ietaW, EAST, barrelTopo,
			       endcapTopo, barrelGeom, endcapGeom, debug);
      vdetW = spr::newECALIdEW(vdetnew, 0, ietaE, ietaW, WEST, barrelTopo,
			       endcapTopo, barrelGeom, endcapGeom, debug);
      for (unsigned int i2=0; i2<vdetW.size(); i2++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i2]) == 0 &&
	    std::count(vdetnew.begin(),vdetnew.end(),vdetW[i2]) == 0)
	  vdets.push_back(vdetW[i2]);
      }
      for (unsigned int i2=0; i2<vdetE.size(); i2++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i2]) == 0 &&
	    std::count(vdetnew.begin(),vdetnew.end(),vdetE[i2]) == 0)
	  vdets.push_back(vdetE[i2]);
      }
      last = vdets.size();
      vdets.insert(vdets.end(), vdetnew.begin(), vdetnew.end());
      if (debug) {
	std::cout << "newECALIdNS::Addition results a set of " 
		  << (vdets.size()-last0)  << " new  cells" << std::endl;
	for (unsigned int i1=last0; i1<vdets.size(); i1++) {
	  if (vdets[i1].subdetId() == EcalBarrel) {
	    EBDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " "  << id << std::endl;
	  } else if (vdets[i1].subdetId() == EcalEndcap) {
	    EEDetId id = vdets[i1];
	    std::cout << "newECALIdNS::Cell " << i1 << " " << id << std::endl;
	  } else {
	    std::cout << "newECALIdNS::Cell " << i1 << " 0x" << std::hex 
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
      last0 = last;
    }
    if (dir == NORTH) iphiN = iphi;
    else              iphiS = iphi;

    if (iphi > 0) {
      last = last0;
      return spr::newECALIdNS(vdets, last, ietaE, ietaW, iphiN, iphiS, dir,
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
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last,
				 int ieta, const CaloDirection& dir, 
				 const CaloSubdetectorTopology& barrelTopo, 
				 const CaloSubdetectorTopology& endcapTopo, 
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdEW::Add " << ieta << " rows of cells along " 
		<< dir << " for " << (dets.size()-last) << " cells" 
		<< std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << " Unknown Type" << std::endl;
	}
      }
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    if (ieta > 0) {
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::pair<DetId,bool> result = spr::simpleMove(dets[i1], dir, 
						       barrelTopo,endcapTopo,
						       barrelGeom,endcapGeom,
						       debug);
	if (result.second == true) {
	  if (std::count(vdets.begin(),vdets.end(),result.first) == 0)
	    vdets.push_back(result.first);
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
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	} else if (vdets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << vdets[i1]() << " Unknown Type" << std::endl;
	}
      }
    }

    if (ieta > 0) {
      last = dets.size();
      return spr::newECALIdEW(vdets,last,ieta,dir,barrelTopo,endcapTopo,barrelGeom,endcapGeom,debug);
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
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::vector<DetId> newECALIdEW(std::vector<DetId>& dets, unsigned int last,
				 int ietaE,int ietaW,const CaloDirection& dir, 
				 const CaloSubdetectorTopology& barrelTopo, 
				 const CaloSubdetectorTopology& endcapTopo, 
				 const EcalBarrelGeometry& barrelGeom, 
				 const EcalEndcapGeometry& endcapGeom,
				 bool debug) {

    if (debug) {
      std::cout << "newECALIdEW::Add " << ietaE << "|" << ietaW 
		<< " rows of cells along " << dir << " for " 
		<< (dets.size()-last) << " cells" << std::endl;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	if (dets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	} else if (dets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = dets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << dets[i1]() << " Unknown Type" << std::endl;
	}
      }
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    int ieta = ietaW;
    if (dir == EAST) ieta = ietaE;
    if (ieta > 0) {
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::pair<DetId,bool> result = spr::simpleMove(dets[i1], dir, 
						       barrelTopo,endcapTopo,
						       barrelGeom,endcapGeom,
						       debug);
	if (result.second == true) {
	  if (std::count(vdets.begin(),vdets.end(),result.first) == 0)
	    vdets.push_back(result.first);
	}
      }
      ieta--;
    }
    if (dir == EAST) ietaE = ieta;
    else             ietaW = ieta;
    
    if (debug) {
      std::cout << "newECALIdEW::Addition results a set of " 
		<< (vdets.size()-dets.size()) << " new  cells" << std::endl;
      for (unsigned int i1=dets.size(); i1<vdets.size(); i1++) {
	if (vdets[i1].subdetId() == EcalBarrel) {
	  EBDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " "  << id << std::endl;
	} else if (vdets[i1].subdetId() == EcalEndcap) {
	  EEDetId id = vdets[i1];
	  std::cout << "newECALIdEW::Cell " << i1 << " " << id << std::endl;
	} else {
	  std::cout << "newECALIdEW::Cell " << i1 << " 0x" << std::hex 
		    << vdets[i1]() << " Unknown Type" << std::endl;
	}
      }
    }

    if (ieta > 0) {
      last = dets.size();
      return spr::newECALIdEW(vdets, last, ietaE, ietaW, dir, barrelTopo,
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
		      << vdets[i1]() << " Unknown Type" << std::endl;
	  }
	}
      }
      return vdets;
    }
  }

  std::pair<DetId,bool> simpleMove(DetId& det, const CaloDirection& dir, 
				   const CaloSubdetectorTopology& barrelTopo, 
				   const CaloSubdetectorTopology& endcapTopo, 
				   const EcalBarrelGeometry& barrelGeom, 
				   const EcalEndcapGeometry& endcapGeom, 
				   bool debug) {
    DetId cell;
    bool ok=false;
    if (det.subdetId() == EcalBarrel) {
      EBDetId detId = det;
      std::vector<DetId> neighbours = barrelTopo.getNeighbours(detId,dir);
      if (neighbours.size()>0 && !neighbours[0].null()) {
	cell = neighbours[0];
	ok   = true;
      } else {
	const int ietaAbs ( detId.ietaAbs() ) ; // abs value of ieta
	if (EBDetId::MAX_IETA == ietaAbs ) {
	  // get ee nbrs for for end of barrel crystals
	  const EcalBarrelGeometry::OrderedListOfEEDetId&
	    ol( * barrelGeom.getClosestEndcapCells(detId) ) ;
	  // take closest neighbour on the other side, that is in the endcap
	  cell = *(ol.begin() );
	  ok   = true;
	}
      }
    } else if (det.subdetId() == EcalEndcap) {
      EEDetId detId = det;
      std::vector<DetId> neighbours = endcapTopo.getNeighbours(detId,dir);
      if (neighbours.size()>0 && !neighbours[0].null()) {
	cell = neighbours[0];
	ok   = true;
      } else {
	// are we on the outer ring ?
	const int iphi ( detId.iPhiOuterRing() ) ;
	if (iphi!= 0) {
	  // get eb nbrs for for end of endcap crystals
	  const EcalEndcapGeometry::OrderedListOfEBDetId&
	    ol( * endcapGeom.getClosestBarrelCells(detId) ) ;
	  // take closest neighbour on the other side, that is in the barrel.
	  cell = *(ol.begin() );
	  ok   = true;
	}
      }
    }  
    if (debug) {
      std::cout << "simpleMove:: Move DetId 0x" << std::hex << det() 
		<< std::dec << " along " << dir << " to get 0x" << std::hex
		<< cell() << std::dec << " with flag " << ok << std::endl;
    }
    return std::pair<DetId,bool>(cell,ok);
  }
}
