#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Calibration/IsolatedParticles/interface/MatrixHCALDetIds.h"
#include "Calibration/IsolatedParticles/interface/FindDistCone.h"
#include "Calibration/IsolatedParticles/interface/DebugInfo.h"

#include<algorithm>
#include<iostream>

namespace spr{

  std::vector<DetId> matrixHCALIds(std::vector<DetId>& dets, 
				   const HcalTopology* topology, int ieta, 
				   int iphi, bool includeHO, bool debug) {

    if (debug) {
      std::cout << "matrixHCALIds::Add " << ieta << " rows and " << iphi 
		<< " columns of cells for " << dets.size() << " cells" 
		<< std::endl;
      spr::debugHcalDets(0, dets);
    }

    std::vector<DetId> vdetN = spr::newHCALIdNS(dets, 0, topology, true,  ieta,
					       iphi, debug);
    std::vector<DetId> vdetS = spr::newHCALIdNS(dets, 0, topology, false, ieta,
						iphi, debug);
    for (unsigned int i1=0; i1<vdetS.size(); i1++) {
      if (std::count(vdetN.begin(),vdetN.end(),vdetS[i1]) == 0)
	vdetN.push_back(vdetS[i1]);
    }

    vdetS = spr::matrixHCALIdsDepth(vdetN, topology, includeHO, debug);

    if (debug) {
      std::cout << "matrixHCALIds::Total number of cells found is " 
		<< vdetS.size() << std::endl;
      spr::debugHcalDets(0, vdetS);
    }
    return vdetS;
  }

  std::vector<DetId> matrixHCALIds(const DetId& det, const CaloGeometry* geo,
				   const HcalTopology* topology, double dR, 
				   const GlobalVector& trackMom, bool includeHO,
				   bool debug) {
 
    HcalDetId   hcdet = HcalDetId(det);
    GlobalPoint core  = geo->getPosition(hcdet);
    std::vector<DetId> dets, vdetx;
    dets.push_back(det);
    int ietaphi = (int)(dR/15.0)+1;
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ietaphi, 
						  ietaphi, includeHO, debug);
    for (unsigned int i=0; i<vdets.size(); ++i) {
      HcalDetId   hcdet  = HcalDetId(vdets[i]);
      GlobalPoint rpoint = geo->getPosition(hcdet);
      if (spr::getDistInPlaneTrackDir(core, trackMom, rpoint) < dR) {
	vdetx.push_back(vdets[i]);
      }
    }

    if (debug) {
      std::cout << "matrixHCALIds::Final List of cells for dR " << dR
		<< " is with " << vdetx.size() << " from original list of "
		<< vdets.size() << " cells" << std::endl;
      spr::debugHcalDets(0, vdetx);
    }
    return vdetx;
 }

  std::vector<DetId> matrixHCALIds(std::vector<DetId>& dets, 
				   const HcalTopology* topology, int ietaE, 
				   int ietaW,int iphiN,int iphiS, 
				   bool includeHO, bool debug) {

    if (debug) {
      std::cout << "matrixHCALIds::Add " <<ietaE << "|" <<ietaW << " rows and "
		<< iphiN << "|" << iphiS << " columns of cells for " 
		<< dets.size() << " cells" << std::endl;
      spr::debugHcalDets(0, dets);
    }

    std::vector<DetId> vdetN = spr::newHCALIdNS(dets, 0, topology, true, ietaE,
						ietaW, iphiN, iphiS, debug);
    std::vector<DetId> vdetS = spr::newHCALIdNS(dets, 0, topology, false,ietaE,
						ietaW, iphiN, iphiS, debug);
    for (unsigned int i1=0; i1<vdetS.size(); i1++) {
      if (std::count(vdetN.begin(),vdetN.end(),vdetS[i1]) == 0)
	vdetN.push_back(vdetS[i1]);
    }

    vdetS = spr::matrixHCALIdsDepth(vdetN, topology, includeHO, debug);

    if (debug) {
      std::cout << "matrixHCALIds::Total number of cells found is " 
		<< vdetS.size() << std::endl;
      spr::debugHcalDets(0, vdetS);
    }
    return vdetS;
  }

  std::vector<DetId> newHCALIdNS(std::vector<DetId>& dets, unsigned int last,
				 const HcalTopology* topology, bool shiftNorth,
				 int ieta, int iphi, bool debug) {

    if (debug) {
      std::cout << "newHCALIdNS::Add " << iphi << " columns of cells along " 
		<< shiftNorth << " for " << (dets.size()-last) << " cells" 
		<< std::endl;
      spr::debugHcalDets(last, dets);
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    std::vector<DetId> vdetE, vdetW;
    if (last == 0) {
      vdetE = spr::newHCALIdEW(dets, last, topology, true,  ieta, debug);
      vdetW = spr::newHCALIdEW(dets, last, topology, false, ieta, debug);
      for (unsigned int i1=0; i1<vdetW.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i1]) == 0)
	  vdets.push_back(vdetW[i1]);
      }
      for (unsigned int i1=0; i1<vdetE.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i1]) == 0)
	  vdets.push_back(vdetE[i1]);
      }
      if (debug) {
	std::cout <<"newHCALIdNS::With Added cells along E/W results a set of "
		  << (vdets.size()-dets.size()) << " new  cells" << std::endl;
	spr::debugHcalDets(dets.size(), vdets);
      }
    }
    unsigned int last0 = vdets.size();
    if (iphi > 0) {
      std::vector<DetId> vdetnew;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::vector<DetId> vdet;
	if (shiftNorth) vdet = topology->north(dets[i1]);
	else            vdet = topology->south(dets[i1]);
	for (unsigned int i2=0; i2<vdet.size(); i2++) {
	  if (std::count(vdets.begin(),vdets.end(),vdet[i2]) == 0)
	    vdetnew.push_back(vdet[i2]);
	}
      }
      iphi--;
      vdetE = spr::newHCALIdEW(vdetnew, 0, topology, true,  ieta, debug);
      vdetW = spr::newHCALIdEW(vdetnew, 0, topology, false, ieta, debug);
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
	std::cout << "newHCALIdNS::Addition results a set of " 
		  << (vdets.size()-last0)  << " new  cells" << std::endl;
	spr::debugHcalDets(last0, vdets);
      }
      last0 = last;
    }

    if (iphi > 0) {
      last = last0;
      return spr::newHCALIdNS(vdets,last,topology,shiftNorth,ieta,iphi,debug);
    } else {
      if (debug) {
	std::cout << "newHCALIdNS::Final list consists of " << vdets.size()
		  << " cells" << std::endl;
	spr::debugHcalDets(0, vdets);
      }
      return vdets;
    }
  }

  std::vector<DetId> newHCALIdNS(std::vector<DetId>& dets, unsigned int last,
				 const HcalTopology* topology, bool shiftNorth,
				 int ietaE, int ietaW, int iphiN, int iphiS,
				 bool debug) {

    if (debug) {
      std::cout << "newHCALIdNS::Add " << iphiN << "|" << iphiS
		<< " columns of cells along " << shiftNorth << " for " 
		<< (dets.size()-last) << " cells" << std::endl;
      spr::debugHcalDets(last, dets);
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    std::vector<DetId> vdetE, vdetW;
    if (last == 0) {
      vdetE = spr::newHCALIdEW(dets,last, topology, true,  ietaE,ietaW, debug);
      vdetW = spr::newHCALIdEW(dets,last, topology, false, ietaE,ietaW, debug);
      for (unsigned int i1=0; i1<vdetW.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetW[i1]) == 0)
	  vdets.push_back(vdetW[i1]);
      }
      for (unsigned int i1=0; i1<vdetE.size(); i1++) {
	if (std::count(vdets.begin(),vdets.end(),vdetE[i1]) == 0)
	  vdets.push_back(vdetE[i1]);
      }
      if (debug) {
	std::cout <<"newHCALIdNS::With Added cells along E/W results a set of "
		  << (vdets.size()-dets.size()) << " new  cells" << std::endl;
	spr::debugHcalDets(dets.size(), vdets);
      }
    }
    unsigned int last0 = vdets.size();
    int iphi = iphiS;
    if (shiftNorth) iphi = iphiN;
    if (iphi > 0) {
      std::vector<DetId> vdetnew;
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::vector<DetId> vdet;
	if (shiftNorth) vdet = topology->north(dets[i1]);
	else            vdet = topology->south(dets[i1]);
	for (unsigned int i2=0; i2<vdet.size(); i2++) {
	  if (std::count(vdets.begin(),vdets.end(),vdet[i2]) == 0)
	    vdetnew.push_back(vdet[i2]);
	}
      }
      iphi--;
      vdetE = spr::newHCALIdEW(vdetnew,0, topology, true,  ietaE,ietaW, debug);
      vdetW = spr::newHCALIdEW(vdetnew,0, topology, false, ietaE,ietaW, debug);
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
	std::cout << "newHCALIdNS::Addition results a set of " 
		  << (vdets.size()-last0)  << " new  cells" << std::endl;
	spr::debugHcalDets(last0, vdets);
      }
      last0 = last;
    }
    if (shiftNorth) iphiN = iphi;
    else            iphiS = iphi;

    if (iphi > 0) {
      last = last0;
      return spr::newHCALIdNS(vdets,last,topology,shiftNorth,ietaE,ietaW,
			      iphiN,iphiS,debug);
    } else {
      if (debug) {
	std::cout << "newHCALIdNS::Final list consists of " << vdets.size()
		  << " cells" << std::endl;
	spr::debugHcalDets(0, vdets);
      }
      return vdets;
    }
  }

  std::vector<DetId> newHCALIdEW(std::vector<DetId>& dets, unsigned int last,
				 const HcalTopology* topology, bool shiftEast,
				 int ieta, bool debug) {

    if (debug) {
      std::cout << "newHCALIdEW::Add " << ieta << " rows of cells along " 
		<< shiftEast << " for " << (dets.size()-last) << " cells" 
		<< std::endl;
      spr::debugHcalDets(last, dets);
    }

    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    if (ieta > 0) {
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::vector<DetId> vdet;
	if (shiftEast) vdet = topology->east(dets[i1]);
	else           vdet = topology->west(dets[i1]);
	for (unsigned int i2=0; i2<vdet.size(); i2++) {
	  if (std::count(vdets.begin(),vdets.end(),vdet[i2]) == 0)
	    vdets.push_back(vdet[i2]);
	}
      }
      ieta--;
    }
    
    if (debug) {
      std::cout << "newHCALIdEW::Addition results a set of " 
		<< (vdets.size()-dets.size()) << " new  cells" << std::endl;
      spr::debugHcalDets(dets.size(), vdets);
    }

    if (ieta > 0) {
      last = dets.size();
      return spr::newHCALIdEW(vdets, last, topology, shiftEast, ieta, debug);
    } else {
      if (debug) {
	std::cout << "newHCALIdEW::Final list (EW) consists of " <<vdets.size()
		  << " cells" << std::endl;
	spr::debugHcalDets(0, vdets);
      }
      return vdets;
    }
  }

  std::vector<DetId> newHCALIdEW(std::vector<DetId>& dets, unsigned int last,
				 const HcalTopology* topology, bool shiftEast,
				 int ietaE, int ietaW, bool debug) {

    if (debug) {
      std::cout << "newHCALIdEW::Add " << ietaE << "|" << ietaW
		<< " rows of cells along " << shiftEast << " for " 
		<< (dets.size()-last) << " cells" << std::endl;
      spr::debugHcalDets(last, dets);
    }

    int ieta = ietaW;
    if (shiftEast) ieta = ietaE;
    std::vector<DetId> vdets;
    vdets.insert(vdets.end(), dets.begin(), dets.end());
    if (ieta > 0) {
      for (unsigned int i1=last; i1<dets.size(); i1++) {
	std::vector<DetId> vdet;
	if (shiftEast) vdet = topology->east(dets[i1]);
	else           vdet = topology->west(dets[i1]);
	for (unsigned int i2=0; i2<vdet.size(); i2++) {
	  if (std::count(vdets.begin(),vdets.end(),vdet[i2]) == 0)
	    vdets.push_back(vdet[i2]);
	}
      }
      ieta--;
    }
    if (shiftEast) ietaE = ieta;
    else           ietaW = ieta;
    
    if (debug) {
      std::cout << "newHCALIdEW::Addition results a set of " 
		<< (vdets.size()-dets.size()) << " new  cells" << std::endl;
      spr::debugHcalDets(dets.size(), vdets);
    }

    if (ieta > 0) {
      last = dets.size();
      return spr::newHCALIdEW(vdets,last,topology,shiftEast,ietaE,ietaW,debug);
    } else {
      if (debug) {
	std::cout << "newHCALIdEW::Final list (EW) consists of " <<vdets.size()
		  << " cells" << std::endl;
	spr::debugHcalDets(0, vdets);
      }
      return vdets;
    }
  }

  std::vector<DetId> matrixHCALIdsDepth(std::vector<DetId>& dets, 
					const HcalTopology* topology, 
					bool includeHO, bool debug) {

    if (debug) {
      std::cout << "matrixHCALIdsDepth::Add cells with higher depths with HO" 
		<< "Flag set to " << includeHO << " to existing "
		<< dets.size() << " cells" << std::endl;
      spr::debugHcalDets(0, dets);
    }
 
    std::vector<DetId> vdets(dets);
    for (unsigned int i1=0; i1<dets.size(); i1++) {
      HcalDetId vdet = dets[i1];
      for (int idepth = 0; idepth < 3; idepth++) {
        std::vector<DetId> vUpDetId = topology->up(vdet);
        if (vUpDetId.size() != 0) {
          if (includeHO || vUpDetId[0].subdetId() != (int)(HcalOuter)) {
            int n = std::count(vdets.begin(),vdets.end(),vUpDetId[0]);
            if (n == 0) {
	      if (debug) std::cout << "matrixHCALIdsDepth:: Depth " << idepth 
				   << " " << vdet << " " 
				   << (HcalDetId)vUpDetId[0] << std::endl;
              vdets.push_back(vUpDetId[0]);
	    }
          }
          vdet = vUpDetId[0];
        }
      }
    }

    if (debug) {
      std::cout << "matrixHCALIdsDepth::Final list contains " << vdets.size() 
		<< " cells" << std::endl;
      spr::debugHcalDets(0, vdets);
    }
    return vdets;
  }

}
