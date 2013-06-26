#include "Calibration/IsolatedParticles/interface/FindEtaPhi.h"
#include <iostream>

namespace spr{

  spr::EtaPhi getEtaPhi(int ieta, int iphi, bool debug) {

    int ietal = (ieta-1)/2;
    int ietar = ieta - ietal - 1;
    int iphil = (iphi-1)/2;
    int iphir = iphi - iphil - 1;
    spr::EtaPhi etaphi;
    etaphi.ietaE[0] = ietal; etaphi.ietaW[0] = ietar;
    etaphi.iphiN[0] = iphil; etaphi.iphiS[0] = iphir;
    if (ietal == ietar && iphil == iphir) {
      etaphi.ntrys = 1;
    } else if (ietal == ietar || iphil == iphir) {
      etaphi.ntrys = 2;
      etaphi.ietaE[1] = ietar; etaphi.ietaW[1] = ietal;
      etaphi.iphiN[1] = iphir; etaphi.iphiS[1] = iphil;
    } else {
      etaphi.ntrys = 4;
      etaphi.ietaE[1] = ietar; etaphi.ietaW[1] = ietal;
      etaphi.iphiN[1] = iphil; etaphi.iphiS[1] = iphir;
      etaphi.ietaE[2] = ietal; etaphi.ietaW[1] = ietar;
      etaphi.iphiN[2] = iphir; etaphi.iphiS[1] = iphil;
      etaphi.ietaE[3] = ietar; etaphi.ietaW[1] = ietal;
      etaphi.iphiN[3] = iphir; etaphi.iphiS[1] = iphil;
    }

    if (debug) {
      std::cout << "getEtaPhi:: Has " <<  etaphi.ntrys << " possibilites for "
		<< ieta << "X" << iphi << " matrix" << std::endl;
      for (int itry=0; itry<etaphi.ntrys; itry++) {
	std::cout << "Trial " << itry <<" with etaE|etaW " <<etaphi.ietaE[itry]
		  <<"|" << etaphi.ietaW[itry] << " and phiN|PhiS " 
		  << etaphi.iphiN[itry] <<"|" <<etaphi.iphiS[itry] <<std::endl;
      }
    }
    return etaphi;
  }
}
