#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include<iostream>

namespace spr{

  std::pair<double,bool> eECALmatrix(const DetId& detId, edm::Handle<EcalRecHitCollection>& hitsEB, edm::Handle<EcalRecHitCollection>& hitsEE, const EcalChannelStatus& chStatus, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, double ebThr, double eeThr, bool debug) {

    std::vector<DetId> vdets = spr::matrixECALIds(detId, ieta, iphi, geo, caloTopology, debug);

    const EcalRecHitCollection * recHitsEB = 0;
    if (hitsEB.isValid())  recHitsEB = hitsEB.product();
    bool flag = true;
    if (debug) {
      std::cout << "Inside eECALmatrix " << 2*ieta+1 << "X" << 2*iphi+1
                << " nXtals " << vdets.size() << std::endl;
   }

    double energySum = 0.0;
    for (unsigned int i1=0; i1<vdets.size(); i1++) {
      if (vdets[i1] != DetId(0)) {
	bool ok = true;
	std::vector<EcalRecHitCollection::const_iterator> hit;
        if (vdets[i1].subdetId()==EcalBarrel) {
          hit = spr::findHit(hitsEB,vdets[i1]);
	  ok  = (EcalSeverityLevelAlgo::severityLevel(vdets[i1], (*recHitsEB), chStatus) != EcalSeverityLevelAlgo::kWeird);
        } else if (vdets[i1].subdetId()==EcalEndcap) {
          hit = spr::findHit(hitsEE,vdets[i1]);
        }
        if (debug) std::cout << "Xtal 0x" <<std::hex << vdets[i1]() <<std::dec;
        double ener=0, ethr=ebThr;
	if (vdets[i1].subdetId() !=EcalBarrel) ethr = eeThr;
	for (unsigned int ihit=0; ihit<hit.size(); ihit++) {
	  double en=0;
	  if (vdets[i1].subdetId()==EcalBarrel) {
	    if (hit[ihit] != hitsEB->end()) en = hit[ihit]->energy();
	  } else if (vdets[i1].subdetId()==EcalEndcap) {
	    if (hit[ihit] != hitsEE->end()) en = hit[ihit]->energy();
	  }
	  if (debug) std::cout << " " << ihit << " " << en;
	  ener += en;
	}
	if (!ok) {
	  flag = false;
	  if (debug) std::cout << " detected to be a spike";
	}
        if (debug) std::cout << "\n";
        if (ener > ethr) energySum += ener;
      }
    }
    if (debug) std::cout << "energyECAL: energySum = " << energySum << " flag = " << flag << std::endl;
    return std::pair<double,bool>(energySum,flag);
  }
}
