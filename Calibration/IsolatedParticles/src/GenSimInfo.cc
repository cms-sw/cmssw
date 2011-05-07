#include "Calibration/IsolatedParticles/interface/MatrixECALDetIds.h"
#include "Calibration/IsolatedParticles/interface/MatrixHCALDetIds.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/GenSimInfo.h"

#include<iostream>

namespace spr{

  void eGenSimInfo(const DetId& coreDet, HepMC::GenEvent::particle_const_iterator trkItr, std::vector<spr::propagatedGenTrackID>& trackIds, const CaloGeometry* geo, const CaloTopology* caloTopology, int ieta, int iphi, genSimInfo & info, bool debug) {
    
    if (debug) std::cout << "eGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta() << "/" << (*trkItr)->momentum().phi() << " with ieta:iphi " << ieta << ":" << iphi << std::endl;

    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, ieta, iphi, geo, caloTopology, debug);
    spr::cGenSimInfo(vdets, trkItr, trackIds, info, debug);
  }


  void eGenSimInfo(const DetId& coreDet, HepMC::GenEvent::particle_const_iterator trkItr, std::vector<spr::propagatedGenTrackID>& trackIds, const CaloGeometry* geo, const CaloTopology* caloTopology, double dR, const GlobalVector& trackMom, spr::genSimInfo & info, bool debug) {

    if (debug) std::cout << "eGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta() << "/" << (*trkItr)->momentum().phi() << " with dR,tMom " << dR << " " << trackMom << std::endl;

    std::vector<DetId> vdets = spr::matrixECALIds(coreDet, dR, trackMom, geo, caloTopology, debug);
    spr::cGenSimInfo(vdets, trkItr, trackIds, info, debug);
  }

  void hGenSimInfo(const DetId& coreDet, HepMC::GenEvent::particle_const_iterator trkItr, std::vector<spr::propagatedGenTrackID>& trackIds, const HcalTopology* topology, int ieta, int iphi, genSimInfo & info, bool includeHO, bool debug) {
    
    if (debug) std::cout << "hGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta() << "/" << (*trkItr)->momentum().phi() << " with ieta:iphi " << ieta << ":" << iphi << std::endl;

    std::vector<DetId> dets;
    dets.push_back(coreDet);
    std::vector<DetId> vdets = spr::matrixHCALIds(dets, topology, ieta, iphi, includeHO, debug);
     spr::cGenSimInfo(vdets, trkItr, trackIds, info, debug);
  }

  void hGenSimInfo(const DetId& coreDet, HepMC::GenEvent::particle_const_iterator trkItr, std::vector<spr::propagatedGenTrackID>& trackIds, const CaloGeometry* geo, const HcalTopology* topology, double dR, const GlobalVector& trackMom, spr::genSimInfo & info, bool includeHO, bool debug) {
    
    if (debug) std::cout << "hGenSimInfo:: For track " << (*trkItr)->momentum().rho() << "/" << (*trkItr)->momentum().eta() << "/" << (*trkItr)->momentum().phi() << " with dR,tMom " << dR << " " << trackMom << std::endl;

    std::vector<DetId> vdets = spr::matrixHCALIds(coreDet, geo, topology, dR, trackMom, includeHO, debug);
     spr::cGenSimInfo(vdets, trkItr, trackIds, info, debug);
  }

  void cGenSimInfo(std::vector<DetId> vdets, HepMC::GenEvent::particle_const_iterator trkItr, std::vector<spr::propagatedGenTrackID>& trackIds, spr::genSimInfo & info, bool debug) {

    for (unsigned int i=0; i<trackIds.size(); ++i) {
      HepMC::GenEvent::particle_const_iterator trkItr2 = trackIds[i].trkItr;
      // avoid the track under consideration
      if ( (trkItr2 != trkItr) && trackIds[i].ok) {
	int charge = trackIds[i].charge;
	int pdgid  = trackIds[i].pdgId;
	double p   = (*trkItr2)->momentum().rho();
	const DetId anyCell = trackIds[i].detIdHCAL;
	if (!spr::chargeIsolation(anyCell,vdets)) {
	  if      (pdgid==22 ) info.photonEne  += p;
	  else if (pdgid==11)  info.eleEne     += p;
	  else if (pdgid==13)  info.muEne      += p;
	  else if (std::abs(charge)>0)   { 
	    info.isChargedIso = false; 
	    info.cHadronEne  += p;
	    if (p>1.0) info.cHadronEne_[0] += p;
	    if (p>2.0) info.cHadronEne_[1] += p;
	    if (p>3.0) info.cHadronEne_[2] += p;
	    if (info.maxNearP<p) info.maxNearP=p;
	  } else if (std::abs(charge)==0)  { 
	    info.nHadronEne += p;
	  }
	}
      }
    }
    if (debug) {
      std::cout << "Isolation variables: isChargedIso :" << info.isChargedIso 
		<< " maxNearP " << info.maxNearP << " Energy e/mu/g/ch/nh "
		<< info.eleEne << "," << info.muEne << "," << info.photonEne
		<< info.cHadronEne << "," << info.nHadronEne << " charge" 
		<< info.cHadronEne_[0] << "," << info.cHadronEne_[1] << ","
		<< info.cHadronEne_[2] << std::endl;
    }
  }
}
