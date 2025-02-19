#include <cmath>
#include <cassert>
#include <fstream>
#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/EcalBarrelGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/Records/interface/EcalEndcapGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "PhotonFixCMS.h"

PhotonFixCMS::PhotonFixCMS(const reco::Photon &p):
  pf(p.energy(),p.superCluster()->eta(),p.superCluster()->phi(),p.r9()) {
}

bool PhotonFixCMS::initialise(const edm::EventSetup &iSetup, const std::string &s) {
  
  if(PhotonFix::initialised()) return false;
 
  PhotonFix::initialiseParameters(s);
  
  // Get ECAL geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<CaloGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  
  // EB
  const CaloSubdetectorGeometry *barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  
  double bc[170][360][2];
  for(int iz(0);iz<2;iz++) {
    for(int ie(0);ie<85;ie++) {
      int id = ie+1;
      if (iz==0) id = ie-85; 
      for(int ip(0);ip<360;ip++) {
	EBDetId eb(id,ip+1);
	
	const CaloCellGeometry *cellGeometry = barrelGeometry->getGeometry(eb);
	GlobalPoint crystalPos = cellGeometry->getPosition();
	bc[85*iz+ie][ip][0]=crystalPos.eta();
	bc[85*iz+ie][ip][1]=crystalPos.phi();
      }
    }
  }
  
  for(unsigned i(0);i<169;i++) {
    for(unsigned j(0);j<360;j++) {
      unsigned k((j+1)%360);
     
      double eta = 0.25*(    bc[i][j][0]+bc[i+1][j][0]+
				     bc[i][k][0]+bc[i+1][k][0]);
      double phi = PhotonFix::GetaPhi(PhotonFix::GetaPhi(bc[i][j][1],bc[i+1][j][1]),
				PhotonFix::GetaPhi(bc[i][k][1],bc[i+1][k][1]));

      PhotonFix::barrelCGap(i,j,0,eta);
      PhotonFix::barrelCGap(i,j,1,phi);
      
      if((i%5)==4 && (j%2)==1) {
	  PhotonFix::barrelSGap(i/5,j/2,0,eta);
    PhotonFix::barrelSGap(i/5,j/2,1,phi);	
      }
      
      if((j%20)==19) {
	  if(i== 19) {PhotonFix::barrelMGap(0,j/20,0,eta); PhotonFix::barrelMGap(0,j/20,1,phi);}
	  if(i== 39) {PhotonFix::barrelMGap(1,j/20,0,eta); PhotonFix::barrelMGap(1,j/20,1,phi);}
	  if(i== 59) {PhotonFix::barrelMGap(2,j/20,0,eta); PhotonFix::barrelMGap(2,j/20,1,phi);}
	  if(i== 84) {PhotonFix::barrelMGap(3,j/20,0,eta); PhotonFix::barrelMGap(3,j/20,1,phi);}
	  if(i==109) {PhotonFix::barrelMGap(4,j/20,0,eta); PhotonFix::barrelMGap(4,j/20,1,phi);}
	  if(i==129) {PhotonFix::barrelMGap(5,j/20,0,eta); PhotonFix::barrelMGap(5,j/20,1,phi);}
	  if(i==149) {PhotonFix::barrelMGap(6,j/20,0,eta); PhotonFix::barrelMGap(6,j/20,1,phi);}
	
      }
    }
  }
  
  // EE
  const CaloSubdetectorGeometry *endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  
  double ec[2][100][100][2];
  bool valid[100][100];
  int val_count=0;
  for(int iz(0);iz<2;iz++) {
    for(int ix(0);ix<100;ix++) {
      for(int iy(0);iy<100;iy++) {
    valid[ix][iy] = EEDetId::validDetId(ix+1,iy+1,2*iz-1);
if(iz==0) PhotonFix::endcapCrystal(ix,iy,valid[ix][iy]);
	if(valid[ix][iy]) {
	  EEDetId ee(ix+1,iy+1,2*iz-1);
    val_count+=1;
	  
	  const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(ee);
	  GlobalPoint crystalPos = cellGeometry->getPosition();
	  ec[iz][ix][iy][0]=asinh(crystalPos.x()/fabs(crystalPos.z()));
	  ec[iz][ix][iy][1]=asinh(crystalPos.y()/fabs(crystalPos.z()));
	}
      }
    }
  }
  std::cout << "GG valid " << val_count << std::endl;
  double c[2];
  for(unsigned iz(0);iz<2;iz++) {
    unsigned nC(0),nS(0);
    for(unsigned i(0);i<99;i++) {
      for(unsigned j(0);j<99;j++) {
	if(valid[i][j  ] && valid[i+1][j  ] && 
	   valid[i][j+1] && valid[i+1][j+1]) {
	  for(unsigned k(0);k<2;k++) {

      c[k] = 0.25*(ec[iz][i][j][k]+ec[iz][i+1][j][k]+ec[iz][i][j+1][k]+ec[iz][i+1][j+1][k]);

	    PhotonFix::endcapCGap(iz,nC,k,c[k]);	 
    }
	  
	  if((i%5)==4 && (j%5)==4) {
	    for(unsigned k(0);k<2;k++) {
	    PhotonFix::endcapSGap(iz,nS,k,c[k]);	 
	    }
	    nS++;
	  }
	  nC++;
	}
      }
    }
    std::cout << "Endcap number of crystal, submodule boundaries = "
	      << nC << ", " << nS << std::endl;
  }
  
  // Hardcode EE D-module gap to 0,0
	PhotonFix::endcapMGap(0,0,0,0.0);	 
	PhotonFix::endcapMGap(0,0,1,0.0);	 
	PhotonFix::endcapMGap(1,0,0,0.0);	 
	PhotonFix::endcapMGap(1,0,1,0.0);	 
  
  return true;
}


double PhotonFixCMS::fixedEnergy() const {
  
  return pf.fixedEnergy();
}

double PhotonFixCMS::sigmaEnergy() const {
  
  return pf.sigmaEnergy();
}

const PhotonFix& PhotonFixCMS::photonFix() const {
  return pf;
}

