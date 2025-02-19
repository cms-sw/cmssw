#include "Calibration/EcalCalibAlgos/interface/EcalGeomPhiSymHelper.h"

#include "FWCore/Framework/interface/ESHandle.h"


// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

//Channel status

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include <fstream>

void EcalGeomPhiSymHelper::setup(const CaloGeometry* geometry, 
				 const EcalChannelStatus* chStatus,
				 int statusThresold){

  
  for (int ieta=0; ieta<kBarlRings; ieta++)    nBads_barl[ieta] = 0;
  for (int ring=0; ring<kEndcEtaRings; ring++) nBads_endc[ring] = 0;

  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {

      cellPhi_[ix][iy]=0.;
      cellArea_[ix][iy]=0.;
      endcapRing_[ix][iy]=-1;
    }
  }
 

  // loop over all barrel crystals
  const std::vector<DetId>& barrelCells = geometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  std::vector<DetId>::const_iterator barrelIt;

  for (barrelIt=barrelCells.begin(); barrelIt!=barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);

    int sign = eb.zside()>0 ? 1 : 0;
    
    int chs= (*chStatus)[*barrelIt].getStatusCode() & 0x001F;
    if( chs <=  statusThresold)
      goodCell_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] = true;
    	    
    if( !goodCell_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] )
      nBads_barl[abs(eb.ieta())-1]++;
    
  }



  const CaloSubdetectorGeometry *endcapGeometry = 
    geometry->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
 
  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {
      cellPos_[ix][iy] = GlobalPoint(0.,0.,0.);
      cellPhi_[ix][iy]=0.;
      cellArea_[ix][iy]=0.;
      endcapRing_[ix][iy]=-1;
    }
  }

  const std::vector<DetId>& endcapCells = geometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  std::vector<DetId>::const_iterator endcapIt;
  for (endcapIt=endcapCells.begin(); endcapIt!=endcapCells.end(); endcapIt++) {

    const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(*endcapIt);
    EEDetId ee(*endcapIt);
    int ix=ee.ix()-1;
    int iy=ee.iy()-1;

    int sign = ee.zside()>0 ? 1 : 0;
    
    // store all crystal positions
    cellPos_[ix][iy] = cellGeometry->getPosition();
    cellPhi_[ix][iy] = cellGeometry->getPosition().phi();

    // calculate and store eta-phi area for each crystal front face Shoelace formuls
       const CaloCellGeometry::CornersVec& cellCorners (cellGeometry->getCorners()) ;
    cellArea_[ix][iy]=0.;

    for (int i=0; i<4; i++) {
      int iplus1 = i==3 ? 0 : i+1;
      cellArea_[ix][iy] += 
	cellCorners[i].eta()*float(cellCorners[iplus1].phi()) - 
	cellCorners[iplus1].eta()*float(cellCorners[i].phi());

    }
   

    cellArea_[ix][iy] = fabs(cellArea_[ix][iy])/2.;
/*
    const double deltaPhi =
      (dynamic_cast<const EcalEndcapGeometry*>(endcapGeometry))->deltaPhi(ee);

    const double deltaEta =
      (dynamic_cast<const EcalEndcapGeometry*>(endcapGeometry))->deltaEta(ee) ;

    cellArea_[ix][iy] = deltaEta*deltaPhi;
*/
    int chs= (*chStatus)[*endcapIt].getStatusCode() & 0x001F;
    if( chs <=  statusThresold)
      goodCell_endc[ix][iy][sign] = true;
    	    

  }
    
  // get eta boundaries for each endcap ring
  etaBoundary_[0]=1.479;
  etaBoundary_[39]=3.;  //It was 4. !!!
  for (int ring=1; ring<kEndcEtaRings; ring++) {
    double eta_ring_minus1= cellPos_[ring-1][50].eta()  ;
    double eta_ring = cellPos_[ring][50].eta()  ; 
    etaBoundary_[ring]=(eta_ring+eta_ring_minus1)/2.;
    std::cout << "Eta ring " << ring << " : " << eta_ring << std::endl; 
  }


    
 

  // determine to which ring each endcap crystal belongs,
  // the number of crystals in each ring,
  // and the mean eta-phi area of the crystals in each ring

  for (int ring=0; ring<kEndcEtaRings; ring++) {
    nRing_[ring]=0;
    meanCellArea_[ring]=0.;
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (fabs(cellPos_[ix][iy].eta())>etaBoundary_[ring] &&
	    fabs(cellPos_[ix][iy].eta())<etaBoundary_[ring+1]) {
	  meanCellArea_[ring]+=cellArea_[ix][iy];
	  endcapRing_[ix][iy]=ring;
	  nRing_[ring]++;
	  
	  

	  for(int sign=0; sign<kSides; sign++){
	    if( !goodCell_endc[ix][iy][sign] )
	      nBads_endc[ring]++;
	  } //sign

	} //if
      } //ix
    } //iy

    meanCellArea_[ring]/=nRing_[ring];


  } //ring

    // fill phi_endc[ip][ring] vector
    for (int ring=0; ring<kEndcEtaRings; ring++) {
    
      for (int i=0; i<kMaxEndciPhi; i++) 
	phi_endc_[i][ring]=0.;
    
      float philast=-999.;
      for (int ip=0; ip<nRing_[ring]; ip++) {
	float phimin=999.;
	for (int ix=0; ix<kEndcWedgesX; ix++) {
	  for (int iy=0; iy<kEndcWedgesY; iy++) {
	    if (endcapRing_[ix][iy]==ring) {
	      if (cellPhi_[ix][iy]<phimin && cellPhi_[ix][iy]>philast) {
		phimin=cellPhi_[ix][iy];
	      } //if edges
	    } //if ring
	  } //iy
	} //ix	
	phi_endc_[ip][ring]=phimin;
	philast=phimin;
      } //ip
  
    } //ring

    // Print out detid->ring association 
    std::fstream eeringsf("endcaprings.dat",std::ios::out);
    for (endcapIt=endcapCells.begin(); endcapIt!=endcapCells.end();endcapIt++){
      EEDetId eedet(*endcapIt);
      eeringsf<< eedet.hashedIndex()<< " " 
	      << endcapRing_[eedet.ix()-1][eedet.iy()-1] << " " 
              << cellPhi_ [eedet.ix()-1][eedet.iy()-1] << " "
              << cellArea_[eedet.ix()-1][eedet.iy()-1]/
	         meanCellArea_[endcapRing_[eedet.ix()-1][eedet.iy()-1]] <<   std::endl;
              
    }
}
