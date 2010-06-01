#ifndef _Calibration_EcalCalibAlgos_EcalGeomPhiSymHelper_h_
#define _Calibration_EcalCalibAlgos_EcalGeomPhiSymHelper_h_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

static const int  kBarlRings  = 85;
static const int  kBarlWedges = 360;
static const int  kSides      = 2;

static const int  kEndcWedgesX = 100;
static const int  kEndcWedgesY = 100;

static const int  kEndcEtaRings  = 39;
static const int kMaxEndciPhi = 360;


class  CaloGeometry;

class EcalGeomPhiSymHelper {

 public:
  

  void setup(const CaloGeometry* geometry, 
	     const EcalChannelStatus* chstatus,
	     int statusThreshold);

  GlobalPoint cellPos_[kEndcWedgesX][kEndcWedgesY];
  double cellPhi_     [kEndcWedgesX][kEndcWedgesY];  
  double cellArea_    [kEndcWedgesX][kEndcWedgesY];
  double phi_endc_    [kMaxEndciPhi][kEndcEtaRings]; 
  double meanCellArea_[kEndcEtaRings];
  double etaBoundary_ [kEndcEtaRings+1];
  int endcapRing_     [kEndcWedgesX][kEndcWedgesY];  
  int nRing_          [kEndcEtaRings];
 
  // informations about good cells
  bool goodCell_barl[kBarlRings][kBarlWedges][kSides];
  bool goodCell_endc[kEndcWedgesX][kEndcWedgesX][kSides];   
  int nBads_barl[kBarlRings];
  int nBads_endc[kEndcEtaRings];

};


#endif
