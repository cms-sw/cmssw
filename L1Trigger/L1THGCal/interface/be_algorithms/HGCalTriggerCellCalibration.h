#ifndef __L1Trigger_L1THGCal_HGCalTriggerCellCalibration_h__
#define __L1Trigger_L1THGCal_HGCalTriggerCellCalibration_h__

#include <stdint.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

class HGCalTriggerCellCalibration{

public:
  
    HGCalTriggerCellCalibration(const edm::ParameterSet &conf);    
    l1t::HGCalTriggerCell calibrate(l1t::HGCalTriggerCell&, int subdet, int cellThickness); 
    void print();

private:
    
    double LSB;
    uint32_t trgCellTruncBit;
    std::vector<double> fCperMIP_ee;
    std::vector<double> fCperMIP_fh;
    std::vector<double> dEdX_weights;
    std::vector<double> thickCorr;

};

#endif
