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
    void calibrate(l1t::HGCalTriggerCell&, int cellThickness); 
    void print();

private:
    
    double LSB_;
    std::vector<double> fCperMIP_ee_;
    std::vector<double> fCperMIP_fh_;
    std::vector<double> dEdX_weights_;
    std::vector<double> thickCorr_;

};

#endif
