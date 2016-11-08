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

    HGCalTriggerCellCalibration(const edm::ParameterSet& conf);    
 
    l1t::HGCalTriggerCell calibTrgCell(l1t::HGCalTriggerCell&, const edm::EventSetup& es); 
    //void calibTrgCellCollection( l1t::HGCalTriggerCellBxCollection& trgCellColl, const edm::EventSetup& es);
    void print();
    edm::ESHandle<HGCalGeometry> hgceeGeoHandle_;
    edm::ESHandle<HGCalGeometry> hgchefGeoHandle_;

private:
    
    double LSB_;
    uint32_t trgCellTruncBit_;
    std::vector<double> fCxMIP_ee_;
    std::vector<double> fCxMIP_fh_;
    std::vector<double> dEdX_weights_;
    std::vector<double> thickCorr_;

};

#endif
