#ifndef __L1Trigger_L1THGCal_HGCalTriggerCellCalibration_h__
#define __L1Trigger_L1THGCal_HGCalTriggerCellCalibration_h__

#include <cstdint>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerCellCalibration{

public:
  
    HGCalTriggerCellCalibration(const edm::ParameterSet &conf);    
    void eventSetup(const edm::EventSetup& es) {triggerTools_.eventSetup(es);}
    void calibrateInMipT(l1t::HGCalTriggerCell&);
    void calibrateMipTinGeV(l1t::HGCalTriggerCell&); 
    void calibrateInGeV(l1t::HGCalTriggerCell&); 
    void print();

private:
    
    double LSB_silicon_fC_;
    double LSB_scintillator_MIP_;
    double fCperMIP_;
    double thickCorr_;
    std::vector<double> dEdX_weights_;

    HGCalTriggerTools triggerTools_;

};

#endif
