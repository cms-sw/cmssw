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

private:
    // FIXME: currently there is no access to the HGCal DDDConstants
    // So cannot retrieve the following constants.
    static const unsigned kLayersEE_ = 28;
    static const unsigned kLayersFH_ = 12;
    static const unsigned kLayersBH_ = 12;
    static const unsigned kLayers_ = kLayersEE_+kLayersFH_+kLayersBH_;

public:
  
    HGCalTriggerCellCalibration(const edm::ParameterSet &conf);    
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

};

#endif
