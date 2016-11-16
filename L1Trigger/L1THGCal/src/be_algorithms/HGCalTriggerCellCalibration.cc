#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"

//class constructor
HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& conf){
    
    const edm::ParameterSet beCodecConfig = conf.getParameterSet("calib_parameters");

    LSB_ = beCodecConfig.getParameter<double>("cellLSB"); 
    fCperMIP_ee_ = beCodecConfig.getParameter<std::vector<double>>("fCperMIPee");
    fCperMIP_fh_ = beCodecConfig.getParameter<std::vector<double>>("fCperMIPfh");
    dEdX_weights_ = beCodecConfig.getParameter<std::vector<double>>("dEdXweights");
    thickCorr_ = beCodecConfig.getParameter<std::vector<double>>("thickCorr");
}

void HGCalTriggerCellCalibration::calibrate(l1t::HGCalTriggerCell& trgCell, int cellThickness){
        HGCalDetId trgdetid(trgCell.detId());
        int trgCellLayer = trgdetid.layer();
        int subdet =  (ForwardSubdetector)trgdetid.subdetId();

        //get the hardware pT in fC:
        int hwPt = trgCell.hwPt();
        //set the lowest signal bit:
        double amplitude = hwPt * LSB_;  
        if( subdet == HGCEE ){ 
            //convert the charge amplitude in MIP:
            amplitude = amplitude / fCperMIP_ee_[cellThickness-1];
        }else if( subdet == HGCHEF ){
            //convert the charge amplitude in MIP:
            amplitude = amplitude / fCperMIP_fh_[cellThickness-1];
            trgCellLayer = trgCellLayer + 28;
        }else if( subdet == HGCHEB ){
            edm::LogWarning("DataNotFound") << "WARNING: the BH trgCells are not yet implemented !! ";
        }
        
        //weight the amplitude by the absorber coefficient in MeV + bring it in GeV and correct for the sensor thickness
        double trgCell_E = amplitude * dEdX_weights_[trgCellLayer] * 0.001 *  thickCorr_[cellThickness-1];

        //assign the new energy to the four-vector of the trigger cell and return the trigger cell
        math::PtEtaPhiMLorentzVector calibP4(trgCell_E/cosh(trgCell.eta()), trgCell.eta(), trgCell.phi(), trgCell.p4().M() );
        trgCell.setP4(calibP4);
        
} 
