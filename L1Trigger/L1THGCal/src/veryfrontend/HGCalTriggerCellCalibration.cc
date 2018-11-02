#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalTriggerCellCalibration.h"

//class constructor
HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& beCodecConfig){
    
    LSB_silicon_fC_ = beCodecConfig.getParameter<double>("siliconCellLSB_fC"); 
    LSB_scintillator_MIP_ = beCodecConfig.getParameter<double>("scintillatorCellLSB_MIP"); 
    fCperMIP_ = beCodecConfig.getParameter<double>("fCperMIP");
    dEdX_weights_ = beCodecConfig.getParameter<std::vector<double>>("dEdXweights");
    thickCorr_ = beCodecConfig.getParameter<double>("thickCorr");
    


    if(fCperMIP_ <= 0){
        edm::LogWarning("DivisionByZero") << "WARNING: the MIP->fC correction factor is zero or negative. It won't be applied to correct trigger cell energies.";
    }
    if(thickCorr_ <= 0){
        edm::LogWarning("DivisionByZero") << "WARNING: the cell-thickness correction factor is zero or negative. It won't be applied to correct trigger cell energies.";
    }

}


void HGCalTriggerCellCalibration::calibrateInMipT(l1t::HGCalTriggerCell& trgCell)
{
    
    HGCalDetId trgdetid( trgCell.detId() );

    /* get the hardware pT in ADC counts: */
    int hwPt = trgCell.hwPt();

    // Convert ADC to charge in fC (in EE+FH) or in MIPs (in BH)
    double amplitude = hwPt * (trgdetid.subdetId()==HGCHEB ? LSB_scintillator_MIP_ :  LSB_silicon_fC_);  

    // The responses of the different cell thicknesses have been equalized
    // to the 200um response in the front-end. So there is only one global
    // fCperMIP and thickCorr here
    /* convert the charge amplitude in MIP: */
    double trgCellMipP = amplitude;
    if( trgdetid.subdetId()!=HGCHEB && fCperMIP_ > 0 ){
        trgCellMipP /= fCperMIP_; 
    }

    /* compute the transverse-mip */
    double trgCellMipPt = trgCellMipP/cosh( trgCell.eta() ); 

    /* setting pT [mip] */
    trgCell.setMipPt( trgCellMipPt ) ;
} 


void HGCalTriggerCellCalibration::calibrateMipTinGeV(l1t::HGCalTriggerCell& trgCell)
{
    const double MevToGeV(0.001);

    HGCalDetId trgdetid( trgCell.detId() );
    unsigned trgCellLayer = triggerTools_.layerWithOffset(trgdetid);
    int subdet = trgdetid.subdetId();

    if(dEdX_weights_.at(trgCellLayer)==0.){
        throw cms::Exception("BadConfiguration")
            <<"Trigger cell energy forced to 0 by calibration coefficients.\n"
            <<"The configuration should be changed. "
            <<"Discarded layers should be defined in hgcalTriggerGeometryESProducer.TriggerGeometry.DisconnectedLayers and not with calibration coefficients = 0\n";
    }
   
    /* weight the amplitude by the absorber coefficient in MeV/mip + bring it in GeV */
    double trgCellEt = trgCell.mipPt() * dEdX_weights_.at(trgCellLayer) * MevToGeV;


    /* correct for the cell-thickness */
    if( subdet!=HGCHEB && thickCorr_ > 0 ){
        trgCellEt /= thickCorr_; 
    }
    /* assign the new energy to the four-vector of the trigger cell */
    math::PtEtaPhiMLorentzVector calibP4(trgCellEt, 
                                         trgCell.eta(), 
                                         trgCell.phi(), 
                                         0.);
    
    /* overwriting the 4p with the calibrated 4p */     
    trgCell.setP4( calibP4 );

}

void HGCalTriggerCellCalibration::calibrateInGeV(l1t::HGCalTriggerCell& trgCell)
{

    /* calibrate from ADC count to transverse mip */
    calibrateInMipT(trgCell);

    /* calibrate from mip count to GeV */
    calibrateMipTinGeV(trgCell);

}
 
