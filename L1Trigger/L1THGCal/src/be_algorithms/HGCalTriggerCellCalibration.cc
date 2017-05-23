#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"

//class constructor
HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& beCodecConfig){
    
    LSB_ = beCodecConfig.getParameter<double>("cellLSB"); 
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

    // Convert ADC to charge in fC
    double amplitude = hwPt * LSB_;  

    // The responses of the different cell thicknesses have been equalized
    // to the 200um response in the front-end. So there is only one global
    // fCperMIP and thickCorr here
    /* convert the charge amplitude in MIP: */
    double trgCellMipP = amplitude;
    if( fCperMIP_ > 0 ){
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
    int trgCellLayer = trgdetid.layer();
    int subdet = trgdetid.subdetId();

    /* get the transverse momentum in mip units */
    double mipP = trgCell.mipPt() * cosh( trgCell.eta() );

    if( subdet == HGCHEF ){
            trgCellLayer = trgCellLayer + 28;
    }
   
    /* weight the amplitude by the absorber coefficient in MeV/mip + bring it in GeV */
    double trgCellE = mipP * dEdX_weights_.at(trgCellLayer) * MevToGeV;

    /* correct for the cell-thickness */
    if( thickCorr_ > 0 ){
        trgCellE /= thickCorr_; 
    }

    /* assign the new energy to the four-vector of the trigger cell */
    math::PtEtaPhiMLorentzVector calibP4(trgCellE/cosh( trgCell.eta() ), 
                                         trgCell.eta(), 
                                         trgCell.phi(), 
                                         trgCell.p4().M() );
    
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
 
