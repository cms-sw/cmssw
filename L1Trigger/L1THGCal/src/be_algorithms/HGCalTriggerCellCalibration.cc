#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"

//class constructor
HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& beCodecConfig){
    
    LSB_ = beCodecConfig.getParameter<double>("cellLSB"); 
    fCperMIP_ee_ = beCodecConfig.getParameter<std::vector<double>>("fCperMIPee");
    fCperMIP_fh_ = beCodecConfig.getParameter<std::vector<double>>("fCperMIPfh");
    dEdX_weights_ = beCodecConfig.getParameter<std::vector<double>>("dEdXweights");
    thickCorr_ = beCodecConfig.getParameter<std::vector<double>>("thickCorr");
    

    for(auto corr : thickCorr_){
        if(corr <= 0){
            edm::LogWarning("DivisionByZero") << "WARNING: the cell-thickness correction factor is zero or negative. It won't be applied to correct trigger cell energies.";
        }

    }
}


void HGCalTriggerCellCalibration::calibrateInMipT(l1t::HGCalTriggerCell& trgCell, int cellThickness)
{
    
    HGCalDetId trgdetid( trgCell.detId() );
    int subdet = trgdetid.subdetId();

    /* get the hardware pT in ADC counts: */
    int hwPt = trgCell.hwPt();

    /* set the lowest signal bit and convert in charge amplitude: */
    double amplitude = hwPt * LSB_;  

    /* convert the charge amplitude in MIP: */
    if( subdet == HGCEE ){ 
        amplitude = amplitude / fCperMIP_ee_.at(cellThickness-1);
    }else if( subdet == HGCHEF ){
        amplitude = amplitude / fCperMIP_fh_.at(cellThickness-1);
    }else if( subdet == HGCHEB ){
        edm::LogWarning("DataNotFound") << "WARNING: the BH trgCells are not yet implemented";
    }

    /* correct the charge amplitude for the sensor thickness */
    double trgCellMipP = amplitude;
    if( thickCorr_.at( cellThickness-1 ) > 0 ){
        trgCellMipP = trgCellMipP / thickCorr_.at( cellThickness-1 ); 
    }
     
    double trgCellMipPt = trgCellMipP/cosh( trgCell.eta() ); 

    /* setting pT [mip] */
    trgCell.setMipPt( trgCellMipPt ) ;
} 


void HGCalTriggerCellCalibration::calibrateMipTinGeV(l1t::HGCalTriggerCell& trgCell )
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
   
    //weight the amplitude by the absorber coefficient in MeV/mip + bring it in GeV
    double trgCellE = mipP * dEdX_weights_.at(trgCellLayer) * MevToGeV;

    //assign the new energy to the four-vector of the trigger cell
    math::PtEtaPhiMLorentzVector calibP4(trgCellE/cosh( trgCell.eta() ), 
                                         trgCell.eta(), 
                                         trgCell.phi(), 
                                         trgCell.p4().M() );
    
    // overwriting the 4p with the calibrated 4p     
    trgCell.setP4( calibP4 );

}

void HGCalTriggerCellCalibration::calibrateInGeV(l1t::HGCalTriggerCell& trgCell, int cellThickness)
{

    /* calibrate from ADC count to transverse mip */
    calibrateInMipT(trgCell, cellThickness);

    /* calibrate from mip count to GeV */
    calibrateMipTinGeV(trgCell);

}
 
