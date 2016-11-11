#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"
#include "TMath.h"

//class constructor
HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& conf){
    
    const edm::ParameterSet beCodecConfig = conf.getParameterSet("calib_constant");

    LSB = beCodecConfig.getParameter<double>("linLSB");
    trgCellTruncBit = beCodecConfig.getParameter<uint32_t>("triggerCellTruncationBits");
    fCperMIP_ee = beCodecConfig.getParameter<std::vector<double>>("fCperMIPee");
    fCperMIP_fh = beCodecConfig.getParameter<std::vector<double>>("fCperMIPfh");
    dEdX_weights = beCodecConfig.getParameter<std::vector<double>>("dEdXweights");
    thickCorr = beCodecConfig.getParameter<std::vector<double>>("thickCorr");
}

//member function implementation

///ciao

void HGCalTriggerCellCalibration::print(){
    std::cout << LSB                 << std::endl; 
    std::cout << trgCellTruncBit     << std::endl; 
    std::cout << fCperMIP_ee.size()  << std::endl; 
    std::cout << fCperMIP_fh.size()  << std::endl; 
    std::cout << dEdX_weights.size() << std::endl; 
    std::cout << thickCorr.size()    << std::endl;
}


l1t::HGCalTriggerCell HGCalTriggerCellCalibration::calibrate(l1t::HGCalTriggerCell& trgCell, const edm::EventSetup& es){
        HGCalDetId trgdetid(trgCell.detId());
        int subdet =  (ForwardSubdetector)trgdetid.subdetId() - 3;
        int trgCellLayer = trgdetid.layer();
        int TrgCellTruncBit = (int)trgCellTruncBit;

        es.get<IdealGeometryRecord>().get("HGCalEESensitive",hgceeGeoHandle);  
        es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",hgchefGeoHandle); 
        const HGCalGeometry* geom = nullptr;


        //get the hardware pT in fC:
        int hwPt = trgCell.hwPt();
        //set the lowest signal bit:
        double amplitude = hwPt * LSB * TMath::Power(2,TrgCellTruncBit) ;  
        
        //std::cout << "trunc power " << TMath::Power(2,TrgCellTruncBit_) <<  "  Charge " << amplitude << " fC" <<std::endl;

        if( subdet == 0 ){ 
            geom = hgceeGeoHandle.product();     
        }else if( subdet == 1 ){
            geom = hgchefGeoHandle.product();
            trgCellLayer = trgCellLayer + 28;
        }else if( subdet == 2 ){
            //std::cout << "ATTENTION: the BH trgCells are not yet implemented !! "<< std::endl;
        }
        //std::cout << "subdet = " << subdet << "---- trgcell " << trgdetid << " --> Layer = " << trgCellLayer << std::endl;
        
        const HGCalTopology& topo = geom->topology();
        const HGCalDDDConstants& dddConst = topo.dddConstants();
        int cellThickness = dddConst.waferTypeL((unsigned int)trgdetid.wafer() );        

        if( subdet == 0 ){ 
            //convert the charge amplitude in MIP:
            amplitude = amplitude / fCperMIP_ee[cellThickness-1];
        }else if( subdet == 1 ){
            //convert the charge amplitude in MIP:
            amplitude = amplitude / fCperMIP_fh[cellThickness-1];
        }else if( subdet == 2 ){
            //std::cout << "ATTENTION: the BH trgCells are not yet implemented !! "<< std::endl;
        }
        
        //weight the amplitude by the absorber coefficient in MeV + bring it in GeV and correct for the sensor thickness
        double trgCell_E = amplitude * dEdX_weights[trgCellLayer] * 0.001 *  thickCorr[cellThickness-1];
        /*
        std::cout << "fCxMIP_ee_[cellThickness-1] = "   << fCxMIP_ee_[cellThickness-1]   << "  cellThickness = " << cellThickness << std::endl;
        std::cout << "fCxMIP_fh_[cellThickness-1] = "   << fCxMIP_fh_[cellThickness-1]   << "  cellThickness = " << cellThickness << std::endl;
        std::cout << "dEdX_weights_[trgCellLayer] = " << dEdX_weights_[trgCellLayer] << "  trgCellLayer = "  << trgCellLayer  << std::endl;
        std::cout << "thickCorr_[cellThickness] "       << thickCorr_[cellThickness-1]   << "  cellThickness = " << cellThickness << std::endl; 
        std::cout << "hwPt = " << trgCell.hwPt() << " Amplitude " << amplitude << " MIP --> E = " << trgCell_E << " GeV" <<  std::endl;
        */
        //double sinTheta = TMath::Sin(2*TMath::ATan(TMath::Exp(-fabs( ) ) ));
        //assign the new energy to the four-vector of the trigger cell and return the trigger cell
        math::PtEtaPhiMLorentzVector calibP4(trgCell_E/cosh(trgCell.eta()), trgCell.eta(), trgCell.phi(), trgCell.p4().M() );
        trgCell.setP4(calibP4);
        
        //std::cout << "calibrated E = " << trgCell.p4().E() << std::endl; 
        return trgCell;
} 
