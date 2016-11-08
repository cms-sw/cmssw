#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalTriggerCellCalibration.h"
#include "TMath.h"

//class constructor
HGCalTriggerCellCalibration::HGCalTriggerCellCalibration(const edm::ParameterSet& conf){

    const edm::ParameterSet beCodecConfig = conf.getParameterSet("FECodec");
    
    LSB_ = beCodecConfig.getParameter<double>("linLSB");
    trgCellTruncBit_= beCodecConfig.getParameter<uint32_t>("triggerCellTruncationBits");
    fCxMIP_ee_ = beCodecConfig.getParameter<std::vector<double>>("fCxMIPee");
    fCxMIP_fh_ = beCodecConfig.getParameter<std::vector<double>>("fCxMIPfh");
    dEdX_weights_ = beCodecConfig.getParameter<std::vector<double>>("dEdXweights");
    thickCorr_ = beCodecConfig.getParameter<std::vector<double>>("thickCorr");
}

//member function implementation

///ciao

void HGCalTriggerCellCalibration::print(){
    std::cout << LSB_                 << std::endl; 
    std::cout << trgCellTruncBit_     << std::endl; 
    std::cout << fCxMIP_ee_.size()    << std::endl; 
    std::cout << dEdX_weights_.size() << std::endl; 
    std::cout << thickCorr_.size()    << std::endl;
}


l1t::HGCalTriggerCell HGCalTriggerCellCalibration::calibTrgCell(l1t::HGCalTriggerCell& trgCell, const edm::EventSetup& es){
        HGCalDetId trgdetid(trgCell.detId());
        int subdet =  (ForwardSubdetector)trgdetid.subdetId() - 3;
        int trgCellLayer = trgdetid.layer();
        int TrgCellTruncBit_ = (int)trgCellTruncBit_;

        es.get<IdealGeometryRecord>().get("HGCalEESensitive",hgceeGeoHandle_);  
        es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",hgchefGeoHandle_); 
        const HGCalGeometry* geom = nullptr;


        //get the hardware pT in fC:
        int hwPt = trgCell.hwPt();
        //set the lowest signal bit:
        double amplitude = hwPt * LSB_ * TMath::Power(2,TrgCellTruncBit_) ;  
        
        //std::cout << "trunc power " << TMath::Power(2,TrgCellTruncBit_) <<  "  Charge " << amplitude << " fC" <<std::endl;

        if( subdet == 0 ){ 
            std::cout << "Subdetector EE" << std::endl;
            geom = hgceeGeoHandle_.product();     
        }else if( subdet == 1 ){
            std::cout << "Subdetector FH" << std::endl;
            geom = hgchefGeoHandle_.product();
            trgCellLayer = trgCellLayer + 28;
        }else if( subdet == 2 ){
            std::cout << "ATTENTION: the BH trgCells are not yet implemented !! "<< std::endl;
        }
        //std::cout << "subdet = " << subdet << "---- trgcell " << trgdetid << " --> Layer = " << trgCellLayer << std::endl;
        
        const HGCalTopology& topo = geom->topology();
        const HGCalDDDConstants& dddConst = topo.dddConstants();
        int cellThickness = dddConst.waferTypeL((unsigned int)trgdetid.wafer() );        

        if( subdet == 0 ){ 
            //convert the charge amplitude in MIP:
            amplitude = amplitude / fCxMIP_ee_[cellThickness-1];
        }else if( subdet == 1 ){
            //convert the charge amplitude in MIP:
            amplitude = amplitude / fCxMIP_fh_[cellThickness-1];
        }else if( subdet == 2 ){
            std::cout << "ATTENTION: the BH trgCells are not yet implemented !! "<< std::endl;
        }
        
        //weight the amplitude by the absorber coefficient in MeV + bring it in GeV and correct for the sensor thickness
        double trgCell_E = amplitude * dEdX_weights_[trgCellLayer] * 0.001 *  thickCorr_[cellThickness-1];
        /*
        std::cout << "fCxMIP_ee_[cellThickness-1] = "   << fCxMIP_ee_[cellThickness-1]   << "  cellThickness = " << cellThickness << std::endl;
        std::cout << "fCxMIP_fh_[cellThickness-1] = "   << fCxMIP_fh_[cellThickness-1]   << "  cellThickness = " << cellThickness << std::endl;
        std::cout << "dEdX_weights_[trgCellLayer] = " << dEdX_weights_[trgCellLayer] << "  trgCellLayer = "  << trgCellLayer  << std::endl;
        std::cout << "thickCorr_[cellThickness] "       << thickCorr_[cellThickness-1]   << "  cellThickness = " << cellThickness << std::endl; 
        std::cout << "hwPt = " << trgCell.hwPt() << " Amplitude " << amplitude << " MIP --> E = " << trgCell_E << " GeV" <<  std::endl;
        */
        double sinTheta = TMath::Sin(2*TMath::ATan(TMath::Exp(-fabs(trgCell.p4().Eta() ) ) ));
        //assign the new energy to the four-vector of the trigger cell and return the trigger cell
        math::PtEtaPhiMLorentzVector calibP4(trgCell_E*sinTheta, trgCell.p4().Eta(), trgCell.p4().Phi(), trgCell.p4().M() );
        trgCell.setP4(calibP4);
        
        //std::cout << "calibrated E = " << trgCell.p4().E() << std::endl; 
        return trgCell;
} 

//calibrate all the trigger cell in one step
/*
void HGCalTriggerCellCalibration::calibTrgCellCollection( l1t::HGCalTriggerCellBxCollection& trgCellColl, const edm::EventSetup& es){    
    for(unsigned itc = 0; itc < trgCellColl.size(); ++itc){
        std::cout << "HI THERE I AM CALIBRATING EVERYTHING!!"<< std::endl;
        this->calibTrgCell(trgCellColl[itc], es);
    }
}
*/
