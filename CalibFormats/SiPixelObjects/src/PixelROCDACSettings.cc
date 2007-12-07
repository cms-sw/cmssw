//
// This class provide the data structure for the
// ROC DAC parameters
//
// At this point I do not see a reason to make an
// abstract layer for this code.
//

#include "CalibFormats/SiPixelObjects/interface/PixelROCDACSettings.h"
#include <fstream>
#include <iostream>

using namespace pos;

PixelROCDACSettings::PixelROCDACSettings(){}


void PixelROCDACSettings::getDACs(std::vector<unsigned int>& dacs) const{

    dacs.clear();
    dacs.push_back(Vdd_);
    dacs.push_back(Vana_);
    dacs.push_back(Vsf_);
    dacs.push_back(Vcomp_);
    dacs.push_back(Vleak_);
    dacs.push_back(VrgPr_);
    dacs.push_back(VwllPr_);
    dacs.push_back(VrgSh_);
    dacs.push_back(VwllSh_);
    dacs.push_back(VHldDel_);
    dacs.push_back(Vtrim_);
    dacs.push_back(VcThr_);
    dacs.push_back(VIbias_bus_);
    dacs.push_back(VIbias_sf_);
    dacs.push_back(VOffsetOp_);
    dacs.push_back(VbiasOp_);
    dacs.push_back(VOffsetRO_);
    dacs.push_back(VIon_);
    dacs.push_back(VIbias_PH_);
    dacs.push_back(VIbias_DAC_);
    dacs.push_back(VIbias_roc_);
    dacs.push_back(VIColOr_);
    dacs.push_back(Vnpix_);
    dacs.push_back(VsumCol_);
    dacs.push_back(Vcal_);
    dacs.push_back(CalDel_);
    dacs.push_back(WBC_);
    dacs.push_back(ChipContReg_);
}

void PixelROCDACSettings::setDAC(unsigned int dacaddress, unsigned int dacvalue) {

	switch (dacaddress) {
		case 1: Vdd_=dacvalue; break;
		case 2: Vana_=dacvalue; break;
		case 3: Vsf_=dacvalue; break;
		case 4: Vcomp_=dacvalue; break;
		case 5: Vleak_=dacvalue; break;
		case 6: VrgPr_=dacvalue; break;
		case 7: VwllPr_=dacvalue; break;
		case 8: VrgSh_=dacvalue; break;
		case 9: VwllSh_=dacvalue; break;
		case 10: VHldDel_=dacvalue; break;
		case 11: Vtrim_=dacvalue; break;
		case 12: VcThr_=dacvalue; break;
		case 13: VIbias_bus_=dacvalue; break;
		case 14: VIbias_sf_=dacvalue; break;
		case 15: VOffsetOp_=dacvalue; break;
		case 16: VbiasOp_=dacvalue; break;
		case 17: VOffsetRO_=dacvalue; break;
		case 18: VIon_=dacvalue; break;
		case 19: VIbias_PH_=dacvalue; break;
		case 20: VIbias_DAC_=dacvalue; break;
		case 21: VIbias_roc_=dacvalue; break;
		case 22: VIColOr_=dacvalue; break;
		case 23: Vnpix_=dacvalue; break;
		case 24: VsumCol_=dacvalue; break;
		case 25: Vcal_=dacvalue; break;
		case 26: CalDel_=dacvalue; break;
		case 254: WBC_=dacvalue; break;
		case 253: ChipContReg_=dacvalue; break;
		default: std::cout<<"DAC Address "<<dacaddress<<" does not exist!"<<std::endl;
	}

}

void PixelROCDACSettings::writeBinary(std::ofstream& out) const{

    out << (char)rocid_.rocname().size();
    out.write(rocid_.rocname().c_str(),rocid_.rocname().size());

    out << Vdd_;
    out << Vana_;
    out << Vsf_;
    out << Vcomp_;
    out << Vleak_;
    out << VrgPr_;
    out << VwllPr_;
    out << VrgSh_;
    out << VwllSh_;
    out << VHldDel_;
    out << Vtrim_;
    out << VcThr_;
    out << VIbias_bus_;
    out << VIbias_sf_;
    out << VOffsetOp_;
    out << VbiasOp_;
    out << VOffsetRO_;
    out << VIon_;
    out << VIbias_PH_;
    out << VIbias_DAC_;
    out << VIbias_roc_;
    out << VIColOr_;
    out << Vnpix_;
    out << VsumCol_;
    out << Vcal_;
    out << CalDel_;
    out << WBC_;
    out << ChipContReg_;	


}


int PixelROCDACSettings::readBinary(std::ifstream& in, const PixelROCName& rocid){
    
    rocid_=rocid;

    in.read((char*)&Vdd_,1);
    in.read((char*)&Vana_,1);
    in.read((char*)&Vsf_,1);
    in.read((char*)&Vcomp_,1);
    in.read((char*)&Vleak_,1);
    in.read((char*)&VrgPr_,1);
    in.read((char*)&VwllPr_,1);
    in.read((char*)&VrgSh_,1);
    in.read((char*)&VwllSh_,1);
    in.read((char*)&VHldDel_,1);
    in.read((char*)&Vtrim_,1);
    in.read((char*)&VcThr_,1);
    in.read((char*)&VIbias_bus_,1);
    in.read((char*)&VIbias_sf_,1);
    in.read((char*)&VOffsetOp_,1);
    in.read((char*)&VbiasOp_,1);
    in.read((char*)&VOffsetRO_,1);
    in.read((char*)&VIon_,1);
    in.read((char*)&VIbias_PH_,1);
    in.read((char*)&VIbias_DAC_,1);
    in.read((char*)&VIbias_roc_,1);
    in.read((char*)&VIColOr_,1);
    in.read((char*)&Vnpix_,1);
    in.read((char*)&VsumCol_,1);
    in.read((char*)&Vcal_,1);
    in.read((char*)&CalDel_,1);
    in.read((char*)&WBC_,1);
    in.read((char*)&ChipContReg_,1);	
    
    return 1;

}

void PixelROCDACSettings::writeASCII(std::ostream& out) const{

    out << "ROC:           "<<rocid_.rocname()<<std::endl;

    out << "Vdd:           "<<(int)Vdd_<<std::endl;
    out << "Vana:          "<<(int)Vana_<<std::endl;
    out << "Vsf:           "<<(int)Vsf_<<std::endl;
    out << "Vcomp:         "<<(int)Vcomp_<<std::endl;
    out << "Vleak:         "<<(int)Vleak_<<std::endl;
    out << "VrgPr:         "<<(int)VrgPr_<<std::endl;
    out << "VwllPr:        "<<(int)VwllPr_<<std::endl;
    out << "VrgSh:         "<<(int)VrgSh_<<std::endl;
    out << "VwllSh:        "<<(int)VwllSh_<<std::endl;
    out << "VHldDel:       "<<(int)VHldDel_<<std::endl;
    out << "Vtrim:         "<<(int)Vtrim_<<std::endl;
    out << "VcThr:         "<<(int)VcThr_<<std::endl;
    out << "VIbias_bus:    "<<(int)VIbias_bus_<<std::endl;
    out << "VIbias_sf:     "<<(int)VIbias_sf_<<std::endl;
    out << "VOffsetOp:     "<<(int)VOffsetOp_<<std::endl;
    out << "VbiasOp:       "<<(int)VbiasOp_<<std::endl;
    out << "VOffsetRO:     "<<(int)VOffsetRO_<<std::endl;
    out << "VIon:          "<<(int)VIon_<<std::endl;
    out << "VIbias_PH:     "<<(int)VIbias_PH_<<std::endl;
    out << "VIbias_DAC:    "<<(int)VIbias_DAC_<<std::endl;
    out << "VIbias_roc:    "<<(int)VIbias_roc_<<std::endl;
    out << "VIColOr:       "<<(int)VIColOr_<<std::endl;
    out << "Vnpix:         "<<(int)Vnpix_<<std::endl;
    out << "VsumCol:       "<<(int)VsumCol_<<std::endl;
    out << "Vcal:          "<<(int)Vcal_<<std::endl;
    out << "CalDel:        "<<(int)CalDel_<<std::endl;
    out << "WBC:           "<<(int)WBC_<<std::endl;
    out << "ChipContReg:   "<<(int)ChipContReg_<<std::endl;	


}


int PixelROCDACSettings::read(std::ifstream& in, const PixelROCName& rocid){
    
    rocid_=rocid;

    unsigned int tmp;

    std::string tag;
    in >> tag >> tmp; Vdd_=tmp;
    in >> tag >> tmp; Vana_=tmp;
    in >> tag >> tmp; Vsf_=tmp;
    in >> tag >> tmp; Vcomp_=tmp;
    in >> tag >> tmp; Vleak_=tmp;
    in >> tag >> tmp; VrgPr_=tmp;
    in >> tag >> tmp; VwllPr_=tmp;
    in >> tag >> tmp; VrgSh_=tmp;
    in >> tag >> tmp; VwllSh_=tmp;
    in >> tag >> tmp; VHldDel_=tmp;
    in >> tag >> tmp; Vtrim_=tmp;
    in >> tag >> tmp; VcThr_=tmp;
    in >> tag >> tmp; VIbias_bus_=tmp;
    in >> tag >> tmp; VIbias_sf_=tmp;
    in >> tag >> tmp; VOffsetOp_=tmp;
    in >> tag >> tmp; VbiasOp_=tmp;
    in >> tag >> tmp; VOffsetRO_=tmp;
    in >> tag >> tmp; VIon_=tmp;
    in >> tag >> tmp; VIbias_PH_=tmp;
    in >> tag >> tmp; VIbias_DAC_=tmp;
    in >> tag >> tmp; VIbias_roc_=tmp;
    in >> tag >> tmp; VIColOr_=tmp;
    in >> tag >> tmp; Vnpix_=tmp;
    in >> tag >> tmp; VsumCol_=tmp;
    in >> tag >> tmp; Vcal_=tmp;
    in >> tag >> tmp; CalDel_=tmp;
    in >> tag >> tmp; WBC_=tmp;
    in >> tag >> tmp; ChipContReg_=tmp;

    return 0;
}


std::string PixelROCDACSettings::getConfigCommand(){

  std::string s;

  return s;

}

std::ostream& pos::operator<<(std::ostream& s, const PixelROCDACSettings& dacs){
  
  s << "Vdd          :" << (unsigned int)dacs.Vdd_ << std::endl;
  s << "Vana         :" << (unsigned int)dacs.Vana_ << std::endl;
  s << "Vsf          :" << (unsigned int)dacs.Vsf_ << std::endl;
  s << "Vcomp        :" << (unsigned int)dacs.Vcomp_ << std::endl;
  s << "Vleak        :" << (unsigned int)dacs.Vleak_ << std::endl;
  s << "VrgPr        :" << (unsigned int)dacs.VrgPr_ << std::endl;
  s << "VwllPr       :" << (unsigned int)dacs.VwllPr_ << std::endl;
  s << "VrgSh        :" << (unsigned int)dacs.VrgSh_ << std::endl;
  s << "VwllSh       :" << (unsigned int)dacs.VwllSh_ << std::endl;
  s << "VHldDel      :" << (unsigned int)dacs.VHldDel_ << std::endl;
  s << "Vtrim        :" << (unsigned int)dacs.Vtrim_ << std::endl;
  s << "VcThr        :" << (unsigned int)dacs.VcThr_ << std::endl;
  s << "VIbias_bus   :" << (unsigned int)dacs.VIbias_bus_ << std::endl;
  s << "VIbias_sf    :" << (unsigned int)dacs.VIbias_sf_ << std::endl;
  s << "VOffsetOp    :" << (unsigned int)dacs.VOffsetOp_ << std::endl;
  s << "VbiasOp      :" << (unsigned int)dacs.VbiasOp_ << std::endl;
  s << "VOffsetRO    :" << (unsigned int)dacs.VOffsetRO_ << std::endl;
  s << "VIon         :" << (unsigned int)dacs.VIon_ << std::endl;
  s << "VIbias_PH    :" << (unsigned int)dacs.VIbias_PH_ << std::endl;
  s << "VIbias_DAC   :" << (unsigned int)dacs.VIbias_DAC_ << std::endl;
  s << "VIbias_roc   :" << (unsigned int)dacs.VIbias_roc_ << std::endl;
  s << "VIColOr      :" << (unsigned int)dacs.VIColOr_ << std::endl;
  s << "Vnpix        :" << (unsigned int)dacs.Vnpix_ << std::endl;
  s << "VsumCol      :" << (unsigned int)dacs.VsumCol_ << std::endl;
  s << "Vcal         :" << (unsigned int)dacs.Vcal_ << std::endl;
  s << "CalDel       :" << (unsigned int)dacs.CalDel_ << std::endl;
  s << "WBC          :" << (unsigned int)dacs.WBC_ << std::endl;
  s << "ChipContReg  :" << (unsigned int)dacs.ChipContReg_ << std::endl;
  
  return s;

}

//Added by Umesh
void PixelROCDACSettings::setDac(std::string dacName, int dacValue){
  if(dacName == "VDD"){
//     std::cout << "VDD" << std::endl;
    Vdd_ = dacValue;
  }
  else if(dacName == "VANA"){
//     std::cout << "VANA" << std::endl;
    Vana_ = dacValue;
  }
  else if(dacName == "VSF"){
//     std::cout << "VSF" << std::endl;
    Vsf_ = dacValue;
  }
  else if(dacName == "VCOMP"){
//     std::cout << "VCOMP" << std::endl;
    Vcomp_ = dacValue;
  }
  else if(dacName == "VLEAK"){
//     std::cout << "VLEAK" << std::endl;
    Vleak_ = dacValue;
  }
  else if(dacName == "VRGPR"){
//     std::cout << "VRGPR" << std::endl;
    VrgPr_ = dacValue;
  }
  else if(dacName == "VWLLSH"){
//     std::cout << "VWLLSH" << std::endl;
    VwllPr_ = dacValue;
  }
  else if(dacName == "VRGSH"){
//     std::cout << "VRGSH" << std::endl;
    VrgSh_ = dacValue;
  }
  else if(dacName == "VWLLSH"){
//     std::cout << "VWLLSH" << std::endl;
    VwllSh_ = dacValue;
  }
  else if(dacName == "VHLDDEL"){
//     std::cout << "VHLDDEL" << std::endl;
    VHldDel_ = dacValue;
  }
  else if(dacName == "VTRIM"){
//     std::cout << "VTRIM" << std::endl;
    Vtrim_ = dacValue;
  }
  else if(dacName == "VCTHR"){
//     std::cout << "VCTHR" << std::endl;
    VcThr_ = dacValue;
  }
  else if(dacName == "VIBIAS_BUS"){
//     std::cout << "VIBIAS_BUS" << std::endl;
    VIbias_bus_ = dacValue;
  }
  else if(dacName == "VIBIAS_SF"){
//     std::cout << "VIBIAS_SF" << std::endl;
    VIbias_sf_ = dacValue;
  }
  else if(dacName == "VOFFSETOP"){
//     std::cout << "VOFFSETOP" << std::endl;
    VOffsetOp_ = dacValue;
  }
  else if(dacName == "VBIASOP"){
//     std::cout << "VBIASOP" << std::endl;
    VbiasOp_ = dacValue;
  }
  else if(dacName == "VOFFSETRO"){
//     std::cout << "VOFFSETRO" << std::endl;
    VOffsetRO_ = dacValue;
  }
  else if(dacName == "VION"){
//     std::cout << "VION" << std::endl;
    VIon_ = dacValue;
  }
  else if(dacName == "VIBIAS_PH"){
//     std::cout << "VIBIAS_PH" << std::endl;
    VIbias_PH_ = dacValue;
  }
  else if(dacName == "VIBIAS_DAC"){
//     std::cout << "VIBIAS_DAC" << std::endl;
    VIbias_DAC_ = dacValue;
  }
  else if(dacName == "VIBIAS_ROC"){
//     std::cout << "VIBIAS_ROC" << std::endl;
    VIbias_roc_ = dacValue;
  }
  else if(dacName == "VICOLOR"){
//     std::cout << "VICOLOR" << std::endl;
    VIColOr_ = dacValue;
  }
  else if(dacName == "VNPIX"){
//     std::cout << "VNPIX" << std::endl;
    Vnpix_ = dacValue;
  }
  else if(dacName == "VSUMCOL"){
//     std::cout << "VSUMCOL" << std::endl;
    VsumCol_ = dacValue;
  }
  else if(dacName == "VCAL"){
//     std::cout << "VCAL" << std::endl;
    Vcal_ = dacValue;
  }
  else if(dacName == "CALDEL"){
//     std::cout << "CALDEL" << std::endl;
    CalDel_ = dacValue;
  }
  else if(dacName == "TEMP_RANGE"){
//     std::cout << "TEMP_RANGE" << std::endl;
    Temp_Range_ = dacValue;
  }
  else if(dacName == "WBC"){
//     std::cout << "WBC" << std::endl;
    WBC_ = dacValue;
  }

}
