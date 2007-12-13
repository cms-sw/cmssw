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
using namespace std;

PixelROCDACSettings::PixelROCDACSettings(){}


void PixelROCDACSettings::getDACs(vector<unsigned int>& dacs) const
{
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
    dacs.push_back(TempRange_);
    dacs.push_back(WBC_);
    dacs.push_back(ChipContReg_);
}

// Added by Dario
void PixelROCDACSettings::getDACs(std::map<std::string, unsigned int>& dacs) const
{
    dacs.clear();
    dacs["Vdd"        ] = Vdd_        ;
    dacs["Vana"       ] = Vana_       ;     
    dacs["Vsf"        ] = Vsf_        ;      
    dacs["Vcomp"      ] = Vcomp_      ;      
    dacs["Vleak"      ] = Vleak_      ;      
    dacs["VrgPr"      ] = VrgPr_      ;      
    dacs["VwllPr"     ] = VwllPr_     ;     
    dacs["VrgSh"      ] = VrgSh_      ;      
    dacs["VwllSh"     ] = VwllSh_     ;     
    dacs["VHldDel"    ] = VHldDel_    ;    
    dacs["Vtrim"      ] = Vtrim_      ;      
    dacs["VcThr"      ] = VcThr_      ;      
    dacs["VIbiasbus"  ] = VIbias_bus_ ;
    dacs["VIbiassf"   ] = VIbias_sf_  ; 
    dacs["VOffsetOp"  ] = VOffsetOp_  ; 
    dacs["VbiasOp"    ] = VbiasOp_    ;    
    dacs["VOffsetRO"  ] = VOffsetRO_  ; 
    dacs["VIon"       ] = VIon_       ;       
    dacs["VIbiasPH"   ] = VIbias_PH_  ; 
    dacs["VIbiasDAC"  ] = VIbias_DAC_ ;
    dacs["VIbiasroc"  ] = VIbias_roc_ ;
    dacs["VIColOr"    ] = VIColOr_    ;    
    dacs["Vnpix"      ] = Vnpix_      ;      
    dacs["VsumCol"    ] = VsumCol_    ;    
    dacs["Vcal"       ] = Vcal_       ;       
    dacs["CalDel"     ] = CalDel_     ;     
    dacs["TempRange"  ] = TempRange_  ; 
    dacs["WBC"        ] = WBC_        ;
    dacs["ChipContReg"] = ChipContReg_;
}

void PixelROCDACSettings::setDAC(unsigned int dacaddress, unsigned int dacvalue) 
{
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
	        case 27: TempRange_=dacvalue; break;
		case 254: WBC_=dacvalue; break;
		case 253: ChipContReg_=dacvalue; break;
		default: cout<<"DAC Address "<<dacaddress<<" does not exist!"<<endl;
	}

}

void PixelROCDACSettings::writeBinary(ofstream& out) const
{
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
    out << TempRange_;
    out << WBC_;
    out << ChipContReg_;	
}


int PixelROCDACSettings::readBinary(ifstream& in, const PixelROCName& rocid){
    
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
    in.read((char*)&TempRange_,1);
    in.read((char*)&WBC_,1);
    in.read((char*)&ChipContReg_,1);	
    
    return 1;

}

void PixelROCDACSettings::writeASCII(ostream& out) const{

    out << "ROC:           "<<rocid_.rocname()<<endl;

    out << "Vdd:           "<<(int)Vdd_<<endl;
    out << "Vana:          "<<(int)Vana_<<endl;
    out << "Vsf:           "<<(int)Vsf_<<endl;
    out << "Vcomp:         "<<(int)Vcomp_<<endl;
    out << "Vleak:         "<<(int)Vleak_<<endl;
    out << "VrgPr:         "<<(int)VrgPr_<<endl;
    out << "VwllPr:        "<<(int)VwllPr_<<endl;
    out << "VrgSh:         "<<(int)VrgSh_<<endl;
    out << "VwllSh:        "<<(int)VwllSh_<<endl;
    out << "VHldDel:       "<<(int)VHldDel_<<endl;
    out << "Vtrim:         "<<(int)Vtrim_<<endl;
    out << "VcThr:         "<<(int)VcThr_<<endl;
    out << "VIbias_bus:    "<<(int)VIbias_bus_<<endl;
    out << "VIbias_sf:     "<<(int)VIbias_sf_<<endl;
    out << "VOffsetOp:     "<<(int)VOffsetOp_<<endl;
    out << "VbiasOp:       "<<(int)VbiasOp_<<endl;
    out << "VOffsetRO:     "<<(int)VOffsetRO_<<endl;
    out << "VIon:          "<<(int)VIon_<<endl;
    out << "VIbias_PH:     "<<(int)VIbias_PH_<<endl;
    out << "VIbias_DAC:    "<<(int)VIbias_DAC_<<endl;
    out << "VIbias_roc:    "<<(int)VIbias_roc_<<endl;
    out << "VIColOr:       "<<(int)VIColOr_<<endl;
    out << "Vnpix:         "<<(int)Vnpix_<<endl;
    out << "VsumCol:       "<<(int)VsumCol_<<endl;
    out << "Vcal:          "<<(int)Vcal_<<endl;
    out << "CalDel:        "<<(int)CalDel_<<endl;
    out << "TempRange:     "<<(int)TempRange_<<endl;
    out << "WBC:           "<<(int)WBC_<<endl;
    out << "ChipContReg:   "<<(int)ChipContReg_<<endl;	


}

void PixelROCDACSettings::checkTag(string tag, 
				   string dacName,
				   const PixelROCName& rocid){
  
  dacName+=":";
  if (tag!=dacName) {
    cout << "Read ROC name:"<<tag<<endl;
    cout << "But expected to find:"<<dacName<<endl;
    cout << "When reading DAC settings for ROC "<<rocid<<endl;
    assert(0);
  }

}


int PixelROCDACSettings::read(ifstream& in, const PixelROCName& rocid){
    
    rocid_=rocid;

    unsigned int tmp;
    string tag;

    in >> tag; 
    checkTag(tag,"Vdd",rocid);
    in >> tmp; Vdd_=tmp;
    in >> tag; 
    checkTag(tag,"Vana",rocid);
    in >> tmp; Vana_=tmp;
    in >> tag; 
    checkTag(tag,"Vsf",rocid);
    in >> tmp; Vsf_=tmp;
    in >> tag; 
    checkTag(tag,"Vcomp",rocid);
    in >> tmp; Vcomp_=tmp;
    in >> tag; 
    checkTag(tag,"Vleak",rocid);
    in >> tmp; Vleak_=tmp;
    in >> tag; 
    checkTag(tag,"VrgPr",rocid);
    in >> tmp; VrgPr_=tmp;
    in >> tag; 
    checkTag(tag,"VwllPr",rocid);
    in >> tmp; VwllPr_=tmp;
    in >> tag; 
    checkTag(tag,"VrgSh",rocid);
    in >> tmp; VrgSh_=tmp;
    in >> tag; 
    checkTag(tag,"VwllSh",rocid);
    in >> tmp; VwllSh_=tmp;
    in >> tag; 
    checkTag(tag,"VHldDel",rocid);
    in >> tmp; VHldDel_=tmp;
    in >> tag; 
    checkTag(tag,"Vtrim",rocid);
    in >> tmp; Vtrim_=tmp;
    in >> tag; 
    checkTag(tag,"VcThr",rocid);
    in >> tmp; VcThr_=tmp;
    in >> tag; 
    checkTag(tag,"VIbias_bus",rocid);
    in >> tmp; VIbias_bus_=tmp;
    in >> tag; 
    checkTag(tag,"VIbias_sf",rocid);
    in >> tmp; VIbias_sf_=tmp;
    in >> tag; 
    checkTag(tag,"VOffsetOp",rocid);
    in >> tmp; VOffsetOp_=tmp;
    in >> tag; 
    checkTag(tag,"VbiasOp",rocid);
    in >> tmp; VbiasOp_=tmp;
    in >> tag; 
    checkTag(tag,"VOffsetRO",rocid);
    in >> tmp; VOffsetRO_=tmp;
    in >> tag; 
    checkTag(tag,"VIon",rocid);
    in >> tmp; VIon_=tmp;
    in >> tag; 
    checkTag(tag,"VIbias_PH",rocid);
    in >> tmp; VIbias_PH_=tmp;
    in >> tag; 
    checkTag(tag,"VIbias_DAC",rocid);
    in >> tmp; VIbias_DAC_=tmp;
    in >> tag; 
    checkTag(tag,"VIbias_roc",rocid);
    in >> tmp; VIbias_roc_=tmp;
    in >> tag; 
    checkTag(tag,"VIColOr",rocid);
    in >> tmp; VIColOr_=tmp;
    in >> tag; 
    checkTag(tag,"Vnpix",rocid);
    in >> tmp; Vnpix_=tmp;
    in >> tag; 
    checkTag(tag,"VsumCol",rocid);
    in >> tmp; VsumCol_=tmp;
    in >> tag; 
    checkTag(tag,"Vcal",rocid);
    in >> tmp; Vcal_=tmp;
    in >> tag; 
    checkTag(tag,"CalDel",rocid);
    in >> tmp; CalDel_=tmp;
    in >> tag; 
    if (tag=="WBC:"){
      static bool first=true;
      if (first){
	cout << "**********************************************"<<endl;
	cout << "Did not find TempRange setting in DAC settings"<<endl;
	cout << "Will use a default value of 4."<<endl;
	cout << "This message will only be printed out once"<<endl;
	cout << "**********************************************"<<endl;
	TempRange_=4;
	first=false;
      }
      in >> tmp; WBC_=tmp;
    } else {	
      checkTag(tag,"TempRange",rocid);
      in >> tmp; TempRange_=tmp;
      in >> tag; 
      checkTag(tag,"WBC",rocid);
      in >> tmp; WBC_=tmp;
    }
    in >> tag; 
    checkTag(tag,"ChipContReg",rocid);
    in >> tmp; ChipContReg_=tmp;

    return 0;
}


string PixelROCDACSettings::getConfigCommand(){

  string s;

  return s;

}

ostream& pos::operator<<(ostream& s, const PixelROCDACSettings& dacs){
  
  s << "Vdd          :" << (unsigned int)dacs.Vdd_ << endl;
  s << "Vana         :" << (unsigned int)dacs.Vana_ << endl;
  s << "Vsf          :" << (unsigned int)dacs.Vsf_ << endl;
  s << "Vcomp        :" << (unsigned int)dacs.Vcomp_ << endl;
  s << "Vleak        :" << (unsigned int)dacs.Vleak_ << endl;
  s << "VrgPr        :" << (unsigned int)dacs.VrgPr_ << endl;
  s << "VwllPr       :" << (unsigned int)dacs.VwllPr_ << endl;
  s << "VrgSh        :" << (unsigned int)dacs.VrgSh_ << endl;
  s << "VwllSh       :" << (unsigned int)dacs.VwllSh_ << endl;
  s << "VHldDel      :" << (unsigned int)dacs.VHldDel_ << endl;
  s << "Vtrim        :" << (unsigned int)dacs.Vtrim_ << endl;
  s << "VcThr        :" << (unsigned int)dacs.VcThr_ << endl;
  s << "VIbias_bus   :" << (unsigned int)dacs.VIbias_bus_ << endl;
  s << "VIbias_sf    :" << (unsigned int)dacs.VIbias_sf_ << endl;
  s << "VOffsetOp    :" << (unsigned int)dacs.VOffsetOp_ << endl;
  s << "VbiasOp      :" << (unsigned int)dacs.VbiasOp_ << endl;
  s << "VOffsetRO    :" << (unsigned int)dacs.VOffsetRO_ << endl;
  s << "VIon         :" << (unsigned int)dacs.VIon_ << endl;
  s << "VIbias_PH    :" << (unsigned int)dacs.VIbias_PH_ << endl;
  s << "VIbias_DAC   :" << (unsigned int)dacs.VIbias_DAC_ << endl;
  s << "VIbias_roc   :" << (unsigned int)dacs.VIbias_roc_ << endl;
  s << "VIColOr      :" << (unsigned int)dacs.VIColOr_ << endl;
  s << "Vnpix        :" << (unsigned int)dacs.Vnpix_ << endl;
  s << "VsumCol      :" << (unsigned int)dacs.VsumCol_ << endl;
  s << "Vcal         :" << (unsigned int)dacs.Vcal_ << endl;
  s << "CalDel       :" << (unsigned int)dacs.CalDel_ << endl;
  s << "TempRange    :" << (unsigned int)dacs.TempRange_ << endl;
  s << "WBC          :" << (unsigned int)dacs.WBC_ << endl;
  s << "ChipContReg  :" << (unsigned int)dacs.ChipContReg_ << endl;
  
  return s;

}

//Added by Umesh
void PixelROCDACSettings::setDac(string dacName, int dacValue){
  if(dacName == "VDD"){
//     cout << "VDD" << endl;
    Vdd_ = dacValue;
  }
  else if(dacName == "VANA"){
//     cout << "VANA" << endl;
    Vana_ = dacValue;
  }
  else if(dacName == "VSF"){
//     cout << "VSF" << endl;
    Vsf_ = dacValue;
  }
  else if(dacName == "VCOMP"){
//     cout << "VCOMP" << endl;
    Vcomp_ = dacValue;
  }
  else if(dacName == "VLEAK"){
//     cout << "VLEAK" << endl;
    Vleak_ = dacValue;
  }
  else if(dacName == "VRGPR"){
//     cout << "VRGPR" << endl;
    VrgPr_ = dacValue;
  }
  else if(dacName == "VWLLSH"){
//     cout << "VWLLSH" << endl;
    VwllPr_ = dacValue;
  }
  else if(dacName == "VRGSH"){
//     cout << "VRGSH" << endl;
    VrgSh_ = dacValue;
  }
  else if(dacName == "VWLLSH"){
//     cout << "VWLLSH" << endl;
    VwllSh_ = dacValue;
  }
  else if(dacName == "VHLDDEL"){
//     cout << "VHLDDEL" << endl;
    VHldDel_ = dacValue;
  }
  else if(dacName == "VTRIM"){
//     cout << "VTRIM" << endl;
    Vtrim_ = dacValue;
  }
  else if(dacName == "VCTHR"){
//     cout << "VCTHR" << endl;
    VcThr_ = dacValue;
  }
  else if(dacName == "VIBIAS_BUS"){
//     cout << "VIBIAS_BUS" << endl;
    VIbias_bus_ = dacValue;
  }
  else if(dacName == "VIBIAS_SF"){
//     cout << "VIBIAS_SF" << endl;
    VIbias_sf_ = dacValue;
  }
  else if(dacName == "VOFFSETOP"){
//     cout << "VOFFSETOP" << endl;
    VOffsetOp_ = dacValue;
  }
  else if(dacName == "VBIASOP"){
//     cout << "VBIASOP" << endl;
    VbiasOp_ = dacValue;
  }
  else if(dacName == "VOFFSETRO"){
//     cout << "VOFFSETRO" << endl;
    VOffsetRO_ = dacValue;
  }
  else if(dacName == "VION"){
//     cout << "VION" << endl;
    VIon_ = dacValue;
  }
  else if(dacName == "VIBIAS_PH"){
//     cout << "VIBIAS_PH" << endl;
    VIbias_PH_ = dacValue;
  }
  else if(dacName == "VIBIAS_DAC"){
//     cout << "VIBIAS_DAC" << endl;
    VIbias_DAC_ = dacValue;
  }
  else if(dacName == "VIBIAS_ROC"){
//     cout << "VIBIAS_ROC" << endl;
    VIbias_roc_ = dacValue;
  }
  else if(dacName == "VICOLOR"){
//     cout << "VICOLOR" << endl;
    VIColOr_ = dacValue;
  }
  else if(dacName == "VNPIX"){
//     cout << "VNPIX" << endl;
    Vnpix_ = dacValue;
  }
  else if(dacName == "VSUMCOL"){
//     cout << "VSUMCOL" << endl;
    VsumCol_ = dacValue;
  }
  else if(dacName == "VCAL"){
//     cout << "VCAL" << endl;
    Vcal_ = dacValue;
  }
  else if(dacName == "CALDEL"){
//     cout << "CALDEL" << endl;
    CalDel_ = dacValue;
  }
  else if(dacName == "TEMPRANGE"){
//     cout << "TEMPRANGE" << endl;
    TempRange_ = dacValue;
  }
  else if(dacName == "WBC"){
//     cout << "WBC" << endl;
    WBC_ = dacValue;
  }

}
