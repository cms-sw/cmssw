//
// This class provide the data structure for the
// ROC DAC parameters
//
// At this point I do not see a reason to make an
// abstract layer for this code.
//

#include "CalibFormats/SiPixelObjects/interface/PixelROCDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACNames.h"
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
    dacs[k_DACName_Vdd        ] = Vdd_        ;
    dacs[k_DACName_Vana       ] = Vana_       ;     
    dacs[k_DACName_Vsf        ] = Vsf_        ;      
    dacs[k_DACName_Vcomp      ] = Vcomp_      ;      
    dacs[k_DACName_Vleak      ] = Vleak_      ;      
    dacs[k_DACName_VrgPr      ] = VrgPr_      ;      
    dacs[k_DACName_VwllPr     ] = VwllPr_     ;     
    dacs[k_DACName_VrgSh      ] = VrgSh_      ;      
    dacs[k_DACName_VwllSh     ] = VwllSh_     ;     
    dacs[k_DACName_VHldDel    ] = VHldDel_    ;    
    dacs[k_DACName_Vtrim      ] = Vtrim_      ;      
    dacs[k_DACName_VcThr      ] = VcThr_      ;      
    dacs[k_DACName_VIbias_bus ] = VIbias_bus_ ;
    dacs[k_DACName_VIbias_sf  ] = VIbias_sf_  ; 
    dacs[k_DACName_VOffsetOp  ] = VOffsetOp_  ; 
    dacs[k_DACName_VbiasOp    ] = VbiasOp_    ;    
    dacs[k_DACName_VOffsetRO  ] = VOffsetRO_  ; 
    dacs[k_DACName_VIon       ] = VIon_       ;       
    dacs[k_DACName_VIbias_PH  ] = VIbias_PH_  ; 
    dacs[k_DACName_VIbias_DAC ] = VIbias_DAC_ ;
    dacs[k_DACName_VIbias_roc ] = VIbias_roc_ ;
    dacs[k_DACName_VIColOr    ] = VIColOr_    ;    
    dacs[k_DACName_Vnpix      ] = Vnpix_      ;      
    dacs[k_DACName_VsumCol    ] = VsumCol_    ;    
    dacs[k_DACName_Vcal       ] = Vcal_       ;       
    dacs[k_DACName_CalDel     ] = CalDel_     ;     
    dacs[k_DACName_TempRange  ] = TempRange_  ; 
    dacs[k_DACName_WBC        ] = WBC_        ;
    dacs[k_DACName_ChipContReg] = ChipContReg_;
}

// Added by Dario
void PixelROCDACSettings::setDACs(std::map<std::string, unsigned int>& dacs) 
{
    Vdd_	 = dacs[k_DACName_Vdd	     ] ;
    Vana_	 = dacs[k_DACName_Vana       ] ;    
    Vsf_	 = dacs[k_DACName_Vsf	     ] ;     
    Vcomp_	 = dacs[k_DACName_Vcomp      ] ;     
    Vleak_	 = dacs[k_DACName_Vleak      ] ;     
    VrgPr_	 = dacs[k_DACName_VrgPr      ] ;     
    VwllPr_	 = dacs[k_DACName_VwllPr     ] ;    
    VrgSh_	 = dacs[k_DACName_VrgSh      ] ;     
    VwllSh_	 = dacs[k_DACName_VwllSh     ] ;    
    VHldDel_	 = dacs[k_DACName_VHldDel    ] ;   
    Vtrim_	 = dacs[k_DACName_Vtrim      ] ;     
    VcThr_	 = dacs[k_DACName_VcThr      ] ;     
    VIbias_bus_  = dacs[k_DACName_VIbias_bus ] ;
    VIbias_sf_   = dacs[k_DACName_VIbias_sf  ] ; 
    VOffsetOp_   = dacs[k_DACName_VOffsetOp  ] ; 
    VbiasOp_	 = dacs[k_DACName_VbiasOp    ] ;   
    VOffsetRO_   = dacs[k_DACName_VOffsetRO  ] ; 
    VIon_	 = dacs[k_DACName_VIon       ] ;      
    VIbias_PH_   = dacs[k_DACName_VIbias_PH  ] ; 
    VIbias_DAC_  = dacs[k_DACName_VIbias_DAC ] ;
    VIbias_roc_  = dacs[k_DACName_VIbias_roc ] ;
    VIColOr_	 = dacs[k_DACName_VIColOr    ] ;   
    Vnpix_	 = dacs[k_DACName_Vnpix      ] ;     
    VsumCol_	 = dacs[k_DACName_VsumCol    ] ;   
    Vcal_	 = dacs[k_DACName_Vcal       ] ;      
    CalDel_	 = dacs[k_DACName_CalDel     ] ;    
    TempRange_   = dacs[k_DACName_TempRange  ] ; 
    WBC_	 = dacs[k_DACName_WBC	     ] ;
    ChipContReg_ = dacs[k_DACName_ChipContReg] ;
}

// Added by Dario
void PixelROCDACSettings::compareDACs(std::map<std::string, unsigned int>& dacs, 
                                      std::map<std::string, bool>& changes, 
				      std::map<std::string, unsigned int>& previous) 
{
    changes[k_DACName_Vdd	 ] = false;
    changes[k_DACName_Vana	 ] = false;
    changes[k_DACName_Vsf	 ] = false;
    changes[k_DACName_Vcomp	 ] = false;
    changes[k_DACName_Vleak	 ] = false;
    changes[k_DACName_VrgPr	 ] = false;
    changes[k_DACName_VwllPr	 ] = false;
    changes[k_DACName_VrgSh	 ] = false;
    changes[k_DACName_VwllSh	 ] = false;
    changes[k_DACName_VHldDel	 ] = false;
    changes[k_DACName_Vtrim	 ] = false;
    changes[k_DACName_VcThr	 ] = false;
    changes[k_DACName_VIbias_bus ] = false;
    changes[k_DACName_VIbias_sf  ] = false;
    changes[k_DACName_VOffsetOp  ] = false;
    changes[k_DACName_VbiasOp	 ] = false;
    changes[k_DACName_VOffsetRO  ] = false;
    changes[k_DACName_VIon	 ] = false;
    changes[k_DACName_VIbias_PH  ] = false;
    changes[k_DACName_VIbias_DAC ] = false;
    changes[k_DACName_VIbias_roc ] = false;
    changes[k_DACName_VIColOr	 ] = false;
    changes[k_DACName_Vnpix	 ] = false;
    changes[k_DACName_VsumCol	 ] = false;
    changes[k_DACName_Vcal	 ] = false;
    changes[k_DACName_CalDel	 ] = false;
    changes[k_DACName_TempRange  ] = false;
    changes[k_DACName_WBC	 ] = false;
    changes[k_DACName_ChipContReg] = false;

    if( Vdd_	     != dacs[k_DACName_Vdd	  ] ) {changes[k_DACName_Vdd	    ] = true; previous[k_DACName_Vdd	    ] = Vdd_	    ;} 
    if( Vana_	     != dacs[k_DACName_Vana	  ] ) {changes[k_DACName_Vana	    ] = true; previous[k_DACName_Vana	    ] = Vana_	    ;} 
    if( Vsf_	     != dacs[k_DACName_Vsf	  ] ) {changes[k_DACName_Vsf	    ] = true; previous[k_DACName_Vsf	    ] = Vsf_	    ;}         
    if( Vcomp_       != dacs[k_DACName_Vcomp	  ] ) {changes[k_DACName_Vcomp      ] = true; previous[k_DACName_Vcomp      ] = Vcomp_      ;}         
    if( Vleak_       != dacs[k_DACName_Vleak	  ] ) {changes[k_DACName_Vleak      ] = true; previous[k_DACName_Vleak      ] = Vleak_      ;}         
    if( VrgPr_       != dacs[k_DACName_VrgPr	  ] ) {changes[k_DACName_VrgPr      ] = true; previous[k_DACName_VrgPr      ] = VrgPr_      ;}         
    if( VwllPr_      != dacs[k_DACName_VwllPr	  ] ) {changes[k_DACName_VwllPr     ] = true; previous[k_DACName_VwllPr     ] = VwllPr_     ;}        
    if( VrgSh_       != dacs[k_DACName_VrgSh	  ] ) {changes[k_DACName_VrgSh      ] = true; previous[k_DACName_VrgSh      ] = VrgSh_      ;}         
    if( VwllSh_      != dacs[k_DACName_VwllSh	  ] ) {changes[k_DACName_VwllSh     ] = true; previous[k_DACName_VwllSh     ] = VwllSh_     ;}        
    if( VHldDel_     != dacs[k_DACName_VHldDel    ] ) {changes[k_DACName_VHldDel    ] = true; previous[k_DACName_VHldDel    ] = VHldDel_    ;} 
    if( Vtrim_       != dacs[k_DACName_Vtrim	  ] ) {changes[k_DACName_Vtrim      ] = true; previous[k_DACName_Vtrim      ] = Vtrim_      ;}         
    if( VcThr_       != dacs[k_DACName_VcThr	  ] ) {changes[k_DACName_VcThr      ] = true; previous[k_DACName_VcThr      ] = VcThr_      ;}         
    if( VIbias_bus_  != dacs[k_DACName_VIbias_bus ] ) {changes[k_DACName_VIbias_bus ] = true; previous[k_DACName_VIbias_bus ] = VIbias_bus_ ;} 
    if( VIbias_sf_   != dacs[k_DACName_VIbias_sf  ] ) {changes[k_DACName_VIbias_sf  ] = true; previous[k_DACName_VIbias_sf  ] = VIbias_sf_  ;} 
    if( VOffsetOp_   != dacs[k_DACName_VOffsetOp  ] ) {changes[k_DACName_VOffsetOp  ] = true; previous[k_DACName_VOffsetOp  ] = VOffsetOp_  ;} 
    if( VbiasOp_     != dacs[k_DACName_VbiasOp    ] ) {changes[k_DACName_VbiasOp    ] = true; previous[k_DACName_VbiasOp    ] = VbiasOp_    ;} 
    if( VOffsetRO_   != dacs[k_DACName_VOffsetRO  ] ) {changes[k_DACName_VOffsetRO  ] = true; previous[k_DACName_VOffsetRO  ] = VOffsetRO_  ;} 
    if( VIon_	     != dacs[k_DACName_VIon	  ] ) {changes[k_DACName_VIon	    ] = true; previous[k_DACName_VIon	    ] = VIon_	    ;}  	
    if( VIbias_PH_   != dacs[k_DACName_VIbias_PH  ] ) {changes[k_DACName_VIbias_PH  ] = true; previous[k_DACName_VIbias_PH  ] = VIbias_PH_  ;} 
    if( VIbias_DAC_  != dacs[k_DACName_VIbias_DAC ] ) {changes[k_DACName_VIbias_DAC ] = true; previous[k_DACName_VIbias_DAC ] = VIbias_DAC_ ;} 
    if( VIbias_roc_  != dacs[k_DACName_VIbias_roc ] ) {changes[k_DACName_VIbias_roc ] = true; previous[k_DACName_VIbias_roc ] = VIbias_roc_ ;} 
    if( VIColOr_     != dacs[k_DACName_VIColOr    ] ) {changes[k_DACName_VIColOr    ] = true; previous[k_DACName_VIColOr    ] = VIColOr_    ;} 
    if( Vnpix_       != dacs[k_DACName_Vnpix	  ] ) {changes[k_DACName_Vnpix      ] = true; previous[k_DACName_Vnpix      ] = Vnpix_      ;}         
    if( VsumCol_     != dacs[k_DACName_VsumCol    ] ) {changes[k_DACName_VsumCol    ] = true; previous[k_DACName_VsumCol    ] = VsumCol_    ;} 
    if( Vcal_	     != dacs[k_DACName_Vcal	  ] ) {changes[k_DACName_Vcal	    ] = true; previous[k_DACName_Vcal	    ] = Vcal_	    ;}  	
    if( CalDel_      != dacs[k_DACName_CalDel	  ] ) {changes[k_DACName_CalDel     ] = true; previous[k_DACName_CalDel     ] = CalDel_     ;}        
    if( TempRange_   != dacs[k_DACName_TempRange  ] ) {changes[k_DACName_TempRange  ] = true; previous[k_DACName_TempRange  ] = TempRange_  ;} 
    if( WBC_	     != dacs[k_DACName_WBC	  ] ) {changes[k_DACName_WBC	    ] = true; previous[k_DACName_WBC	    ] = WBC_	    ;} 
    if( ChipContReg_ != dacs[k_DACName_ChipContReg] ) {changes[k_DACName_ChipContReg] = true; previous[k_DACName_ChipContReg] = ChipContReg_;} 
}		   								            
										       
void PixelROCDACSettings::setDAC(unsigned int dacaddress, unsigned int dacvalue)       
{
        std::string mthn = "[PixelROCDACSettings::setDAC()]\t\t\t\t    " ;
	switch (dacaddress) {
		case   1: Vdd_         = dacvalue;  break;
		case   2: Vana_        = dacvalue;  break;
		case   3: Vsf_         = dacvalue;  break;
		case   4: Vcomp_       = dacvalue;  break;
		case   5: Vleak_       = dacvalue;  break;
		case   6: VrgPr_       = dacvalue;  break;
		case   7: VwllPr_      = dacvalue;  break;
		case   8: VrgSh_       = dacvalue;  break;
		case   9: VwllSh_      = dacvalue;  break;
		case  10: VHldDel_     = dacvalue;  break;
		case  11: Vtrim_       = dacvalue;  break;
		case  12: VcThr_       = dacvalue;  break;
		case  13: VIbias_bus_  = dacvalue;  break;
		case  14: VIbias_sf_   = dacvalue;  break;
		case  15: VOffsetOp_   = dacvalue;  break;
		case  16: VbiasOp_     = dacvalue;  break;
		case  17: VOffsetRO_   = dacvalue;  break;
		case  18: VIon_        = dacvalue;  break;
		case  19: VIbias_PH_   = dacvalue;  break;
		case  20: VIbias_DAC_  = dacvalue;  break;
		case  21: VIbias_roc_  = dacvalue;  break;
		case  22: VIColOr_     = dacvalue;  break;
		case  23: Vnpix_       = dacvalue;  break;
		case  24: VsumCol_     = dacvalue;  break;
		case  25: Vcal_        = dacvalue;  break;
		case  26: CalDel_      = dacvalue;  break;
	        case  27: TempRange_   = dacvalue;  break;
		case 254: WBC_         = dacvalue;  break;
		case 253: ChipContReg_ = dacvalue;  break;
		default: cout << __LINE__ << "]\t" << mthn 
		              << "DAC Address " << dacaddress << " does not exist!" << endl;
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

    out << "ROC:           " << rocid_.rocname()   <<endl;

    out << k_DACName_Vdd << ":           " << (int)Vdd_	         <<endl;
    out << k_DACName_Vana << ":          " << (int)Vana_	 <<endl;
    out << k_DACName_Vsf << ":           " << (int)Vsf_	         <<endl;
    out << k_DACName_Vcomp << ":         " << (int)Vcomp_	 <<endl;
    out << k_DACName_Vleak << ":         " << (int)Vleak_	 <<endl;
    out << k_DACName_VrgPr << ":         " << (int)VrgPr_	 <<endl;
    out << k_DACName_VwllPr << ":        " << (int)VwllPr_	 <<endl;
    out << k_DACName_VrgSh << ":         " << (int)VrgSh_	 <<endl;
    out << k_DACName_VwllSh << ":        " << (int)VwllSh_	 <<endl;
    out << k_DACName_VHldDel << ":       " << (int)VHldDel_	 <<endl;
    out << k_DACName_Vtrim << ":         " << (int)Vtrim_	 <<endl;
    out << k_DACName_VcThr << ":         " << (int)VcThr_	 <<endl;
    out << k_DACName_VIbias_bus << ":    " << (int)VIbias_bus_   <<endl;
    out << k_DACName_VIbias_sf << ":     " << (int)VIbias_sf_    <<endl;
    out << k_DACName_VOffsetOp << ":     " << (int)VOffsetOp_    <<endl;
    out << k_DACName_VbiasOp << ":       " << (int)VbiasOp_	 <<endl;
    out << k_DACName_VOffsetRO << ":     " << (int)VOffsetRO_    <<endl;
    out << k_DACName_VIon << ":          " << (int)VIon_	 <<endl;
    out << k_DACName_VIbias_PH << ":     " << (int)VIbias_PH_    <<endl;
    out << k_DACName_VIbias_DAC << ":    " << (int)VIbias_DAC_   <<endl;
    out << k_DACName_VIbias_roc << ":    " << (int)VIbias_roc_   <<endl;
    out << k_DACName_VIColOr << ":       " << (int)VIColOr_	 <<endl;
    out << k_DACName_Vnpix << ":         " << (int)Vnpix_	 <<endl;
    out << k_DACName_VsumCol << ":       " << (int)VsumCol_      <<endl;
    out << k_DACName_Vcal << ":          " << (int)Vcal_	 <<endl;
    out << k_DACName_CalDel << ":        " << (int)CalDel_	 <<endl;
    out << k_DACName_TempRange << ":     " << (int)TempRange_    <<endl;
    out << k_DACName_WBC << ":           " << (int)WBC_	         <<endl;
    out << k_DACName_ChipContReg << ":   " << (int)ChipContReg_  <<endl;

}

//=============================================================================================
void PixelROCDACSettings::writeXML(ofstream *out) const {
  std::string mthn = "[PixelROCDACSettings::writeXML()]\t\t\t    " ;

  *out << "  <DATA>"                                                  << endl ;
  *out << "   <ROC_NAME>"    << rocid_.rocname()  << "</ROC_NAME>"    << endl ;
  *out << "   <VDD>"	     << (int)Vdd_	  << "</VDD>"	      << endl ;
  *out << "   <VANA>"	     << (int)Vana_	  << "</VANA>"        << endl ;
  *out << "   <VSF>"	     << (int)Vsf_	  << "</VSF>"	      << endl ;
  *out << "   <VCOMP>"       << (int)Vcomp_	  << "</VCOMP>"       << endl ;
  *out << "   <VLEAK>"       << (int)Vleak_	  << "</VLEAK>"       << endl ;
  *out << "   <VRGPR>"       << (int)VrgPr_	  << "</VRGPR>"       << endl ;
  *out << "   <VWLLPR>"      << (int)VwllPr_	  << "</VWLLPR>"      << endl ;
  *out << "   <VRGSH>"       << (int)VrgSh_	  << "</VRGSH>"       << endl ;
  *out << "   <VWLLSH>"      << (int)VwllSh_	  << "</VWLLSH>"      << endl ;
  *out << "   <VHLDDEL>"     << (int)VHldDel_	  << "</VHLDDEL>"     << endl ;
  *out << "   <VTRIM>"       << (int)Vtrim_	  << "</VTRIM>"       << endl ;
  *out << "   <VCTHR>"       << (int)VcThr_	  << "</VCTHR>"       << endl ;
  *out << "   <VIBIAS_BUS>"  << (int)VIbias_bus_  << "</VIBIAS_BUS>"  << endl ;
  *out << "   <VIBIAS_SF>"   << (int)VIbias_sf_   << "</VIBIAS_SF>"   << endl ;
  *out << "   <VOFFSETOP>"   << (int)VOffsetOp_   << "</VOFFSETOP>"   << endl ;
  *out << "   <VBIASOP>"     << (int)VbiasOp_	  << "</VBIASOP>"     << endl ;
  *out << "   <VOFFSETRO>"   << (int)VOffsetRO_   << "</VOFFSETRO>"   << endl ;
  *out << "   <VION>"	     << (int)VIon_	  << "</VION>"        << endl ;
  *out << "   <VIBIAS_PH>"   << (int)VIbias_PH_   << "</VIBIAS_PH>"   << endl ;
  *out << "   <VIBIAS_DAC>"  << (int)VIbias_DAC_  << "</VIBIAS_DAC>"  << endl ;
  *out << "   <VIBIAS_ROC>"  << (int)VIbias_roc_  << "</VIBIAS_ROC>"  << endl ;
  *out << "   <VICOLOR>"     << (int)VIColOr_	  << "</VICOLOR>"     << endl ;
  *out << "   <VNPIX>"       << (int)Vnpix_	  << "</VNPIX>"       << endl ;
  *out << "   <VSUMCOL>"     << (int)VsumCol_	  << "</VSUMCOL>"     << endl ;
  *out << "   <VCAL>"	     << (int)Vcal_	  << "</VCAL>"        << endl ;
  *out << "   <CALDEL>"      << (int)CalDel_	  << "</CALDEL>"      << endl ;
  *out << "   <TEMPRANGE>"   << (int)TempRange_   << "</TEMPRANGE>"   << endl ;
  *out << "   <WBC>"	     << (int)WBC_	  << "</WBC>"	      << endl ;
  *out << "   <CHIPCONTREG>" << (int)ChipContReg_ << "</CHIPCONTREG>" << endl ;
  *out << "  </DATA>"						      << endl ;
  *out << " "							      << endl ;

}

//=============================================================================================
void PixelROCDACSettings::checkTag(string tag, 
				   string dacName,
				   const PixelROCName& rocid){
  
  std::string mthn = "[PixelROCDACSettings::checkTag()]\t\t\t\t    " ;
  dacName+=":";
  if (tag!=dacName) {
    cout << __LINE__ << "]\t" << mthn << "Read ROC name       : "	      << tag     << endl;
    cout << __LINE__ << "]\t" << mthn << "But expected to find: "	      << dacName << endl;
    cout << __LINE__ << "]\t" << mthn << "When reading DAC settings for ROC " << rocid   << endl;
    assert(0);
  }

}

int PixelROCDACSettings::read(std::istringstream& in, const PixelROCName& rocid)
{
    std::string mthn = "[PixelROCDACSettings::read()]\t\t\t\t    " ;

    rocid_=rocid;

    unsigned int tmp;
    string tag;

    //    cout << "[PixelROCDACSettings::read()] |" << in.str() << "|" << endl ;
    in >> tag; 
    checkTag(tag,k_DACName_Vdd,rocid);
    in >> tmp; Vdd_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vana,rocid);
    in >> tmp; Vana_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vsf,rocid);
    in >> tmp; Vsf_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vcomp,rocid);
    in >> tmp; Vcomp_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vleak,rocid);
    in >> tmp; Vleak_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VrgPr,rocid);
    in >> tmp; VrgPr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VwllPr,rocid);
    in >> tmp; VwllPr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VrgSh,rocid);
    in >> tmp; VrgSh_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VwllSh,rocid);
    in >> tmp; VwllSh_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VHldDel,rocid);
    in >> tmp; VHldDel_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vtrim,rocid);
    in >> tmp; Vtrim_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VcThr,rocid);
    in >> tmp; VcThr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_bus,rocid);
    in >> tmp; VIbias_bus_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_sf,rocid);
    in >> tmp; VIbias_sf_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VOffsetOp,rocid);
    in >> tmp; VOffsetOp_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VbiasOp,rocid);
    in >> tmp; VbiasOp_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VOffsetRO,rocid);
    in >> tmp; VOffsetRO_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIon,rocid);
    in >> tmp; VIon_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_PH,rocid);
    in >> tmp; VIbias_PH_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_DAC,rocid);
    in >> tmp; VIbias_DAC_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_roc,rocid);
    in >> tmp; VIbias_roc_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIColOr,rocid);
    in >> tmp; VIColOr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vnpix,rocid);
    in >> tmp; Vnpix_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VsumCol,rocid);
    in >> tmp; VsumCol_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vcal,rocid);
    in >> tmp; Vcal_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_CalDel,rocid);
    in >> tmp; CalDel_=tmp;
    in >> tag; 
    if (tag==k_DACName_WBC+":"){
      static bool first=true;
      if (first){
	cout << __LINE__ << "]\t" << mthn << "**********************************************" << endl;
	cout << __LINE__ << "]\t" << mthn << "Did not find TempRange setting in DAC settings" << endl;
	cout << __LINE__ << "]\t" << mthn << "Will use a default value of 4."                 << endl;
	cout << __LINE__ << "]\t" << mthn << "This message will only be printed out once"     << endl;
	cout << __LINE__ << "]\t" << mthn << "**********************************************" << endl;
	TempRange_=4;
	first=false;
      }
      in >> tmp; WBC_=tmp;
    } else {	
      checkTag(tag,k_DACName_TempRange,rocid);
      in >> tmp; TempRange_=tmp;
      in >> tag; 
      checkTag(tag,k_DACName_WBC,rocid);
      in >> tmp; WBC_=tmp;
    }
    in >> tag; 
    checkTag(tag,k_DACName_ChipContReg,rocid);
    in >> tmp; ChipContReg_=tmp;

    return 0;
}

int PixelROCDACSettings::read(ifstream& in, const PixelROCName& rocid){
    
    std::string mthn = "[PixelROCDACSettings::read()]\t\t\t\t    " ;
    rocid_=rocid;

    unsigned int tmp;
    string tag;

    in >> tag; 
    checkTag(tag,k_DACName_Vdd,rocid);
    in >> tmp; Vdd_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vana,rocid);
    in >> tmp; Vana_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vsf,rocid);
    in >> tmp; Vsf_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vcomp,rocid);
    in >> tmp; Vcomp_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vleak,rocid);
    in >> tmp; Vleak_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VrgPr,rocid);
    in >> tmp; VrgPr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VwllPr,rocid);
    in >> tmp; VwllPr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VrgSh,rocid);
    in >> tmp; VrgSh_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VwllSh,rocid);
    in >> tmp; VwllSh_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VHldDel,rocid);
    in >> tmp; VHldDel_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vtrim,rocid);
    in >> tmp; Vtrim_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VcThr,rocid);
    in >> tmp; VcThr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_bus,rocid);
    in >> tmp; VIbias_bus_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_sf,rocid);
    in >> tmp; VIbias_sf_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VOffsetOp,rocid);
    in >> tmp; VOffsetOp_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VbiasOp,rocid);
    in >> tmp; VbiasOp_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VOffsetRO,rocid);
    in >> tmp; VOffsetRO_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIon,rocid);
    in >> tmp; VIon_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_PH,rocid);
    in >> tmp; VIbias_PH_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_DAC,rocid);
    in >> tmp; VIbias_DAC_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIbias_roc,rocid);
    in >> tmp; VIbias_roc_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VIColOr,rocid);
    in >> tmp; VIColOr_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vnpix,rocid);
    in >> tmp; Vnpix_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_VsumCol,rocid);
    in >> tmp; VsumCol_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_Vcal,rocid);
    in >> tmp; Vcal_=tmp;
    in >> tag; 
    checkTag(tag,k_DACName_CalDel,rocid);
    in >> tmp; CalDel_=tmp;
    in >> tag; 
    if (tag==k_DACName_WBC+":"){
      static bool first=true;
      if (first){
	cout << __LINE__ << "]\t" << mthn << "**********************************************" << endl;
	cout << __LINE__ << "]\t" << mthn << "Did not find TempRange setting in DAC settings" << endl;
	cout << __LINE__ << "]\t" << mthn << "Will use a default value of 4."                 << endl;
	cout << __LINE__ << "]\t" << mthn << "This message will only be printed out once"     << endl;
	cout << __LINE__ << "]\t" << mthn << "**********************************************" << endl;
	TempRange_=4;
	first=false;
      }
      in >> tmp; WBC_=tmp;
    } else {	
      checkTag(tag,k_DACName_TempRange,rocid);
      in >> tmp; TempRange_=tmp;
      in >> tag; 
      checkTag(tag,k_DACName_WBC,rocid);
      in >> tmp; WBC_=tmp;
    }
    in >> tag; 
    checkTag(tag,k_DACName_ChipContReg,rocid);
    in >> tmp; ChipContReg_=tmp;

    return 0;
}


string PixelROCDACSettings::getConfigCommand(){

  string s;

  return s;

}

ostream& pos::operator<<(ostream& s, const PixelROCDACSettings& dacs){
  
  s << k_DACName_Vdd << "          :" << (unsigned int)dacs.Vdd_ << endl;
  s << k_DACName_Vana << "         :" << (unsigned int)dacs.Vana_ << endl;
  s << k_DACName_Vsf << "          :" << (unsigned int)dacs.Vsf_ << endl;
  s << k_DACName_Vcomp << "        :" << (unsigned int)dacs.Vcomp_ << endl;
  s << k_DACName_Vleak << "        :" << (unsigned int)dacs.Vleak_ << endl;
  s << k_DACName_VrgPr << "        :" << (unsigned int)dacs.VrgPr_ << endl;
  s << k_DACName_VwllPr << "       :" << (unsigned int)dacs.VwllPr_ << endl;
  s << k_DACName_VrgSh << "        :" << (unsigned int)dacs.VrgSh_ << endl;
  s << k_DACName_VwllSh << "       :" << (unsigned int)dacs.VwllSh_ << endl;
  s << k_DACName_VHldDel << "      :" << (unsigned int)dacs.VHldDel_ << endl;
  s << k_DACName_Vtrim << "        :" << (unsigned int)dacs.Vtrim_ << endl;
  s << k_DACName_VcThr << "        :" << (unsigned int)dacs.VcThr_ << endl;
  s << k_DACName_VIbias_bus << "   :" << (unsigned int)dacs.VIbias_bus_ << endl;
  s << k_DACName_VIbias_sf << "    :" << (unsigned int)dacs.VIbias_sf_ << endl;
  s << k_DACName_VOffsetOp << "    :" << (unsigned int)dacs.VOffsetOp_ << endl;
  s << k_DACName_VbiasOp << "      :" << (unsigned int)dacs.VbiasOp_ << endl;
  s << k_DACName_VOffsetRO << "    :" << (unsigned int)dacs.VOffsetRO_ << endl;
  s << k_DACName_VIon << "         :" << (unsigned int)dacs.VIon_ << endl;
  s << k_DACName_VIbias_PH << "    :" << (unsigned int)dacs.VIbias_PH_ << endl;
  s << k_DACName_VIbias_DAC << "   :" << (unsigned int)dacs.VIbias_DAC_ << endl;
  s << k_DACName_VIbias_roc << "   :" << (unsigned int)dacs.VIbias_roc_ << endl;
  s << k_DACName_VIColOr << "      :" << (unsigned int)dacs.VIColOr_ << endl;
  s << k_DACName_Vnpix << "        :" << (unsigned int)dacs.Vnpix_ << endl;
  s << k_DACName_VsumCol << "      :" << (unsigned int)dacs.VsumCol_ << endl;
  s << k_DACName_Vcal << "         :" << (unsigned int)dacs.Vcal_ << endl;
  s << k_DACName_CalDel << "       :" << (unsigned int)dacs.CalDel_ << endl;
  s << k_DACName_TempRange << "    :" << (unsigned int)dacs.TempRange_ << endl;
  s << k_DACName_WBC << "          :" << (unsigned int)dacs.WBC_ << endl;
  s << k_DACName_ChipContReg << "  :" << (unsigned int)dacs.ChipContReg_ << endl;
  
  return s;

}

//Added by Umesh
void PixelROCDACSettings::setDac(string dacName, int dacValue){
  if(ToLower(dacName) == ToLower(k_DACName_Vdd)){
    Vdd_ = dacValue;
  }
  else if(ToLower(dacName) == ToLower(k_DACName_Vana)){
    Vana_ = dacValue;
  }
  else if(ToLower(dacName) == ToLower(k_DACName_Vsf)){
    Vsf_ = dacValue;
  }
  else if(ToLower(dacName) == ToLower(k_DACName_Vcomp)){
    Vcomp_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_Vleak)){
    Vleak_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VrgPr)){
    VrgPr_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VwllPr)){
    VwllPr_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VrgSh)){
    VrgSh_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VwllSh)){
    VwllSh_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VHldDel)){
    VHldDel_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_Vtrim)){
    Vtrim_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VcThr)){
    VcThr_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIbias_bus)){
    VIbias_bus_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIbias_sf)){
    VIbias_sf_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VOffsetOp)){
    VOffsetOp_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VbiasOp)){
    VbiasOp_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VOffsetRO)){
    VOffsetRO_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIon)){
    VIon_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIbias_PH)){
    VIbias_PH_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIbias_DAC)){
    VIbias_DAC_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIbias_roc)){
    VIbias_roc_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VIColOr)){
    VIColOr_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_Vnpix)){;
    Vnpix_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_VsumCol)){
    VsumCol_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_Vcal)){
    Vcal_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_CalDel)){
    CalDel_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_TempRange)){
    TempRange_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_WBC)){
    WBC_ = dacValue;
  }
  else if(ToLower(dacName)==ToLower(k_DACName_ChipContReg)){
    ChipContReg_ = dacValue;
  }
  else
  {
    cout << "ERROR in PixelROCDACSettings::setDac: DAC name " << dacName << " does not exist." << endl;
    assert(0);
  }

}

unsigned int PixelROCDACSettings::getDac(string dacName) const {
  
  if(dacName == k_DACName_Vdd){
    return Vdd_;
  }
  else if(dacName == k_DACName_Vana){
    return Vana_;
  }
  else if(dacName == k_DACName_Vsf){
    return Vsf_;
  }
  else if(dacName == k_DACName_Vcomp){
    return Vcomp_;
  }
  else if(dacName == k_DACName_Vleak){
    return Vleak_;
  }
  else if(dacName == k_DACName_VrgPr){
    return VrgPr_;
  }
  else if(dacName == k_DACName_VwllPr){
    return VwllPr_;
  }
  else if(dacName == k_DACName_VrgSh){
    return VrgSh_;
  }
  else if(dacName == k_DACName_VwllSh){
    return VwllSh_;
  }
  else if(dacName == k_DACName_VHldDel){
    return VHldDel_;
  }
  else if(dacName == k_DACName_Vtrim){
    return Vtrim_;
  }
  else if(dacName == k_DACName_VcThr){
    return VcThr_;
  }
  else if(dacName == k_DACName_VIbias_bus){
    return VIbias_bus_;
  }
  else if(dacName == k_DACName_VIbias_sf){
    return VIbias_sf_;
  }
  else if(dacName == k_DACName_VOffsetOp){
    return VOffsetOp_;
  }
  else if(dacName == k_DACName_VbiasOp){
    return VbiasOp_;
  }
  else if(dacName == k_DACName_VOffsetRO){
    return VOffsetRO_;
  }
  else if(dacName == k_DACName_VIon){
    return VIon_;
  }
  else if(dacName == k_DACName_VIbias_PH){
    return VIbias_PH_;
  }
  else if(dacName == k_DACName_VIbias_DAC){
    return VIbias_DAC_;
  }
  else if(dacName == k_DACName_VIbias_roc){
    return VIbias_roc_;
  }
  else if(dacName == k_DACName_VIColOr){
    return VIColOr_;
  }
  else if(dacName == k_DACName_Vnpix){
    return Vnpix_;
  }
  else if(dacName == k_DACName_VsumCol){
    return VsumCol_;
  }
  else if(dacName == k_DACName_Vcal){
    return Vcal_;
  }
  else if(dacName == k_DACName_CalDel){
    return CalDel_;
  }
  else if(dacName == k_DACName_TempRange){
    return TempRange_;
  }
  else if(dacName == k_DACName_WBC){
    return WBC_;
  }
  else if(dacName == k_DACName_ChipContReg){
    return ChipContReg_;
  }
  else {
    cout << "ERROR in PixelROCDACSettings::getDac: DAC name " << dacName << " does not exist." << endl;
    assert(0);
  }
}			  


string PixelROCDACSettings::ToLower(string generic)
{
  string result ;
  for(unsigned int i = 0; i < generic.length() ; i++)
    {
      result.append(1,(char)tolower(generic[i]) );
    }
  return result ;
}
