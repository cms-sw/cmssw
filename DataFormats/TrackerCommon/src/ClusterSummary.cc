#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ClusterSummary::ClusterSummary() :
  nModules_tmp  (100,0),
  clusSize_tmp  (100,0),
  clusCharge_tmp(100,0)
{}

ClusterSummary::~ClusterSummary()
{}

// copy ctor
ClusterSummary::ClusterSummary(const ClusterSummary& src) :
    modules       (src.modules       ),
    nModules      (src.nModules      ),
    clusSize      (src.clusSize      ),
    clusCharge    (src.clusCharge    ),
    nModules_tmp  (src.nModules_tmp  ),
    clusSize_tmp  (src.clusSize_tmp  ),
    clusCharge_tmp(src.clusCharge_tmp)
{}

// copy assingment operator
ClusterSummary& ClusterSummary::operator=(const ClusterSummary& rhs)
{
  modules       = rhs.modules       ;
  nModules      = rhs.nModules      ;
  clusSize      = rhs.clusSize      ;
  clusCharge    = rhs.clusCharge    ;
  nModules_tmp  = rhs.nModules_tmp  ;
  clusSize_tmp  = rhs.clusSize_tmp  ;
  clusCharge_tmp= rhs.clusCharge_tmp;
  return *this;
}

// move ctor
ClusterSummary::ClusterSummary(ClusterSummary&& other) : ClusterSummary()
{
    *this = other;
}


bool ClusterSummary::checkSubDet(const int input){

  switch (input){
  case TRACKER :
  case TIB    :  case TID    :   case TIDMR_2:  case TECM_5 :  case TECP_8 :  case TECPR_4:
  case TIB_1  :  case TIDM   :   case TIDMR_3:  case TECM_6 :  case TECP_9 :  case TECPR_5:
  case TIB_2  :  case TIDP   :   case TIDPR_1:  case TECM_7 :  case TECMR_1:  case TECPR_6:
  case TIB_3  :  case TIDM_1 :   case TIDPR_2:  case TECM_8 :  case TECMR_2:  case TECPR_7:
  case TIB_4  :  case TIDM_2 :   case TIDPR_3:  case TECM_9 :  case TECMR_3:
  case TOB    :  case TIDM_3 :   case TEC    :  case TECP_1 :  case TECMR_4:
  case TOB_1  :  case TIDP_1 :   case TECM   :  case TECP_2 :  case TECMR_5:
  case TOB_2  :  case TIDP_2 :   case TECP   :  case TECP_3 :  case TECMR_6:
  case TOB_3  :  case TIDP_3 :   case TECM_1 :  case TECP_4 :  case TECMR_7:
  case TOB_4  :  case TIDMR_1:   case TECM_2 :  case TECP_5 :  case TECPR_1:
  case TOB_5  :                  case TECM_3 :  case TECP_6 :  case TECPR_2:
  case TOB_6  :                  case TECM_4 :  case TECP_7 :  case TECPR_3:
    return false;
  case PIXEL   : case FPIXP_3 :
  case FPIX    : case BPIX    :
  case FPIX_1  : case BPIX_1  :
  case FPIX_2  : case BPIX_2  :
  case FPIX_3  : case BPIX_3  :
  case FPIXM   :
  case FPIXP   :
  case FPIXM_1 :
  case FPIXM_2 :
  case FPIXM_3 :
  case FPIXP_1 :
  case FPIXP_2 :
    return true;
  default:
    throw cms::Exception("ClusterSummary::checkSubDet")  << "Invalid detector: " << input;
  }

  return false;

}

std::string ClusterSummary::getSubDetName(const CMSTracker subdet){
  switch (subdet){
  case TRACKER : return "TRACKER"  ; case TIDM    : return "TIDM"     ; case TEC     : return "TEC"      ; case TECP_3  : return "TECP_3"   ;  case TECPR_1 : return "TECPR_1"  ; case FPIXM_1 : return "FPIXM_1"  ;
  case TIB     : return "TIB"      ; case TIDP    : return "TIDP"     ; case TECM    : return "TECM"     ; case TECP_4  : return "TECP_4"   ;  case TECPR_2 : return "TECPR_2"  ; case FPIXM_2 : return "FPIXM_2"  ;
  case TIB_1   : return "TIB_1"    ; case TIDM_1  : return "TIDM_1"   ; case TECP    : return "TECP"     ; case TECP_5  : return "TECP_5"   ;  case TECPR_3 : return "TECPR_3"  ; case FPIXM_3 : return "FPIXM_3"  ;
  case TIB_2   : return "TIB_2"    ; case TIDM_2  : return "TIDM_2"   ; case TECM_1  : return "TECM_1"   ; case TECP_6  : return "TECP_6"   ;  case TECPR_4 : return "TECPR_4"  ; case FPIXP_1 : return "FPIXP_1"  ;
  case TIB_3   : return "TIB_3"    ; case TIDM_3  : return "TIDM_3"   ; case TECM_2  : return "TECM_2"   ; case TECP_7  : return "TECP_7"   ;  case TECPR_5 : return "TECPR_5"  ; case FPIXP_2 : return "FPIXP_2"  ;
  case TIB_4   : return "TIB_4"    ; case TIDP_1  : return "TIDP_1"   ; case TECM_3  : return "TECM_3"   ; case TECP_8  : return "TECP_8"   ;  case TECPR_6 : return "TECPR_6"  ; case FPIXP_3 : return "FPIXP_3"  ;
  case TOB     : return "TOB"      ; case TIDP_2  : return "TIDP_2"   ; case TECM_4  : return "TECM_4"   ; case TECP_9  : return "TECP_9"   ;  case TECPR_7 : return "TECPR_7"  ; case BPIX    : return "BPIX"     ;
  case TOB_1   : return "TOB_1"    ; case TIDP_3  : return "TIDP_3"   ; case TECM_5  : return "TECM_5"   ; case TECMR_1 : return "TECMR_1"  ;  case PIXEL   : return "PIXEL"    ; case BPIX_1  : return "BPIX_1"   ;
  case TOB_2   : return "TOB_2"    ; case TIDMR_1 : return "TIDMR_1"  ; case TECM_6  : return "TECM_6"   ; case TECMR_2 : return "TECMR_2"  ;  case FPIX    : return "FPIX"     ; case BPIX_2  : return "BPIX_2"   ;
  case TOB_3   : return "TOB_3"    ; case TIDMR_2 : return "TIDMR_2"  ; case TECM_7  : return "TECM_7"   ; case TECMR_3 : return "TECMR_3"  ;  case FPIX_1  : return "FPIX_1"   ; case BPIX_3  : return "BPIX_3"   ;
  case TOB_4   : return "TOB_4"    ; case TIDMR_3 : return "TIDMR_3"  ; case TECM_8  : return "TECM_8"   ; case TECMR_4 : return "TECMR_4"  ;  case FPIX_2  : return "FPIX_2"   ;
  case TOB_5   : return "TOB_5"    ; case TIDPR_1 : return "TIDPR_1"  ; case TECM_9  : return "TECM_9"   ; case TECMR_5 : return "TECMR_5"  ;  case FPIX_3  : return "FPIX_3"   ;
  case TOB_6   : return "TOB_6"    ; case TIDPR_2 : return "TIDPR_2"  ; case TECP_1  : return "TECP_1"   ; case TECMR_6 : return "TECMR_6"  ;  case FPIXM   : return "FPIXM"    ;
  case TID     : return "TID"      ; case TIDPR_3 : return "TIDPR_3"  ; case TECP_2  : return "TECP_2"   ; case TECMR_7 : return "TECMR_7"  ;  case FPIXP   : return "FPIXP"    ;
  default:
    return "UNKOWN";
  }
  return "UNKOWN";

}

std::string ClusterSummary::getVarName(const VariablePlacement var){
  switch (var){
    case NMODULES     : return "NMODULES";
    case CLUSTERSIZE  : return "CLUSTERSIZE";
    case CLUSTERCHARGE: return "CLUSTERCHARGE";
    default:
      return "UNKOWN";
  }
  return "UNKOWN";
}

int ClusterSummary::GetModuleLocation ( int mod, bool warn ) const {

  int sortMod = mod;
  while (sortMod > 9 ){
    sortMod /= 10;
  }

  if(sortMod < 5){
    for(unsigned int iM = 0; iM < modules.size(); ++iM){
      if(mod == modules[iM])
        return iM;
    }
  } else {
    for(unsigned int iM =  modules.size(); iM-- > 0;){
      if(mod == modules[iM])
        return iM;
    }
  }

  if(!warn)
    return -1;

    edm::LogWarning("NoModule") << "No information for requested module "<<mod<<". Please check in the Provinence Infomation for proper modules.";
    return -1;
}

void ClusterSummary::PrepairGenericVariable() {
  nModules   = nModules_tmp  ;
  clusSize   = clusSize_tmp  ;
  clusCharge = clusCharge_tmp;

  nModules.erase(std::remove(nModules.begin(), nModules.end(), 0), nModules.end());
  clusSize.erase(std::remove(clusSize.begin(), clusSize.end(), 0), clusSize.end());
  clusCharge.erase(std::remove(clusCharge.begin(), clusCharge.end(), 0), clusCharge.end());
} 


std::vector<std::string> ClusterSummary::DecodeProvInfo(std::string ProvInfo) const {

  std::vector<std::string> v_moduleTypes;

  std::string mod = ProvInfo;
  std::string::size_type i = 0;
  std::string::size_type j = mod.find(',');

  if ( j == std::string::npos ){
    v_moduleTypes.push_back(mod);
  }
  else{

    while (j != std::string::npos) {
      v_moduleTypes.push_back(mod.substr(i, j-i));
      i = ++j;
      j = mod.find(',', j);
      if (j == std::string::npos)
	v_moduleTypes.push_back(mod.substr(i, mod.length( )));
    }

  }

  return v_moduleTypes;

}
