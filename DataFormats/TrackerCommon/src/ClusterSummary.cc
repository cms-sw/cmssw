#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

int ClusterSummary::GetModuleLocation ( int mod ) const {

  int placeInModsVector = -1;
    
  int cnt = 0;
  int pixelcnt = 0; 
  for(std::vector<int>::const_iterator it = modules_.begin(); it != modules_.end(); ++it) {
    /*
    if ( mod == (*it) ) { 
      placeInModsVector = cnt; 
      break;
    }
    else ++cnt;
    */
    
    int mod_tmp = *it;
    while (mod_tmp > 9 ){
      mod_tmp /= 10;
    }
     
    if ( mod_tmp < 5 ){

      if ( mod == (*it) ) { 
	placeInModsVector = cnt; 
	break;
      }
      else ++cnt;
    }
    else{      
      if ( mod == (*it) ) { 
	placeInModsVector = pixelcnt; 
	break;
      }
      else ++pixelcnt;
    }   
  }

  if (placeInModsVector == -1){

    edm::LogWarning("NoModule") << "No information for requested module "<<mod<<". Please check in the Provinence Infomation for proper modules.";
      
    return -1;

  }

  return placeInModsVector;

}



int ClusterSummary::GetVariableLocation ( std::string var ) const {

  int placeInUserVector = -1;
    

  int cnt = 0;
  for(std::vector<std::string>::const_iterator it = userContent.begin(); it != userContent.end(); ++it) {

    if ( var == (*it) ) { 
      placeInUserVector = cnt; 
      break;
    }
    else ++cnt;
      
  }


  /*
  if ( var == "cHits" )
    placeInUserVector = NMODULES;
  else if (var == "cSize" )
    placeInUserVector = CLUSTERSIZE;
  else if (var == "cCharge" )
    placeInUserVector = CLUSTERCHARGE;
  else if (var == "pHits" )
    placeInUserVector = NMODULESPIXELS;
  else if (var == "pSize" )
    placeInUserVector = CLUSTERSIZEPIXELS;
  else if (var == "pCharge" )
    placeInUserVector = CLUSTERCHARGEPIXELS;
  else
    placeInUserVector = -1;
  */
  if (placeInUserVector == -1){
    std::ostringstream err;
    err<<"No information for requested var "<<var<<". Please check if you have chosen a proper variable.";
      
    throw cms::Exception( "Missing Variable", err.str());
  }

  return placeInUserVector;

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



std::pair<int,int> ClusterSummary::ModuleSelection::IsStripSelected(int DetId, const TrackerTopology *tTopo){

  // true if the module mod is among the selected modules.  
  int isselected = 0;
  int enumVal = 99999;
  
  int subdetid = SiStripDetId(DetId).subDetector();
  
  std::string::size_type result = geosearch.find("_");

  if(result != std::string::npos) { 

    /****
	 Check to the layers in the modules
    ****/
  
    std::string modStr = geosearch; //Convert to string to use needed methods	 
    size_t pos = modStr.find("_", 0); //find the '_'
    std::string Mod = modStr.substr(0, pos); //find the module
    std::string Layer = modStr.substr(pos+1, modStr.length()); //find the Layer

    std::stringstream ss(Layer);
    unsigned int layer_id = 0;
	 
    ss >> layer_id;
	 
    if (SiStripDetId::TIB == subdetid && Mod == "TIB"){
	   
      if (layer_id == tTopo->tibLayer(DetId)){

	if (layer_id == 1) enumVal = ClusterSummary::TIB_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TIB_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TIB_3;
	else if (layer_id == 4) enumVal = ClusterSummary::TIB_4;

	isselected = 1;
      }
    } 
	 
    else if (SiStripDetId::TOB == subdetid && Mod == "TOB"){

      if (layer_id == tTopo->tobLayer(DetId)){

	if (layer_id == 1) enumVal = ClusterSummary::TOB_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TOB_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TOB_3;
	else if (layer_id == 4) enumVal = ClusterSummary::TOB_4;
	else if (layer_id == 5) enumVal = ClusterSummary::TOB_5;
	else if (layer_id == 6) enumVal = ClusterSummary::TOB_6;
	  
	isselected = 1;
      }
    } 

    else if (SiStripDetId::TEC == subdetid && Mod == "TECM"){

      if (layer_id == tTopo->tecWheel(DetId) && tTopo->tecIsZMinusSide(DetId)){
	  
	if (layer_id == 1) enumVal = ClusterSummary::TECM_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TECM_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TECM_3;
	else if (layer_id == 4) enumVal = ClusterSummary::TECM_4;
	else if (layer_id == 5) enumVal = ClusterSummary::TECM_5;
	else if (layer_id == 6) enumVal = ClusterSummary::TECM_6;
	else if (layer_id == 7) enumVal = ClusterSummary::TECM_7;
	else if (layer_id == 8) enumVal = ClusterSummary::TECM_8;
	else if (layer_id == 9) enumVal = ClusterSummary::TECM_9;

	isselected = 1;
      }
    } 

    else if (SiStripDetId::TEC == subdetid && Mod == "TECP"){

      if (layer_id == tTopo->tecWheel(DetId) && !tTopo->tecIsZMinusSide(DetId)){

	if (layer_id == 1) enumVal = ClusterSummary::TECP_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TECP_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TECP_3;
	else if (layer_id == 4) enumVal = ClusterSummary::TECP_4;
	else if (layer_id == 5) enumVal = ClusterSummary::TECP_5;
	else if (layer_id == 6) enumVal = ClusterSummary::TECP_6;
	else if (layer_id == 7) enumVal = ClusterSummary::TECP_7;
	else if (layer_id == 8) enumVal = ClusterSummary::TECP_8;
	else if (layer_id == 9) enumVal = ClusterSummary::TECP_9;

	isselected = 1;
      }
    } 

    // TEC minus ring
    else if (SiStripDetId::TEC == subdetid && Mod == "TECMR"){

      if (layer_id == tTopo->tecRing(DetId) && tTopo->tecIsZMinusSide(DetId)){

	if (layer_id == 1) enumVal = ClusterSummary::TECMR_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TECMR_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TECMR_3;
	else if (layer_id == 4) enumVal = ClusterSummary::TECMR_4;
	else if (layer_id == 5) enumVal = ClusterSummary::TECMR_5;
	else if (layer_id == 6) enumVal = ClusterSummary::TECMR_6;
	else if (layer_id == 7) enumVal = ClusterSummary::TECMR_7;

	isselected = 1;
      }
    } 

    // TEC plus ring
    else if (SiStripDetId::TEC == subdetid && Mod == "TECPR"){

      if (layer_id == tTopo->tecRing(DetId) && !tTopo->tecIsZMinusSide(DetId)){

	if (layer_id == 1) enumVal = ClusterSummary::TECPR_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TECPR_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TECPR_3;
	else if (layer_id == 4) enumVal = ClusterSummary::TECPR_4;
	else if (layer_id == 5) enumVal = ClusterSummary::TECPR_5;
	else if (layer_id == 6) enumVal = ClusterSummary::TECPR_6;
	else if (layer_id == 7) enumVal = ClusterSummary::TECPR_7;

	isselected = 1;
      }
    } 

    else if (SiStripDetId::TID == subdetid && Mod == "TIDM"){

      if (layer_id == tTopo->tidWheel(DetId) && tTopo->tidIsZMinusSide(DetId)){
	if (layer_id == 1) enumVal = ClusterSummary::TIDM_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TIDM_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TIDM_3;

	isselected = 1;
      }
    } 

    else if (SiStripDetId::TID == subdetid && Mod == "TIDP"){

      if (layer_id == tTopo->tidWheel(DetId) && !tTopo->tidIsZMinusSide(DetId)){
	if (layer_id == 1) enumVal = ClusterSummary::TIDP_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TIDP_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TIDP_3;

	isselected = 1;
      }
    } 
         
    // TID minus ring
    else if (SiStripDetId::TID == subdetid && Mod == "TIDMR"){
      if (layer_id == tTopo->tidRing(DetId) && tTopo->tidIsZMinusSide(DetId)){
  
	if (layer_id == 1) enumVal = ClusterSummary::TIDMR_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TIDMR_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TIDMR_3;

	isselected = 1;
      }
    } 

    // TID plus ring
    else if (SiStripDetId::TID == subdetid && Mod == "TIDPR"){
      if (layer_id == tTopo->tidRing(DetId) && !tTopo->tidIsZMinusSide(DetId)){

	if (layer_id == 1) enumVal = ClusterSummary::TIDPR_1;
	else if (layer_id == 2) enumVal = ClusterSummary::TIDPR_2;
	else if (layer_id == 3) enumVal = ClusterSummary::TIDPR_3;

	isselected = 1;
      }
    } 
  }
    
  /****
       Check the top and bottom for the TEC and TID
  ****/

  else if( SiStripDetId::TEC == subdetid && geosearch.compare("TECM")==0 ) {
       
    if (tTopo->tecIsZMinusSide(DetId)){
      isselected = 1;
      enumVal = ClusterSummary::TECM;
    }
  }

  else if( SiStripDetId::TEC == subdetid && geosearch.compare("TECP")==0 ) {
      
    if (!tTopo->tecIsZMinusSide(DetId)){
      isselected = 1;
      enumVal = ClusterSummary::TECP;
    }    
  }


  else if( SiStripDetId::TID == subdetid && geosearch.compare("TIDM")==0 ) {
    if (tTopo->tidIsZMinusSide(DetId)){
      isselected = 1;
      enumVal = ClusterSummary::TIDM;
    }
  }


  else if( SiStripDetId::TID == subdetid && geosearch.compare("TIDP")==0 ) {
       
    if (!tTopo->tidIsZMinusSide(DetId)){
      isselected = 1;
      enumVal = ClusterSummary::TIDP;
    }
  }

  /****
       Check the full TOB, TIB, TID, TEC modules
  ****/

  else if( SiStripDetId::TIB == subdetid && geosearch.compare("TIB")==0 ) {
    isselected = 1;
    enumVal = ClusterSummary::TIB;
  }
  else if( SiStripDetId::TID == subdetid && geosearch.compare("TID")==0 ) {
    isselected = 1;
    enumVal = ClusterSummary::TID;
  } 
  else if( SiStripDetId::TOB == subdetid && geosearch.compare("TOB")==0) {
    isselected = 1;
    enumVal = ClusterSummary::TOB;
  } 
  else if( SiStripDetId::TEC == subdetid && geosearch.compare("TEC")==0) {
    isselected = 1;
    enumVal = ClusterSummary::TEC;
  }
  else if( geosearch.compare("TRACKER")==0) {
    isselected = 1;
    enumVal = ClusterSummary::TRACKER;
  }
  

  return  std::make_pair(isselected, enumVal);
}








std::pair<int,int> ClusterSummary::ModuleSelection::IsPixelSelected(int detid, const TrackerTopology *tTopo){

  // true if the module mod is among the selected modules.  
  int isselected = 0;
  int enumVal = 99999;
  
  DetId detId = DetId(detid);       // Get the Detid object
  unsigned int detType=detId.det(); // det type, pixel=1
  unsigned int subdetid=detId.subdetId(); //subdetector type, barrel=1, foward=2

  if(detType!=1) return std::make_pair(0,99999); // look only at pixels

  std::string::size_type result = geosearch.find("_");

  if(result != std::string::npos) { 
  
    std::string modStr = geosearch; //Convert to string to use needed methods	 
    size_t pos = modStr.find("_", 0); //find the '_'
    std::string Mod = modStr.substr(0, pos); //find the module
    std::string Layer = modStr.substr(pos+1, modStr.length()); //find the Layer

    std::stringstream ss(Layer);
    unsigned int layer_id = 0;
	 
    ss >> layer_id;

    /****
	 Check the Layers of the Barrel
    ****/

    if (subdetid == 1 && Mod == "BPIX"){
	   
      if (layer_id == tTopo->pxbLayer(detid)) {

	if (layer_id == 1) enumVal = ClusterSummary::BPIX_1;
	else if (layer_id == 2) enumVal = ClusterSummary::BPIX_2;
	else if (layer_id == 3) enumVal = ClusterSummary::BPIX_3;

	isselected = 1;
      }
    } 

    /****
	 Check the Disk of the endcaps
    ****/
    else if (subdetid == 2 && Mod == "FPIX"){
      
      if (layer_id == tTopo->pxfDisk(detid)) {

	if (layer_id == 1) enumVal = ClusterSummary::FPIX_1;
	else if (layer_id == 2) enumVal = ClusterSummary::FPIX_2;
	else if (layer_id == 3) enumVal = ClusterSummary::FPIX_3;

	isselected = 1;
	
      }
    }

    /****
	 Check the sides of each Disk of the endcaps
    ****/

    else if (subdetid == 2 && Mod == "FPIXM"){
      
      if (layer_id == tTopo->pxfDisk(detid) && tTopo->pxfSide(detid)==1) {

	if (layer_id == 1) enumVal = ClusterSummary::FPIXM_1;
	else if (layer_id == 2) enumVal = ClusterSummary::FPIXM_2;
	else if (layer_id == 3) enumVal = ClusterSummary::FPIXM_3;

	isselected = 1;
	
      }
    }

    else if (subdetid == 2 && Mod == "FPIXP"){
      if (layer_id == tTopo->pxfDisk(detid) && tTopo->pxfSide(detid)==2) {

	if (layer_id == 1) enumVal = ClusterSummary::FPIXP_1;
	else if (layer_id == 2) enumVal = ClusterSummary::FPIXP_2;
	else if (layer_id == 3) enumVal = ClusterSummary::FPIXP_3;

	isselected = 1;
	
      }
    }
  }
   
  /****
       Check the top and bottom of the endcaps
  ****/

  else if( subdetid == 2 && geosearch.compare("FPIXM")==0 ) {
       
    if (tTopo->pxfSide(detid) ==1) {
      isselected = 1;
      enumVal = ClusterSummary::FPIXM;
    }
  }

  else if( subdetid == 2 && geosearch.compare("FPIXP")==0 ) {
      
    if (tTopo->pxfSide(detid) ==2) {
      isselected = 1;
      enumVal = ClusterSummary::FPIXP;
    }    
  }


  /****
       Check the full Barrel and Endcaps
  ****/
    
  else if(subdetid == 1 && geosearch.compare("BPIX")==0 ) {
    isselected = 1;
    enumVal = ClusterSummary::BPIX;
  }
  else if(subdetid == 2 && geosearch.compare("FPIX")==0 ) {
    isselected = 1;
    enumVal = ClusterSummary::FPIX;
  }
  else if( geosearch.compare("PIXEL")==0) {
    isselected = 1;
    enumVal = ClusterSummary::PIXEL;
  }


  return  std::make_pair(isselected, enumVal);
}

ClusterSummary::ModuleSelection::ModuleSelection(std::string gs){
  geosearch = gs;
}

ClusterSummary::ModuleSelection::~ModuleSelection() {}
