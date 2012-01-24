#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"

int ClusterSummary::GetModuleLocation ( int mod ) const {
  std::vector<int> mods = GetUserModules();

  int placeInModsVector = -1;
    
  int cnt = 0;
  for(std::vector<int>::iterator it = mods.begin(); it != mods.end(); ++it) {

    if ( mod == (*it) ) { 
      placeInModsVector = cnt; 
      break;
    }
    else ++cnt;
      
  }

  if (placeInModsVector == -1){
    std::ostringstream err;
    err<<"No information for requested module "<<mod<<". Please check in the Provinence Infomation for proper modules.";
      
    throw cms::Exception( "Missing Module", err.str());

  }

  return placeInModsVector;

}


std::vector<int> ClusterSummary::GetNumberOfModules() const { 
    
  std::vector<int> nType;
  std::vector<double> Allvars =  GetUserVariables();

  int cnt = 0;
  for ( unsigned int i = 0; i < modules_.size(); i++ ){
    nType . push_back( Allvars.at(NMODULES + iterator_.at(0)*cnt) );
    cnt ++;
  }
    
  return nType; 
}


int ClusterSummary::GetNumberOfModules( int mod ) const { 
    
  std::vector<double> Allvars =  GetUserVariables();

  int nType;
  int placeInModsVector = GetModuleLocation( mod );
   
  nType = Allvars.at(NMODULES + iterator_.at(0)*placeInModsVector);
    
  return nType; 

}


std::vector<double> ClusterSummary::GetClusterSize() const { 
        
  std::vector<double> clusterSize;
  std::vector<double> Allvars =  GetUserVariables();

  int cnt = 0;
  for ( unsigned int i = 0; i < modules_.size(); i++ ){
    clusterSize . push_back( Allvars.at(CLUSTERSIZE + iterator_.at(0)*cnt) );
    cnt ++;
  }

  return clusterSize; 
}


double ClusterSummary::GetClusterSize( int mod ) const { 
        
  std::vector<double> Allvars =  GetUserVariables();
    
  double clusterSize;
  int placeInModsVector = GetModuleLocation( mod );
   
  clusterSize = Allvars.at(CLUSTERSIZE + iterator_.at(0)*placeInModsVector);

  return clusterSize; 
}


std::vector<double> ClusterSummary::GetClusterCharge() const { 
          
  std::vector<double> clusterCharge;
  std::vector<double> Allvars =  GetUserVariables();

  int cnt = 0;
  for ( unsigned int i = 0; i < modules_.size(); i++ ){
    clusterCharge . push_back( Allvars.at(CLUSTERCHARGE + iterator_.at(0)*cnt) );
    cnt ++;
  }
    
  return clusterCharge; 

}
  

double ClusterSummary::GetClusterCharge( int mod ) const { 
    
  std::vector<double> Allvars =  GetUserVariables();
    
  double clusterCharge;
  int placeInModsVector = GetModuleLocation( mod );
   
  clusterCharge = Allvars.at(CLUSTERCHARGE + iterator_.at(0)*placeInModsVector);

    
  return clusterCharge; 

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



std::pair<int,int> ClusterSummary::ModuleSelection::IsSelected(int DetId){

  // true if the module mod is among the selected modules.  
  int isselected = 0;
  int enumVal = 99999;
  
  SiStripDetId subdet(DetId);
  int subdetid = subdet.subDetector();
  
  while(1) {

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
      int layer_id = 0;
	 
      ss >> layer_id;
	 
      if (SiStripDetId::TIB == subdetid && Mod == "TIB"){
	   
	TIBDetId tib(DetId);
	int layer    = tib.layer(); 
	if (layer_id == layer){

	  if (layer_id == 1) enumVal = ClusterSummary::TIB_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TIB_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TIB_3;
	  else if (layer_id == 4) enumVal = ClusterSummary::TIB_4;

	  isselected = 1;
	  break;
	}
      } 
	 
      else if (SiStripDetId::TOB == subdetid && Mod == "TOB"){

	TOBDetId tob(DetId);
	int layer    = tob.layer(); 
	if (layer_id == layer){

	  if (layer_id == 1) enumVal = ClusterSummary::TOB_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TOB_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TOB_3;
	  else if (layer_id == 4) enumVal = ClusterSummary::TOB_4;
	  else if (layer_id == 5) enumVal = ClusterSummary::TOB_5;
	  else if (layer_id == 6) enumVal = ClusterSummary::TOB_6;
	  
	  isselected = 1;
	  break;
	}
      } 

      else if (SiStripDetId::TEC == subdetid && Mod == "TECM"){

	TECDetId tec(DetId);
	int side          = (tec.isZMinusSide())?-1:1; 
	int layerwheel    = tec.wheel(); 

	if (layer_id == layerwheel && side == -1){
	  
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
	  break;
	}
      } 

      else if (SiStripDetId::TEC == subdetid && Mod == "TECP"){

	TECDetId tec(DetId);
	int side          = (tec.isZMinusSide())?-1:1; 
	int layerwheel    = tec.wheel(); 

	if (layer_id == layerwheel && side == 1){

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
	  break;
	}
      } 

      // TEC minus ring
      else if (SiStripDetId::TEC == subdetid && Mod == "TECMR"){

	TECDetId tec(DetId);
	int side          = (tec.isZMinusSide())?-1:1; 
	int ring    = tec.ringNumber();  

	if (layer_id == ring && side == -1){

	  if (layer_id == 1) enumVal = ClusterSummary::TECMR_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TECMR_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TECMR_3;
	  else if (layer_id == 4) enumVal = ClusterSummary::TECMR_4;
	  else if (layer_id == 5) enumVal = ClusterSummary::TECMR_5;
	  else if (layer_id == 6) enumVal = ClusterSummary::TECMR_6;
	  else if (layer_id == 7) enumVal = ClusterSummary::TECMR_7;

	  isselected = 1;
	  break;
	}
      } 

      // TEC plus ring
      else if (SiStripDetId::TEC == subdetid && Mod == "TECPR"){

	TECDetId tec(DetId);
	int side          = (tec.isZMinusSide())?-1:1; 
	int ring    = tec.ringNumber();  
	if (layer_id == ring && side == 1){

	  if (layer_id == 1) enumVal = ClusterSummary::TECPR_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TECPR_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TECPR_3;
	  else if (layer_id == 4) enumVal = ClusterSummary::TECPR_4;
	  else if (layer_id == 5) enumVal = ClusterSummary::TECPR_5;
	  else if (layer_id == 6) enumVal = ClusterSummary::TECPR_6;
	  else if (layer_id == 7) enumVal = ClusterSummary::TECPR_7;

	  isselected = 1;
	  break;
	}
      } 

      else if (SiStripDetId::TID == subdetid && Mod == "TIDM"){

	TIDDetId tid(DetId);
	int side          = (tid.isZMinusSide())?-1:1; 
	int layerwheel    = tid.wheel(); 

	if (layer_id == layerwheel && side == -1){

	  if (layer_id == 1) enumVal = ClusterSummary::TIDM_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TIDM_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TIDM_3;

	  isselected = 1;
	  break;
	}
      } 

      else if (SiStripDetId::TID == subdetid && Mod == "TIDP"){

	TIDDetId tid(DetId);
	int side          = (tid.isZMinusSide())?-1:1; 
	int layerwheel    = tid.wheel(); 

	if (layer_id == layerwheel && side == 1){

	  if (layer_id == 1) enumVal = ClusterSummary::TIDP_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TIDP_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TIDP_3;

	  isselected = 1;
	  break;
	}
      } 
         
       // TID minus ring
      else if (SiStripDetId::TID == subdetid && Mod == "TIDMR"){
	TIDDetId tid(DetId);
	int side          = (tid.isZMinusSide())?-1:1; 
	int ring    = tid.ringNumber(); 
	if (layer_id == ring && side == -1){
	  
	  if (layer_id == 1) enumVal = ClusterSummary::TIDMR_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TIDMR_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TIDMR_3;

	  isselected = 1;
	  break;
	}
      } 

      // TID plus ring
      else if (SiStripDetId::TID == subdetid && Mod == "TIDPR"){
	TIDDetId tid(DetId);
	int side          = (tid.isZMinusSide())?-1:1; 
	int ring    = tid.ringNumber(); 
	
	if (layer_id == ring && side == 1){

	  if (layer_id == 1) enumVal = ClusterSummary::TIDPR_1;
	  else if (layer_id == 2) enumVal = ClusterSummary::TIDPR_2;
	  else if (layer_id == 3) enumVal = ClusterSummary::TIDPR_3;

	  isselected = 1;
	  break;
	}
      } 
    }
    
    /****
	 Check the top and bottom for the TEC and TID
    ****/

    else if( SiStripDetId::TEC == subdetid && geosearch.compare("TECM")==0 ) {
       
      TECDetId tec(DetId);
      int side          = (tec.isZMinusSide())?-1:1;  

      if (side == -1){
	isselected = 1;
	enumVal = ClusterSummary::TECM;
	break;
      }
    }

    else if( SiStripDetId::TEC == subdetid && geosearch.compare("TECP")==0 ) {
      
      TECDetId tec(DetId);
      int side          = (tec.isZMinusSide())?-1:1;  

      if (side == 1){
	isselected = 1;
	enumVal = ClusterSummary::TECP;
	break;
      }    
    }


    else if( SiStripDetId::TID == subdetid && geosearch.compare("TIDM")==0 ) {
       
      TIDDetId tid(DetId);
      int side          = (tid.isZMinusSide())?-1:1;  

      if (side == -1){
	isselected = 1;
	enumVal = ClusterSummary::TIDM;
	break;
      }
    }


    else if( SiStripDetId::TID == subdetid && geosearch.compare("TIDP")==0 ) {
       
      TIDDetId tid(DetId);
      int side          = (tid.isZMinusSide())?-1:1;  

      if (side == 1){
	isselected = 1;
	enumVal = ClusterSummary::TIDP;
	break;
      }
    }

    /****
	 Check the full TOB, TIB, TID, TEC modules
    ****/

    else if( SiStripDetId::TIB == subdetid && geosearch.compare("TIB")==0 ) {
      isselected = 1;
      enumVal = ClusterSummary::TIB;
      break;   
    }
    else if( SiStripDetId::TID == subdetid && geosearch.compare("TID")==0 ) {
      isselected = 1;
      enumVal = ClusterSummary::TID;
      break;
    } 
    else if( SiStripDetId::TOB == subdetid && geosearch.compare("TOB")==0) {
      isselected = 1;
      enumVal = ClusterSummary::TOB;
      break;
    } 
    else if( SiStripDetId::TEC == subdetid && geosearch.compare("TEC")==0) {
      isselected = 1;
      enumVal = ClusterSummary::TEC;
      break;
    }
    else if( geosearch.compare("TRACKER")==0) {
      isselected = 1;
       enumVal = ClusterSummary::TRACKER;
      break;
    }

  
    break;
    //gst = strtok(0," ,");
  }
  
  //delete geosearchtmp;

  return  std::make_pair(isselected, enumVal);
}
