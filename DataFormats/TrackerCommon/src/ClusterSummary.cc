#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ClusterSummary::ClusterSummary()
    : userContent(nullptr), genericVariablesTmp_(6, std::vector<double>(100,0) )
{
}

ClusterSummary::~ClusterSummary()
{
    delete userContent;
    userContent = nullptr;
}

// swap function
void ClusterSummary::swap(ClusterSummary& other)
{
    other.userContent.exchange(
            userContent.exchange(other.userContent.load(std::memory_order_acquire), std::memory_order_acq_rel),
            std::memory_order_acq_rel);
    std::swap(iterator_, other.iterator_);
    std::swap(modules_, other.modules_);
    std::swap(genericVariables_, other.genericVariables_);
    std::swap(genericVariablesTmp_, other.genericVariablesTmp_);
}

// copy ctor
ClusterSummary::ClusterSummary(const ClusterSummary& src)
    : userContent(nullptr), iterator_(src.iterator_), modules_(src.modules_),
    genericVariables_(src.genericVariables_),
    genericVariablesTmp_(src.genericVariablesTmp_)
{
}

// copy assingment operator
ClusterSummary& ClusterSummary::operator=(const ClusterSummary& rhs)
{
    ClusterSummary temp(rhs);
    temp.swap(*this);
    return *this;
}

// move ctor
ClusterSummary::ClusterSummary(ClusterSummary&& other) 
    : ClusterSummary() 
{
    other.swap(*this);
}

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

void ClusterSummary::PrepairGenericVariable() { 

    genericVariables_ = genericVariablesTmp_;

    for (unsigned int i = 0; i < (*userContent.load(std::memory_order_acquire)).size(); ++i){
      genericVariables_[i].erase(std::remove(genericVariables_[i].begin(), genericVariables_[i].end(), 0), genericVariables_[i].end());
    }
} 

// Setter and Getter for the User Content. You can also return the size and what is stored in the UserContent 
void ClusterSummary::SetUserContent(const std::vector<std::string>& Content)  const
{
    if(!userContent.load(std::memory_order_acquire)) {
      auto ptr = new std::vector<std::string>;
      for(auto i=Content.begin(); i!=Content.end(); ++i) {
          ptr->push_back(*i);
      }
      //atomically try to swap this to become mItemsById
      std::vector<std::string>* expect = nullptr;
      bool exchanged = userContent.compare_exchange_strong(expect, ptr, std::memory_order_acq_rel);
      if(!exchanged) {
          delete ptr;
      }
    }
}
std::vector<std::string> ClusterSummary::GetUserContent()
{
    return (*userContent.load(std::memory_order_acquire));
}
int ClusterSummary::GetUserContentSize()
{
    return (*userContent.load(std::memory_order_acquire)).size();
}
void  ClusterSummary::GetUserContentInfo() const  { 
    std::cout << "Saving info for " ;
    for (unsigned int i = 0; i < (*userContent.load(std::memory_order_acquire)).size(); ++i) {
        std::cout << (*userContent.load(std::memory_order_acquire)).at(i) << " " ;
    }
    std::cout << std::endl;
}

int ClusterSummary::GetVariableLocation ( std::string var ) const {

  int placeInUserVector = -1;
    

  int cnt = 0;
  auto obj = (*userContent.load(std::memory_order_acquire));
  for(auto it=obj.begin(); it!=obj.end(); ++it) {

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



std::pair<int,int> ClusterSummary::ModuleSelection::IsStripSelected(int DetId){

  // true if the module mod is among the selected modules.  
  int isselected = 0;
  int enumVal = 99999;
  
  SiStripDetId subdet(DetId);
  int subdetid = subdet.subDetector();
  
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
    }
  }

  else if( SiStripDetId::TEC == subdetid && geosearch.compare("TECP")==0 ) {
      
    TECDetId tec(DetId);
    int side          = (tec.isZMinusSide())?-1:1;  

    if (side == 1){
      isselected = 1;
      enumVal = ClusterSummary::TECP;
    }    
  }


  else if( SiStripDetId::TID == subdetid && geosearch.compare("TIDM")==0 ) {
       
    TIDDetId tid(DetId);
    int side          = (tid.isZMinusSide())?-1:1;  

    if (side == -1){
      isselected = 1;
      enumVal = ClusterSummary::TIDM;
    }
  }


  else if( SiStripDetId::TID == subdetid && geosearch.compare("TIDP")==0 ) {
       
    TIDDetId tid(DetId);
    int side          = (tid.isZMinusSide())?-1:1;  

    if (side == 1){
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








std::pair<int,int> ClusterSummary::ModuleSelection::IsPixelSelected(int detid){

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
    int layer_id = 0;
	 
    ss >> layer_id;

    /****
	 Check the Layers of the Barrel
    ****/

    if (subdetid == 1 && Mod == "BPIX"){
	   
      PXBDetId pdetId = PXBDetId(detid);
      // Barell layer = 1,2,3
      int layer=pdetId.layer();

      if (layer_id == layer){

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
      
      PXFDetId pdetId = PXFDetId(detid);
      int disk=pdetId.disk(); //1,2,3

      if (layer_id == disk){

	if (disk == 1) enumVal = ClusterSummary::FPIX_1;
	else if (disk == 2) enumVal = ClusterSummary::FPIX_2;
	else if (disk == 3) enumVal = ClusterSummary::FPIX_3;

	isselected = 1;
	
      }
    }

    /****
	 Check the sides of each Disk of the endcaps
    ****/

    else if (subdetid == 2 && Mod == "FPIXM"){
      
      PXFDetId pdetId = PXFDetId(detid);
      int side=pdetId.side(); //size=1 for -z, 2 for +z
      int disk=pdetId.disk(); //1,2,3

      if (layer_id == disk && side == 1 ){

	if (disk == 1) enumVal = ClusterSummary::FPIXM_1;
	else if (disk == 2) enumVal = ClusterSummary::FPIXM_2;
	else if (disk == 3) enumVal = ClusterSummary::FPIXM_3;

	isselected = 1;
	
      }
    }

    else if (subdetid == 2 && Mod == "FPIXP"){
      
      PXFDetId pdetId = PXFDetId(detid);
      int side=pdetId.side(); //size=1 for -z, 2 for +z
      int disk=pdetId.disk(); //1,2,3

      if (layer_id == disk && side == 2){

	if (disk == 1) enumVal = ClusterSummary::FPIXP_1;
	else if (disk == 2) enumVal = ClusterSummary::FPIXP_2;
	else if (disk == 3) enumVal = ClusterSummary::FPIXP_3;

	isselected = 1;
	
      }
    }
  }
   
  /****
       Check the top and bottom of the endcaps
  ****/

  else if( subdetid == 2 && geosearch.compare("FPIXM")==0 ) {
       
    PXFDetId pdetId = PXFDetId(detid);
    int side=pdetId.side(); //size=1 for -z, 2 for +z

    if (side == 1){
      isselected = 1;
      enumVal = ClusterSummary::FPIXM;
    }
  }

  else if( subdetid == 2 && geosearch.compare("FPIXP")==0 ) {
      
    PXFDetId pdetId = PXFDetId(detid);
    int side=pdetId.side(); //size=1 for -z, 2 for +z

    if (side == 2){
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
