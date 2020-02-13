#include "L1Trigger/DTPhase2Trigger/interface/MuonPathAnalyzerInChamber.h"
#include <cmath> 

using namespace edm;
using namespace std;



// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAnalyzerInChamber::MuonPathAnalyzerInChamber(const ParameterSet& pset) :
  MuonPathAnalyzer(pset),
  debug(pset.getUntrackedParameter<Bool_t>("debug")),
  chi2Th(pset.getUntrackedParameter<double>("chi2Th")),    
  z_filename(pset.getParameter<edm::FileInPath>("z_filename")),
  shift_filename(pset.getParameter<edm::FileInPath>("shift_filename")),  
  bxTolerance(30),
  minQuality(LOWQGHOST),
  chiSquareThreshold(50),
  minHits4Fit(pset.getUntrackedParameter<int>("minHits4Fit"))
{
  // Obtention of parameters
  
  if (debug) cout <<"MuonPathAnalyzer: constructor" << endl;

  
  setChiSquareThreshold(chi2Th*100.); 

  //z
  int rawId;
  std::ifstream ifin2(z_filename.fullPath());
  double z;
  if (ifin2.fail()) {
      throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerInChamber::MuonPathAnalyzerInChamber() -  Cannot find " << z_filename.fullPath();
  }
  while (ifin2.good()){
    ifin2 >> rawId >> z;
    zinfo[rawId]=z;
  }

  //shift
  std::ifstream ifin3(shift_filename.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
      << "MuonPathAnalyzerInChamber::MuonPathAnalyzerInChamber() -  Cannot find " << shift_filename.fullPath();
  }
  while (ifin3.good()){
    ifin3 >> rawId >> shift;
    shiftinfo[rawId]=shift;
  }

}

MuonPathAnalyzerInChamber::~MuonPathAnalyzerInChamber() {
  if (debug) cout <<"MuonPathAnalyzer: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAnalyzerInChamber::initialise(const edm::EventSetup& iEventSetup) {
  if(debug) cout << "MuonPathAnalyzerInChamber::initialiase" << endl;
  iEventSetup.get<MuonGeometryRecord>().get(dtGeo);//1103
}

void MuonPathAnalyzerInChamber::run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, std::vector<MuonPath*> &muonpaths, std::vector<MuonPath*> &outmuonpaths) {

  if (debug) cout <<"MuonPathAnalyzerInChamber: run" << endl;
  
  // fit per SL (need to allow for multiple outputs for a single mpath)
  for(auto muonpath = muonpaths.begin();muonpath!=muonpaths.end();++muonpath) {
    analyze(*muonpath, outmuonpaths);
  }

}

void MuonPathAnalyzerInChamber::finish() {
  if (debug) cout <<"MuonPathAnalyzer: finish" << endl;
};

const int MuonPathAnalyzerInChamber::LAYER_ARRANGEMENTS[4][3] = {
    {0, 1, 2}, {1, 2, 3},                       // Grupos consecutivos
    {0, 1, 3}, {0, 2, 3}                        // Grupos salteados
};


//------------------------------------------------------------------
//--- Métodos get / set
//------------------------------------------------------------------
void MuonPathAnalyzerInChamber::setBXTolerance(int t) { bxTolerance = t; }
int  MuonPathAnalyzerInChamber::getBXTolerance(void)  { return bxTolerance; }

void MuonPathAnalyzerInChamber::setChiSquareThreshold(float ch2Thr) {
    chiSquareThreshold = ch2Thr;
}

void MuonPathAnalyzerInChamber::setMinimumQuality(MP_QUALITY q) {
    if (minQuality >= LOWQGHOST) minQuality = q;
}
MP_QUALITY MuonPathAnalyzerInChamber::getMinimumQuality(void) { return minQuality; }


//------------------------------------------------------------------
//--- Métodos privados
//------------------------------------------------------------------
void MuonPathAnalyzerInChamber::analyze(MuonPath *inMPath,std::vector<MuonPath*>& outMPath) {

  if(debug) std::cout<<"DTp2:analyze \t\t\t\t starts"<<std::endl;
  
  // Clonamos el objeto analizado.
  if (debug) cout << inMPath->getNPrimitives() << endl;
  MuonPath *mPath = new MuonPath(*inMPath);
  
  if (debug) {
    std::cout << "DTp2::analyze, looking at mPath: " << std::endl;
    for (int i=0; i<mPath->getNPrimitives(); i++) 
      std::cout << mPath->getPrimitive(i)->getLayerId() << " , " 
		<< mPath->getPrimitive(i)->getSuperLayerId() << " , " 
		<< mPath->getPrimitive(i)->getChannelId() << " , " 
		<< mPath->getPrimitive(i)->getLaterality() << std::endl;
  }

  if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t is Analyzable? "<<std::endl;
  if (!mPath->isAnalyzable())  return;
  if(debug) std::cout<<"DTp2:analyze \t\t\t\t\t yes it is analyzable "<<mPath->isAnalyzable()<<std::endl;
  
  // first of all, get info from primitives, so we can reduce the number of latereralities:
  buildLateralities(mPath);
  //  setCellLayout(mPath);  
  setWirePosAndTimeInMP(mPath);
  
  MuonPath *mpAux(NULL);
  int bestI = -1; 
  float best_chi2=99999.;
  for (int i = 0; i < totalNumValLateralities; i++) {// LOOP for all lateralities:   
    if (debug) cout << "DTp2:analyze \t\t\t\t\t Start with combination " << i << endl; 
    int NTotalHits=8;
    float xwire[8];
    int present_layer[8];
    for (int ii=0; ii<8; ii++){ 
      xwire[ii]      = mPath->getXWirePos(ii); 
      if (xwire[ii]==0) { 
        present_layer[ii]=0;
        NTotalHits--;      
      } else {            
        present_layer[ii]=1;
      }
    }
    
    while (NTotalHits >= minHits4Fit){
      mPath->setChiSq(0); 
      calculateFitParameters(mPath,lateralities[i], present_layer);
      if (mPath->getChiSq() != 0) break;
      NTotalHits--;
    }
    if ( mPath->getChiSq() > chiSquareThreshold ) continue;
    

    evaluateQuality(mPath);
    
    if ( mPath->getQuality() < minQuality ) continue;
    
   /* 
    int selected_Id=0;
    for (int i=0; i<mPath->getNPrimitives(); i++) {
      if (mPath->getPrimitive(i)->isValidTime()) {
	selected_Id= mPath->getPrimitive(i)->getCameraId();
	mPath->setRawId(selected_Id);
	break;
      }
    }
    
    DTLayerId thisLId(selected_Id);
    DTSuperLayerId MuonPathSLId(thisLId.wheel(),thisLId.station(),thisLId.sector(),thisLId.superLayer());
    DTChamberId ChId(MuonPathSLId.wheel(),MuonPathSLId.station(),MuonPathSLId.sector());
    DTWireId wireId(MuonPathSLId,2,1);
    
    //computing phi and phiB
    double z=0;
    double z1=11.75;
    double z3=-1.*z1;
    if (ChId.station() == 3 or ChId.station() == 4){
      z1=9.95;
      z3=-13.55;
    }
    
    if(MuonPathSLId.superLayer()==1) z=z1;
    if(MuonPathSLId.superLayer()==3) z=z3;
    
    double jm_x=(mPath->getHorizPos()/10.)+shiftinfo[wireId.rawId()]; 
*/   
    double z=0;
    double jm_x=(mPath->getHorizPos());
    int selected_Id=0;
    for (int i=0; i<mPath->getNPrimitives(); i++) {
      if (mPath->getPrimitive(i)->isValidTime()) {
        selected_Id= mPath->getPrimitive(i)->getCameraId();
        mPath->setRawId(selected_Id);
        break;
      }
    }
    DTLayerId thisLId(selected_Id);
    if(thisLId.station()>=3)z=-1.8;   
 
    DTSuperLayerId MuonPathSLId(thisLId.wheel(),thisLId.station(),thisLId.sector(),thisLId.superLayer());
    GlobalPoint jm_x_cmssw_global = dtGeo->chamber(MuonPathSLId)->toGlobal(LocalPoint(jm_x,0.,z));//jm_x is already extrapolated to the middle of the SL
    int thisec = MuonPathSLId.sector();
    if(thisec==13) thisec = 4;
    if(thisec==14) thisec = 10;
    double phi= jm_x_cmssw_global.phi()-0.5235988*(thisec-1);
    double psi=atan(mPath->getTanPhi());
    mPath->setPhi(jm_x_cmssw_global.phi()-0.5235988*(thisec-1));
    mPath->setPhiB(hasPosRF(MuonPathSLId.wheel(),MuonPathSLId.sector()) ? psi-phi : -psi-phi);
    
    if(mPath->getChiSq() < best_chi2 && mPath->getChiSq() > 0 ){
      mpAux = new MuonPath(mPath);
      bestI = i; 
      best_chi2 = mPath->getChiSq();
    }
  }
  if (mpAux!=NULL){ 
    outMPath.push_back(std::move(mpAux));
    if (debug) std::cout<<"DTp2:analize \t\t\t\t\t Laterality "<< bestI << " is the one with smaller chi2"<<std::endl;
  } else {
    if (debug) std::cout<<"DTp2:analize \t\t\t\t\t No Laterality found with chi2 smaller than threshold"<<std::endl;
  }
  if (debug) std::cout<<"DTp2:analize \t\t\t\t\t Ended working with this set of lateralities"<<std::endl; 
}

void MuonPathAnalyzerInChamber::setCellLayout(MuonPath *mpath) {
  
  for (int i=0; i<=mpath->getNPrimitives(); i++) {
    if (mpath->getPrimitive(i)->isValidTime())   
      cellLayout[i] = mpath->getPrimitive(i)->getChannelId();
    else 
      cellLayout[i] = -99; 
  }
  
  // putting info back into the mpath:
  mpath->setCellHorizontalLayout(cellLayout);
  for (int i=0; i<=mpath->getNPrimitives(); i++){
    if (cellLayout[i]>=0) {
      mpath->setBaseChannelId(cellLayout[i]);
      break;
    }
  }
}

/**
 * For a combination of up to 8 cells, build all the lateralities to be tested,
 * and discards all  construye de forma automática todas las posibles
 * combinaciones de lateralidad (LLLL, LLRL,...) que son compatibles con una
 * trayectoria recta. Es decir, la partícula no hace un zig-zag entre los hilos
 * de diferentes celdas, al pasar de una a otra.
 */
void MuonPathAnalyzerInChamber::buildLateralities(MuonPath *mpath) {
  
  if (debug) cout << "MPAnalyzer::buildLateralities << setLateralitiesFromPrims " << endl;
  mpath->setLateralCombFromPrimitives();
  
  totalNumValLateralities = 0;
  lateralities.clear();
  latQuality.clear();
  
  /* We generate all the possible laterality combinations compatible with the built 
     group in the previous step*/
  lateralities.push_back(TLateralities());
  for (int ilat = 0; ilat <  NLayers; ilat++) {
    // Get value from input
    LATERAL_CASES lr = (mpath->getLateralComb())[ilat];
    if (debug) std::cout << "[DEBUG] Input[" << ilat << "]: " << lr << std::endl;
    
    // If left/right fill number
    if (lr != NONE ) {
      if (debug) std::cout << "[DEBUG]   - Adding it to " << lateralities.size() << " lists..." << std::endl;
      for (unsigned int iall = 0; iall < lateralities.size(); iall++) {
        lateralities[iall][ilat]  = lr;
	
      }
    }
    // both possibilites
    else {
      // Get the number of possible options now
      auto ncurrentoptions = lateralities.size();
      
      // Duplicate them
      if (debug) std::cout << "[DEBUG]   - Duplicating " << ncurrentoptions << " lists..." << std::endl;
      copy(lateralities.begin(), lateralities.end(), back_inserter(lateralities));     
      if (debug) std::cout << "[DEBUG]   - Now we have " << lateralities.size() << " lists..." << std::endl;

      // Asign LEFT to first ncurrentoptions and RIGHT to the last
      for (unsigned int iall = 0; iall < ncurrentoptions; iall++) {
        lateralities[iall][ilat]  = LEFT;
        lateralities[iall+ncurrentoptions][ilat]  = RIGHT;
      }
    } // else
  } // Iterate over input array
  
  totalNumValLateralities = (int) lateralities.size(); 
  /*
    for (unsigned int iall=0; iall<lateralities.size(); iall++) {
    latQuality.push_back(LATQ_TYPE());

    latQuality[iall].valid            = false;
    latQuality[iall].bxValue          = 0;
    latQuality[iall].quality          = NOPATH;
    latQuality[iall].invalidateHitIdx = -1;
    }
  */
  if (totalNumValLateralities>128) {
    // ADD PROTECTION!
    cout << "[WARNING]: TOO MANY LATERALITIES TO CHECK !!" << endl;
    cout << "[WARNING]: skipping this muon" << endl;
    lateralities.clear();
    latQuality.clear();
    totalNumValLateralities = 0;
  }
  
  // Dump values
  if (debug) {
    for (unsigned int iall = 0; iall < lateralities.size(); iall++) {
      std::cout << iall << " -> [";
      for (int ilat = 0; ilat < NLayers; ilat++) {
	if (ilat != 0)
	  std::cout << ",";
	std::cout << lateralities[iall][ilat];
      }
      std::cout << "]" << std::endl;
    } 
  }
}
void MuonPathAnalyzerInChamber::setLateralitiesInMP(MuonPath *mpath, TLateralities lat){
  LATERAL_CASES tmp[8]; 
  for (int i=0; i<8; i++)
    tmp[i] = lat[i];
    
  mpath->setLateralComb(tmp);
}
void MuonPathAnalyzerInChamber::setWirePosAndTimeInMP(MuonPath *mpath){
  int selected_Id=0;
  for (int i=0; i<mpath->getNPrimitives(); i++) {
    if (mpath->getPrimitive(i)->isValidTime()) {
      selected_Id= mpath->getPrimitive(i)->getCameraId();
      mpath->setRawId(selected_Id);
      break;
    }
  }
  DTLayerId thisLId(selected_Id);
  DTChamberId chId(thisLId.wheel(),thisLId.station(),thisLId.sector());
  if (debug) cout << "Id " << chId.rawId() << " Wh " << chId.wheel() << " St " << chId.station() << " Se " << chId.sector() <<  endl; 
  mpath->setRawId(chId.rawId());
  
  DTSuperLayerId MuonPathSLId1(thisLId.wheel(),thisLId.station(),thisLId.sector(),1);
  DTSuperLayerId MuonPathSLId3(thisLId.wheel(),thisLId.station(),thisLId.sector(),3);
  DTWireId wireId1(MuonPathSLId1,2,1);
  DTWireId wireId3(MuonPathSLId3,2,1);

  if (debug) cout << "shift1=" << shiftinfo[wireId1.rawId()] << " shift3=" << shiftinfo[wireId3.rawId()] << endl;  

  float delta = 42000; //um
  float zwire[8]={-13.7, -12.4, -11.1, -9.8002, 9.79999, 11.1, 12.4, 13.7}; // mm
  for (int i=0; i<mpath->getNPrimitives(); i++){ 
    if (mpath->getPrimitive(i)->isValidTime())  {
      if (i<4)mpath->setXWirePos(10000*shiftinfo[wireId1.rawId()]+(mpath->getPrimitive(i)->getChannelId() + 0.5*(double)((i+1)%2)) * delta,i);
      if (i>=4)mpath->setXWirePos(10000*shiftinfo[wireId3.rawId()]+(mpath->getPrimitive(i)->getChannelId() + 0.5*(double)((i+1)%2)) * delta,i);
     // mpath->setXWirePos((mpath->getPrimitive(i)->getChannelId() + 0.5*(double)(i%2)) * delta,i);
      mpath->setZWirePos(zwire[i]*1000, i); // in um
      mpath->settWireTDC(mpath->getPrimitive(i)->getTDCTime()*DRIFT_SPEED,i);
    }
    else {
      mpath->setXWirePos(0.,i);
      mpath->setZWirePos(0.,i);
      mpath->settWireTDC(-1*DRIFT_SPEED,i);
    }
    if (debug) cout << mpath->getPrimitive(i)->getTDCTime() << " ";
  }
  if (debug) cout << endl;
}
void MuonPathAnalyzerInChamber::calculateFitParameters(MuonPath *mpath, TLateralities laterality, int present_layer[8]) {
  
  // First prepare mpath for fit: 
  float xwire[8],zwire[8],tTDCvdrift[8];
  double b[8];
  for (int i=0; i<8; i++){ 
    xwire[i]      = mpath->getXWirePos(i); 
    zwire[i]      = mpath->getZWirePos(i);
    tTDCvdrift[i] = mpath->gettWireTDC(i);
    b[i]          = 1;
  }
  
  //// NOW Start FITTING:  
  
  // fill hit position
  float xhit[8];
  for  (int lay=0; lay<8; lay++){
    if (debug) cout << "In fitPerLat " << lay << " xwire " << xwire[lay] << " zwire "<< zwire[lay]<<" tTDCvdrift "<< tTDCvdrift[lay]<< endl;
    xhit[lay]=xwire[lay]+(-1+2*laterality[lay])*1000*tTDCvdrift[lay];
    if (debug) cout << "In fitPerLat " << lay << " xhit "<< xhit[lay]<< endl;
  }  
      
  //Proceed with calculation of fit parameters
  double cbscal={0.000000d};
  double zbscal={0.000000d};
  double czscal={0.000000d};
  double bbscal={0.000000d};
  double zzscal={0.000000d};
  double ccscal={0.000000d};
  
  for  (int lay=0; lay<8; lay++){
    if (present_layer[lay]==0) continue;
    if (debug) cout<< " For layer " << lay+1 << " xwire[lay] " << xwire[lay] << " zwire " << zwire[lay] << " b " << b[lay] << endl;
    if (debug) cout<< " xhit[lat][lay] " << xhit[lay] << endl;
    cbscal=(-1+2*laterality[lay])*b[lay]+cbscal;
    zbscal=zwire[lay]*b[lay]+zbscal; //it actually does not depend on laterality
    czscal=(-1+2*laterality[lay])*zwire[lay]+czscal;
    
    bbscal=b[lay]*b[lay]+bbscal; //it actually does not depend on laterality
    zzscal=zwire[lay]*zwire[lay]+zzscal; //it actually does not depend on laterality
    ccscal=(-1+2*laterality[lay])*(-1+2*laterality[lay])+ccscal;
  }
  
  
  double cz= {0.000000d};
  double cb= {0.000000d};
  double zb= {0.000000d};
  double zc= {0.000000d};
  double bc= {0.000000d};
  double bz= {0.000000d};
  
  cz=(cbscal*zbscal-czscal*bbscal)/(zzscal*bbscal-zbscal*zbscal);
  cb=(czscal*zbscal-cbscal*zzscal)/(zzscal*bbscal-zbscal*zbscal);
  
  zb=(czscal*cbscal-zbscal*ccscal)/(bbscal*ccscal-cbscal*cbscal);
  zc=(zbscal*cbscal-czscal*bbscal)/(bbscal*ccscal-cbscal*cbscal);
  
  bc=(zbscal*czscal-cbscal*zzscal)/(ccscal*zzscal-czscal*czscal);
  bz=(cbscal*czscal-zbscal*ccscal)/(ccscal*zzscal-czscal*czscal);
  
  
  double c_tilde[8]; 
  double z_tilde[8];
  double b_tilde[8];
  
  for  (int lay=0; lay<8; lay++){
    if (present_layer[lay]==0) continue;
    if (debug) cout<< " For layer " << lay+1 << " xwire[lay] " << xwire[lay] << " zwire " << zwire[lay] << " b " << b[lay] << endl;
    c_tilde[lay]=(-1+2*laterality[lay])+cz*zwire[lay]+cb*b[lay];       	
    z_tilde[lay]=zwire[lay]+zb*b[lay]+zc*(-1+2*laterality[lay]);
    b_tilde[lay]=b[lay]+bc*(-1+2*laterality[lay])+bz*zwire[lay];
    
  }
  
  //Calculate results per lat
  double xctilde={0.000000d};
  double xztilde={0.000000d};
  double xbtilde={0.000000d};
  double ctildectilde={0.000000d};
  double ztildeztilde={0.000000d};
  double btildebtilde={0.000000d};
  
  double rect0vdrift={0.000000d};
  double recslope={0.000000d};
  double recpos={0.000000d};
  
  for  (int lay=0; lay<8; lay++){
    if (present_layer[lay]==0) continue;
    xctilde=xhit[lay]*c_tilde[lay]+xctilde;
    ctildectilde=c_tilde[lay]*c_tilde[lay]+ctildectilde;
    xztilde=xhit[lay]*z_tilde[lay]+xztilde;
    ztildeztilde=z_tilde[lay]*z_tilde[lay]+ztildeztilde;
    xbtilde=xhit[lay]*b_tilde[lay]+xbtilde;
    btildebtilde=b_tilde[lay]*b_tilde[lay]+btildebtilde;
  }
  
  //Results for t0vdrift (BX), slope and position per lat
  rect0vdrift=xctilde/ctildectilde;
  recslope=xztilde/ztildeztilde;
  recpos=xbtilde/btildebtilde;
  if(debug) {
    cout<< " In fitPerLat Reconstructed values per lat " << " rect0vdrift "<< rect0vdrift;
    cout <<"rect0 "<< rect0vdrift/DRIFT_SPEED <<" recBX " << rect0vdrift/DRIFT_SPEED/25 << " recslope " << recslope << " recpos " << recpos  << endl;
  }
  
  //Get t*v and residuals per layer, and chi2 per laterality
  double rectdriftvdrift[8]={0.000000d};
  double recres[8]={0.000000d};
  double recchi2={0.000000d};
  int sign_tdriftvdrift={0};    
  int incell_tdriftvdrift={0};    
  int physical_slope={0}; 

  // Select the worst hit in order to get rid of it
  double maxDif = -1; 
  int maxInt = -1; 
    
  for  (int lay=0; lay<8; lay++){
    if (present_layer[lay]==0) continue;
    rectdriftvdrift[lay]= tTDCvdrift[lay]- rect0vdrift/1000;
    if (debug) cout << rectdriftvdrift[lay] << endl; 
    recres[lay]=xhit[lay]-zwire[lay]*recslope-b[lay]*recpos-(-1+2*laterality[lay])*rect0vdrift;
     //if (debug) cout <<"Pr14 "<< recres[lay] << endl;  
    if ((present_layer[lay]==1)&&(rectdriftvdrift[lay] <-0.1)){
      sign_tdriftvdrift=-1;
      if (-0.1 - rectdriftvdrift[lay] > maxDif) {
        maxDif = -0.1 - rectdriftvdrift[lay];
        maxInt = lay; 
      }
    }		  
    if ((present_layer[lay]==1)&&(abs(rectdriftvdrift[lay]) >21.1)){
      incell_tdriftvdrift=-1; //Changed to 2.11 to account for resolution effects
      if (rectdriftvdrift[lay] - 21.1 > maxDif) {
        maxDif = rectdriftvdrift[lay] - 21.1;
        maxInt = lay; 
      }
    }		  
  }

  if (fabs(recslope/10)>1.3)  physical_slope=-1;
  
  if (physical_slope==-1 && debug)  cout << "Combination with UNPHYSICAL slope " <<endl;
  if (sign_tdriftvdrift==-1 && debug) cout << "Combination with negative tdrift-vdrift " <<endl;
  if (incell_tdriftvdrift==-1 && debug) cout << "Combination with tdrift-vdrift larger than half cell " <<endl;
  
  for  (int lay=0; lay<8; lay++){
    if (present_layer[lay]==0) continue;
    recchi2=recres[lay]*recres[lay] + recchi2;
  }
  if(debug) cout << "In fitPerLat Chi2 " << recchi2 << " with sign " << sign_tdriftvdrift << " within cell " << incell_tdriftvdrift << " physical_slope "<< physical_slope << endl;
  
  //LATERALITY IS NOT VALID
  if (true && maxInt!=-1)  { 
    present_layer[maxInt] = 0; 
    if(debug) cout << "We get rid of hit in layer " << maxInt << endl; 
  }  
  
  // LATERALITY IS VALID... 
  if(!(sign_tdriftvdrift==-1) && !(incell_tdriftvdrift==-1) && !(physical_slope==-1)){
    mpath->setBxTimeValue((rect0vdrift/DRIFT_SPEED)/1000);
    mpath->setTanPhi(-1*recslope/10);
    mpath->setHorizPos(recpos/10000);
    mpath->setChiSq(recchi2/100000000);
    setLateralitiesInMP(mpath,laterality);
    if(debug) cout << "In fitPerLat " << "t0 " <<  mpath->getBxTimeValue() <<" slope " << mpath->getTanPhi() <<" pos "<< mpath->getHorizPos() <<" chi2 "<< mpath->getChiSq() << " rawId " << mpath->getRawId() << endl;
  }
  //std::cout<<"Pr 1 " <<mpath->getChiSq() << endl;
  
}
/**
 * Recorre las calidades calculadas para todas las combinaciones de lateralidad
 * válidas, para determinar la calidad final asignada al "MuonPath" con el que
 * se está trabajando.
 */
void MuonPathAnalyzerInChamber::evaluateQuality(MuonPath *mPath) {
  
  // Por defecto.
  mPath->setQuality(NOPATH);
  
  if (mPath->getNPrimitivesUp() >= 4 && mPath->getNPrimitivesDown() >= 4) {
    mPath->setQuality(HIGHHIGHQ);
  }
  else if ((mPath->getNPrimitivesUp() == 4 && mPath->getNPrimitivesDown() == 3) || 
	   (mPath->getNPrimitivesUp() == 3 && mPath->getNPrimitivesDown() == 4)
	   ) {
    mPath->setQuality(HIGHLOWQ);
  } 
  else if ((mPath->getNPrimitivesUp() == 4 && mPath->getNPrimitivesDown() <= 2 && mPath->getNPrimitivesDown()>0) || 
	   (mPath->getNPrimitivesUp() <= 2 && mPath->getNPrimitivesUp()>0 && mPath->getNPrimitivesDown() == 4)
	   ){
    mPath->setQuality(CHIGHQ); //Falta añadir que el 4+0 no esta aqui
  }
  else if ((mPath->getNPrimitivesUp() == 3 && mPath->getNPrimitivesDown() == 3)
	   ){
    mPath->setQuality(LOWLOWQ);
  }
  else if ((mPath->getNPrimitivesUp() == 3 && mPath->getNPrimitivesDown() <= 2 && mPath->getNPrimitivesDown()>0) || 
	   (mPath->getNPrimitivesUp() <= 2 &&  mPath->getNPrimitivesUp()>0 && mPath->getNPrimitivesDown() == 3) || 
	   (mPath->getNPrimitivesUp() == 2 && mPath->getNPrimitivesDown() == 2)	   
	   ){
    mPath->setQuality(CLOWQ); //Falta añadir que el 3+0 no esta aqui
  }  
  else if (mPath->getNPrimitivesUp() >= 4 || mPath->getNPrimitivesDown() >= 4) {
    mPath->setQuality(HIGHQ);    
  }
  else if (mPath->getNPrimitivesUp() == 3 || mPath->getNPrimitivesDown() == 3) {
    mPath->setQuality(LOWQ);    
  }
  //std::cout<<mPath->getNPrimitivesUp()<<'+'<<mPath->getNPrimitivesDown()<<'='<<mPath->getQuality()<<endl;
}

