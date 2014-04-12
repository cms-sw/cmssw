// Aram Avetisyan; Brown University; February 15, 2008

#include "CaloOnlineTools/HcalOnlineDb/interface/DBlmapReader.h"

using namespace std;
using namespace oracle::occi;
using namespace hcal;

int i, j;
vector<int> tempVector;
stringstream sstemp;

void DBlmapReader::lrTestFunction(void){
  
  std::cout<<"Hello"<<std::endl;
  return;
}

VectorLMAP* DBlmapReader::GetLMAP(int LMversion = 30){
  HCALConfigDB * db = new HCALConfigDB();
  const std::string _accessor = "occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22";
  db -> connect( _accessor );
  
  oracle::occi::Connection * myConn = db -> getConnection();

  int CHAcount = 0; 
  VectorLMAP* lmapHBEFO = new VectorLMAP();
  
  sstemp.str("");
  sstemp<<"'"<<LMversion<<"'";
  std::string verstring = sstemp.str();
  sstemp.str("");

  try {
    Statement* stmt = myConn -> createStatement();

    std::string query = (" SELECT C.VERSION, ");
    query += " H.SIDE, H.ETA, H.PHI, ";
    query += " H.DELTA_PHI, H.DEPTH, H.SUBDETECTOR, H.RBX, H.WEDGE, ";
    query += " H.SECTOR, H.RM_SLOT, H.HPD_PIXEL, H.QIE_SLOT, H.ADC, ";
    query += " H.RM_FIBER, H.FIBER_CHANNEL, H.LETTER_CODE, H.CRATE, H.HTR_SLOT, ";
    query += " H.HTR_FPGA, H.HTR_FIBER, H.DCC_SL, H.SPIGOT, H.DCC_SLOT, ";
    query += " H.SLB_SITE, H.SLB_CHANNEL, H.SLB_CHANNEL2, H.SLB_CABLE, H.RCT_CRATE, ";
    query += " H.RCT_CARD, H.RCT_CONNECTOR, H.RCT_NAME, H.FED_ID ";
    query += " FROM CMS_HCL_HCAL_CONDITION_OWNER.HCAL_HARDWARE_LOGICAL_MAPS_V3 H inner join ";
    query += " CMS_HCL_CORE_CONDITION_OWNER.COND_DATA_SETS C ";
    query += " on C.CONDITION_DATA_SET_ID=H.CONDITION_DATA_SET_ID ";
    query += " inner join CMS_HCL_CORE_CONDITION_OWNER.KINDS_OF_CONDITIONS K ";
    query += " on K.KIND_OF_CONDITION_ID=C.KIND_OF_CONDITION_ID ";
    query += " where C.IS_RECORD_DELETED='F' and K.IS_RECORD_DELETED='F' ";
    query += " and K.NAME='HCAL hardware logical channel maps v3' ";
    query += " and C.VERSION=";
    query += verstring;
    
    //SELECT
    ResultSet *rs = stmt->executeQuery(query.c_str());

    while (rs->next() && CHAcount < 10000) {

      lmapHBEFO -> versionC . push_back( rs -> getInt(1) );
      lmapHBEFO -> sideC    . push_back( rs -> getInt(2) );
      lmapHBEFO -> etaC     . push_back( rs -> getInt(3) );
      lmapHBEFO -> phiC     . push_back( rs -> getInt(4) );
      lmapHBEFO -> dphiC    . push_back( rs -> getInt(5) );
      
      lmapHBEFO -> depthC  . push_back( rs -> getInt(6) );
      lmapHBEFO -> detC    . push_back( rs -> getString(7) );
      lmapHBEFO -> rbxC    . push_back( rs -> getString(8) );
      lmapHBEFO -> wedgeC  . push_back( rs -> getInt(9) );
      lmapHBEFO -> sectorC . push_back( rs -> getInt(10) );

      lmapHBEFO -> rmC    . push_back( rs -> getInt(11) );
      lmapHBEFO -> pixelC . push_back( rs -> getInt(12) );
      lmapHBEFO -> qieC   . push_back( rs -> getInt(13) );
      lmapHBEFO -> adcC   . push_back( rs -> getInt(14) );
      lmapHBEFO -> rm_fiC . push_back( rs -> getInt(15) );

      lmapHBEFO -> fi_chC    . push_back( rs -> getInt(16) );
      lmapHBEFO -> let_codeC . push_back( rs -> getString(17) );
      lmapHBEFO -> crateC    . push_back( rs -> getInt(18) );
      lmapHBEFO -> htrC      . push_back( rs -> getInt(19) );
      lmapHBEFO -> fpgaC     . push_back( rs -> getString(20) );

      lmapHBEFO -> htr_fiC . push_back( rs -> getInt(21) );
      lmapHBEFO -> dcc_slC . push_back( rs -> getInt(22) );
      lmapHBEFO -> spigoC  . push_back( rs -> getInt(23) );
      lmapHBEFO -> dccC    . push_back( rs -> getInt(24) );
      lmapHBEFO -> slbC    . push_back( rs -> getInt(25) );

      lmapHBEFO -> slbinC  . push_back( rs -> getString(26) );
      lmapHBEFO -> slbin2C . push_back( rs -> getString(27) );
      lmapHBEFO -> slnamC  . push_back( rs -> getString(28) );
      lmapHBEFO -> rctcraC . push_back( rs -> getInt(29) );
      lmapHBEFO -> rctcarC . push_back( rs -> getInt(30) );
    
      lmapHBEFO -> rctconC . push_back( rs -> getInt(31) );
      lmapHBEFO -> rctnamC . push_back( rs -> getString(32) );
      lmapHBEFO -> fedidC  . push_back( rs -> getInt(33) );

      lmapHBEFO -> orderC . push_back( CHAcount );
      
      CHAcount++;
    }
    //Always terminate statement
    myConn -> terminateStatement(stmt);
  }
  catch (SQLException& e) {
    std::cout<<"Couldn't get statement"<<std::endl;
  }
  db -> disconnect();
  
  return lmapHBEFO;
}

void DBlmapReader::PrintLMAP(FILE* HBEFmap, FILE* HOmap, VectorLMAP* lmapHBEFO){

  int CHAcount = lmapHBEFO->orderC.size();

  lmapHBEFO = SortByHardware(lmapHBEFO);
  for (i = 0; i < CHAcount; i++){
    if (lmapHBEFO -> orderC[i] >= CHAcount){
      std::cout<<"Bad order vector";
      break;
    }
    
    if (lmapHBEFO -> detC[lmapHBEFO -> orderC[i]] != "HO") printHBHEHF(i, HBEFmap, lmapHBEFO);
  }
  
  lmapHBEFO = SortByGeometry(lmapHBEFO);
  for (i = 0; i < CHAcount; i++){
    if (lmapHBEFO -> orderC[i] >= CHAcount){
      std::cout<<"Bad order vector";
      break;
    }
    if (lmapHBEFO -> detC[lmapHBEFO -> orderC[i]] == "HO") printHO(i, HOmap, lmapHBEFO);
  }  
  
  std::cout<<CHAcount<<std::endl;
  return;
}

void DBlmapReader::PrintEMAPfromLMAP(FILE* emap, VectorLMAP* lmapHBEFO){

  int CHAcount = lmapHBEFO->orderC.size();

  lmapHBEFO = SortByHardware(lmapHBEFO);
  for (i = 0; i < CHAcount; i++){
     if (lmapHBEFO ->detC[lmapHBEFO -> orderC[i]] != "HO") printEMAProw(i, emap, lmapHBEFO);
  }

  lmapHBEFO = SortByGeometry(lmapHBEFO);
  for (i = 0; i < CHAcount; i++){
     if (lmapHBEFO -> detC[lmapHBEFO -> orderC[i]] == "HO") printEMAProw(i, emap, lmapHBEFO);
  }
  
  return;
}


void printHBHEHF(int channel, FILE * HBEFmap, VectorLMAP * lmap){
  
  if (channel % 21 == 0){
    fprintf(HBEFmap,"# side    eta    phi   dphi  depth    det     rbx  wedge     rm  pixel   qie    adc");
    fprintf(HBEFmap,"  rm_fi  fi_ch  crate    htr   fpga  htr_fi  dcc_sl  spigo    dcc    slb  slbin  slbin2");
    fprintf(HBEFmap,"           slnam    rctcra rctcar rctcon               rctnam     fedid\n");
  }
  
  j = lmap -> orderC[channel];

  fprintf(HBEFmap,"%6d %6d %6d %6d %6d ", lmap->sideC[j],           lmap->etaC[j],            lmap->phiC[j],    lmap->dphiC[j],     lmap -> depthC[j]);
  fprintf(HBEFmap,"%6s %7s %6d %6d %6d",  lmap->detC[j].c_str(),    lmap->rbxC[j].c_str(),    lmap->wedgeC[j],  lmap->rmC[j],       lmap->pixelC[j]);
  fprintf(HBEFmap,"%6d %6d %6d %6d %6d ", lmap->qieC[j],            lmap->adcC[j],            lmap->rm_fiC[j],  lmap->fi_chC[j],    lmap->crateC[j]);
  fprintf(HBEFmap,"%6d %6s%8d %7d ",      lmap->htrC[j],            lmap->fpgaC[j].c_str(),   lmap->htr_fiC[j], lmap->dcc_slC[j]);
  fprintf(HBEFmap,"%6d %6d %6d %6s",      lmap->spigoC[j],          lmap->dccC[j],            lmap->slbC[j],    lmap->slbinC[j].c_str());
  fprintf(HBEFmap,"%8s %15s    %6d %6d ", lmap->slbin2C[j].c_str(), lmap->slnamC[j].c_str(),  lmap->rctcraC[j], lmap->rctcarC[j]);
  fprintf(HBEFmap,"%6d %20s    %6d\n",    lmap->rctconC[j],         lmap->rctnamC[j].c_str(), lmap->fedidC[j]);
}
  
void printHO(int channel, FILE * HOmap, VectorLMAP * lmap){
  //HO goes last, after 6912 entries; 6912 % 21 = 3
  if (channel % 21 == 3){    
    fprintf(HOmap,"# side    eta    phi   dphi  depth    det     rbx  sector    rm  pixel   qie    adc");
    fprintf(HOmap,"  rm_fi  fi_ch let_code  crate    htr   fpga  htr_fi  dcc_sl  spigo    dcc    slb  slbin  slbin2");
    fprintf(HOmap,"           slnam    rctcra rctcar rctcon               rctnam     fedid\n");
  }

  j = lmap -> orderC[channel];

  fprintf(HOmap,"%6d %6d %6d %6d %6d ", lmap->sideC[j],           lmap->etaC[j],              lmap->phiC[j],    lmap->dphiC[j],     lmap -> depthC[j]);
  fprintf(HOmap,"%6s %7s %6d %6d %6d",  lmap->detC[j].c_str(),    lmap->rbxC[j].c_str(),      lmap->sectorC[j], lmap->rmC[j],       lmap->pixelC[j]);
  fprintf(HOmap,"%6d %6d %6d ",         lmap->qieC[j],            lmap->adcC[j],              lmap->rm_fiC[j]);
  fprintf(HOmap,"%6d %8s %6d ",         lmap->fi_chC[j],          lmap->let_codeC[j].c_str(), lmap->crateC[j]);
  fprintf(HOmap,"%6d %6s%8d %7d ",      lmap->htrC[j],            lmap->fpgaC[j].c_str(),     lmap->htr_fiC[j], lmap->dcc_slC[j]);
  fprintf(HOmap,"%6d %6d %6d\n",        lmap->spigoC[j],          lmap->dccC[j],              lmap->fedidC[j]);


  // New Format (will update as soon as database update is complete

//   fprintf(HOmap,"# side    eta    phi   dphi  depth    det     rbx  sector    rm  pixel   qie    adc");
//   fprintf(HOmap,"  rm_fi  fi_ch let_code  crate    htr   fpga  htr_fi  dcc_sl  spigo    dcc  fedid    geo  block     lc\n");
//   fprintf(HOmap,"%6d %6d %6d %6d %6d %6s %7s %6d %6d %6d",iside,ieta,iphi,idphi,idepth,det.c_str(),rbx.c_str(),isector,irm,ipixel);
//   fprintf(HOmap,"%6d %6d %6d %6d %8s %6d %6d %6s",iqie,iadc,irm_fi,ifi_ch,letter.c_str(),icrate,ihtr,fpga.c_str());
//   fprintf(HOmap,"%8d %7d %6d %6d %6d %6d %6d %6d\n",ihtr_fi,idcc_sl,ispigot,idcc,ifed,geo,block,lc);
  
}
 
void printEMAProw(int channel, FILE * emap, VectorLMAP * lmap){
  j = lmap -> orderC[channel];

  HcalSubdetector _subdet;
  if      ( lmap->detC[j] == "HB" ) _subdet = HcalBarrel;
  else if ( lmap->detC[j] == "HE" ) _subdet = HcalEndcap;
  else if ( lmap->detC[j] == "HO" ) _subdet = HcalOuter;
  else if ( lmap->detC[j] == "HF" ) _subdet = HcalForward;
  else{
    _subdet = HcalBarrel;
    std::cerr<<"Bad Subdet"<<std::endl;
  }
  HcalDetId _hcaldetid( _subdet, (lmap->sideC[j])*(lmap->etaC[j]), lmap->phiC[j], lmap->depthC[j] );
  int hcalID = _hcaldetid . rawId(); 

  char tb = lmap->fpgaC[j][0];
  fprintf(emap,"%10d %3d %3d %2c %4d %5d",hcalID, lmap->crateC[j], lmap->htrC[j], tb, (lmap->fedidC[j] - 700), lmap->spigoC[j]);
  fprintf(emap,"%5d %8d %8s %5d %4d %6d\n", lmap->htr_fiC[j], lmap->fi_chC[j], lmap->detC[j].c_str(), (lmap->etaC[j]*lmap->sideC[j]), lmap->phiC[j], lmap->depthC[j]);

  return;
}

bool SortComp(int x, int y){
  return tempVector[x] < tempVector[y];
}

VectorLMAP* SortByHardware(VectorLMAP* lmapHBEFO){

  int CHAcount = lmapHBEFO->orderC.size();
  tempVector.clear();

  //Sort by fiber channel
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> fi_chC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by HTR fiber
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> htr_fiC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by FPGA
  for (i = 0; i < CHAcount; i++){
    if (lmapHBEFO -> fpgaC[i] == "top") tempVector.push_back (0);
    else                                tempVector.push_back (1);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by HTR
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> htrC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by crate
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> crateC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by subdetector
  for (i = 0; i < CHAcount; i++){
    if      (lmapHBEFO -> detC[i] == "HB" || lmapHBEFO -> detC[i] == "HE") tempVector.push_back (0);
    else if (lmapHBEFO -> detC[i] == "HF")                                 tempVector.push_back (1);
    else                                                                   tempVector.push_back (2);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  return lmapHBEFO;
}

VectorLMAP* SortByGeometry(VectorLMAP* lmapHBEFO){

  int CHAcount = lmapHBEFO->orderC.size();
  tempVector.clear();
  
  //Sort by eta
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> etaC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by phi
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> phiC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  //Sort by side
  for (i = 0; i < CHAcount; i++){
    tempVector.push_back (lmapHBEFO -> sideC[i]);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();
  
  //Sort by subdetector
  for (i = 0; i < CHAcount; i++){
    if      (lmapHBEFO -> detC[i] == "HB" || lmapHBEFO -> detC[i] == "HE") tempVector.push_back (0);
    else if (lmapHBEFO -> detC[i] == "HF")                                 tempVector.push_back (1);
    else                                                                   tempVector.push_back (2);
  }
  stable_sort(lmapHBEFO -> orderC.begin( ), lmapHBEFO -> orderC.end( ), SortComp);
  tempVector.clear();

  return lmapHBEFO;
}

