// 
//
// still missing: * check crystals with occupancy zero - to be cross-checked with what was masked in the DAQ
//                * info about what is in/out of daq 
//
// ------------------------------------------------------------


// analyses:
// pedestal runs -> PEDESTAL analysis only (occupancy not available from DB):
// laser runs    -> LASER analysis only
// physics runs  -> PEDESTAL ONLINE + PEDESTAL (from calib sequence) + OCCUPANCY + LASER (from calib sequence) analysis


// pedestal / pedestal online analysis:
// a) if one crystal is dead or noisy at least in one gain is recorded
// b) cuts: * pedestal value within 100-300 
//          * pedestal RMS above 2 (EB) or 2.5 (EE)
//          * pedestal RMS not zero
//          * channel working at all gains [in local pedestal runs]
//          * channel not stuck at G0 [ stuck = at gain zero in more than 1000 events ]


// laser analysis:
// a) a light module is ON if at least TWO crystals have amplitude>100. The analysis is done on light modules which are on.
// b) cuts: * APD > 400 in EB
//          * APD > 100 in EE


// occupancy analysis:
// cuts:
// a) a channel in EB (EE) is noisy if it has > 1 permill of total events above high threshold in EB (EE)  



// cosmics/physics runs analysis: combine pedestal online + laser + occupancy analyses






#include "CondTools/Ecal/interface/EcalLaserHandler.h"

#include "CondTools/Ecal/interface/EcalChannelStatusHandler.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "TProfile2D.h"
#include "TTree.h"
#include "TFile.h"
#include "TLine.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGaxis.h"
#include "TColor.h"


popcon::EcalChannelStatusHandler::EcalChannelStatusHandler(const edm::ParameterSet & ps) {

  std::cout << "EcalChannelStatus Source handler constructor\n" << std::endl;

  m_firstRun       = static_cast<unsigned int>(atoi( ps.getParameter<std::string>("firstRun").c_str()));
  m_lastRun        = static_cast<unsigned int>(atoi( ps.getParameter<std::string>("lastRun").c_str()));
  m_sid            = ps.getParameter<std::string>("OnlineDBSID");
  m_user           = ps.getParameter<std::string>("OnlineDBUser");
  m_pass           = ps.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = ps.getParameter<std::string>("LocationSource");
  m_location       = ps.getParameter<std::string>("Location");
  m_gentag         = ps.getParameter<std::string>("GenTag");
  m_runtype        = ps.getParameter<std::string>("RunType");  
  m_name           = ps.getUntrackedParameter<std::string>("name","EcalChannelStatusHandler");
  
  std::cout << m_sid << "/" << m_user << "/" << m_pass << "/" << m_location << "/" << m_gentag << std::endl;
}


popcon::EcalChannelStatusHandler::~EcalChannelStatusHandler() { }




// ------------------------------------------------
// START PEDESTALS ROUTINES

// return pedestal values at gain 12 
float popcon::EcalChannelStatusHandler::checkPedestalValueGain12( EcalPedestals::Item* item ) {
  float result = item->mean_x12;
  return result;
}

// return pedestal values at gain 6 - pedestal runs
float popcon::EcalChannelStatusHandler::checkPedestalValueGain6( EcalPedestals::Item* item ) {
  float result = item->mean_x6;
  return result;
}

// return pedestal values at gain 1 - pedestal runs
float popcon::EcalChannelStatusHandler::checkPedestalValueGain1( EcalPedestals::Item* item ) {
  float result = item->mean_x1;
  return result;
}

// return pedestal RMS at gain 12
float popcon::EcalChannelStatusHandler::checkPedestalRMSGain12( EcalPedestals::Item* item ) {
  float result = item->rms_x12;
  return result;
}

// return pedestal RMS at gain 6 - pedestal runs
float popcon::EcalChannelStatusHandler::checkPedestalRMSGain6( EcalPedestals::Item* item ) {
  float result = item->rms_x6;
  return result;
}

// return pedestal RMS at gain 1 - pedestal runs
float popcon::EcalChannelStatusHandler::checkPedestalRMSGain1( EcalPedestals::Item* item ) {
  float result = item->rms_x1;
  return result;
}





// ------------------------------------------------
// START LASER ROUTINES 

// choose 'good' EB and EE light modules. Based on DQM infos 
void popcon::EcalChannelStatusHandler::nBadLaserModules( std::map<EcalLogicID, MonLaserBlueDat> dataset_mon ) {
  
  // NB: EcalLogicId, sm_num = ecid_xt.getID1() --> for barrel is 1->18 = EB+; 19->36 = EB-  
  // NB: EcalLogicId, sm_num = ecid_xt.getID1() --> for endcap is 1=Z+, -1=Z- ;              

  // good and bad light modules and reference crystals in EB
  // theLM=0 -> L-shaped            ( TTs: 1,2,3,4; 7-8; 11-12 ... )
  // theLM=1 -> not L-shaped        ( TTs: 5-6,9-10 ... ) 
  for (int theSm=0; theSm<36; theSm++){
    for (int theLM=0; theLM<2; theLM++){
      isEBRef1[theSm][theLM]        = false;
      isEBRef2[theSm][theLM]        = false;
      isGoodLaserEBSm[theSm][theLM] = false;
    }}
  
  // good and bad light modules and reference crystals in EE [NB: only in EE-5 and EE+5 we need 2 LM... redundant]
  // theLM=0 -> light module A (ix<=50)
  // theLM=1 -> light module B (ix>50)
  for (int theSector=0; theSector<18; theSector++){
    for (int theLM=0; theLM<2; theLM++){
      isEERef1[theSector][theLM]        = false;
      isEERef2[theSector][theLM]        = false;
      isGoodLaserEESm[theSector][theLM] = false;
    }}

  
  typedef std::map<EcalLogicID, MonLaserBlueDat>::const_iterator CImon;
  EcalLogicID ecid_xt;
  MonLaserBlueDat rd_blue;

  for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {

    ecid_xt = p->first;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2();
    int yt_num = ecid_xt.getID3();

    // in which Fed/triggerTower/sectorSide I am
    int theFed =-100;
    int theTT  =-100; 
    int theIx  =-100;
    if(ecid_xt.getName()=="EB_crystal_number") {
      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
      EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
      theFed = 600+elecId.dccId();
      theTT  = elecId.towerId();   
    }
    if(ecid_xt.getName()=="EE_crystal_number") {
      if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
        EEDetId eedetid(xt_num,yt_num,sm_num);
        EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
        theFed = 600+elecId.dccId();
	theIx  = eedetid.ix(); 
      }
    }

    // to have numbers from 0 to 17 for the EE feds and 0-36 for EB                                             
    //  0 = EE-7;  1 = EE-8;  2 = EE-9;  3 = EE-1;  4 = EE-2;  5 = EE-3;  6 = EE-4;  7 = EE-5;  8 = EE-6;         
    //  9 = EE+7; 10 = EE+8; 11 = EE+9; 12 = EE+1; 13 = EE+2; 14 = EE+3; 15 = EE+4; 16 = EE+5; 17 = EE+6              
    //  0 = EB-1 --> 17 = EB-18; 18 = EB+1 --> 35 = EB+18                                                                               
    int thisFed=-100;
    if (ecid_xt.getName()=="EE_crystal_number") {
      if (theFed>600 && theFed<610) thisFed = theFed -601;
      if (theFed>645 && theFed<655) thisFed = theFed -646 + 9;
    }
    if (ecid_xt.getName()=="EB_crystal_number") { thisFed = theFed - 610; }


    // in which light module I am
    int theTTieta=-100;
    int theTTiphi=-100;
    int theLM=-100;
    if(ecid_xt.getName()=="EB_crystal_number") {
      theTTieta = (theTT-1)/4 +1;                 
      theTTiphi = (theTT-1)%4 +1;
      if (theTTieta==1 ||  theTTiphi==3 || theTTiphi==4)  theLM=0;     // L-shaped
      if (theTTieta>1  && (theTTiphi==1 || theTTiphi==2)) theLM=1;     // not L-shaped
    }
    if( (ecid_xt.getName()=="EE_crystal_number") && (thisFed==7 || thisFed==16) ) {   
      if (theIx<=50) theLM=0;
      if (theIx>50)  theLM=1;
    }

    
    // taking laser infos
    rd_blue = p->second;
    float myApdMean = rd_blue.getAPDMean();
    
    // barrel: is there at least one crystal on in this LM?
    if (ecid_xt.getName()=="EB_crystal_number") {
      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
      if ( !isEBRef1[thisFed][theLM] && !isEBRef2[thisFed][theLM] && myApdMean>100 ) isEBRef1[thisFed][theLM] = true;
      if (  isEBRef1[thisFed][theLM] && !isEBRef2[thisFed][theLM] && myApdMean>100 ) isEBRef2[thisFed][theLM] = true;
    }
    
    // endcap: is there at least one crystal on in this LM?
    if (ecid_xt.getName()=="EE_crystal_number") {
      EEDetId eedetid(xt_num,yt_num,sm_num);

      // just 1 LM per Fed
      if (thisFed!=7 && thisFed!=16) {
	if ( !isEERef1[thisFed][0] && !isEERef2[thisFed][0] && myApdMean>100 ) { isEERef1[thisFed][0] = true; isEERef1[thisFed][1] = true; }
	if (  isEERef1[thisFed][0] && !isEERef2[thisFed][0] && myApdMean>100 ) { isEERef2[thisFed][0] = true; isEERef2[thisFed][1] = true; }
      }
      
      // 2 LMs per Fed
      if (thisFed==7 || thisFed==16) {
	if ( !isEERef1[thisFed][theLM] && !isEERef2[thisFed][theLM] && myApdMean>100 ) isEERef1[thisFed][theLM] = true;
	if (  isEERef1[thisFed][theLM] && !isEERef2[thisFed][theLM] && myApdMean>100 ) isEERef2[thisFed][theLM] = true;
      }
    }
  }
  
  // check if the light module is on: at least two crystals must be on
  for (int theSm=0; theSm<36; theSm++){
    for (int theLM=0; theLM<2; theLM++){
      if (isEBRef1[theSm][theLM] && isEBRef2[theSm][theLM]) isGoodLaserEBSm[theSm][theLM] = true; 
      // std::cout << "Barrel: SM " << theSm << ", LM " << theLM << ": good = " << isGoodLaserEBSm[theSm][theLM] << std::endl;
    }}
  
  for (int theSector=0; theSector<18; theSector++){
    for (int theLM=0; theLM<2; theLM++){
      if (isEERef1[theSector][theLM] && isEERef2[theSector][theLM]) isGoodLaserEESm[theSector][theLM] = true; 
      // std::cout << "Endcap: SM " << theSector << ", LM " << theLM << ": good = " << isGoodLaserEESm[theSector][theLM] << std::endl;
    }}
}



// ----------------------------------------------------------
// START MASKING ROUTINES
void popcon::EcalChannelStatusHandler::pedOnlineMasking() {

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_ONLINE_HIGH_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_ONLINE_HIGH_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR");
  
  std::map<EcalLogicID, RunCrystalErrorsDat> theMask;
  EcalErrorMask::fetchDataSet(&theMask);
  
  if ( !theMask.empty() ) {

    std::map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
    for (m=theMask.begin(); m!=theMask.end(); m++) {
      
      EcalLogicID ecid_xt = m->first;
      int sm_num = ecid_xt.getID1();
      int xt_num = ecid_xt.getID2(); 
      int yt_num = ecid_xt.getID3(); 

      if ( (m->second).getErrorBits() & bits03 ) {

	if(ecid_xt.getName()=="EB_crystal_number") {
	  EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	  maskedOnlinePedEB.insert(std::pair<DetId, float>(ebdetid, 9999.));
	} else {
	  if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	    EEDetId eedetid(xt_num,yt_num,sm_num);
	    maskedOnlinePedEE.insert(std::pair<DetId, float>(eedetid, 9999.));
	  }}}
    }
  }
}


void popcon::EcalChannelStatusHandler::pedMasking() {

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");  

  std::map<EcalLogicID, RunCrystalErrorsDat> theMask;
  EcalErrorMask::fetchDataSet(&theMask);
  
  if ( !theMask.empty() ) {

    std::map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
    for (m=theMask.begin(); m!=theMask.end(); m++) {
      
      EcalLogicID ecid_xt = m->first;
      int sm_num = ecid_xt.getID1();
      int xt_num = ecid_xt.getID2(); 
      int yt_num = ecid_xt.getID3(); 

      if ( (m->second).getErrorBits() & bits03 ) {

	if(ecid_xt.getName()=="EB_crystal_number") {
	  EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	  maskedPedEB.insert(std::pair<DetId, float>(ebdetid, 9999.));
	} else {
	  if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	    EEDetId eedetid(xt_num,yt_num,sm_num);
	    maskedPedEE.insert(std::pair<DetId, float>(eedetid, 9999.));
	  }}}
    }
  }
}


void popcon::EcalChannelStatusHandler::laserMasking() {

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("LASER_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("LASER_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("LASER_MEAN_OVER_PN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("LASER_RMS_OVER_PN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("LASER_MEAN_TIMING_WARNING");
  bits03 |= EcalErrorDictionary::getMask("LASER_RMS_TIMING_WARNING");

  std::map<EcalLogicID, RunCrystalErrorsDat> theMask;
  EcalErrorMask::fetchDataSet(&theMask);
  
  if ( !theMask.empty() ) {

    std::map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
    for (m=theMask.begin(); m!=theMask.end(); m++) {
      
      EcalLogicID ecid_xt = m->first;
      int sm_num = ecid_xt.getID1();
      int xt_num = ecid_xt.getID2(); 
      int yt_num = ecid_xt.getID3(); 

      if ( (m->second).getErrorBits() & bits03 ) {

	if(ecid_xt.getName()=="EB_crystal_number") {
	  EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	  maskedLaserEB.insert(std::pair<DetId, float>(ebdetid, 9999.));
	} else {
	  if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	    EEDetId eedetid(xt_num,yt_num,sm_num);
	    maskedLaserEE.insert(std::pair<DetId, float>(eedetid, 9999.));
	  }}}
    }
  }
}


void popcon::EcalChannelStatusHandler::physicsMasking() {

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PHYSICS_BAD_CHANNEL_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PHYSICS_BAD_CHANNEL_ERROR");

  std::map<EcalLogicID, RunCrystalErrorsDat> theMask;
  EcalErrorMask::fetchDataSet(&theMask);
  
  if ( !theMask.empty() ) {

    std::map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
    for (m=theMask.begin(); m!=theMask.end(); m++) {
      
      EcalLogicID ecid_xt = m->first;
      int sm_num = ecid_xt.getID1();
      int xt_num = ecid_xt.getID2(); 
      int yt_num = ecid_xt.getID3(); 

      if ( (m->second).getErrorBits() & bits03 ) {

	if(ecid_xt.getName()=="EB_crystal_number") {
	  EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	  maskedPhysicsEB.insert(std::pair<DetId, float>(ebdetid, 9999.));
	} else {
	  if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	    EEDetId eedetid(xt_num,yt_num,sm_num);
	    maskedPhysicsEE.insert(std::pair<DetId, float>(eedetid, 9999.));
	  }}}
    }
  }
}


// ----------------------------------------------------------
// START DAQ EXCLUDED FEDs ROUTINES
void popcon::EcalChannelStatusHandler::daqOut(const RunIOV& _myRun) {
  RunIOV myRun = _myRun;
  std::map<EcalLogicID, RunFEConfigDat> feconfig;
  econn->fetchDataSet(&feconfig, &myRun);
  
  typedef std::map<EcalLogicID, RunFEConfigDat>::const_iterator feConfIter;
  EcalLogicID ecid_xt;
  RunFEConfigDat rd_fe;
  
  int fe_conf_id=0;
  for (feConfIter p=feconfig.begin(); p!=feconfig.end(); p++) {
    ecid_xt = p->first;
    rd_fe   = p->second;
    fe_conf_id=rd_fe.getConfigId();
  }
	
  // reading this configuration
  ODFEDAQConfig myconfig;
  myconfig.setId(fe_conf_id);
  econn->fetchConfigSet(&myconfig);
  
  // list of bad channels
  int myTT=myconfig.getBadTTId();
  ODBadTTInfo mybadTT;
  mybadTT.setId(myTT);
  econn->fetchConfigSet(&mybadTT);
  
  std::vector< ODBadTTDat > badTT_dat;
  econn->fetchConfigDataSet(&badTT_dat, &mybadTT);
  
  for(size_t iTT=0; iTT<badTT_dat.size(); iTT++){
    int fed_id = badTT_dat[iTT].getFedId();
    int tt_id  = badTT_dat[iTT].getTTId();
    if (tt_id<69) *daqFile << fed_id << " " << tt_id << std::endl;
    
    // taking the channel list for towers out of daq
    if((fed_id<=609 || fed_id>=646) && tt_id<69) { // endcap    
      
      // moving from cms fed to db fed convention
      //int db_fedId = -999;
      //if (fed_id>=604 && fed_id<=609) db_fedId = fed_id - 603 + 9;  
      //if (fed_id>=601 && fed_id<=603) db_fedId = fed_id - 600 + 15;  
      //if (fed_id>=649 && fed_id<=654) db_fedId = fed_id - 648;
      //if (fed_id>=646 && fed_id<=648) db_fedId = fed_id - 645 + 6;  	    
      // db_fedId = fed_id;    // fixme: do we need 1-18 or 6XX?
      
      std::vector<EcalLogicID> badCrystals;
      badCrystals=econn->getEcalLogicIDSet("EE_readout_tower",fed_id, fed_id, tt_id, tt_id, EcalLogicID::NULLID,EcalLogicID::NULLID, "EE_crystal_number");    
 
      for(size_t mycrys=0; mycrys<badCrystals.size(); mycrys++){
	EcalLogicID ecid_xt = badCrystals[mycrys];
	int zSide = 999;
	int log_id = ecid_xt.getLogicID();
	int yt2    = log_id%1000;                                     //  EE_crystal_number:   2010100060 -> z=-1, x=100, y=60
	int xt2    = (log_id%1000000)/1000;   	                      //  EE_crystal_number:   2012020085 -> z=+1, x=20,  y=85
	int zt2    = (log_id/1000000)%10;
	if (zt2==0) zSide = -1;
	if (zt2==2) zSide = 1;
	*daqFile2 << xt2 << " " << yt2 << " " << zSide << std::endl;
      }
    }
    
    if(fed_id>=610 && fed_id<=645 && tt_id<69) { // barrel
      
      // moving from cms fed to db fed convention	    
      int db_fedId = -999;
      if (fed_id>=610 && fed_id<=627) db_fedId = fed_id - 609 + 18;
      if (fed_id>=628 && fed_id<=645) db_fedId = fed_id - 627;
      
      std::vector<EcalLogicID> badCrystals;   
      badCrystals=econn->getEcalLogicIDSet("EB_trigger_tower",db_fedId, db_fedId, tt_id, tt_id, EcalLogicID::NULLID, EcalLogicID::NULLID, "EB_crystal_number");
      
      for(size_t mycrys=0; mycrys<badCrystals.size(); mycrys++){
	EcalLogicID ecid_xt = badCrystals[mycrys];
	int sm_num  = ecid_xt.getID1();
	int log_id  = ecid_xt.getLogicID();
	int xt2_num = log_id%10000;
	EBDetId ebdetid(sm_num,xt2_num,EBDetId::SMCRYSTALMODE);
	*daqFile2 << ebdetid.hashedIndex() << std::endl;
      } 
    }
  }
}


// ----------------------------------------------------------
//
// ANALYSIS
//
// ----------------------------------------------------------


// ----------------------------------------------------------
// LOCAL pedestal runs  

void popcon::EcalChannelStatusHandler::pedAnalysis( std::map<EcalLogicID, MonPedestalsDat> dataset_mon, std::map<EcalLogicID, MonCrystalConsistencyDat> wrongGain_mon ) {

  // to take the list of masked crystals
  pedMasking();

  // to iterate
  std::map<DetId,float>::const_iterator theIter;
  
  // logic id
  EcalLogicID ecid_xt;
  
  // to check all problems except gain zero
  typedef std::map<EcalLogicID, MonPedestalsDat>::const_iterator CImon;
  MonPedestalsDat rd_ped;
  
  for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {

    uint16_t status_now = 0;
    ecid_xt    = p->first;
    rd_ped     = p->second;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2(); 
    int yt_num = ecid_xt.getID3(); 
    
    EcalPedestals::Item ped_item;
    ped_item.mean_x1  = rd_ped.getPedMeanG1() ;
    ped_item.mean_x6  = rd_ped.getPedMeanG6();
    ped_item.mean_x12 = rd_ped.getPedMeanG12();
    ped_item.rms_x1   = rd_ped.getPedRMSG1();
    ped_item.rms_x6   = rd_ped.getPedRMSG6() ;
    ped_item.rms_x12  = rd_ped.getPedRMSG12();

    // check if pedestal RMS is bad at least at 1 gain -> noisy or very noisy channel
    float lowerCut = 999.;
    if(ecid_xt.getName()=="EB_crystal_number") { lowerCut = 2.0; }
    if(ecid_xt.getName()=="EE_crystal_number") { lowerCut = 2.5; }
    if ( (checkPedestalRMSGain12(&ped_item)>lowerCut) || (checkPedestalRMSGain6(&ped_item)>lowerCut) || (checkPedestalRMSGain1(&ped_item)>lowerCut) ) status_now = 3;   
      
    // check if pedestal value is bad at least at 1 gain -> problem in dac settings
    if ( 
	( checkPedestalValueGain12(&ped_item)>300 || checkPedestalValueGain12(&ped_item)<100 ) ||  
	( checkPedestalValueGain6(&ped_item)>300  || checkPedestalValueGain6(&ped_item)<100  ) ||  
	( checkPedestalValueGain1(&ped_item)>300  || checkPedestalValueGain1(&ped_item)<100  ) 
	)  status_now = 1;   
    
    // check if pedestal rms is zero at least at 1 gain -> dead at that channel  
    if ( checkPedestalRMSGain12(&ped_item)==0 || checkPedestalRMSGain6(&ped_item)==0 || checkPedestalRMSGain1(&ped_item)==0 ) status_now = 11;
    
    // check if the channel is fixed at G1
    if ( checkPedestalValueGain12(&ped_item)<0 && checkPedestalValueGain6(&ped_item)<0 && checkPedestalValueGain1(&ped_item)>0 ) status_now = 9;

    // check if the channel is fixed at G6
    if ( checkPedestalValueGain12(&ped_item)<0 && checkPedestalValueGain6(&ped_item)>0 && checkPedestalValueGain1(&ped_item)<0 ) status_now = 8;
    
    // check if the channel is not working at G12
    if ( checkPedestalValueGain12(&ped_item)<0 && checkPedestalValueGain6(&ped_item)>0 && checkPedestalValueGain1(&ped_item)>0 ) status_now = 8;
    

    // output in case of problems:
    if(status_now>0) {    
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
	int thisFed = 600+elecId.dccId();	

	*ResFileEB << thisFed           << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 
		   << ped_item.mean_x12 << "\t\t" << ped_item.rms_x12 << "\t\t"
		   << ped_item.mean_x6  << "\t\t" << ped_item.rms_x6  << "\t\t"
		   << ped_item.mean_x1  << "\t\t" << ped_item.rms_x1  << std::endl;

	
	// file with new problems only
	bool isOld=false;
	for(theIter=maskedPedEB.begin(); theIter!=maskedPedEB.end(); ++theIter) {
	  if ((*theIter).first==ebdetid) isOld=true;
	}
	if (!isOld) { 
	  *ResFileNewEB << thisFed           << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 
			<< ped_item.mean_x12 << "\t\t" << ped_item.rms_x12 << "\t\t"
			<< ped_item.mean_x6  << "\t\t" << ped_item.rms_x6  << "\t\t"
			<< ped_item.mean_x1  << "\t\t" << ped_item.rms_x1  << std::endl;
	}
	
      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
	  int thisFed = 600+elecId.dccId();
	  *ResFileEE << thisFed           << "\t\t" 
		     << eedetid.ix()      << "\t\t" << eedetid.iy()     << "\t\t" 
		     << eedetid.zside()   << "\t\t" << eedetid.hashedIndex()  << "\t\t"  		     
		     << ped_item.mean_x12 << "\t\t" << ped_item.rms_x12 << "\t\t"
		     << ped_item.mean_x6  << "\t\t" << ped_item.rms_x6  << "\t\t"
		     << ped_item.mean_x1  << "\t\t" << ped_item.rms_x1  << std::endl;
	  
	  bool isOld=false;
	  for(theIter=maskedPedEE.begin(); theIter!=maskedPedEE.end(); ++theIter) {
	    if ((*theIter).first==eedetid) isOld=true;
	  }
	  if (!isOld) { 
	    *ResFileNewEE << thisFed           << "\t\t" 
			  << eedetid.ix()      << "\t\t" << eedetid.iy()     << "\t\t" 
			  << eedetid.zside()   << "\t\t" << eedetid.hashedIndex()  << "\t\t"  		     
			  << ped_item.mean_x12 << "\t\t" << ped_item.rms_x12 << "\t\t"
			  << ped_item.mean_x6  << "\t\t" << ped_item.rms_x6  << "\t\t"
			  << ped_item.mean_x1  << "\t\t" << ped_item.rms_x1  << std::endl;
	  }
	}
      }
    }
  }

  
  
  // to check if a crystal is at gain zero in at least 1000 events (which is not the best! fixme)   
  typedef std::map<EcalLogicID, MonCrystalConsistencyDat>::const_iterator WGmonIter;
  MonCrystalConsistencyDat rd_wgain;
  
  for (WGmonIter p = wrongGain_mon.begin(); p != wrongGain_mon.end(); p++) {
    ecid_xt    = p->first;
    rd_wgain   = p->second;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2(); 
    int yt_num = ecid_xt.getID3(); 
    
    // output if problematic
    if (rd_wgain.getProblemsGainZero() > 1000) { 
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);
	EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
	int thisFed = 600+elecId.dccId();	
	*ResFileEB << thisFed  << "\t\t"  << ebdetid.ic() << "\t\t" << ebdetid.hashedIndex() << "\t\t" << "at gain zero"  << std::endl; 		   
		   
	// file with new problems only
	bool isOld=false;
	for(theIter=maskedPedEB.begin(); theIter!=maskedPedEB.end(); ++theIter) {
	  if ((*theIter).first==ebdetid) isOld=true;
	}
	if (!isOld) { 
	  *ResFileNewEB << thisFed << "\t\t"  << ebdetid.ic() << "\t\t" << ebdetid.hashedIndex() << "\t\t" << "at gain zero"  << std::endl; 		   
	}

      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
	  int thisFed = 600+elecId.dccId();	
	  *ResFileEE << thisFed                << "\t\t" << eedetid.ix()    << "\t\t" 
		     << eedetid.iy()           << "\t\t" << eedetid.zside() << "\t\t" 		     
		     << eedetid.hashedIndex()  << "\t\t"  		      
		     << "at gain zero"    << std::endl; 

	  bool isOld=false;
	  for(theIter=maskedPedEE.begin(); theIter!=maskedPedEE.end(); ++theIter) {
	    if ((*theIter).first==eedetid) isOld=true;
	  }
	  if (!isOld) { 
	    *ResFileNewEE << thisFed                << "\t\t" << eedetid.ix()    << "\t\t" 
			  << eedetid.iy()           << "\t\t" << eedetid.zside() << "\t\t" 		     
			  << eedetid.hashedIndex()  << "\t\t"  		      
			  << "at gain zero"    << std::endl; 
	  }
	}
      }   
    }
  }

}


// ------------------------------------------------------------
//
// LOCAL laser runs

void popcon::EcalChannelStatusHandler::laserAnalysis( std::map<EcalLogicID, MonLaserBlueDat> dataset_mon ) { 
  
  // to take the list of masked crystals
  laserMasking();

  // to iterate
  std::map<DetId,float>::const_iterator theIter;

  typedef std::map<EcalLogicID, MonLaserBlueDat>::const_iterator CImon;
  EcalLogicID ecid_xt;
  MonLaserBlueDat rd_blue;
  
  for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
    ecid_xt    = p->first;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2(); 
    int yt_num = ecid_xt.getID3(); 
    
    // in which Fed/triggerTower/sectorSide I am
    int theFed = -100;
    int theTT  = -100; 
    int theIx  = -100;
    if(ecid_xt.getName()=="EB_crystal_number") {
      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
      EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
      theFed = 600+elecId.dccId();
      theTT  = elecId.towerId();   
    }   
    if(ecid_xt.getName()=="EE_crystal_number") {   
      if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	EEDetId eedetid(xt_num,yt_num,sm_num);
	EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
	theFed = 600+elecId.dccId();
	theIx  = eedetid.ix(); 
      }
    }
    
    // to have numbers from 0 to 17 for the EE feds and 0-36 for EB 
    int thisFed=-100;
    if (ecid_xt.getName()=="EE_crystal_number") {
      if (theFed>600 && theFed<610) thisFed = theFed -601;    
      if (theFed>645 && theFed<655) thisFed = theFed -646 + 9;    
    }
    if (ecid_xt.getName()=="EB_crystal_number") { thisFed = theFed - 610; }

    // in which light module I am
    int theTTieta =-100;
    int theTTiphi =-100;
    int theLM     =-100;
    if(ecid_xt.getName()=="EB_crystal_number") {
      theTTieta = (theTT-1)/4 +1;                
      theTTiphi = (theTT-1)%4 +1;
      if (theTTieta==1 ||  theTTiphi==3 || theTTiphi==4)  theLM=0;     // L-shaped
      if (theTTieta>1  && (theTTiphi==1 || theTTiphi==2)) theLM=1;     // not L-shaped
    }
    if( (ecid_xt.getName()=="EE_crystal_number") && (thisFed==7 || thisFed==16) ) {   
      if (theIx<=50) theLM=0;  
      if (theIx>50)  theLM=1;
    }

    // check if APD mean is bad
    uint16_t status_now = 0;
    rd_blue = p->second;
    float myApdMean = rd_blue.getAPDMean();
    
    if(ecid_xt.getName()=="EB_crystal_number") {
      if ( (myApdMean < 400) && isGoodLaserEBSm[thisFed][theLM] ) status_now = 2;      
    }    

    if(ecid_xt.getName()=="EE_crystal_number") {
      if ( myApdMean<100 ) {
	if (  thisFed!=7 && thisFed!=16  && isGoodLaserEESm[thisFed][0] )     status_now = 2;   
	if ( (thisFed==7 || thisFed==16) && isGoodLaserEESm[thisFed][theLM] ) status_now = 2;   
      }    
    }

    // output if problematic
    if (status_now>0) { 
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	*ResFileEB << theFed  << "\t\t" << ebdetid.ic() << "\t\t" << ebdetid.hashedIndex() << "\t\t" << myApdMean << std::endl;		   
	
	// file with new problems only
	bool isOld=false;
	for(theIter=maskedLaserEB.begin(); theIter!=maskedLaserEB.end(); ++theIter) {
	  if ((*theIter).first==ebdetid) isOld=true;
	}
	if (!isOld) { *ResFileNewEB << thisFed << "\t\t" << ebdetid.ic() << "\t\t" << ebdetid.hashedIndex() << "\t\t" << myApdMean << std::endl; }
	
      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  *ResFileEE << theFed        << "\t\t" << eedetid.ix()        << "\t\t" 
		     << eedetid.iy()  << "\t\t" << eedetid.zside()     << "\t\t" 
		     << eedetid.hashedIndex()   << "\t\t" << myApdMean << std::endl;
	  
	  // file with new problems only
	  bool isOld=false;
	  for(theIter=maskedLaserEE.begin(); theIter!=maskedLaserEE.end(); ++theIter) {
	    if ((*theIter).first==eedetid) isOld=true;
	  }
	  if (!isOld) {
	    *ResFileEE << theFed        << "\t\t" << eedetid.ix()        << "\t\t" 
		       << eedetid.iy()  << "\t\t" << eedetid.zside()     << "\t\t" 
		       << eedetid.hashedIndex()   << "\t\t" << myApdMean << std::endl; 
	  }
	}       
      }      
    }
  }
}



// ------------------------------------------------------------
//
// COSMICS/PHYSICS ANALYSIS: infos from pedestal online, laser and occupancy analysis

void popcon::EcalChannelStatusHandler::cosmicsAnalysis( std::map<EcalLogicID, MonPedestalsOnlineDat> pedestalO_mon, 
							std::map<EcalLogicID, MonCrystalConsistencyDat> wrongGain_mon, 
							std::map<EcalLogicID, MonLaserBlueDat> laser_mon, 
							std::map<EcalLogicID, MonOccupancyDat> occupancy_mon ) { 

  // to take the list of masked crystals for the diffrent analyses
  pedOnlineMasking();
  laserMasking();
  physicsMasking();
  
  std::map<DetId,float>::const_iterator theIter;
  std::map<DetId, float> badPedOnEB,        badPedOnEE;
  std::map<DetId, float> badPedOnRmsEB,     badPedOnRmsEE;
  std::map<DetId, float> badGainEB,         badGainEE;
  std::map<DetId, float> badLaserEB,        badLaserEE;  
  std::map<DetId, float> badOccHighEB,      badOccHighEE;  

  typedef std::map<EcalLogicID, MonPedestalsOnlineDat>::const_iterator CImonPedO;  
  typedef std::map<EcalLogicID, MonCrystalConsistencyDat>::const_iterator CImonCons;
  typedef std::map<EcalLogicID, MonOccupancyDat>::const_iterator CImonOcc;
  typedef std::map<EcalLogicID, MonLaserBlueDat>::const_iterator CImonLaser;
  MonPedestalsOnlineDat    rd_ped0;  
  MonCrystalConsistencyDat rd_wgain;
  MonOccupancyDat          rd_occ;
  MonLaserBlueDat          rd_blue;

  // logic id
  EcalLogicID ecid_xt;
  
  // to be used after: total number of entries above high threshold 
  float totEntriesAboveHighThrEB=0.; 
  float totEntriesAboveHighThrEE=0.; 
  for (CImonOcc p=occupancy_mon.begin(); p!=occupancy_mon.end(); p++) {
    ecid_xt = p->first;
    rd_occ  = p->second;
    float highOcc=rd_occ.getEventsOverHighThreshold();
    if (ecid_xt.getName()=="EB_crystal_number" && highOcc>-1) totEntriesAboveHighThrEB = totEntriesAboveHighThrEB + highOcc;
    if (ecid_xt.getName()=="EE_crystal_number" && highOcc>-1) totEntriesAboveHighThrEE = totEntriesAboveHighThrEE + highOcc;
  }


  // A) creating the list of all bad channels: searching for problems based on pedestal online except gain zero
  for (CImonPedO p = pedestalO_mon.begin(); p != pedestalO_mon.end(); p++) {
    
    bool isWrong = false;
    ecid_xt = p->first;
    rd_ped0 = p->second;
    int sm_num=ecid_xt.getID1();
    int xt_num=ecid_xt.getID2(); 
    int yt_num=ecid_xt.getID3(); 
    
    EcalPedestals::Item ped_item;
    ped_item.mean_x12 = rd_ped0.getADCMeanG12();
    ped_item.rms_x12  = rd_ped0.getADCRMSG12();
    
    float lowerCut=999.;
    if(ecid_xt.getName()=="EB_crystal_number") { lowerCut = 2.0; }
    if(ecid_xt.getName()=="EE_crystal_number") { lowerCut = 2.5; }
    if ( checkPedestalRMSGain12(&ped_item)>lowerCut ) isWrong = true;
    if ( checkPedestalValueGain12(&ped_item)>300 )    isWrong = true;   
    if ( checkPedestalValueGain12(&ped_item)<100 )    isWrong = true;
    if ( checkPedestalRMSGain12(&ped_item)==0 )       isWrong = true;      

    // is this channel already in the list?
    if (isWrong) { 
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	theIter = badPedOnEB.find(ebdetid);
	if (theIter==badPedOnEB.end()) {
	  badPedOnEB.insert   (std::pair<DetId, float>(ebdetid, rd_ped0.getADCMeanG12()));
	  badPedOnRmsEB.insert(std::pair<DetId, float>(ebdetid, rd_ped0.getADCRMSG12()));	  
	}
      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  theIter = badPedOnEE.find(eedetid);
	  if (theIter==badPedOnEE.end()) {   
	    badPedOnEE.insert   (std::pair<DetId, float>(eedetid, rd_ped0.getADCMeanG12()));
	    badPedOnRmsEE.insert(std::pair<DetId, float>(eedetid, rd_ped0.getADCRMSG12()));	  
	  }
	}
      }    
    }
  }


  // B) creating the list of all bad channels: searching for channels at gain zero at least in 100 events
  for (CImonCons p = wrongGain_mon.begin(); p != wrongGain_mon.end(); p++) {
    
    bool isWrong = false;
    ecid_xt    = p->first;
    rd_wgain   = p->second;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2(); 
    int yt_num = ecid_xt.getID3(); 
    
    if (rd_wgain.getProblemsGainZero() > 1000) {isWrong = true; }
    
    if (isWrong) { 
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	theIter = badGainEB.find(ebdetid);
	if (theIter==badGainEB.end()) badGainEB.insert(std::pair<DetId, float>(ebdetid, 999.));
      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  theIter = badGainEE.find(eedetid);
	  if (theIter==badGainEE.end()) badGainEE.insert(std::pair<DetId, float>(eedetid, 999.));	  
	}    
      }
    }
  }


  // C) creating the list of all bad channels: searching for channels with bad occupancy [ too high ]
  for (CImonOcc p=occupancy_mon.begin(); p!=occupancy_mon.end(); p++) {
    
    // logic id
    ecid_xt = p->first;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2(); 
    int yt_num = ecid_xt.getID3(); 

    // occupancy
    rd_occ = p->second;    

    bool isWrong = false;
    float occAvg = -999.;
    if ( ecid_xt.getName()=="EB_crystal_number" ) { 
      occAvg = rd_occ.getEventsOverHighThreshold()/totEntriesAboveHighThrEB;
      if (occAvg>0.001) isWrong=true;  
    }
    if ( ecid_xt.getName()=="EE_crystal_number" ) { 
      occAvg = rd_occ.getEventsOverHighThreshold()/totEntriesAboveHighThrEE;
      if (occAvg>0.001) isWrong=true;  
    }

    if (isWrong) { 
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	theIter = badOccHighEB.find(ebdetid);
	if (theIter==badOccHighEB.end()) badOccHighEB.insert(std::pair<DetId, float>(ebdetid, occAvg));
      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  theIter = badOccHighEE.find(eedetid);
	  if (theIter==badOccHighEE.end()) badOccHighEE.insert(std::pair<DetId, float>(eedetid, occAvg));
	}
      }    
    }
  }


  // D) creating the list of all bad channels: searching for channels with bad laser amplitude [among those covered by the calibration sequence]
  for (CImonLaser p=laser_mon.begin(); p!=laser_mon.end(); p++) {

    // logic id
    ecid_xt    = p->first;
    int sm_num = ecid_xt.getID1();
    int xt_num = ecid_xt.getID2(); 
    int yt_num = ecid_xt.getID3(); 
    
    // in which Fed/triggerTower/sectorSide I am
    int theFed = -100;
    int theTT  = -100; 
    int theIx  = -100;
    if(ecid_xt.getName()=="EB_crystal_number") {
      EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
      EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
      theFed = 600+elecId.dccId();
      theTT  = elecId.towerId();   
    }   
    if(ecid_xt.getName()=="EE_crystal_number") {   
      if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	EEDetId eedetid(xt_num,yt_num,sm_num);
	EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
	theFed = 600+elecId.dccId();
	theIx  = eedetid.ix(); 
      }
    }
    
    // to have numbers from 0 to 17 for the EE feds and 0-36 for EB 
    int thisFed=-100;
    if (ecid_xt.getName()=="EE_crystal_number") {
      if (theFed>600 && theFed<610) thisFed = theFed -601;    
      if (theFed>645 && theFed<655) thisFed = theFed -646 + 9;    
    }
    if (ecid_xt.getName()=="EB_crystal_number") { thisFed = theFed - 610; }

    // in which light module I am
    int theTTieta =-100;
    int theTTiphi =-100;
    int theLM     =-100;
    if(ecid_xt.getName()=="EB_crystal_number") {
      theTTieta = (theTT-1)/4 +1;              
      theTTiphi = (theTT-1)%4 +1;
      if (theTTieta==1 ||  theTTiphi==3 || theTTiphi==4)  theLM=0;     // L-shaped
      if (theTTieta>1  && (theTTiphi==1 || theTTiphi==2)) theLM=1;     // not L-shaped
    }
    if( (ecid_xt.getName()=="EE_crystal_number") && (thisFed==7 || thisFed==16) ) {   
      if (theIx<=50) theLM=0;  
      if (theIx>50)  theLM=1;
    }
    
    // APD mean value
    rd_blue = p->second;
    float myApdMean = rd_blue.getAPDMean();
    
    bool isWrong = false;
    if(ecid_xt.getName()=="EB_crystal_number") {
      if ( (myApdMean < 400) && isGoodLaserEBSm[thisFed][theLM] ) isWrong=true;
    }        
    if(ecid_xt.getName()=="EE_crystal_number") {
      if ( myApdMean<100 ) {
	if (  thisFed!=7 && thisFed!=16  && isGoodLaserEESm[thisFed][0] )     isWrong=true;
	if ( (thisFed==7 || thisFed==16) && isGoodLaserEESm[thisFed][theLM] ) isWrong=true;
      }    
    }
    
    if (isWrong) { 
      if(ecid_xt.getName()=="EB_crystal_number") {
	EBDetId ebdetid(sm_num,xt_num,EBDetId::SMCRYSTALMODE);	      
	theIter = badLaserEB.find(ebdetid);
	if (theIter==badLaserEB.end()) badLaserEB.insert(std::pair<DetId, float>(ebdetid, myApdMean));
      } else {      
	if(EEDetId::validDetId(xt_num,yt_num,sm_num)){
	  EEDetId eedetid(xt_num,yt_num,sm_num);
	  theIter = badLaserEE.find(eedetid);
	  if (theIter==badLaserEE.end()) badLaserEE.insert(std::pair<DetId, float>(eedetid, myApdMean));
	}
      }    
    } 
  }


  // check if the crystal is in the vector and fill the summary file
  std::map<DetId, float>::const_iterator theIterPedOn;
  std::map<DetId, float>::const_iterator theIterPedOnRms;
  std::map<DetId, float>::const_iterator theIterGain;
  std::map<DetId, float>::const_iterator theIterLaser;
  std::map<DetId, float>::const_iterator theIterOccHigh;


  // EB, first check - loop over pedestal online
  for(theIterPedOn=badPedOnEB.begin(); theIterPedOn!=badPedOnEB.end(); ++theIterPedOn) {
    float thePedOn    = (*theIterPedOn).second;
    float thePedOnRms = 9999.;
    float theGain     = 9999.;
    float theLaser    = 9999.;
    float theOccHigh  = 9999.;

    theIterPedOnRms = badPedOnRmsEB.find((*theIterPedOn).first);
    theIterGain     = badGainEB.find    ((*theIterPedOn).first);
    theIterLaser    = badLaserEB.find   ((*theIterPedOn).first);
    theIterOccHigh  = badOccHighEB.find ((*theIterPedOn).first);

    if (theIterPedOnRms!=badPedOnRmsEB.end()) thePedOnRms = (*theIterPedOnRms).second;
    if (theIterLaser!=badLaserEB.end())       theLaser    = (*theIterLaser).second;
    if (theIterOccHigh!=badOccHighEB.end())   theOccHigh  = (*theIterOccHigh).second;
    
    int thisFed=-100;
    EBDetId ebdetid((*theIterPedOn).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
    thisFed = 600+elecId.dccId();

    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    bool isNewLaser=true;
    bool isNewPhysics=true;
    for(theIter=maskedOnlinePedEB.begin(); theIter!=maskedOnlinePedEB.end(); ++theIter) { if ((*theIter).first==ebdetid) isNewPed=false; }

    for(theIter=maskedLaserEB.begin();     theIter!=maskedLaserEB.end(); ++theIter)     { if ((*theIter).first==ebdetid) isNewLaser=false; }

    for(theIter=maskedPhysicsEB.begin();   theIter!=maskedPhysicsEB.end(); ++theIter)   { if ((*theIter).first==ebdetid) isNewPhysics=false; }

    if ( isNewPed || (theLaser!=9999 && isNewLaser) || (theOccHigh!=9999 && isNewPhysics) ) isNew=true;
      
    if (theIterGain!=badGainEB.end()) {
      *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		 << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		 << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh  << std::endl;
      
      if (isNew) { 
	*ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		      << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		      << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh  << std::endl;
	
	float thisEtaFill=float(0);
	if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
	if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
	newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
      }
    } else {
      *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		 << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		 << theGain     << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh  << std::endl;
      
      if (isNew) { 
	*ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		      << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		      << theGain     << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh  << std::endl;

	float thisEtaFill=float(0);
	if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
	if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
	newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
      } 
    }
  }
  


  // EB, second check - loop over laser
  for(theIterLaser=badLaserEB.begin(); theIterLaser!=badLaserEB.end(); ++theIterLaser) {

    // remove already included channels
    theIterPedOnRms = badPedOnRmsEB.find((*theIterLaser).first);
    if (theIterPedOnRms!=badPedOnRmsEB.end()) continue; 

    float thePedOn    = 9999.;
    float thePedOnRms = 9999.;
    float theGain     = 9999.;
    float theLaser    = (*theIterLaser).second;
    float theOccHigh  = 9999.;
    
    theIterGain     = badGainEB.find    ((*theIterLaser).first);
    theIterOccHigh  = badOccHighEB.find ((*theIterLaser).first);
    if (theIterOccHigh!=badOccHighEB.end()) theOccHigh = (*theIterOccHigh).second;
    
    int thisFed=-100;
    EBDetId ebdetid((*theIterLaser).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
    thisFed = 600+elecId.dccId();

    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    bool isNewLaser=true;
    bool isNewPhysics=true;
    for(theIter=maskedOnlinePedEB.begin(); theIter!=maskedOnlinePedEB.end(); ++theIter) { if ((*theIter).first==ebdetid) isNewPed=false; }

    for(theIter=maskedLaserEB.begin();     theIter!=maskedLaserEB.end();     ++theIter) { if ((*theIter).first==ebdetid) isNewLaser=false; }

    for(theIter=maskedPhysicsEB.begin();   theIter!=maskedPhysicsEB.end();   ++theIter) { if ((*theIter).first==ebdetid) isNewPhysics=false; }

    if ( (isNewPed && theIterGain!=badGainEB.end() ) || (isNewLaser) || (theOccHigh!=9999 && isNewPhysics) ) isNew=true;
        
    if (theIterGain!=badGainEB.end()) {
      *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		 << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		 << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh  << std::endl;

      if (isNew) { 
	*ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		      << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		      << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh  << std::endl;

	float thisEtaFill=float(0);
	if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
	if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
	newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
      }
    } else {
      *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		 << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		 << theGain     << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh  << std::endl;
      
      if (isNew) { 
	*ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		      << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		      << theGain     << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh  << std::endl;

	float thisEtaFill=float(0);
	if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
	if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
	newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
      }
    } 
  }
  

  // EB, third check: loop over occupancy
  for(theIterOccHigh = badOccHighEB.begin(); theIterOccHigh != badOccHighEB.end(); ++theIterOccHigh) {

    // remove already included  channels
    theIterPedOnRms = badPedOnRmsEB.find((*theIterOccHigh).first);
    theIterLaser    = badLaserEB.find   ((*theIterOccHigh).first);
    if (theIterPedOnRms!=badPedOnRmsEB.end()) continue; 
    if (theIterLaser!=badLaserEB.end())       continue; 

    float thePedOn    = 9999.;
    float thePedOnRms = 9999.;
    float theGain     = 9999.;
    float theLaser    = 9999.;
    float theOccHigh  = (*theIterOccHigh).second;
    theIterGain   = badGainEB.find  ((*theIterOccHigh).first);
    
    int thisFed=-100;
    EBDetId ebdetid((*theIterOccHigh).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
    thisFed = 600+elecId.dccId();
    
    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    bool isNewPhysics=true;
    for(theIter=maskedOnlinePedEB.begin(); theIter!=maskedOnlinePedEB.end(); ++theIter) { if ((*theIter).first==ebdetid) isNewPed=false; }

    for(theIter=maskedPhysicsEB.begin();   theIter!=maskedPhysicsEB.end();   ++theIter) { if ((*theIter).first==ebdetid) isNewPhysics=false; }

    if ( (isNewPed && theIterGain!=badGainEB.end() ) || (isNewPhysics) ) isNew=true;
    
    if (theIterGain!=badGainEB.end()) {
      *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		 << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		 << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh  << std::endl;
      
      if (isNew) { 
	*ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		      << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		      << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh  << std::endl;

	float thisEtaFill=float(0);
	if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
	if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
	newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
      }
    } else {
      *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		 << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		 << theGain     << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh  << std::endl;
      
      if (isNew) { 
	*ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		      << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		      << theGain     << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh  << std::endl;

	float thisEtaFill=float(0);
	if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
	if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
	newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
      } 
    }
  }
  
  // EB, fourth check: loop over consistency
  for(theIterGain = badGainEB.begin(); theIterGain != badGainEB.end(); ++theIterGain) {
    
    // remove already included  channels
    theIterPedOnRms = badPedOnRmsEB.find((*theIterGain).first);
    theIterLaser    = badLaserEB.find   ((*theIterGain).first);
    theIterOccHigh  = badOccHighEB.find ((*theIterGain).first);
    if (theIterPedOnRms!=badPedOnRmsEB.end()) continue; 
    if (theIterLaser!=badLaserEB.end())       continue; 
    if (theIterOccHigh!=badOccHighEB.end())   continue;
    
    float thePedOn    = 9999.;
    float thePedOnRms = 9999.;
    float theLaser    = 9999.;
    float theOccHigh  = 9999.;
    
    int thisFed=-100;
    EBDetId ebdetid((*theIterGain).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(ebdetid);
    thisFed = 600+elecId.dccId();

    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    for(theIter=maskedOnlinePedEB.begin(); theIter!=maskedOnlinePedEB.end(); ++theIter) { if ((*theIter).first==ebdetid) isNewPed=false; }

    if ( isNewPed && theIterGain!=badGainEB.end() ) isNew=true;
    
    *ResFileEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
	       << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
	       << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
	       << theOccHigh  << std::endl;
    
    if (isNew) { 
      *ResFileNewEB << thisFed     << "\t\t" << ebdetid.ic()     << "\t\t" << ebdetid.hashedIndex() << "\t\t" 		 
		    << thePedOn    << "\t\t" << thePedOnRms      << "\t\t" 
		    << "gainZero"  << "\t\t" << theLaser         << "\t\t" 
		    << theOccHigh  << std::endl;

      float thisEtaFill=float(0);
      if (ebdetid.ieta()>0) thisEtaFill = ebdetid.ieta() - 0.5;
      if (ebdetid.ieta()<0) thisEtaFill = ebdetid.ieta();
      newBadEB_ -> Fill( (ebdetid.iphi()-0.5), thisEtaFill, 2);
    }
  } 

  
  // EE, first check: loop over pedestal online
  for(theIterPedOn = badPedOnEE.begin(); theIterPedOn != badPedOnEE.end(); ++theIterPedOn) {

    float thePedOn    = (*theIterPedOn).second;
    float thePedOnRms = 9999.;
    float theGain     = 9999.;
    float theLaser    = 9999.;
    float theOccHigh  = 9999.;

    theIterPedOnRms = badPedOnRmsEE.find((*theIterPedOn).first);
    theIterGain     = badGainEE.find    ((*theIterPedOn).first);
    theIterLaser    = badLaserEE.find   ((*theIterPedOn).first);
    theIterOccHigh  = badOccHighEE.find ((*theIterPedOn).first);

    if (theIterPedOnRms!=badPedOnRmsEE.end()) thePedOnRms = (*theIterPedOnRms).second;
    if (theIterLaser   !=badLaserEE.end())    theLaser    = (*theIterLaser).second;
    if (theIterOccHigh !=badOccHighEE.end())  theOccHigh  = (*theIterOccHigh).second;
    
    int thisFed=-100;
    EEDetId eedetid((*theIterPedOn).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
    thisFed = 600+elecId.dccId();

    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    bool isNewLaser=true;
    bool isNewPhysics=true;
    for(theIter=maskedOnlinePedEE.begin(); theIter!=maskedOnlinePedEE.end(); ++theIter) { if ((*theIter).first==eedetid) isNewPed=false; }

    for(theIter=maskedLaserEE.begin();     theIter!=maskedLaserEE.end();     ++theIter) { if ((*theIter).first==eedetid) isNewLaser=false; }
      
    for(theIter=maskedPhysicsEE.begin();   theIter!=maskedPhysicsEE.end();   ++theIter) { if ((*theIter).first==eedetid) isNewPhysics=false; }
      
    if ( isNewPed || (theLaser!=9999 && isNewLaser) || (theOccHigh!=9999 && isNewPhysics) ) isNew=true;
    
    if (theIterGain!=badGainEE.end()) {
      *ResFileEE << thisFed                << "\t\t" << eedetid.ix()     << "\t\t" 
		 << eedetid.iy()           << "\t\t" << eedetid.zside()  << "\t\t" 
		 << eedetid.hashedIndex()  << "\t\t"  		     
		 << thePedOn               << "\t\t" << thePedOnRms      << "\t\t" 
		 << "gainZero"             << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh             << std::endl;

      if (isNew) {
	*ResFileNewEE << thisFed                << "\t\t" << eedetid.ix()     << "\t\t" 
		      << eedetid.iy()           << "\t\t" << eedetid.zside()  << "\t\t" 
		      << eedetid.hashedIndex()  << "\t\t"  		     
		      << thePedOn               << "\t\t" << thePedOnRms      << "\t\t" 
		      << "gainZero"             << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh             << std::endl;

	if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
	if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      }
    } else {
      *ResFileEE << thisFed                << "\t\t" << eedetid.ix()     << "\t\t" 
		 << eedetid.iy()           << "\t\t" << eedetid.zside()  << "\t\t" 
		 << eedetid.hashedIndex()  << "\t\t"  		     
		 << thePedOn               << "\t\t" << thePedOnRms      << "\t\t" 
		 << theGain                << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh             << std::endl;

      if (isNew) {
	*ResFileNewEE << thisFed                << "\t\t" << eedetid.ix()     << "\t\t" 
		      << eedetid.iy()           << "\t\t" << eedetid.zside()  << "\t\t" 
		      << eedetid.hashedIndex()  << "\t\t"  		     
		      << thePedOn               << "\t\t" << thePedOnRms      << "\t\t" 
		      << theGain                << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh             << std::endl;
	
	if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
	if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      }
    } 
  }
  
  
  // EE, second check: loop over laser
  for(theIterLaser = badLaserEE.begin(); theIterLaser != badLaserEE.end(); ++theIterLaser) {

    // remove already included  channels
    theIterPedOnRms = badPedOnRmsEE.find((*theIterLaser).first);
    if (theIterPedOnRms!=badPedOnRmsEE.end()) continue; 

    float thePedOn    = 9999.;
    float thePedOnRms = 9999.;
    float theGain     = 9999.;
    float theLaser    = (*theIterLaser).second;
    float theOccHigh  = 9999.;
    
    theIterGain     = badGainEE.find    ((*theIterLaser).first);
    theIterOccHigh  = badOccHighEE.find ((*theIterLaser).first);
    if (theIterOccHigh!=badOccHighEE.end()) theOccHigh = (*theIterOccHigh).second;
    
    int thisFed=-100;
    EEDetId eedetid((*theIterLaser).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
    thisFed = 600+elecId.dccId();
    
    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    bool isNewLaser=true;
    bool isNewPhysics=true;
    for(theIter=maskedOnlinePedEE.begin(); theIter!=maskedOnlinePedEE.end(); ++theIter) { if ((*theIter).first==eedetid) isNewPed=false; }
      
    for(theIter=maskedLaserEE.begin();     theIter!=maskedLaserEE.end();     ++theIter) { if ((*theIter).first==eedetid) isNewLaser=false; }
      
    for(theIter=maskedPhysicsEE.begin();   theIter!=maskedPhysicsEE.end();   ++theIter) { if ((*theIter).first==eedetid) isNewPhysics=false; }

    if ( (isNewPed && theIterGain!=badGainEE.end() ) || (isNewLaser) || (theOccHigh!=9999 && isNewPhysics) ) isNew=true;
    
    if (theIterGain!=badGainEE.end()) {
      *ResFileEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		 << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		 << eedetid.hashedIndex()  << "\t\t" 		 
		 << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		 << "gainZero"   << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh   << std::endl;

      if (isNew) { 
	*ResFileNewEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		      << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		      << eedetid.hashedIndex()  << "\t\t" 		 
		      << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		      << "gainZero"   << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh   << std::endl;

	if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
	if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      }
    } else {
      *ResFileEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		 << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		 << eedetid.hashedIndex()  << "\t\t" 		 
		 << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		 << theGain      << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh   << std::endl;

      if (isNew) { 
	*ResFileNewEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		      << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		      << eedetid.hashedIndex()  << "\t\t" 		 
		      << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		      << theGain      << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh   << std::endl;

	if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
	if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      }
    } 
  }


  // third check: loop over occupancy
  for(theIterOccHigh = badOccHighEE.begin(); theIterOccHigh != badOccHighEE.end(); ++theIterOccHigh) {

    // remove already included  channels
    theIterPedOnRms = badPedOnRmsEE.find((*theIterOccHigh).first);
    theIterLaser    = badLaserEE.find   ((*theIterOccHigh).first);
    if (theIterPedOnRms!=badPedOnRmsEE.end()) continue; 
    if (theIterLaser!=badLaserEE.end())       continue; 

    float thePedOn    = 9999.;
    float thePedOnRms = 9999.;
    float theGain     = 9999.;
    float theLaser    = 9999.;
    float theOccHigh  = (*theIterOccHigh).second;
    theIterGain    = badGainEE.find  ((*theIterOccHigh).first);
    
    int thisFed=-100;
    EEDetId eedetid((*theIterOccHigh).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
    thisFed = 600+elecId.dccId();
    
    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    bool isNewPhysics=true;
    for(theIter=maskedOnlinePedEE.begin(); theIter!=maskedOnlinePedEE.end(); ++theIter) { if ((*theIter).first==eedetid) isNewPed=false; }

    for(theIter=maskedPhysicsEE.begin();   theIter!=maskedPhysicsEE.end();   ++theIter) { if ((*theIter).first==eedetid) isNewPhysics=false; }
      
    if ( (isNewPed && theIterGain!=badGainEE.end() ) || (isNewPhysics) ) isNew=true;

    if (theIterGain!=badGainEE.end()) {
      *ResFileEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		 << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		 << eedetid.hashedIndex()  << "\t\t" 		 
		 << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		 << "gainZero"   << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh   << std::endl;

      if (isNew) { 
	*ResFileNewEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		      << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		      << eedetid.hashedIndex()  << "\t\t" 		 
		      << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		      << "gainZero"   << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh   << std::endl;

	if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
	if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      }
    } else {
      *ResFileEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		 << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		 << eedetid.hashedIndex()  << "\t\t" 		 
		 << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		 << theGain      << "\t\t" << theLaser         << "\t\t" 
		 << theOccHigh   << std::endl;
      
      if (isNew) { 
	*ResFileNewEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		      << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		      << eedetid.hashedIndex()  << "\t\t" 		 
		      << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		      << theGain      << "\t\t" << theLaser         << "\t\t" 
		      << theOccHigh   << std::endl;
	
	if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
	if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      } 
    }
  }

  // EE, fourth check: loop over consistency
  for(theIterGain = badGainEE.begin(); theIterGain != badGainEE.end(); ++theIterGain) {
    
    // remove already included  channels
    theIterPedOnRms = badPedOnRmsEE.find((*theIterGain).first);
    theIterLaser    = badLaserEE.find   ((*theIterGain).first);
    theIterOccHigh  = badOccHighEE.find ((*theIterGain).first);
    if (theIterPedOnRms!=badPedOnRmsEE.end()) continue; 
    if (theIterLaser!=badLaserEE.end())       continue; 
    if (theIterOccHigh!=badOccHighEE.end())   continue;    
    float thePedOn    = 9999.;
    float thePedOnRms = 9999.;
    float theLaser    = 9999.;
    float theOccHigh  = 9999.;
    
    int thisFed=-100;
    EEDetId eedetid((*theIterGain).first);
    EcalElectronicsId elecId = ecalElectronicsMap_.getElectronicsId(eedetid);
    thisFed = 600+elecId.dccId();
    
    // new problems only
    bool isNew=false;
    bool isNewPed=true;
    for(theIter=maskedOnlinePedEE.begin(); theIter!=maskedOnlinePedEE.end(); ++theIter) { if ((*theIter).first==eedetid) isNewPed=false; }
      
    if ( isNewPed && theIterGain!=badGainEE.end() ) isNew=true;
    
    *ResFileEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
	       << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
	       << eedetid.hashedIndex()  << "\t\t" 		 
	       << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
	       << "gainZero"   << "\t\t" << theLaser         << "\t\t" 
	       << theOccHigh   << std::endl;
    
    if (isNew) { 
      *ResFileNewEE << thisFed      << "\t\t" << eedetid.ix()     << "\t\t"    
		    << eedetid.iy() << "\t\t" << eedetid.zside()  << "\t\t"       
		    << eedetid.hashedIndex()  << "\t\t" 		 
		    << thePedOn     << "\t\t" << thePedOnRms      << "\t\t" 
		    << "gainZero"   << "\t\t" << theLaser         << "\t\t" 
		    << theOccHigh   << std::endl;

      if (eedetid.zside()>0) newBadEEP_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
      if (eedetid.zside()<0) newBadEEM_ -> Fill( (eedetid.ix()-0.5), (eedetid.iy()-0.5), 4);
    }
  } 
}




// core of the work = analyzing runs
void popcon::EcalChannelStatusHandler::getNewObjects() {
  
  std::ostringstream ss; 
  ss << "ECAL ";

  // here we retrieve all the runs of a given type after the last from online DB   
  unsigned int max_since=0;
  max_since=static_cast<unsigned int>(tagInfo().lastInterval.first);
  std::cout << "max_since : "  << max_since << std::endl;

  std::cout << "Retrieving run list from ONLINE DB ... " << std::endl;
  econn = new EcalCondDBInterface( m_sid, m_user, m_pass );
  std::cout << "Connection done" << std::endl;
  
  if (!econn) { 
    std::cout << " Problem with OMDS: connection parameters " << m_sid << "/" << m_user << "/" << m_pass << std::endl;
    throw cms::Exception("OMDS not available");
  } 

  // histos
  newBadEB_  = new TProfile2D("newBadEB_",  "new bad channels, EB",  360, 0., 360., 170, -85.,  85.);
  newBadEEP_ = new TProfile2D("newBad_EEP_","new bad channels, EE+", 100, 0., 100., 100,   0., 100.);
  newBadEEM_ = new TProfile2D("newBad_EEM_","new bad channels, EE-", 100, 0., 100., 100,   0., 100.);
  
  // these are the online conditions DB classes 
  RunList my_runlist ;
  RunTag  my_runtag;
  LocationDef my_locdef;
  RunTypeDef my_rundef;
  my_locdef.setLocation(m_location);
  my_rundef.setRunType(m_runtype);  
  my_runtag.setLocationDef(my_locdef);
  my_runtag.setRunTypeDef(my_rundef);
  my_runtag.setGeneralTag(m_gentag);   
  

  // range of validity
  unsigned int min_run=0;
  if(m_firstRun<max_since) {
    min_run=max_since+1;    // we have to add 1 to the last transferred one
  } else { min_run=m_firstRun; }
  unsigned int max_run=m_lastRun;

  
  // here we retrieve the Monitoring run records  
  MonVersionDef monverdef;
  monverdef.setMonitoringVersion("test01");
  MonRunTag mon_tag;
  // if (m_runtype=="PEDESTAL") mon_tag.setGeneralTag("CMSSW");        
  // if (m_runtype=="LASER")    mon_tag.setGeneralTag("CMSSW");        
  // if (m_runtype=="COSMIC" || m_runtype=="BEAM" || m_runtype=="PHYSICS" || m_runtype=="HALO" || m_runtype=="GLOBAL_COSMICS" ) mon_tag.setGeneralTag("CMSSW-online");        
  if (m_runtype=="PEDESTAL") mon_tag.setGeneralTag("CMSSW-offline-private");        
  if (m_runtype=="LASER")    mon_tag.setGeneralTag("CMSSW-offline-private");        
  if (m_runtype=="COSMIC" || m_runtype=="BEAM" || m_runtype=="PHYSICS" || m_runtype=="HALO" || m_runtype=="GLOBAL_COSMICS" ) mon_tag.setGeneralTag("CMSSW-online-private");        
  mon_tag.setMonVersionDef(monverdef);
  MonRunList mon_list;
  mon_list.setMonRunTag(mon_tag);
  mon_list.setRunTag(my_runtag);
  mon_list = econn->fetchMonRunList(my_runtag, mon_tag, min_run, max_run );
  


  // ----------------------------------------------------------------
  // preparing the output files
  char outfile[800];
  sprintf(outfile,"BadChannelsEB_run%d.txt",min_run);
  ResFileEB    = new std::ofstream(outfile,std::ios::out);
  sprintf(outfile,"BadChannelsEE_run%d.txt",min_run);
  ResFileEE    = new std::ofstream(outfile,std::ios::out);
  sprintf(outfile,"BadNewChannelsEB_run%d.txt",min_run);
  ResFileNewEB = new std::ofstream(outfile,std::ios::out);
  sprintf(outfile,"BadNewChannelsEE_run%d.txt",min_run);
  ResFileNewEE = new std::ofstream(outfile,std::ios::out);
  sprintf(outfile,"DaqConfig_run%d.txt",min_run);
  daqFile = new std::ofstream(outfile,std::ios::out);
  sprintf(outfile,"DaqConfig_channels_run%d.txt",min_run);
  daqFile2 = new std::ofstream(outfile,std::ios::out);

  *daqFile  << "fed" << "\t\t" << "tower" << std::endl; 

  if (m_runtype=="PEDESTAL") { 
    *ResFileEB << "Fed"         << "\t\t"   << "Ic"     << "\t\t" << "hIndex"   << "\t\t"   	       
	       << "MeanG12"     << "\t\t"   << "RmsG12" << "\t\t" 
	       << "MeanG6"      << "\t\t"   << "RmsG6"  << "\t\t" 
	       << "MeanG1"      << "\t\t"   << "RmsG1"  << std::endl;

    *ResFileEE << "Fed"         << "\t\t"   << "Ix"     << "\t\t" 
	       << "Iy"          << "\t\t"   << "Iz"     << "\t\t" << "hIndex"   << "\t\t"
	       << "MeanG12"     << "\t\t"   << "RmsG12" << "\t\t" 
	       << "MeanG6"      << "\t\t"   << "RmsG6"  << "\t\t" 
	       << "MeanG1"      << "\t\t"   << "RmsG1"  << std::endl;

    *ResFileNewEB << "Fed"         << "\t\t"   << "Ic"     << "\t\t" << "hIndex"   << "\t\t"   	       
		  << "MeanG12"     << "\t\t"   << "RmsG12" << "\t\t" 
		  << "MeanG6"      << "\t\t"   << "RmsG6"  << "\t\t" 
		  << "MeanG1"      << "\t\t"   << "RmsG1"  << std::endl;

    *ResFileNewEE << "Fed"         << "\t\t"   << "Ix"     << "\t\t" 
		  << "Iy"          << "\t\t"   << "Iz"     << "\t\t" << "hIndex"   << "\t\t"
		  << "MeanG12"     << "\t\t"   << "RmsG12" << "\t\t" 
		  << "MeanG6"      << "\t\t"   << "RmsG6"  << "\t\t" 
		  << "MeanG1"      << "\t\t"   << "RmsG1"  << std::endl;
  }
  

  if (m_runtype=="LASER") {   
    *ResFileEB    << "Fed" << "\t\t" << "Ic" << "\t\t" << "hIndex" << "\t\t" << "apd" << std::endl;
    *ResFileEE    << "Fed" << "\t\t" << "Ix" << "\t\t" << "Iy"     << "\t\t" << "Iz"  << "\t\t" << "hIndex" << "\t\t" << "apd"  << std::endl;	       
    *ResFileNewEB << "Fed" << "\t\t" << "Ic" << "\t\t" << "hIndex" << "\t\t" << "apd" << std::endl;
    *ResFileNewEE << "Fed" << "\t\t" << "Ix" << "\t\t" << "Iy"     << "\t\t" << "Iz"  << "\t\t" << "hIndex" << "\t\t" << "apd"  << std::endl;	       
  }


  if (m_runtype=="COSMIC" || m_runtype=="BEAM" || m_runtype=="PHYSICS" || m_runtype=="HALO" || m_runtype=="GLOBAL_COSMICS" ) {
    
    *ResFileEB << "Fed"             << "\t\t"   << "Ic"            << "\t\t" << "hIndex"   << "\t\t"   
	       << "pedOnline"       << "\t\t"   << "pedOnlineRMS " << "\t\t" 
	       << "gain0"           << "\t\t"   << "apd"           << "\t\t" 
	       << "highThrOcc"      << "\t\t"   << std::endl;

    *ResFileEE << "Fed"             << "\t\t"   << "Ix" 
	       << "Iy"              << "\t\t"   << "Iz"            << "\t\t" << "hIndex"   << "\t\t"
	       << "pedOnline"       << "\t\t"   << "pedOnlineRMS " << "\t\t" 
	       << "gain0"           << "\t\t"   << "apd"           << "\t\t" 
	       << "highThrOcc"      << "\t\t"   << std::endl;

    *ResFileNewEB << "Fed"             << "\t\t"   << "Ic"            << "\t\t" << "hIndex"   << "\t\t"   
		  << "pedOnline"       << "\t\t"   << "pedOnlineRMS " << "\t\t" 
		  << "gain0"           << "\t\t"   << "apd"           << "\t\t" 
		  << "highThrOcc"      << "\t\t"   << std::endl;
    
    *ResFileNewEE << "Fed"             << "\t\t"   << "Ix" 
		  << "Iy"              << "\t\t"   << "Iz"            << "\t\t" << "hIndex"   << "\t\t"
		  << "pedOnline"       << "\t\t"   << "pedOnlineRMS " << "\t\t" 
		  << "gain0"           << "\t\t"   << "apd"           << "\t\t" 
		  << "highThrOcc"      << "\t\t"   << std::endl;
  }





  // -------------------------------------------------------------------
  // analysis for the wanted runs
  std::vector<MonRunIOV> mon_run_vec = mon_list.getRuns();
  int mon_runs = mon_run_vec.size();
  std::cout << "number of Mon runs is " << mon_runs << std::endl;
  if(mon_runs==0) std::cout << "PROBLEM! 0 runs analyzed by DQM" << std::endl;
  if(mon_runs==0) ss   << "PROBLEM! 0 runs analyzed by DQM" << std::endl;

  // initialize std::maps with masked channels
  maskedOnlinePedEB.clear();
  maskedOnlinePedEE.clear();
  maskedPedEB.clear();
  maskedPedEE.clear();
  maskedLaserEB.clear();
  maskedLaserEE.clear();
  maskedPhysicsEB.clear();
  maskedPhysicsEE.clear();

    // to iterate
  std::map<DetId,float>::const_iterator theIter;
  
  // using db info written by DQM
  if(mon_runs>0){   
    
    for(int dqmRun=0; dqmRun<mon_runs; dqmRun++){
      
      unsigned long iDqmRun=(unsigned long) mon_run_vec[dqmRun].getRunIOV().getRunNumber();  
      
      std::cout << "retrieve the DQM data for run number: " << iDqmRun << ", subrun number: " << mon_run_vec[dqmRun].getSubRunNumber() << std::endl;

      
      if (mon_run_vec[dqmRun].getSubRunNumber()==mon_runs){      // fixme: check it still works after DMQ soft reset modifications
	
	// retrieve daq configuration for this run
	RunIOV myRun;
	myRun=mon_run_vec[dqmRun].getRunIOV();
	daqOut(myRun);


	// retrieve the data for a given run
	RunIOV runiov_prime = mon_run_vec[dqmRun].getRunIOV();
	

	// here we read the list of masked channel in the DB for this run and create masked channels std::maps
	std::cout << "Fetching masked channels from DB" << std::endl;
	EcalErrorMask::readDB(econn, &runiov_prime);



	
	// -----------------------------------------------------------------------------------
	// here we do all the different types of analyses

	
	// PEDESTAL ANALYSIS for local runs: check pedestals only
	if (m_runtype=="PEDESTAL") {   

	  // retrieve the pedestals from OMDS for this run 
	  std::map<EcalLogicID, MonPedestalsDat> dataset_mon;    
	  econn->fetchDataSet(&dataset_mon, &mon_run_vec[dqmRun]);
	  std::cout << "running pedestal analysis" << std::endl;
	  std::cout << "OMDS record for pedestals, run " << iDqmRun << " is made of " << dataset_mon.size() << " entries" << std::endl;

	  // retrieve the crystal consistency from OMDS for this run 
	  std::map<EcalLogicID, MonCrystalConsistencyDat> wrongGain_mon;
	  econn->fetchDataSet(&wrongGain_mon, &mon_run_vec[dqmRun]);
	  std::cout << "OMDS record for consistency, run " << iDqmRun << " is made of " << wrongGain_mon.size() << " entries" << std::endl;

	  // check if enough data and perform analysis
	  if (!dataset_mon.empty()) { 
	    pedAnalysis( dataset_mon, wrongGain_mon );
	  } else {
	    std::cout << "Not enought data for pedestal analysis" << std::endl;
	    ss   << "Not enought data for pedestal analysis" << std::endl;
	  }	  
	}
	
	
	
	// LASER ANALYSIS for local runs: check APD values only
	if (m_runtype=="LASER") {   
	    
	  // retrieve the APD / PNs from OMDS for this run 
	  std::map<EcalLogicID, MonLaserBlueDat > dataset_mon;
	  econn->fetchDataSet(&dataset_mon, &mon_run_vec[dqmRun]);
	  std::cout << "running the laser analysis based on DQM data" << std::endl;
	  std::cout << "OMDS record for run " << iDqmRun << " is made of " << dataset_mon.size() << " records" << std::endl;	      
	  
	  // check if enough data and select good light modules / perform analysis
	  if (!dataset_mon.empty()) { 
	    nBadLaserModules( dataset_mon );
	    laserAnalysis( dataset_mon );
	  } else {
	    std::cout << "Not enought data for dqm-based laser analysis" << std::endl;
	    ss   << "Not enought data for dqm-based laser analysis" << std::endl;
	  }
	}
	
	
	
	// global analysis for global runs
	if (m_runtype=="COSMIC" || m_runtype=="BEAM" || m_runtype=="PHYSICS" || m_runtype=="HALO" || m_runtype=="GLOBAL_COSMICS" ) {   

	  // retrieve the pedestal online from OMDS for this run 
	  std::map<EcalLogicID, MonPedestalsOnlineDat> pedonline_mon;    
	  econn->fetchDataSet(&pedonline_mon, &mon_run_vec[dqmRun]);
	  std::cout << "running pedestal online analysis" << std::endl;
	  std::cout << "OMDS record for pedestals, run " << iDqmRun << " is made of " << pedonline_mon.size() << std::endl;

	  // retrieve the crystal consistency from OMDS for this run 
	  std::map<EcalLogicID, MonCrystalConsistencyDat> wrongGain_mon;
	  econn->fetchDataSet(&wrongGain_mon, &mon_run_vec[dqmRun]);
	  std::cout << "OMDS record for consistency, run " << iDqmRun << " is made of " << wrongGain_mon.size() << " entries" << std::endl;
	  
	  // retrieve the occupancy info from OMDS for this run 
	  std::map<EcalLogicID, MonOccupancyDat> occupancy_mon;
	  econn->fetchDataSet(&occupancy_mon, &mon_run_vec[dqmRun]);
	  std::cout << "OMDS record for occupancy, run " << iDqmRun << " is made of " << occupancy_mon.size() << std::endl;

	  // retrieve the APD / PNs from OMDS for this run 
	  std::map<EcalLogicID, MonLaserBlueDat > laser_mon;
	  econn->fetchDataSet(&laser_mon, &mon_run_vec[dqmRun]);
	  std::cout << "running laser analysis" << std::endl;
	  std::cout << "OMDS record for laser, run " << iDqmRun << " is made of " << laser_mon.size() << " records" << std::endl;	     


	  // check if enough data in all the categories and do the analysis
	  if (pedonline_mon.empty()) { 
	    std::cout << "Not enought data for pedestal online analysis" << std::endl; 
	    ss   << "Not enought data for pedestal online analysis" << std::endl; 
	  }
	  if (occupancy_mon.empty()) { 
	    std::cout << "Not enought data for occupancy analysis" << std::endl;       
	    ss   << "Not enought data for occupancy analysis" << std::endl; 
	  }
	  if (laser_mon.empty()) { 
	    std::cout << "Not enought data for laser analysis" << std::endl;
	    ss   << "Not enought data for laser analysis" << std::endl;
	  }
	  if ( !pedonline_mon.empty() || !occupancy_mon.empty() || !wrongGain_mon.empty() || !laser_mon.empty() ) {
	    nBadLaserModules( laser_mon );
	    cosmicsAnalysis( pedonline_mon, wrongGain_mon, laser_mon, occupancy_mon );
	  }


	  // plotting histos with new bad channels
	  int iLineEB=0;
	  TLine lEB;
	  gStyle->SetPalette(1);
	  gStyle->SetOptStat(0);
	  TCanvas c("c","c",1);
	  newBadEB_ -> SetMaximum(11);
	  newBadEB_ -> Draw("colz");
	  iLineEB=0;
	  lEB.DrawLine(0,0,360,0);
	  while (iLineEB<18) { lEB.DrawLine (iLineEB*20, -85, iLineEB*20, 85);  iLineEB++; }
	  c.SaveAs("newBadEB_.png");
	  
	  TLine lEE;
	  lEE.SetLineWidth(1);
	  int ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20, 0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};
	  int iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13, 8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87, 0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};
	  
	  newBadEEP_ -> SetMaximum(11);
	  newBadEEP_ -> Draw("colz");
	  for ( int iLineEEP=0; iLineEEP<201; iLineEEP=iLineEEP+1 ) {
	    if ( (ixSectorsEE[iLineEEP]!=0 || iySectorsEE[iLineEEP]!=0) && (ixSectorsEE[iLineEEP+1]!=0 || iySectorsEE[iLineEEP+1]!=0) ) {
	      lEE.DrawLine(ixSectorsEE[iLineEEP], iySectorsEE[iLineEEP], ixSectorsEE[iLineEEP+1], iySectorsEE[iLineEEP+1]);
	    }}
	  c.SaveAs("newBadEEP_.png");

	  newBadEEM_  -> SetMaximum(11);
	  newBadEEM_ -> Draw("colz");
	  for ( int iLineEEP=0; iLineEEP<201; iLineEEP=iLineEEP+1 ) {
	    if ( (ixSectorsEE[iLineEEP]!=0 || iySectorsEE[iLineEEP]!=0) && (ixSectorsEE[iLineEEP+1]!=0 || iySectorsEE[iLineEEP+1]!=0) ) {
	      lEE.DrawLine(ixSectorsEE[iLineEEP], iySectorsEE[iLineEEP], ixSectorsEE[iLineEEP+1], iySectorsEE[iLineEEP+1]);
	    }}
	  c.SaveAs("newBadEEM_.png");
	  
	} // cosmics analysis
	
      } // subruns 
    }   // runs loop
  }     // we have the DQM info 
  
  delete econn;
  std::cout << "Ecal - > end of getNewObjects -----------\n";	
}


void popcon::EcalChannelStatusHandler::setElectronicsMap(const EcalElectronicsMapping* theEcalElectronicsMap) {

  ecalElectronicsMap_ = (*theEcalElectronicsMap);
}


