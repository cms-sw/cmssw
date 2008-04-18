
#include "EcalTPGParamBuilder.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGDBApp.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

#include <TF1.h>
#include <iomanip>
#include <fstream>


double oneOverEtResolEt(double *x, double *par) { 
  double Et = x[0] ;
  if (Et<1e-6) return 1./par[1] ; // to avoid division by 0.
  double resolEt_overEt = sqrt( (par[0]/sqrt(Et))*(par[0]/sqrt(Et)) + (par[1]/Et)*(par[1]/Et) + par[2]*par[2] ) ;
  return 1./(Et*resolEt_overEt) ;
}

EcalTPGParamBuilder::EcalTPGParamBuilder(edm::ParameterSet const& pSet)
  : xtal_LSB_EB_(0), xtal_LSB_EE_(0), nSample_(5), complement2_(7)
{
  
  readFromDB_ = pSet.getParameter<bool>("readFromDB") ;
  writeToDB_  = pSet.getParameter<bool>("writeToDB") ;
  DBEE_ = pSet.getParameter<bool>("allowDBEE") ;
  string DBsid    = pSet.getParameter<std::string>("DBsid") ;
  string DBuser   = pSet.getParameter<std::string>("DBuser") ;
  string DBpass   = pSet.getParameter<std::string>("DBpass") ;
  uint32_t DBport = pSet.getParameter<unsigned int>("DBport") ;
  DBrunNb_        = pSet.getParameter<unsigned int>("DBrunNb") ;

  if (readFromDB_ || writeToDB_) {
    try {
      cout << "Warning: using the DB is not yet implemented " <<endl ;
      db_ = new EcalTPGDBApp(DBsid, DBuser, DBpass) ;
    } catch (exception &e) {
      cout << "ERROR:  " << e.what() << endl;
    } catch (...) {
      cout << "Unknown error caught" << endl;
    }
  }

  writeToFiles_ =  pSet.getParameter<bool>("writeToFiles") ;
  if (writeToFiles_) {
    std::string outFile = pSet.getParameter<std::string>("outFile") ;
    out_file_ = new std::ofstream(outFile.c_str(), std::ios::out) ;  
    geomFile_   = new std::ofstream("geomFile.txt", std::ios::out) ;  
  }



  useTransverseEnergy_ = pSet.getParameter<bool>("useTransverseEnergy") ;
  
  Et_sat_EB_ = pSet.getParameter<double>("Et_sat_EB") ;
  Et_sat_EE_ = pSet.getParameter<double>("Et_sat_EE") ;
  sliding_ = pSet.getParameter<unsigned int>("sliding") ;
  sampleMax_ = pSet.getParameter<unsigned int>("weight_sampleMax") ;

  LUT_option_ = pSet.getParameter<std::string>("LUT_option") ;
  LUT_threshold_EB_ = pSet.getParameter<double>("LUT_threshold_EB") ;
  LUT_threshold_EE_ = pSet.getParameter<double>("LUT_threshold_EE") ;
  LUT_stochastic_EB_ = pSet.getParameter<double>("LUT_stochastic_EB") ;
  LUT_noise_EB_ =pSet.getParameter<double>("LUT_noise_EB") ;
  LUT_constant_EB_ =pSet.getParameter<double>("LUT_constant_EB") ;
  LUT_stochastic_EE_ = pSet.getParameter<double>("LUT_stochastic_EE") ;
  LUT_noise_EE_ =pSet.getParameter<double>("LUT_noise_EE") ;
  LUT_constant_EE_ =pSet.getParameter<double>("LUT_constant_EE") ;

  TTF_lowThreshold_EB_ = pSet.getParameter<double>("TTF_lowThreshold_EB") ;
  TTF_highThreshold_EB_ = pSet.getParameter<double>("TTF_highThreshold_EB") ;
  TTF_lowThreshold_EE_ = pSet.getParameter<double>("TTF_lowThreshold_EE") ;
  TTF_highThreshold_EE_ = pSet.getParameter<double>("TTF_highThreshold_EE") ;

  FG_lowThreshold_EB_ = pSet.getParameter<double>("FG_lowThreshold_EB") ;
  FG_highThreshold_EB_ = pSet.getParameter<double>("FG_highThreshold_EB") ;
  FG_lowRatio_EB_ = pSet.getParameter<double>("FG_lowRatio_EB") ;
  FG_highRatio_EB_ = pSet.getParameter<double>("FG_highRatio_EB") ;
  FG_lut_EB_ = pSet.getParameter<unsigned int>("FG_lut_EB") ;
  FG_Threshold_EE_ = pSet.getParameter<double>("FG_Threshold_EE") ;
  FG_lut_strip_EE_ = pSet.getParameter<unsigned int>("FG_lut_strip_EE") ;
  FG_lut_tower_EE_ = pSet.getParameter<unsigned int>("FG_lut_tower_EE") ;
}

EcalTPGParamBuilder::~EcalTPGParamBuilder()
{ 
  if (writeToFiles_) {
    (*out_file_ )<<"EOF"<<std::endl ;
    out_file_->close() ;
    delete out_file_ ;
  }
}

void EcalTPGParamBuilder::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) 
{
  using namespace edm;
  using namespace std;

  ////////////////////////////
  // Initialization section //
  ////////////////////////////
  list<uint32_t> towerListEB ;
  list<uint32_t> stripListEB ;
  list<uint32_t> towerListEE ;
  list<uint32_t> stripListEE ;
  list<uint32_t>::iterator itList ;

  // Pedestals
  ESHandle<EcalPedestals> pedHandle;
  evtSetup.get<EcalPedestalsRcd>().get( pedHandle );
  const EcalPedestalsMap & pedMap = pedHandle.product()->getMap() ;   
  map<EcalLogicID, MonPedestalsDat> pedMapDB ;
  int iovId = 0 ;
  if (readFromDB_) iovId = db_->readFromCondDB_Pedestals(pedMapDB, DBrunNb_) ;

  // Intercalib constants
  ESHandle<EcalIntercalibConstants> pIntercalib ;
  evtSetup.get<EcalIntercalibConstantsRcd>().get(pIntercalib) ;
  const EcalIntercalibConstants * intercalib = pIntercalib.product() ;
  const EcalIntercalibConstantMap & calibMap = intercalib->getMap() ;

  // Gain Ratios
  ESHandle<EcalGainRatios> pRatio;
  evtSetup.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatioMap & gainMap = pRatio.product()->getMap();
  
  // ADCtoGeV
  ESHandle<EcalADCToGeVConstant> pADCToGeV ;
  evtSetup.get<EcalADCToGeVConstantRcd>().get(pADCToGeV) ;
  const EcalADCToGeVConstant * ADCToGeV = pADCToGeV.product() ;
  xtal_LSB_EB_ = ADCToGeV->getEBValue() ;
  xtal_LSB_EE_ = ADCToGeV->getEEValue() ;
  std::cout<<"xtal_LSB_EB_ = "<<xtal_LSB_EB_<<std::endl ;
  std::cout<<"xtal_LSB_EE_ = "<<xtal_LSB_EE_<<std::endl ;

  /////////////////////////////////////////
  // Compute linearization coeff section //
  /////////////////////////////////////////

  map<EcalLogicID, FEConfigPedDat> pedset ;
  map<EcalLogicID, FEConfigLinDat> linset ;

  // loop on EB xtals
  if (writeToFiles_) (*out_file_)<<"COMMENT ====== barrel crystals ====== "<<std::endl ;
  std::vector<DetId> ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);
  for (vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it) ;
    double theta = theBarrelGeometry_->getGeometry(id)->getPosition().theta() ;
    if (!useTransverseEnergy_) theta = acos(0.) ;
    const EcalTrigTowerDetId towid= id.tower();
    towerListEB.push_back(towid.rawId()) ;
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id) ;
    stripListEB.push_back(elId.rawId() & 0xfffffff8) ;
    int dccNb = theMapping_->DCCid(towid) ;
    int tccNb = theMapping_->TCCid(towid) ;
    int towerInTCC = theMapping_->iTT(towid) ;
    int stripInTower = elId.pseudoStripId() ;
    int xtalInStrip = elId.channelId() ;

    EcalLogicID logicId ;
    FEConfigPedDat ped ;
    FEConfigLinDat lin ;
    if (writeToFiles_) (*out_file_)<<"CRYSTAL "<<dec<<id.rawId()<<std::endl ;
    if (writeToDB_ || readFromDB_) logicId = db_->getEcalLogicID ("EB_crystal_number", id.ism(), id.ic()) ;

    coeffStruc coeff ;
    if (readFromDB_) {
      getCoeff(coeff, calibMap, id.rawId()) ;
      getCoeff(coeff, gainMap, id.rawId()) ;
      getCoeff(coeff, pedMapDB, logicId) ;      
    } else {
      getCoeff(coeff, calibMap, id.rawId()) ;
      getCoeff(coeff, gainMap, id.rawId()) ;
      getCoeff(coeff, pedMap, id.rawId()) ;
    }

    // compute and fill linearization parameters
    for (int i=0 ; i<3 ; i++) {
      int mult, shift ;
      bool ok = computeLinearizerParam(theta, coeff.gainRatio_[i], coeff.calibCoeff_, "EB", mult , shift) ;
      if (!ok) std::cout << "unable to compute the parameters for "<<dec<<id.rawId()<<std::endl ;  
      else {
	if (writeToFiles_) (*out_file_) << hex <<" 0x"<<coeff.pedestals_[i]<<" 0x"<<mult<<" 0x"<<shift<<std::endl; 
	if (writeToDB_) {
	  if (i==0)  {ped.setPedMeanG12(coeff.pedestals_[i]) ; lin.setMultX12(mult) ; lin.setShift12(shift) ; } 
	  if (i==1)  {ped.setPedMeanG6(coeff.pedestals_[i]) ; lin.setMultX6(mult) ; lin.setShift6(shift) ; } 
	  if (i==2)  {ped.setPedMeanG1(coeff.pedestals_[i]) ; lin.setMultX1(mult) ; lin.setShift1(shift) ; } 
	}
      }
    }
    if (writeToDB_) {
      pedset[logicId] = ped ;
      linset[logicId] = lin ;
    }
  } //ebCells

  // loop on EE xtals
  if (writeToFiles_) (*out_file_)<<"COMMENT ====== endcap crystals ====== "<<std::endl ;
  std::vector<DetId> eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  for (vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it);
    double theta = theEndcapGeometry_->getGeometry(id)->getPosition().theta() ;
    if (!useTransverseEnergy_) theta = acos(0.) ;
    const EcalTrigTowerDetId towid= (*eTTmap_).towerOf(id) ;
    towerListEE.push_back(towid.rawId()) ;
    // special case of towers in inner rings of EE
    if (towid.ietaAbs() == 27 || towid.ietaAbs() == 28) {
      EcalTrigTowerDetId additionalTower(towid.zside(), towid.subDet(), towid.ietaAbs(), towid.iphi()+1) ;
      towerListEE.push_back(additionalTower.rawId()) ;
    }
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id) ;
    stripListEE.push_back(elId.rawId() & 0xfffffff8) ;
    int dccNb = theMapping_->DCCid(towid) ;
    int tccNb = theMapping_->TCCid(towid) ;
    int towerInTCC = theMapping_->iTT(towid) ;
    int stripInTower = elId.pseudoStripId() ;
    int xtalInStrip = elId.channelId() ;

    EcalLogicID logicId ;
    FEConfigPedDat ped ;
    FEConfigLinDat lin ;
    if (writeToFiles_) (*out_file_)<<"CRYSTAL "<<dec<<id.rawId()<<std::endl ;
    if ((writeToDB_ || readFromDB_) && DBEE_) {
      int iz = id.positiveZ() ;
      if (iz ==0) iz = -1 ;
      logicId = db_->getEcalLogicID ("EE_crystal_number", iz, id.ix(), id.iy()) ;
    }

    coeffStruc coeff ;
    if (readFromDB_ && DBEE_) {
      getCoeff(coeff, calibMap, id.rawId()) ;
      getCoeff(coeff, gainMap, id.rawId()) ;
      getCoeff(coeff, pedMapDB, logicId) ;      
    } else {
      getCoeff(coeff, calibMap, id.rawId()) ;
      getCoeff(coeff, gainMap, id.rawId()) ;
      getCoeff(coeff, pedMap, id.rawId()) ;
    }

    // compute and fill linearization parameters
    for (int i=0 ; i<3 ; i++) {
      int mult, shift ;
      bool ok = computeLinearizerParam(theta, coeff.gainRatio_[i], coeff.calibCoeff_, "EE", mult , shift) ;
      if (!ok) std::cout << "unable to compute the parameters for "<<dec<<id.rawId()<<std::endl ;  
      else {
	if (writeToFiles_) (*out_file_) << hex <<" 0x"<<coeff.pedestals_[i]<<" 0x"<<mult<<" 0x"<<shift<<std::endl; 
	if (writeToDB_ && DBEE_) {
	  if (i==0)  {ped.setPedMeanG12(coeff.pedestals_[i]) ; lin.setMultX12(mult) ; lin.setShift12(shift) ; } 
	  if (i==1)  {ped.setPedMeanG6(coeff.pedestals_[i]) ; lin.setMultX6(mult) ; lin.setShift6(shift) ; } 
	  if (i==2)  {ped.setPedMeanG1(coeff.pedestals_[i]) ; lin.setMultX1(mult) ; lin.setShift1(shift) ; } 
	}	
      }
    }
    if (writeToDB_ && DBEE_) {
      pedset[logicId] = ped ;
      linset[logicId] = lin ;
    }
  } //eeCells

  if (writeToDB_) {
    db_->writeToConfDB_TPGPedestals(pedset, iovId, "from_CondDB") ;
    db_->writeToConfDB_TPGLinearCoef(linset, iovId, "from_CondDB") ;
  }

  /////////////////////////////
  // Compute weights section //
  /////////////////////////////

  // loading reference signal representation
  EcalSimParameterMap parameterMap;  
  EBDetId   barrel(1,1);
  double    phase = parameterMap.simParameters(barrel).timePhase();
  EcalShape shape(phase); 
  std::vector<unsigned int> weights = computeWeights(shape) ;

  if (weights.size() == 5) {
    if (writeToFiles_) {
      (*out_file_) <<std::endl ;
      (*out_file_) <<"WEIGHT 0"<<endl ;
      for (uint sample=0 ; sample<5 ; sample++) (*out_file_) << "0x" <<hex<<weights[sample]<<" " ;
      (*out_file_)<<std::endl ; 
    }
    if (writeToDB_) {

    }
  }

  /////////////////////////
  // Compute FG section //
  /////////////////////////

  // barrel
  uint lowRatio, highRatio, lowThreshold, highThreshold, lutFG ;
  computeFineGrainEBParameters(lowRatio, highRatio, lowThreshold, highThreshold, lutFG) ;
  if (writeToFiles_) {
    (*out_file_) <<std::endl ;
    (*out_file_) <<"FG 0"<<std::endl ;
    (*out_file_)<<hex<<"0x"<<lowThreshold<<" 0x"<<highThreshold
		  <<" 0x"<<lowRatio<<" 0x"<<highRatio<<" 0x"<<lutFG
		  <<std::endl ;
  }

  // endcap
  uint threshold, lut_strip, lut_tower ;
  computeFineGrainEEParameters(threshold, lut_strip, lut_tower) ; 


  /////////////////////////
  // Compute LUT section //
  /////////////////////////

  int lut_EB[1024], lut_EE[1024] ;

  // barrel
  computeLUT(lut_EB, "EB") ; 
  if (writeToFiles_) {
    (*out_file_) <<std::endl ;
    (*out_file_) <<"LUT 0"<<std::endl ;
    for (int i=0 ; i<1024 ; i++) (*out_file_)<<"0x"<<hex<<lut_EB[i]<<" " ;
    (*out_file_)<<endl ;
  }
  
  // endcap
  computeLUT(lut_EE, "EE") ;
  // check first if lut_EB and lut_EE are the same
  bool newLUT(false) ;
  for (int i=0 ; i<1024 ; i++) if (lut_EE[i] != lut_EB[i]) newLUT = true ;
  if (newLUT && writeToFiles_) { 
    (*out_file_) <<std::endl ;
    (*out_file_) <<"LUT 1"<<std::endl ;
    for (int i=0 ; i<1024 ; i++) (*out_file_)<<"0x"<<hex<<lut_EE[i]<<" " ;
    (*out_file_)<<endl ;
  }


  ///////////////////////////////////////////////////////////
  // loop on strips and associate them with default values //
  ///////////////////////////////////////////////////////////

  // Barrel
  stripListEB.sort() ;
  stripListEB.unique() ;
  cout<<"Number of EB strips="<<stripListEB.size()<<endl ;
  if (writeToFiles_) {
    (*out_file_) <<std::endl ;
    for (itList = stripListEB.begin(); itList != stripListEB.end(); itList++ ) {
      (*out_file_) <<"STRIP_EB "<<dec<<(*itList)<<endl ;
      (*out_file_) << hex << "0x" <<sliding_<<std::endl ;
      (*out_file_) <<" 0" << std::endl ;
    }
  }

  // Endcap
  stripListEE.sort() ;
  stripListEE.unique() ;
  cout<<"Number of EE strips="<<stripListEE.size()<<endl ;
  if (writeToFiles_) {
    (*out_file_) <<std::endl ;
    for (itList = stripListEE.begin(); itList != stripListEE.end(); itList++ ) {
      (*out_file_) <<"STRIP_EE "<<dec<<(*itList)<<endl ;
      (*out_file_) << hex << "0x" <<sliding_<<std::endl ;
      (*out_file_) <<" 0" << std::endl ;
      (*out_file_)<<hex<<"0x"<<threshold<<" 0x"<<lut_strip<<std::endl ;  
    }
  }


  ///////////////////////////////////////////////////////////
  // loop on towers and associate them with default values //
  ///////////////////////////////////////////////////////////

  // Barrel
  towerListEB.sort() ;
  towerListEB.unique() ;
  cout<<"Number of EB towers="<<towerListEB.size()<<endl ;
  if (writeToFiles_) {
    (*out_file_) <<std::endl ;
    (*geomFile_)<<"BARREL"<<endl ;
    for (itList = towerListEB.begin(); itList != towerListEB.end(); itList++ ) {
      (*out_file_) <<"TOWER_EB "<<dec<<(*itList)<<endl ;
      (*out_file_) <<" 0\n 0\n" ;
      EcalTrigTowerDetId towerId((*itList)) ;
      int dccNb = theMapping_->DCCid(towerId) ;
      int tccNb = theMapping_->TCCid(towerId) ;
      int towerInTCC = theMapping_->iTT(towerId) ;
      (*geomFile_)<<"towerId="<<(*itList)<<" ieta="<<towerId.ietaAbs()<<" iphi="<<towerId.iphi()
		  <<" dccNb="<<dccNb<<" tccNb="<<tccNb<<" towerInTCC="<<towerInTCC<<endl ;
    }
  }

  // Endcap
  towerListEE.sort() ;
  towerListEE.unique() ;
  cout<<"Number of EE towers="<<towerListEE.size()<<endl ;
  if (writeToFiles_) {
    (*out_file_) <<std::endl ;
    (*geomFile_)<<"ENDCAP"<<endl ;
    for (itList = towerListEE.begin(); itList != towerListEE.end(); itList++ ) {
      (*out_file_) <<"TOWER_EE "<<dec<<(*itList)<<endl ;
      if (newLUT) (*out_file_) <<" 1\n" ;
      else  (*out_file_) <<" 0\n" ;
      (*out_file_)<<hex<<"0x"<<lut_tower<<std::endl ;
      EcalTrigTowerDetId towerId((*itList)) ;
      int dccNb = theMapping_->DCCid(towerId) ;
      int tccNb = theMapping_->TCCid(towerId) ;
      int towerInTCC = theMapping_->iTT(towerId) ;
      (*geomFile_)<<"towerId="<<(*itList)<<" ieta="<<towerId.ietaAbs()<<" iphi="<<towerId.iphi()
		  <<" dccNb="<<dccNb<<" tccNb="<<tccNb<<" towerInTCC="<<towerInTCC<<endl ;      
    }
  }

}

void EcalTPGParamBuilder::beginJob(const edm::EventSetup& evtSetup)
{
  using namespace edm;
  using namespace std;

  // geometry
  ESHandle<CaloGeometry> theGeometry;
  ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  evtSetup.get<IdealGeometryRecord>().get( theGeometry );
  evtSetup.get<IdealGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  evtSetup.get<IdealGeometryRecord>().get(eTTmap_);
  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);

  // electronics mapping
  ESHandle< EcalElectronicsMapping > ecalmapping;
  evtSetup.get< EcalMappingRcd >().get(ecalmapping);
  theMapping_ = ecalmapping.product();  

  create_header() ; 

  DetId eb(DetId::Ecal,EcalBarrel) ;
  DetId ee(DetId::Ecal,EcalEndcap) ;

  if (writeToFiles_) {
    (*out_file_)<<"PHYSICS_EB "<<dec<<eb.rawId()<<std::endl ;
    (*out_file_)<<Et_sat_EB_<<" "<<TTF_lowThreshold_EB_<<" "<<TTF_highThreshold_EB_<<std::endl ;
    (*out_file_)<<FG_lowThreshold_EB_<<" "<<FG_highThreshold_EB_<<" "
		  <<FG_lowRatio_EB_<<" "<<FG_highRatio_EB_<<std::endl ;
    (*out_file_) <<std::endl ;

    (*out_file_)<<"PHYSICS_EE "<<dec<<ee.rawId()<<std::endl ;
    (*out_file_)<<Et_sat_EE_<<" "<<TTF_lowThreshold_EE_<<" "<<TTF_highThreshold_EE_<<std::endl ;
    (*out_file_)<<FG_Threshold_EE_<<" "<<-1<<" "
		  <<-1<<" "<<-1<<std::endl ;
    (*out_file_) <<std::endl ;
  }
}



bool EcalTPGParamBuilder::computeLinearizerParam(double theta, double gainRatio, double calibCoeff, std::string subdet, int & mult , int & shift) 
{
  /*
    Linearization coefficient are determined in order to satisfy:
    tpg(ADC_sat) = 1024
    where: 
    tpg() is a model of the linearized tpg response on 10b 
    ADC_sat is the number of ADC count corresponding the Et_sat, the maximum scale of the transverse energy
    
    Since we have:
    Et_sat = xtal_LSB * ADC_sat * gainRatio * calibCoeff * sin(theta)
    and a simple model of tpg() being given by:
    tpg(X) = [ (X*mult) >> (shift+2) ] >> (sliding+shiftDet) 
    we must satisfy:
    [ (Et_sat/(xtal_LSB * gainRatio * calibCoeff * sin(theta)) * mult) >> (shift+2) ] >> (sliding+shiftDet) = 1024 
    that is:
    mult = 1024/Et_sat * xtal_LSB * gainRatio * calibCoeff * sin(theta) * 2^-(sliding+shiftDet+2) * 2^-shift
    mult = factor * 2^-shift
  */

  // case barrel:
  int shiftDet = 2 ;
  double ratio = xtal_LSB_EB_/Et_sat_EB_ ;
  // case endcap:
  if (subdet=="EE") {
    shiftDet = 0 ;
    ratio = xtal_LSB_EE_/Et_sat_EE_ ;
  }

  double factor = 1024 * ratio * gainRatio * calibCoeff * sin(theta) * (1 << (sliding_ + shiftDet + 2)) ;
  // Let's try first with shift = 0 (trivial solution)
  mult = (int)(factor+0.5) ; 
  for (shift = 0 ; shift<15 ; shift++) {
    if (mult>=128  && mult<256) return true ;
    factor *= 2 ; 
    mult = (int)(factor+0.5) ;
  }
  return false ;
}

void EcalTPGParamBuilder::create_header() 
{
  if (!writeToFiles_) return ;
  (*out_file_) <<"COMMENT put your comments here"<<std::endl ; 

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           physics EB structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)"<<std::endl ;
  (*out_file_) <<"COMMENT  FG_lowThreshold (GeV), FG_highThreshold (GeV), FG_lowRatio, FG_highRatio"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           physics EE structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)"<<std::endl ;
  (*out_file_) <<"COMMENT  FG_Threshold (GeV), dummy, dummy, dummy"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           crystal structure (same for EB and EE)"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  ped, mult, shift [gain12]"<<std::endl ;
  (*out_file_) <<"COMMENT  ped, mult, shift [gain6]"<<std::endl ;
  (*out_file_) <<"COMMENT  ped, mult, shift [gain1]"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           strip EB structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  sliding_window"<<std::endl ;
  (*out_file_) <<"COMMENT  weightGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           strip EE structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  sliding_window"<<std::endl ;
  (*out_file_) <<"COMMENT  weightGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT  threshold_fg strip_lut_fg"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           tower EB structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  LUTGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT  FgGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           tower EE structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  LUTGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT  tower_lut_fg"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           Weight structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  weightGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT  w0, w1, w2, w3, w4"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           lut structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  LUTGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT  LUT[1-1024]"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT           fg EB structure"<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;
  (*out_file_) <<"COMMENT  FgGroupId"<<std::endl ;
  (*out_file_) <<"COMMENT  el, eh, tl, th, lut_fg"<<std::endl ;
  (*out_file_) <<"COMMENT ================================="<<std::endl ;
  (*out_file_) <<"COMMENT"<<std::endl ;

  (*out_file_) <<std::endl ;
}


int EcalTPGParamBuilder::uncodeWeight(double weight, int complement2)
{
  int iweight ;
  uint max = (uint)(pow(2.,complement2)-1) ;
  if (weight>0) iweight=int((1<<6)*weight+0.5) ; // +0.5 for rounding pb
  else iweight= max - int(-weight*(1<<6)+0.5) +1 ;
  iweight = iweight & max ;
  return iweight ;
}

double EcalTPGParamBuilder::uncodeWeight(int iweight, int complement2)
{
  double weight = double(iweight)/pow(2., 6.) ;
  // test if negative weight:
  if ( (iweight & (1<<(complement2-1))) != 0) weight = (double(iweight)-pow(2., complement2))/pow(2., 6.) ;
  return weight ;
}

std::vector<unsigned int> EcalTPGParamBuilder::computeWeights(EcalShape & shape)
{
  double timeMax = shape.computeTimeOfMaximum() - shape.computeT0() ; // timeMax w.r.t begining of pulse
  double max = shape(timeMax) ;

  double sumf = 0. ;
  double sumf2 = 0. ;
  for (uint sample = 0 ; sample<nSample_ ; sample++) {
    double time = timeMax - ((double)sampleMax_-(double)sample)*25. ;
    sumf += shape(time)/max ;
    sumf2 += shape(time)/max * shape(time)/max ;
  }
  double lambda = 1./(sumf2-sumf*sumf/nSample_) ;
  double gamma = -lambda*sumf/nSample_ ;
  double * weight = new double[nSample_] ;
  for (uint sample = 0 ; sample<nSample_ ; sample++) {
    double time = timeMax - ((double)sampleMax_-(double)sample)*25. ;
    weight[sample] = lambda*shape(time)/max + gamma ;
  }

//   double ampl = 0. ;
//   for (uint sample = 0 ; sample<nSample_ ; sample++) {
//     double time = timeMax - ((double)sampleMax_-(double)sample)*25. ;
//     ampl += weight[sample]*shape(time) ;
//   }
//   std::cout<<max<<" "<<ampl<<std::endl ;


  int * iweight = new int[nSample_] ;
  for (uint sample = 0 ; sample<nSample_ ; sample++)   iweight[sample] = uncodeWeight(weight[sample], complement2_) ;

  // Let's check:  
  int isumw  = 0 ;  
  for (uint sample = 0 ; sample<nSample_ ; sample++) isumw  += iweight[sample] ;
  uint imax = (uint)(pow(2.,int(complement2_))-1) ;
  isumw = (isumw & imax ) ;

  // Let's correct for bias if any
  if (isumw != 0) {
    double min = 99. ;
    uint index = 0 ;
    if ( (isumw & (1<<(complement2_-1))) != 0) {
      // add 1:
      for (uint sample = 0 ; sample<nSample_ ; sample++) {
	int new_iweight = iweight[sample]+1 ; 
	double new_weight = uncodeWeight(new_iweight, complement2_) ;
	if (fabs(new_weight-weight[sample])<min) {
	  min = fabs(new_weight-weight[sample]) ;
	  index = sample ;
	}
      }
      iweight[index] ++ ; 
    } else {
      // Sub 1:
      for (uint sample = 0 ; sample<nSample_ ; sample++) {
        int new_iweight = iweight[sample]-1 ;    
	double new_weight = uncodeWeight(new_iweight, complement2_) ;
        if (fabs(new_weight-weight[sample])<min) {
          min = fabs(new_weight-weight[sample]) ;
          index = sample ;
        }
      }
      iweight[index] -- ; 
    } 
  }

  std::vector<unsigned int> theWeights ;
  for (uint sample = 0 ; sample<nSample_ ; sample++) theWeights.push_back(iweight[sample]) ;

  delete weight ;
  delete iweight ;
  return theWeights ;
}

void EcalTPGParamBuilder::computeLUT(int * lut, std::string det) 
{
  double Et_sat = Et_sat_EB_ ;
  double LUT_threshold = LUT_threshold_EB_ ;
  double LUT_stochastic = LUT_stochastic_EB_ ;
  double LUT_noise = LUT_noise_EB_ ;
  double LUT_constant = LUT_constant_EB_ ;
  double TTF_lowThreshold = TTF_lowThreshold_EB_ ;
  double TTF_highThreshold = TTF_highThreshold_EB_ ;
  if (det == "EE") {
    Et_sat = Et_sat_EE_ ;
    LUT_threshold = LUT_threshold_EE_ ;
    LUT_stochastic = LUT_stochastic_EE_ ;
    LUT_noise = LUT_noise_EE_ ;
    LUT_constant = LUT_constant_EE_ ;
    TTF_lowThreshold = TTF_lowThreshold_EE_ ;
    TTF_highThreshold = TTF_highThreshold_EE_ ;
  }

  // initialisation with identity
  for (int i=0 ; i<1024 ; i++) {
    lut[i] = i ;
    if (lut[i]>0xff) lut[i] = 0xff ;
  }

  // case linear LUT
  if (LUT_option_ == "Linear") {
    int mylut = 0 ;
    for (int i=0 ; i<1024 ; i++) {
      lut[i] = mylut ;
      if ((i+1)%4 == 0 ) mylut++ ;
    }
  }

  // case LUT following Ecal resolution
  if (LUT_option_ == "EcalResolution") {
    TF1 * func = new TF1("func",oneOverEtResolEt, 0., Et_sat,3) ;
    func->SetParameters(LUT_stochastic, LUT_noise, LUT_constant) ;
    double norm = func->Integral(0., Et_sat) ;
    for (int i=0 ; i<1024 ; i++) {   
      double Et = i*Et_sat/1024. ;
      lut[i] =  int(0xff*func->Integral(0., Et)/norm + 0.5) ;
    }
  }

  // Now, add TTF thresholds to LUT and apply LUT threshold if needed
  for (int j=0 ; j<1024 ; j++) {
    double Et_GeV = Et_sat/1024*(j+0.5) ;
    int ttf = 0x0 ;    
    if (Et_GeV >= TTF_highThreshold) ttf = 3 ;
    if (Et_GeV >= TTF_lowThreshold && Et_GeV < TTF_highThreshold) ttf = 1 ;
    ttf = ttf << 8 ;
    lut[j] += ttf ;
    if (Et_GeV <= LUT_threshold) lut[j] = 0 ;
  }

}

void EcalTPGParamBuilder::getCoeff(coeffStruc & coeff, const EcalIntercalibConstantMap & calibMap, uint rawId)
{
  // get current intercalibration coeff
  coeff.calibCoeff_ = 1. ;
  EcalIntercalibConstantMap::const_iterator icalit = calibMap.find(rawId);
  if( icalit != calibMap.end() ) coeff.calibCoeff_ = (*icalit) ;
  else std::cout<<"getCoeff: "<<rawId<<" not found in EcalIntercalibConstantMap"<<std::endl ;
}

void EcalTPGParamBuilder::getCoeff(coeffStruc & coeff, const EcalGainRatioMap & gainMap, uint rawId)
{
  // get current gain ratio
  coeff.gainRatio_[0]  = 1. ;
  coeff.gainRatio_[1]  = 2. ;
  coeff.gainRatio_[2]  = 12. ;
  EcalGainRatioMap::const_iterator gainIter = gainMap.find(rawId);
  if (gainIter != gainMap.end()) {
    const EcalMGPAGainRatio & aGain = (*gainIter) ;
    coeff.gainRatio_[1] = aGain.gain12Over6() ;
    coeff.gainRatio_[2] = aGain.gain6Over1() * aGain.gain12Over6() ;
  }
  else std::cout<<"getCoeff: "<<rawId<<" not found in EcalGainRatioMap"<<std::endl ;
}

void EcalTPGParamBuilder::getCoeff(coeffStruc & coeff, const EcalPedestalsMap & pedMap, uint rawId)
{
  // get current pedestal
  coeff.pedestals_[0] = 0 ;
  coeff.pedestals_[1] = 0 ;
  coeff.pedestals_[2] = 0 ;
  EcalPedestalsMapIterator pedIter = pedMap.find(rawId);
  if (pedIter != pedMap.end()) {
    EcalPedestals::Item aped = (*pedIter);
    coeff.pedestals_[0] = int(aped.mean_x12 + 0.5) ; 
    coeff.pedestals_[1] = int(aped.mean_x6 + 0.5) ;
    coeff.pedestals_[2] = int(aped.mean_x1 + 0.5) ;
  }
  else std::cout<<"getCoeff: "<<rawId<<" not found in EcalPedestalsMap"<<std::endl ;
}

void EcalTPGParamBuilder::getCoeff(coeffStruc & coeff, const map<EcalLogicID, MonPedestalsDat> & pedMap, const EcalLogicID & logicId)
{
  // get current pedestal
  coeff.pedestals_[0] = 0 ;
  coeff.pedestals_[1] = 0 ;
  coeff.pedestals_[2] = 0 ;

  map<EcalLogicID, MonPedestalsDat>::const_iterator it =  pedMap.find(logicId);
  if (it != pedMap.end()) {
    MonPedestalsDat ped = it->second ;
    coeff.pedestals_[0] = int(ped.getPedMeanG12() + 0.5) ; 
    coeff.pedestals_[1] = int(ped.getPedMeanG6() + 0.5) ; 
    coeff.pedestals_[2] = int(ped.getPedMeanG1() + 0.5) ; 
  } 
  else std::cout<<"getCoeff: "<<logicId.getID1()<<", "<<logicId.getID2()<<", "<<logicId.getID3()
		<<" not found in map<EcalLogicID, MonPedestalsDat>"<<std::endl ;
}

void EcalTPGParamBuilder::computeFineGrainEBParameters(uint & lowRatio, uint & highRatio,
						       uint & lowThreshold, uint & highThreshold, uint & lut)
{
  lowRatio = int(0x80*FG_lowRatio_EB_ + 0.5) ;
  if (lowRatio>0x7f) lowRatio = 0x7f ;
  highRatio = int(0x80*FG_highRatio_EB_ + 0.5) ;
  if (highRatio>0x7f) highRatio = 0x7f ;
  
  // lsb at the stage of the FG calculation is:
  double lsb_FG = Et_sat_EB_/1024./4 ;
  lowThreshold = int(FG_lowThreshold_EB_/lsb_FG+0.5) ;
  if (lowThreshold>0xff) lowThreshold = 0xff ;
  highThreshold = int(FG_highThreshold_EB_/lsb_FG+0.5) ;
  if (highThreshold>0xff) highThreshold = 0xff ;

  // FG lut: FGVB response is LUT(adress) where adress is: 
  // bit3: maxof2/ET >= lowRatio, bit2: maxof2/ET >= highRatio, bit1: ET >= lowThreshold, bit0: ET >= highThreshold
  // FGVB =1 if jet-like (veto active), =0 if E.M.-like
  // the condition for jet-like is: ET>Threshold and  maxof2/ET < Ratio (only TT with enough energy are vetoed)

  // With the following lut, what matters is only max(TLow, Thigh) and max(Elow, Ehigh)
  // So, jet-like if maxof2/ettot<max(TLow, Thigh) && ettot >= max(Elow, Ehigh)
  if (FG_lut_EB_ == 0) lut = 0x0888 ; 
  else lut = FG_lut_EB_ ; // let's use the users value (hope he/she knows what he/she does!)
}

void EcalTPGParamBuilder::computeFineGrainEEParameters(uint & threshold, uint & lut_strip, uint & lut_tower) 
{
  // lsb for EE:
  double lsb_FG = Et_sat_EE_/1024. ; // FIXME is it true????
  threshold = int(FG_Threshold_EE_/lsb_FG+0.5) ;
  lut_strip = FG_lut_strip_EE_  ;
  lut_tower = FG_lut_tower_EE_  ;
}
