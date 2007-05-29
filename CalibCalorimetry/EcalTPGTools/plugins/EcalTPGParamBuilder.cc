
#include "EcalTPGParamBuilder.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"

#include <iomanip>
#include <fstream>


EcalTPGParamBuilder::EcalTPGParamBuilder(edm::ParameterSet const& pSet)
  : xtal_LSB_EB_(0), xtal_LSB_EE_(0), nSample_(5)
{
  Et_sat_ = pSet.getParameter<double>("Et_sat") ;
  sliding_ = pSet.getParameter<unsigned int>("sliding") ;
  sampleMax_ = pSet.getParameter<unsigned int>("weight_sampleMax") ;
  complement2_ = pSet.getParameter<unsigned int>("weight_complement2") ;   
  LUT_option_ = pSet.getParameter<std::string>("LUT_option") ;
  TTF_lowThreshold_ = pSet.getParameter<double>("TTF_lowThreshold") ;
  TTF_highThreshold_ = pSet.getParameter<double>("TTF_highThreshold") ;

  std::string outFileEB = pSet.getParameter<std::string>("outFileEB") ;
  out_fileEB_ = new std::ofstream(outFileEB.c_str(), std::ios::out) ;  
  std::string outFileEE = pSet.getParameter<std::string>("outFileEE") ;
  out_fileEE_ = new std::ofstream(outFileEE.c_str(), std::ios::out) ;  

}

EcalTPGParamBuilder::~EcalTPGParamBuilder()
{ 
  (*out_fileEB_ )<<"EOF"<<std::endl ;
  (*out_fileEE_ )<<"EOF"<<std::endl ;
  out_fileEB_->close() ;
  out_fileEE_->close() ;
  delete out_fileEB_ ;
  delete out_fileEE_ ;
}

void EcalTPGParamBuilder::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) 
{
  using namespace edm;
  using namespace std;
}

void EcalTPGParamBuilder::beginJob(const edm::EventSetup& evtSetup)
{
  using namespace edm;
  using namespace std;


  ////////////////////////////
  // Initialization section //
  ////////////////////////////

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
  

  // Pedestals
  ESHandle<EcalPedestals> pedHandle;
  evtSetup.get<EcalPedestalsRcd>().get( pedHandle );
  const EcalPedestalsMap & pedMap = pedHandle.product()->m_pedestals ;
 
  // Intercalib constants
  ESHandle<EcalIntercalibConstants> pIntercalib ;
  evtSetup.get<EcalIntercalibConstantsRcd>().get(pIntercalib) ;
  const EcalIntercalibConstants * intercalib = pIntercalib.product() ;
  const EcalIntercalibConstants::EcalIntercalibConstantMap & calibMap = intercalib->getMap() ;

   // Gain Ratios
   ESHandle<EcalGainRatios> pRatio;
   evtSetup.get<EcalGainRatiosRcd>().get(pRatio);
   const EcalGainRatios::EcalGainRatioMap & gainMap = pRatio.product()->getMap();

  // ADCtoGeV
  ESHandle<EcalADCToGeVConstant> pADCToGeV ;
  evtSetup.get<EcalADCToGeVConstantRcd>().get(pADCToGeV) ;
  const EcalADCToGeVConstant * ADCToGeV = pADCToGeV.product() ;
  xtal_LSB_EB_ = ADCToGeV->getEBValue() ;
  xtal_LSB_EE_ = ADCToGeV->getEEValue() ;


  /////////////////////////////////////////
  // Compute linearization coeff section //
  /////////////////////////////////////////

  // loop on EB xtals
  create_header(out_fileEB_, "EB") ; 
  std::vector<DetId> ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);
  for (vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it) ;
    double theta = theBarrelGeometry_->getGeometry(id)->getPosition().theta() ;
    const EcalTrigTowerDetId towid= id.tower();
    int tccNb = theMapping_->TCCid(towid) ;
    int towerInTCC = theMapping_->iTT(towid) ;
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id) ;
    int stripInTower = elId.pseudoStripId() ;
    int xtalInStrip = elId.channelId() ;
    (*out_fileEB_)<<"CRYSTAL "<<dec<<tccNb<<" "<<towerInTCC<<" "<<stripInTower<<" "<<xtalInStrip<<std::endl ;

    // get current intercalibration coeff
    double calibCoeff = 1. ;
    EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit = calibMap.find(id.rawId());
    if( icalit != calibMap.end() ){
      EcalIntercalibConstants::EcalIntercalibConstant icalibConst = icalit->second;
      calibCoeff = icalibConst ;
    }

    // get current gain ratio
    double gainRatio[3]  = {1., 2., 12.} ;
    EcalGainRatios::EcalGainRatioMap::const_iterator gainIter = gainMap.find(id.rawId());
    if (gainIter != gainMap.end()) {
      const EcalMGPAGainRatio & aGain = gainIter->second ;
      gainRatio[1] = aGain.gain12Over6() ;
      gainRatio[2] = aGain.gain6Over1() * aGain.gain12Over6() ;
    }

    // get current pedestal
    int pedVec[3] = {0, 0, 0} ;
    EcalPedestalsMapIterator pedIter = pedMap.find(id.rawId());
    if (pedIter != pedMap.end()) {
      EcalPedestals::Item aped = pedIter->second ;
      pedVec[0] = int(aped.mean_x12 + 0.5) ; 
      pedVec[1] = int(aped.mean_x6 + 0.5) ;
      pedVec[2] = int(aped.mean_x1 + 0.5) ;
    }

    // compute and fill linearization parameters
    for (int i=0 ; i<3 ; i++) {
      int mult, shift ;
      bool ok = computeLinearizerParam(theta, gainRatio[i], calibCoeff, "EB", mult , shift) ;
      if (ok) (*out_fileEB_) << hex <<" 0x"<< pedVec[i]<<" 0x"<<mult<<" 0x"<<shift<<std::endl; 
    }
  } //ebCells

  // loop on EE xtals
  create_header(out_fileEE_, "EE") ; 
  vector<DetId> eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  for (vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it);
    double theta = theEndcapGeometry_->getGeometry(id)->getPosition().theta() ;
    const EcalTrigTowerDetId towid= (*eTTmap_).towerOf(id) ;
    int tccNb = theMapping_->TCCid(towid) ;
    int towerInTCC = theMapping_->iTT(towid) ;
    const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(id) ;
    int stripInTower = elId.pseudoStripId() ;
    int xtalInStrip = elId.channelId() ;
    (*out_fileEE_)<<"CRYSTAL "<<dec<<tccNb<<" "<<towerInTCC<<" "<<stripInTower<<" "<<xtalInStrip<<std::endl ;

    // get current intercalibration coeff
    double calibCoeff = 1. ;
    EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit = calibMap.find(id.rawId());
    if( icalit != calibMap.end() ){
      EcalIntercalibConstants::EcalIntercalibConstant icalibConst = icalit->second;
      calibCoeff = icalibConst ;
    }
    
    // get current gain ratio
    double gainRatio[3]  = {1., 2., 12.} ;
    EcalGainRatios::EcalGainRatioMap::const_iterator gainIter = gainMap.find(id.rawId());
    if (gainIter != gainMap.end()) {
      const EcalMGPAGainRatio & aGain = gainIter->second ;
      gainRatio[1] = aGain.gain12Over6() ;
      gainRatio[2] = aGain.gain6Over1() * aGain.gain12Over6() ;
    }
    
    // get current pedestal
    int pedVec[3] = {0, 0, 0} ;
    EcalPedestalsMapIterator pedIter = pedMap.find(id.rawId());
    if (pedIter != pedMap.end()) {
      EcalPedestals::Item aped = pedIter->second ;
      pedVec[0] = int(aped.mean_x12 + 0.5) ; 
      pedVec[1] = int(aped.mean_x6 + 0.5) ;
      pedVec[2] = int(aped.mean_x1 + 0.5) ;
    }

   // compute and fill linearization parameters
    for (int i=0 ; i<3 ; i++) {
      int mult, shift ;
      bool ok = computeLinearizerParam(theta, gainRatio[i], calibCoeff, "EE", mult , shift) ;
      if (ok) (*out_fileEE_) << hex <<" 0x"<< pedVec[i]<<" 0x"<<mult<<" 0x"<<shift<<std::endl; 
    }
  } //eeCells


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
    // barrel
    (*out_fileEB_) <<std::endl ;
    (*out_fileEB_) <<"COMMENT default parameters for strips"<<std::endl ;
    (*out_fileEB_) <<"STRIP "<<dec<<-1<<" "<<-1<<" "<<-1<<std::endl ;
    (*out_fileEB_) << hex << "0x" <<sliding_<<std::endl ;
    for (uint sample=0 ; sample<5 ; sample++) (*out_fileEB_) << "0x" <<hex<<weights[sample]<<" " ;
    (*out_fileEB_)<<std::endl ; 
 
    // endcap
    (*out_fileEE_) <<std::endl ;
    (*out_fileEE_) <<"COMMENT default parameters for strips"<<std::endl ;
    (*out_fileEE_) <<"STRIP "<<dec<<-1<<" "<<-1<<" "<<-1<<std::endl ;
    (*out_fileEE_) << hex << "0x" <<sliding_<<std::endl ;
    for (uint sample=0 ; sample<5 ; sample++) (*out_fileEE_) << "0x" <<hex<<weights[sample]<<" " ;
    (*out_fileEE_)<<std::endl ; 
    (*out_fileEE_) << "0x0 0x0"<<std::endl ;  
  }


  /////////////////////////
  // Compute LUT section //
  /////////////////////////

  int lut[1024] ;

  // barrel
  (*out_fileEB_) <<std::endl ;
  (*out_fileEB_) <<"COMMENT default parameters for towers"<<std::endl ;
  (*out_fileEB_) <<"TOWER "<<dec<<-1<<" "<<-1<<std::endl ;
  computeLUT(lut) ; 
  for (int i=0 ; i<1024 ; i++) (*out_fileEB_)<<"0x"<<hex<<lut[i]<<" " ;
  (*out_fileEB_)<<endl ;
  (*out_fileEB_)<<"0x20 0x40 0x60 0x70 0x0808"<<std::endl ;

  
  // endcap
  (*out_fileEE_) <<std::endl ;
  (*out_fileEE_) <<"COMMENT default parameters for towers"<<std::endl ;
  (*out_fileEE_) <<"TOWER "<<dec<<-1<<" "<<-1<<std::endl ;
  computeLUT(lut) ; 
  for (int i=0 ; i<1024 ; i++) (*out_fileEE_)<<"0x"<<hex<<lut[i]<<" " ;
  (*out_fileEE_)<<endl ;
  (*out_fileEE_)<<"0x0"<<std::endl ;


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
  double xtal_LSB = xtal_LSB_EB_ ;
  // case endcap:
  if (subdet=="EE") {
    shiftDet = 0 ;
    xtal_LSB = xtal_LSB_EE_ ;
  }

  double factor = 1024/Et_sat_ * xtal_LSB * gainRatio * calibCoeff * sin(theta) * (1 << (sliding_ + shiftDet + 2)) ;
  // Let's try first with shift = 0 (trivial solution)
  mult = (int)(factor+0.5) ; 
  for (shift = 0 ; shift<15 ; shift++) {
    if (mult>=128  && mult<256) return true ;
    factor *= 2 ; 
    mult = (int)(factor+0.5) ;
  }
  return false ;
}

void EcalTPGParamBuilder::create_header(std::ofstream * out_file, std::string subdet) 
{
  (*out_file) <<"COMMENT put your comments here"<<std::endl ; 
  (*out_file) <<"COMMENT do not remove xtal, strip and tower coordinates with -1 (used as default)"<<std::endl ;
  (*out_file) <<"COMMENT do not insert comments within physics, xtal, strip or tower structure (data block length fixed)"<<std::endl ;
  (*out_file) <<"COMMENT ================================="<<std::endl ;
  (*out_file) <<"COMMENT           physics structure"<<std::endl ;
  (*out_file) <<"COMMENT"<<std::endl ;
  (*out_file) <<"COMMENT  xtalLSB (GeV), EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)"<<std::endl ;
  (*out_file) <<"COMMENT ================================="<<std::endl ;
  (*out_file) <<"COMMENT"<<std::endl ;
  (*out_file) <<"COMMENT ================================="<<std::endl ;
  (*out_file) <<"COMMENT           crystal structure"<<std::endl ;
  (*out_file) <<"COMMENT"<<std::endl ;
  (*out_file) <<"COMMENT  ped, mult, shift [gain12]"<<std::endl ;
  (*out_file) <<"COMMENT  ped, mult, shift [gain6]"<<std::endl ;
  (*out_file) <<"COMMENT  ped, mult, shift [gain1]"<<std::endl ;
  (*out_file) <<"COMMENT ================================="<<std::endl ;
  (*out_file) <<"COMMENT"<<std::endl ;
  (*out_file) <<"COMMENT ================================="<<std::endl ;
  (*out_file) <<"COMMENT           strip structure"<<std::endl ;
  (*out_file) <<"COMMENT"<<std::endl ;
  (*out_file) <<"COMMENT  sliding_window"<<std::endl ;
  (*out_file) <<"COMMENT  weight[1-5]"<<std::endl ;
  
  if (subdet=="EE") {
    (*out_file) <<"COMMENT  threshold_fg strip_lut_fg"<<std::endl ;
    (*out_file) <<"COMMENT ================================="<<std::endl ;
    (*out_file) <<"COMMENT"<<std::endl ;
    (*out_file) <<"COMMENT ================================="<<std::endl ;
    (*out_file) <<"COMMENT           tower structure"<<std::endl ;
    (*out_file) <<"COMMENT"<<std::endl ;
    (*out_file) <<"COMMENT  LUT[1-1024]"<<std::endl ;
    (*out_file) <<"COMMENT  tower_lut_fg"<<std::endl ;
  }
  if (subdet=="EB") {
    (*out_file) <<"COMMENT ================================="<<std::endl ;
    (*out_file) <<"COMMENT"<<std::endl ;
    (*out_file) <<"COMMENT ================================="<<std::endl ;
    (*out_file) <<"COMMENT           tower structure"<<std::endl ;
    (*out_file) <<"COMMENT"<<std::endl ;
    (*out_file) <<"COMMENT  LUT[1-1024]"<<std::endl ;
    (*out_file) <<"COMMENT  el, eh, tl, th, lut_fg"<<std::endl ;
  }
  
  (*out_file) <<"COMMENT ================================="<<std::endl ;
  (*out_file) <<std::endl ;
  (*out_file) <<std::endl ;
}


int EcalTPGParamBuilder::uncodeWeight(double weight, uint complement2)
{
  int iweight ;
  uint max = (uint)(pow(2.,complement2)-1) ;
  if (weight>0) iweight=int((1<<6)*weight+0.5) ; // +0.5 for rounding pb
  else iweight= max - int(-weight*(1<<6)+0.5) +1 ;
  iweight = iweight & max ;
  return iweight ;
}

double EcalTPGParamBuilder::uncodeWeight(int iweight, uint complement2)
{
  double weight = double(iweight)/pow(2., 6.) ;
  // test if negative weight:
  if (iweight & (1<<(complement2-1)) != 0) weight = (double(iweight)-pow(2., complement2))/pow(2., 6.) ;
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
    std::cout<<time<<" "<<shape(time)<<std::endl ;
  }
  double lambda = 1./(sumf2-sumf*sumf/nSample_) ;
  double gamma = -lambda*sumf/nSample_ ;
  double * weight = new double[nSample_] ;
  for (uint sample = 0 ; sample<nSample_ ; sample++) {
    double time = timeMax - ((double)sampleMax_-(double)sample)*25. ;
    weight[sample] = lambda*shape(time)/max + gamma ;
  }

  int * iweight = new int[nSample_] ;
  for (uint sample = 0 ; sample<nSample_ ; sample++)   iweight[sample] = uncodeWeight(weight[sample], complement2_) ;

  // Let's check:  
  int isumw  = 0 ;  
  for (uint sample = 0 ; sample<nSample_ ; sample++) isumw  += iweight[sample] ;

  // Let's correct for bias if any
  if (isumw != 0) {
    double min = 99. ;
    uint index = 0 ;
    if (isumw &  (1<<(complement2_-1)) != 0) {
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

void EcalTPGParamBuilder::computeLUT(int * lut) 
{
  // initialisation with identity
  for (int i=0 ; i<1024 ; i++) {
    lut[i] = i ;
    if (lut[i]>0xff) lut[i] = 0xff ;
  }

  if (LUT_option_ == "Linear") {
    int mylut = 0 ;
    for (int i=0 ; i<1024 ; i++) {
      lut[i] = mylut ;
      if ((i+1)%4 == 0 ) mylut++ ;
    }
  }

  if (LUT_option_ == "EcalResolution") {
    // to be done! FIXME
  }

  // Now, add TTF thresholds to LUT
  for (int j=0 ; j<1024 ; j++) {
    double Et_GeV = Et_sat_/1024*j ;
    int ttf = 0x0 ;    
    if (Et_GeV >= TTF_highThreshold_) ttf = 3 ;
    if (Et_GeV >= TTF_lowThreshold_ && Et_GeV < TTF_highThreshold_) ttf = 1 ;
    ttf = ttf << 8 ;
    lut[j] += ttf ;
  }
  
}
