#ifndef ECALTPGPARAMBUILDER_H
#define ECALTPGPARAMBUILDER_H

//Author: Pascal Paganini - LLR
//Date: 2006/07/10 15:58:06 $

#define CMSSW_VERSION 340


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/all_fe_config_types.h"

#if (CMSSW_VERSION>=340)
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#else
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#endif


#include <TH1F.h>

#include <vector>
#include <string>
#include <map>
#include <iostream>

class CaloSubdetectorGeometry ;
class EcalElectronicsMapping ;
class EcalTPGDBApp ;

class coeffStruc {
 public:
  coeffStruc() { }
  double calibCoeff_ ;
  double gainRatio_[3] ;
  int pedestals_[3] ;
};

class linStruc {
 public:
  linStruc() { }
  int pedestal_[3] ;
  int mult_[3] ;
  int shift_[3] ;
};

class EcalTPGParamBuilder : public edm::EDAnalyzer {

 public:
  explicit EcalTPGParamBuilder(edm::ParameterSet const& pSet) ;
  ~EcalTPGParamBuilder() ;
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) ;
  virtual void beginJob() ;
  bool checkIfOK (EcalPedestals::Item item) ;

 private:
  bool computeLinearizerParam(double theta, double gainRatio, double calibCoeff, std::string subdet, int & mult , int & shift) ;
  void create_header() ;
  int uncodeWeight(double weight, int complement2 = 7) ;
  double uncodeWeight(int iweight, int complement2 = 7) ;
#if (CMSSW_VERSION>=340)
  std::vector<unsigned int> computeWeights(EcalShapeBase & shape, TH1F * histo) ;
#else
  std::vector<unsigned int> computeWeights(EcalShape & shape, TH1F * histo) ;
#endif
  void computeLUT(int * lut, std::string det="EB")  ;
  void getCoeff(coeffStruc & coeff, const EcalIntercalibConstantMap & calibMap, unsigned int rawId) ;
  void getCoeff(coeffStruc & coeff, const EcalGainRatioMap & gainMap, unsigned int rawId) ;
  void getCoeff(coeffStruc & coeff, const EcalPedestalsMap & pedMap, unsigned int rawId) ;
  void getCoeff(coeffStruc & coeff, const std::map<EcalLogicID, MonPedestalsDat> & pedMap, const EcalLogicID & logicId) ;

  void computeFineGrainEBParameters(unsigned int & lowRatio, unsigned int & highRatio,
				    unsigned int & lowThreshold, unsigned int & highThreshold, unsigned int & lut) ;
  void computeFineGrainEEParameters(unsigned int & threshold, unsigned int & lut_strip, unsigned int & lut_tower) ;
  int getEtaSlice(int tccId, int towerInTCC) ;
  bool realignBaseline(linStruc & lin, float forceBase12) ;
  int getGCTRegionPhi(int ttphi) ;
  int getGCTRegionEta(int tteta) ;
  std::string getDet(int tcc) ;
  std::pair < std::string, int > getCrate(int tcc) ;

  const CaloSubdetectorGeometry * theEndcapGeometry_ ;
  const CaloSubdetectorGeometry * theBarrelGeometry_ ;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  const EcalElectronicsMapping * theMapping_ ;

  bool useTransverseEnergy_ ;
  double xtal_LSB_EB_ , xtal_LSB_EE_ ;
  double Et_sat_EB_,  Et_sat_EE_ ;
  unsigned int sliding_ ;
  unsigned int sampleMax_ ;
  double weight_timeShift_ ;
  bool weight_unbias_recovery_ ;
  unsigned int nSample_ ;
  unsigned int complement2_ ;
  std::string LUT_option_ ;
  double LUT_threshold_EB_, LUT_threshold_EE_ ;
  double LUT_stochastic_EB_, LUT_noise_EB_, LUT_constant_EB_ ;
  double LUT_stochastic_EE_, LUT_noise_EE_, LUT_constant_EE_ ;
  double TTF_lowThreshold_EB_, TTF_highThreshold_EB_ ;
  double TTF_lowThreshold_EE_, TTF_highThreshold_EE_ ;
  double FG_lowThreshold_EB_, FG_highThreshold_EB_, FG_lowRatio_EB_, FG_highRatio_EB_ ; 
  unsigned int FG_lut_EB_ ;
  double FG_Threshold_EE_ ;
  unsigned int FG_lut_strip_EE_, FG_lut_tower_EE_ ;
  int forcedPedestalValue_ ;
  bool forceEtaSlice_ ;
  unsigned int SFGVB_Threshold_, SFGVB_lut_, pedestal_offset_ ;
  bool useInterCalibration_ ;


  std::ofstream * out_file_ ;
  std::ofstream * geomFile_ ;
  EcalTPGDBApp * db_ ;
  bool writeToDB_ ;
  bool writeToFiles_ ;
  unsigned int DBrunNb_ ;
  bool DBEE_ ;

  int ped_conf_id_;
  int lin_conf_id_;
  int lut_conf_id_;
  int fgr_conf_id_;
  int sli_conf_id_;
  int wei_conf_id_;
  int bxt_conf_id_;
  int btt_conf_id_;
  std::string tag_;
  int version_;
  int m_write_ped;
  int m_write_lin;
  int m_write_lut;
  int m_write_wei;
  int m_write_fgr;
  int m_write_sli;
  int m_write_bxt;
  int m_write_btt;

  Float_t * ntupleFloats_ ;
  Char_t ntupleDet_[10] ;
  Char_t ntupleCrate_[10] ;

};
#endif
