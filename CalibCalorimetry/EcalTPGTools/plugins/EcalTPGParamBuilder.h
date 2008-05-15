#ifndef ECALTPGPARAMBUILDER_H
#define ECALTPGPARAMBUILDER_H

//Author: Pascal Paganini - LLR
//Date: 2006/07/10 15:58:06 $

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

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"

#include <vector>
#include <string>
#include <map>
#include <iostream>

class CaloSubdetectorGeometry ;
class EcalElectronicsMapping ;
class EcalTPGCondDBApp ;

class coeffStruc {
 public:
  coeffStruc() { }
  double calibCoeff_ ;
  double gainRatio_[3] ;
  int pedestals_[3] ;
};

class EcalTPGParamBuilder : public edm::EDAnalyzer {

 public:
  explicit EcalTPGParamBuilder(edm::ParameterSet const& pSet) ;
  ~EcalTPGParamBuilder() ;
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) ;
  virtual void beginJob(const edm::EventSetup& evtSetup) ;

 private:
  bool computeLinearizerParam(double theta, double gainRatio, double calibCoeff, std::string subdet, int & mult , int & shift) ;
  void create_header(std::ofstream * out_file, std::string subdet) ;
  int uncodeWeight(double weight, int complement2 = 7) ;
  double uncodeWeight(int iweight, int complement2 = 7) ;
  std::vector<unsigned int> computeWeights(EcalShape & shape) ;
  void computeLUT(int * lut, std::string det="EB")  ;
  void getCoeff(coeffStruc & coeff,
		const EcalIntercalibConstantMap & calibMap, 
		const EcalGainRatioMap & gainMap, 
		const EcalPedestalsMap & pedMap,
		uint rawId) ;
  void computeFineGrainEBParameters(uint & lowRatio, uint & highRatio,
				    uint & lowThreshold, uint & highThreshold, uint & lut) ;
  void computeFineGrainEEParameters(uint & threshold, uint & lut_strip, uint & lut_tower) ;

  const CaloSubdetectorGeometry * theEndcapGeometry_ ;
  const CaloSubdetectorGeometry * theBarrelGeometry_ ;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  const EcalElectronicsMapping * theMapping_ ;

  bool useTransverseEnergy_ ;
  double xtal_LSB_EB_ , xtal_LSB_EE_ ;
  double Et_sat_ ;
  unsigned int sliding_ ;
  unsigned int sampleMax_ ;
  unsigned int nSample_ ;
  unsigned int complement2_ ;
  std::string LUT_option_ ;
  double LUT_threshold_ ;
  double LUT_stochastic_EB_, LUT_noise_EB_, LUT_constant_EB_ ;
  double LUT_stochastic_EE_, LUT_noise_EE_, LUT_constant_EE_ ;
  double TTF_lowThreshold_EB_, TTF_highThreshold_EB_ ;
  double TTF_lowThreshold_EE_, TTF_highThreshold_EE_ ;
  double FG_lowThreshold_EB_, FG_highThreshold_EB_, FG_lowRatio_EB_, FG_highRatio_EB_ ; 
  unsigned int FG_lut_EB_ ;
  double FG_Threshold_EE_ ;
  unsigned int FG_lut_strip_EE_, FG_lut_tower_EE_ ;

  std::ofstream * out_fileEB_ ;
  std::ofstream * out_fileEE_ ;
  std::ofstream * geomFile_ ;
  EcalTPGCondDBApp * db_ ;
  bool readFromDB_ ;
  bool writeToDB_ ;
  bool writeToFiles_ ;

};
#endif
