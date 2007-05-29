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

#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"

#include <vector>
#include <string>
#include <map>
#include <iostream>

class CaloSubdetectorGeometry ;
class EcalTrigTowerConstituentsMap ;
class EcalElectronicsMapping ;

class EcalTPGParamBuilder : public edm::EDAnalyzer {

 public:
  explicit EcalTPGParamBuilder(edm::ParameterSet const& pSet) ;
  ~EcalTPGParamBuilder() ;
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) ;
  virtual void beginJob(const edm::EventSetup& evtSetup) ;

 private:
  bool computeLinearizerParam(double theta, double gainRatio, double calibCoeff, std::string subdet, int & mult , int & shift) ;
  void create_header(std::ofstream * out_file, std::string subdet) ;
  int uncodeWeight(double weight, uint complement2 = 7) ;
  double uncodeWeight(int iweight, uint complement2 = 7) ;
  std::vector<unsigned int> computeWeights(EcalShape & shape) ;
  void computeLUT(int * lut)  ;

  const CaloSubdetectorGeometry * theEndcapGeometry_ ;
  const CaloSubdetectorGeometry * theBarrelGeometry_ ;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
  const EcalElectronicsMapping * theMapping_ ;

  double xtal_LSB_EB_ , xtal_LSB_EE_ ;
  double Et_sat_ ;
  unsigned int sliding_ ;
  unsigned int sampleMax_ ;
  unsigned int nSample_ ;
  unsigned int complement2_ ;
  std::string LUT_option_ ;
  double TTF_lowThreshold_, TTF_highThreshold_ ;

  std::ofstream * out_fileEB_ ;
  std::ofstream * out_fileEE_ ;

};
#endif
