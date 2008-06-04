// -*- C++ -*-
//
// Class:      EcalTPGAnalyzer
// 
//
// Original Author:  Pascal Paganini
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <vector>
#include <string>
#include <TFile.h>
#include <TTree.h>
#include <TH2.h>


class CaloSubdetectorGeometry ;


// Auxiliary class
class towerEner {   
 public:
  float eRec_ ;
  float data_[10] ;
  int tpgEmul_[5] ;
  int tpgADC_; 
  int iphi_, ieta_, iSM_, ttf_, fg_ ;
  towerEner()
    : eRec_(0), tpgADC_(0),  
      iphi_(-999), ieta_(-999), iSM_(0), ttf_(-999), fg_(-999)
  { 
    for (int i=0 ; i<10 ; i ++) data_[i] = 0. ; 
    for (int i=0 ; i<5 ; i ++) tpgEmul_[i] = 0 ; 
  }
};


class EcalTPGAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalTPGAnalyzer(const edm::ParameterSet&);
  ~EcalTPGAnalyzer();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  
private:
  void fillShape(EBDataFrame & df) ;
  void fillShape(towerEner & t) ;
  void fillOccupancyPlots(towerEner & t) ;
  void fillEnergyPlots(towerEner & t) ;
  void fillTPMatchPlots(towerEner & t) ;

private:
  TFile *histfile_;
  TTree *tree_ ;
  TH2F * shape_[36] ;
  TH2F * shapeMax_ ;
  TH2F * occupancyTP_ ;
  TH2F * occupancyTPEmul_ ;
  TH2F * occupancyTPEmulMax_ ;
  TH2F * crystalVsTP_ ;
  TH2F * crystalVsEmulTP_ ;
  TH2F * crystalVsEmulTPMax_ ;
  TH2F * TPVsEmulTP_ ;
  TH1F * TP_ ;  
  TH1F * TPEmul_ ;
  TH1F * TPEmulMax_ ;
  TH1F * TPMatchEmul_ ;
  TH1F * TPEmulMaxIndex_ ;

  std::string label_;
  std::string producer_;
  std::string digi_label_;
  std::string digi_producerEB_, digi_producerEE_ ;
  std::string emul_label_;
  std::string emul_producer_;
  bool allowTP_ ;
  bool useEE_ ;
  int adcCut_, shapeCut_, occupancyCut_ ;
  int tpgRef_ ;

  const CaloSubdetectorGeometry * theEndcapGeometry_ ;
  const CaloSubdetectorGeometry * theBarrelGeometry_ ;
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap_;
};

