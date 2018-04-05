#ifndef HcalGainsCheck_h
#define HcalGainsCheck_h

// 
// R.Ofierzynski 9.12.2007
//
// Code to check pedestals for completeness and to compare to previous pedestals

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

class HcalGainsCheck: public edm::EDAnalyzer
{
 public:
  HcalGainsCheck(edm::ParameterSet const& ps);

  ~HcalGainsCheck() override {}

  void beginJob() override ;
  void endJob() override;
 
  void analyze(const edm::Event& ev, const edm::EventSetup& es) override;

 private:
  //  std::string front;
  //  vecDetId getMissingDetIds(std::vector<HcalPedestals> &);
  std::string dumpupdate;
  std::string dumprefs; 
  std::string rootfile;
  std::string outfile;
  bool emapflag;
  bool validategainsflag;
  double epsilon;

  TFile * f;
  //plots:
  TH2F * ocMapUp;
  TH2F * ocMapRef;
//  TH2F* valMapUp;
//  TH2F* valMapRef;

  TH1F* diffUpRefCap0;
  TH1F* diffUpRefCap1;
  TH1F* diffUpRefCap2;
  TH1F* diffUpRefCap3;
  TH1F* ratioUpRefCap0;
  TH1F* ratioUpRefCap1;
  TH1F* ratioUpRefCap2;
  TH1F* ratioUpRefCap3;
  TH1F* gainsUpCap0;
  TH1F* gainsUpCap1;
  TH1F* gainsUpCap2;
  TH1F* gainsUpCap3;
  TH1F* gainsRefCap0;
  TH1F* gainsRefCap1;
  TH1F* gainsRefCap2;
  TH1F* gainsRefCap3;
  TH1F* gainsUpCap0vsEta;
  TH1F* gainsUpCap1vsEta;
  TH1F* gainsUpCap2vsEta;
  TH1F* gainsUpCap3vsEta;
  TH1F* gainsRefCap0vsEta;
  TH1F* gainsRefCap1vsEta;
  TH1F* gainsRefCap2vsEta;
  TH1F* gainsRefCap3vsEta;

};
#endif
