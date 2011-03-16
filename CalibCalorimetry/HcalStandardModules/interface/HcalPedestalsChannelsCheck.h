// $Id: HcalPedestalsChannelsCheck.h,v 1.6 2009/11/04 08:08:11 devildog Exp $                  

#ifndef HcalPedestalsChannelsCheck_h
#define HcalPedestalsChannelsCheck_h

// 
// R.Ofierzynski 9.12.2007
//
// Code to check pedestals for completeness and to compare to previous pedestals

#include <string>
#include <iostream>
#include <iomanip>
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
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TColor.h"
class HcalPedestalsChannelsCheck: public edm::EDAnalyzer
{
 public:
  HcalPedestalsChannelsCheck(edm::ParameterSet const& ps);

  ~HcalPedestalsChannelsCheck();

  void analyze(const edm::Event& ev, const edm::EventSetup& es);

 private:
  std::string outfile;
  std::string dumprefs;
  std::string dumpupdate;
  double epsilon;
  TH1F * difhist[4];
  TH2F * etaphi[4];
  int runnum;
  //  vecDetId getMissingDetIds(std::vector<HcalPedestals> &);
};
#endif
