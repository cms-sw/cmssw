#ifndef CastorPedestalsAnalysis_H
#define CastorPedestalsAnalysis_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/CastorObjects/interface/CastorPedestals.h"
#include "CondFormats/CastorObjects/interface/CastorPedestalWidths.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "CondFormats/CastorObjects/interface/CastorQIEData.h"
#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/CastorObjects/interface/AllObjects.h"

#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrationWidths.h"

//  #include "CondTools/Hcal/interface/HcalDbOnline.h"

#include "CalibCalorimetry/CastorCalib/interface/CastorDbASCIIIO.h"
// #include "CalibCalorimetry/CastorCalib/interface/CastorCondXML.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TStyle.h"

#include <cmath>
#include <iostream>
#include <map>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

struct NewPedBunch {
  HcalCastorDetId detid;
  bool usedflag;
  float cap[4];
  float capfc[4];
  float sig[4][4];
  float sigfc[4][4];
  float prod[4][4];
  float prodfc[4][4];
  int num[4][4];
};

class CastorPedestalsAnalysis : public edm::EDAnalyzer {
public:
  //Constructor
  CastorPedestalsAnalysis(const edm::ParameterSet &ps);
  //Destructor
  ~CastorPedestalsAnalysis() override;
  //Analysis
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  //Container for data, 1 per channel
  std::vector<NewPedBunch> Bunches;
  //Flag for saving histos
  bool hiSaveFlag;
  bool dumpXML;
  bool verboseflag;
  int runnum;
  int firstTS;
  int lastTS;
  std::string ROOTfilename;
  std::string pedsADCfilename;
  std::string pedsfCfilename;
  std::string widthsADCfilename;
  std::string widthsfCfilename;
  std::string XMLfilename;
  std::string XMLtag;
  std::string ZSfilename;

  edm::ESGetToken<CastorDbService, CastorDbRecord> tok_cond_;
  edm::ESGetToken<CastorElectronicsMap, CastorElectronicsMapRcd> tok_map_;

  TH1F *CASTORMeans;
  TH1F *CASTORWidths;

  // TH2F *dephist[4];
  TH2F *dephist;

  TFile *theFile;
  bool firsttime;

  edm::InputTag castorDigiCollectionTag;
};
#endif
