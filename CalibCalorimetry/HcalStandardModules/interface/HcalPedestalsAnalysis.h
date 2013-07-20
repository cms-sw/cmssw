// $Id: HcalPedestalsAnalysis.h,v 1.15 2012/11/13 03:30:20 dlange Exp $

#ifndef HcalPedestalsAnalysis_H
#define HcalPedestalsAnalysis_H

#include "FWCore/Framework/interface/ESHandle.h"
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
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

#include "CondTools/Hcal/interface/HcalDbOnline.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbXml.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include <math.h>
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
}

   struct NewPedBunch
   {
      HcalDetId detid;
      bool usedflag;
      float cap[4];
      float capfc[4];
      float sig[4][4];
      float sigfc[4][4];
      float prod[4][4];
      float prodfc[4][4];
      int num[4][4];
   };

class HcalTopology;

class HcalPedestalsAnalysis : public edm::EDAnalyzer
{
   public:
   //Constructor
   HcalPedestalsAnalysis(const edm::ParameterSet& ps);
   //Destructor
   virtual ~HcalPedestalsAnalysis();
   //Analysis
   void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
   virtual void endJob();

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
   int ievt;
   std::string ROOTfilename;
   std::string pedsADCfilename;
   std::string pedsfCfilename;
   std::string widthsADCfilename;
   std::string widthsfCfilename;
   std::string XMLfilename;
   std::string XMLtag;

   TH1F *HBMeans;
   TH1F *HBWidths;
   TH1F *HEMeans;
   TH1F *HEWidths;
   TH1F *HFMeans;
   TH1F *HFWidths;
   TH1F *HOMeans;
   TH1F *HOWidths;

   TFile *theFile;
   bool firsttime;
   HcalPedestals* rawPedsItem;
   HcalPedestalWidths* rawWidthsItem;
   HcalPedestals* rawPedsItemfc;
   HcalPedestalWidths* rawWidthsItemfc;
   HcalTopology *theTopology;

   edm::InputTag hbheDigiCollectionTag_;
   edm::InputTag hoDigiCollectionTag_;
   edm::InputTag hfDigiCollectionTag_;
};
#endif

