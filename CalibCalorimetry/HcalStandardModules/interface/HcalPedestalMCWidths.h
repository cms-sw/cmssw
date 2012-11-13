#ifndef HcalPedestalMCWidths_H
#define HcalPedestalMCWidths_H

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
#include "CalibCalorimetry/HcalStandardModules/interface/HcalCondXML.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TStyle.h"

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

   struct MCWidthsBunch
   {
      HcalDetId detid;
      bool usedflag;
      float sig[4][10][10];
      float prod[4][10][10];
      int num[4][10][10];
   };

class HcalPedestalMCWidths : public edm::EDAnalyzer
{
   public:
   //Constructor
   HcalPedestalMCWidths(const edm::ParameterSet& ps);
   //Destructor
   virtual ~HcalPedestalMCWidths();
   //Analysis
   void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

   private:
   //Container for data, 1 per channel
   std::vector<MCWidthsBunch> Bunches;
   //Flag for saving histos
   std::string pedsADCfilename;
   std::string widthsfilename;

   int runnum;

   TProfile *HBMeans[10];
   TProfile *HEMeans[10];
   TProfile *HFMeans[10];
   TProfile *HOMeans[10];

   const HcalTopology* theTopology;

   TFile *theFile;
   bool firsttime;
   bool histflag;

  edm::InputTag hbheDigiCollectionTag_;
  edm::InputTag hoDigiCollectionTag_;
  edm::InputTag hfDigiCollectionTag_;
};
#endif

