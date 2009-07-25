#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"
#include "DataFormats/EcalRawData/interface/ESKCHIPBlock.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DQM/EcalPreshowerMonitorModule/interface/ESIntegrityTask.h"

using namespace cms;
using namespace edm;
using namespace std;

ESIntegrityTask::ESIntegrityTask(const ParameterSet& ps) {

   init_ = false;

   dqmStore_ = Service<DQMStore>().operator->();

   prefixME_      = ps.getUntrackedParameter<string>("prefixME", "");
   enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);
   mergeRuns_     = ps.getUntrackedParameter<bool>("mergeRuns", false);

   dccCollections_   = ps.getParameter<InputTag>("ESDCCCollections");
   kchipCollections_ = ps.getParameter<InputTag>("ESKChipCollections");

}

ESIntegrityTask::~ESIntegrityTask() {
}

void ESIntegrityTask::beginJob(const EventSetup& c) {

   ievt_ = 0;

   if ( dqmStore_ ) {
      dqmStore_->setCurrentFolder(prefixME_ + "/ESIntegrityTask");
      dqmStore_->rmdir(prefixME_ + "/ESIntegrityTask");
   }

}

void ESIntegrityTask::beginRun(const Run& r, const EventSetup& c) {

   if ( ! mergeRuns_ ) this->reset();

}

void ESIntegrityTask::endRun(const Run& r, const EventSetup& c) {

}

void ESIntegrityTask::reset(void) {
   /*
      if ( meFED_ ) meFED_->Reset();
      if ( meGain_ ) meGain_->Reset();
      if ( meDCCErr_ ) meDCCErr_->Reset();
      if ( meOptoRX_ ) meOptoRX_->Reset();
      if ( meOptoBC_ ) meOptoBC_->Reset();
      if ( meFiberStatus_ ) meFiberStatus_->Reset();
      if ( meKF1_ ) meKF1_->Reset();
      if ( meKF2_ ) meKF2_->Reset();
      if ( meKBC_ ) meKBC_->Reset();
      if ( meKEC_ ) meKEC_->Reset();
    */
}

void ESIntegrityTask::setup(void){

   init_ = true;

   char histo[200];

   if (dqmStore_) {
      dqmStore_->setCurrentFolder(prefixME_ + "/ESIntegrityTask");

      sprintf(histo, "ES FEDs used for data taking");
      meFED_ = dqmStore_->book1D(histo, histo, 56, 519.5, 575.5); 
      meFED_->setAxisTitle("ES FED", 1);
      meFED_->setAxisTitle("Num of Events", 2);

      sprintf(histo, "ES Gain used for data taking");
      meGain_ = dqmStore_->book1D(histo, histo, 2, -0.5, 1.5); 
      meGain_->setAxisTitle("ES Gain", 1);
      meGain_->setAxisTitle("Num of Events", 2);

      sprintf(histo, "ES DCC Error codes");
      meDCCErr_ = dqmStore_->book2D(histo, histo, 56, 519.5, 575.5, 6, -0.5, 5.5); 
      meDCCErr_->setAxisTitle("ES FED", 1);
      meDCCErr_->setAxisTitle("ES DCC Error code", 2);

      sprintf(histo, "ES OptoRX used for data taking");
      meOptoRX_ = dqmStore_->book2D(histo, histo, 56, 519.5, 575.5, 3, -0.5, 2.5); 
      meOptoRX_->setAxisTitle("ES FED", 1);
      meOptoRX_->setAxisTitle("ES OptoRX", 2);

      sprintf(histo, "ES OptoRX BC mismatch");
      meOptoBC_ = dqmStore_->book2D(histo, histo, 56, 519.5, 575.5, 3, -0.5, 2.5); 
      meOptoBC_->setAxisTitle("ES FED", 1);
      meOptoBC_->setAxisTitle("ES OptoRX", 2);

      sprintf(histo, "ES Fiber Status");
      meFiberStatus_ = dqmStore_->book2D(histo, histo, 56, 519.5, 575.5, 2, -0.5, 1.5);
      meFiberStatus_->setAxisTitle("ES FED", 1);
      meFiberStatus_->setAxisTitle("ES Fiber Status", 2);

      sprintf(histo, "ES KChip Flag 1 Error codes");
      meKF1_ = dqmStore_->book2D(histo, histo, 1550, -0.5, 1549.5, 16, -0.5, 15.5);
      meKF1_->setAxisTitle("ES KChip", 1);
      meKF1_->setAxisTitle("ES KChip F1 Error Code ", 2);

      sprintf(histo, "ES KChip Flag 2 Error codes");
      meKF2_ = dqmStore_->book2D(histo, histo, 1550, -0.5, 1549.5, 16, -0.5, 15.5);
      meKF2_->setAxisTitle("ES KChip", 1);
      meKF2_->setAxisTitle("ES KChip F1 Error Code ", 2);

      sprintf(histo, "ES KChip BC mismatch with OptoRX");
      meKBC_ = dqmStore_->book1D(histo, histo, 1550, -0.5, 1549.5);
      meKBC_->setAxisTitle("ES KChip", 1);
      meKBC_->setAxisTitle("Num of BC mismatch", 2);

      sprintf(histo, "ES KChip EC mismatch with OptoRX");
      meKEC_ = dqmStore_->book1D(histo, histo, 1550, -0.5, 1549.5);
      meKEC_->setAxisTitle("ES KChip", 1);
      meKEC_->setAxisTitle("Num of EC mismatch", 2);

   }

}

void ESIntegrityTask::cleanup(void){

   if ( ! init_ ) return;

   if ( dqmStore_ ) {

   }

   init_ = false;

}

void ESIntegrityTask::endJob(void){

   LogInfo("ESIntegrityTask") << "analyzed " << ievt_ << " events";

   if ( enableCleanup_ ) this->cleanup();

}

void ESIntegrityTask::analyze(const Event& e, const EventSetup& c){

   if ( ! init_ ) this->setup();

   ievt_++;

   Handle<ESRawDataCollection> dccs;
   if ( e.getByLabel(dccCollections_, dccs) ) {
   } else {
      LogWarning("ESIntegrityTask") << "Error! can't get ES raw data collection !" << std::endl;
   }

   Handle<ESLocalRawDataCollection> kchips;
   if ( e.getByLabel(kchipCollections_, kchips) ) {
   } else {
      LogWarning("ESIntegrityTask") << "Error! can't get ES local raw data collection !" << std::endl;
   }

   // DCC 
   vector<int> fiberStatus;
   for (ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr) {
      ESDCCHeaderBlock dcc = (*dccItr);

      meFED_->Fill(dcc.fedId());

      meDCCErr_->Fill(dcc.fedId(), dcc.getDCCErrors());

      if (dcc.getOptoRX0() == 128) {
	meOptoRX_->Fill(dcc.fedId(), 0);
	if (((dcc.getOptoBC0()-15) & 0x0fff) != dcc.getBX()) meOptoBC_->Fill(dcc.fedId(), 0);
       
      }
      if (dcc.getOptoRX1() == 128) {
	meOptoRX_->Fill(dcc.fedId(), 1);
	if (((dcc.getOptoBC1()-15) & 0x0fff) != dcc.getBX()) meOptoBC_->Fill(dcc.fedId(), 1);
      }
      if (dcc.getOptoRX2() == 128) {
	meOptoRX_->Fill(dcc.fedId(), 2);
	if (((dcc.getOptoBC2()-15) & 0x0fff) != dcc.getBX()) meOptoBC_->Fill(dcc.fedId(), 2);
      }

      fiberStatus = dcc.getFEChannelStatus();

      for (unsigned int i=0; i<fiberStatus.size(); ++i) {
	if (fiberStatus[i]==0 || fiberStatus[i]==7 || fiberStatus[i]==13 || fiberStatus[i]==14 || fiberStatus[i]==9) 
	  meFiberStatus_->Fill(dcc.fedId(), 1);
	else if (fiberStatus[i]==8 || fiberStatus[i]==10 || fiberStatus[i]==11 || fiberStatus[i]==12)
	  meFiberStatus_->Fill(dcc.fedId(), 0);
      }

      runtype_   = dcc.getRunType();
      seqtype_   = dcc.getSeqType();
      dac_       = dcc.getDAC();
      gain_      = dcc.getGain();
      precision_ = dcc.getPrecision();

      meGain_->Fill(gain_);
   }

   // KCHIP's
   for (ESLocalRawDataCollection::const_iterator kItr = kchips->begin(); kItr != kchips->end(); ++kItr) {

      ESKCHIPBlock kchip = (*kItr);

      meKF1_->Fill(kchip.id(), kchip.getFlag1());
      meKF2_->Fill(kchip.id(), kchip.getFlag2());
      if (kchip.getBC() != kchip.getOptoBC()) meKBC_->Fill(kchip.id());
      if (kchip.getEC() != kchip.getOptoEC()) meKEC_->Fill(kchip.id());
   }

}

DEFINE_FWK_MODULE(ESIntegrityTask);
