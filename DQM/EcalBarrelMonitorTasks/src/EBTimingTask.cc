/*
 * \file EBTimingTask.cc
 *
 * $Date: 2010/08/11 13:54:11 $
 * $Revision: 1.63 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBTimingTask.h"

EBTimingTask::EBTimingTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  for (int i = 0; i < 36; i++) {
    meTime_[i] = 0;
    meTimeMap_[i] = 0;
    meTimeAmpli_[i] = 0;
  }

  meTimeAmpliSummary_ = 0;
  meTimeSummary1D_ = 0;
  meTimeSummaryMap_ = 0;
  meTimeSummaryMapProjEta_ = 0;
  meTimeSummaryMapProjPhi_ = 0;

}

EBTimingTask::~EBTimingTask(){

}

void EBTimingTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTimingTask");
    dqmStore_->rmdir(prefixME_ + "/EBTimingTask");
  }

}

void EBTimingTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBTimingTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBTimingTask::reset(void) {

  for (int i = 0; i < 36; i++) {
    if ( meTime_[i] ) meTime_[i]->Reset();
    if ( meTimeMap_[i] ) meTimeMap_[i]->Reset();
    if ( meTimeAmpli_[i] ) meTimeAmpli_[i]->Reset();
  }

  if ( meTimeAmpliSummary_ ) meTimeAmpliSummary_->Reset();
  if ( meTimeSummary1D_ ) meTimeSummary1D_->Reset();
  if ( meTimeSummaryMap_ ) meTimeSummaryMap_->Reset();
  if ( meTimeSummaryMapProjEta_ ) meTimeSummaryMapProjEta_->Reset();
  if ( meTimeSummaryMapProjPhi_ ) meTimeSummaryMapProjPhi_->Reset();

}

void EBTimingTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTimingTask");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBTMT timing 1D %s", Numbers::sEB(i+1).c_str());
      meTime_[i] = dqmStore_->book1D(histo, histo, 50, -50., 50.);
      meTime_[i]->setAxisTitle("time (ns)", 1);
      dqmStore_->tag(meTime_[i], i+1);

      sprintf(histo, "EBTMT timing %s", Numbers::sEB(i+1).c_str());
      meTimeMap_[i] = dqmStore_->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 50, 25., 75., "s");
      meTimeMap_[i]->setAxisTitle("ieta", 1);
      meTimeMap_[i]->setAxisTitle("iphi", 2);
      meTimeMap_[i]->setAxisTitle("time (ns)", 3);
      dqmStore_->tag(meTimeMap_[i], i+1);

      sprintf(histo, "EBTMT timing vs amplitude %s", Numbers::sEB(i+1).c_str());
      meTimeAmpli_[i] = dqmStore_->book2D(histo, histo, 100, 0., 10., 50, -50., 50.);
      meTimeAmpli_[i]->setAxisTitle("energy (GeV)", 1);
      meTimeAmpli_[i]->setAxisTitle("time (ns)", 2);
      dqmStore_->tag(meTimeAmpli_[i], i+1);
    }

    sprintf(histo, "EBTMT timing vs amplitude summary");
    meTimeAmpliSummary_ = dqmStore_->book2D(histo, histo, 100, 0., 10., 50, -50., 50.);
    meTimeAmpliSummary_->setAxisTitle("energy (GeV)", 1);
    meTimeAmpliSummary_->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EBTMT timing 1D summary");
    meTimeSummary1D_ = dqmStore_->book1D(histo, histo, 50, -50., 50.);
    meTimeSummary1D_->setAxisTitle("time (ns)", 1);

    sprintf(histo, "EBTMT timing map");
    meTimeSummaryMap_ = dqmStore_->bookProfile2D(histo, histo, 72, 0., 360., 34, -85, 85, 50, 25., 75., "s");
    meTimeSummaryMap_->setAxisTitle("jphi", 1);
    meTimeSummaryMap_->setAxisTitle("jeta", 2);
    meTimeSummaryMap_->setAxisTitle("time (ns)", 3);

    sprintf(histo, "EBTMT timing projection eta");
    meTimeSummaryMapProjEta_ = dqmStore_->bookProfile(histo, histo, 34, -85., 85., 50, -50., 50., "s");
    meTimeSummaryMapProjEta_->setAxisTitle("jeta", 1);
    meTimeSummaryMapProjEta_->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EBTMT timing projection phi");
    meTimeSummaryMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 72, 0., 360., 50, -50., 50., "s");
    meTimeSummaryMapProjPhi_->setAxisTitle("jphi", 1);
    meTimeSummaryMapProjPhi_->setAxisTitle("time (ns)", 2);

  }

}

void EBTimingTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTimingTask");

    for ( int i = 0; i < 36; i++ ) {
      if ( meTime_[i] ) dqmStore_->removeElement( meTime_[i]->getName() );
      meTime_[i] = 0;

      if ( meTimeMap_[i] ) dqmStore_->removeElement( meTimeMap_[i]->getName() );
      meTimeMap_[i] = 0;

      if ( meTimeAmpli_[i] ) dqmStore_->removeElement( meTimeAmpli_[i]->getName() );
      meTimeAmpli_[i] = 0;
    }

    if ( meTimeAmpliSummary_ ) dqmStore_->removeElement( meTimeAmpliSummary_->getName() );
    meTimeAmpliSummary_ = 0;

    if ( meTimeSummary1D_ ) dqmStore_->removeElement( meTimeSummary1D_->getName() );
    meTimeSummary1D_ = 0;

    if ( meTimeSummaryMap_ ) dqmStore_->removeElement( meTimeSummaryMap_->getName() );
    meTimeSummaryMap_ = 0;

    if ( meTimeSummaryMapProjEta_ ) dqmStore_->removeElement( meTimeSummaryMapProjEta_->getName() );
    meTimeSummaryMapProjEta_ = 0;

    if ( meTimeSummaryMapProjPhi_ ) dqmStore_->removeElement( meTimeSummaryMapProjPhi_->getName() );
    meTimeSummaryMapProjPhi_ = 0;

  }

  init_ = false;

}

void EBTimingTask::endJob(void){

  edm::LogInfo("EBTimingTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBTimingTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool isData = true;
  bool enable = false;
  int runType[36];
  for (int i=0; i<36; i++) runType[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalBarrel );

      runType[ism-1] = dcchItr->getRunType();

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::COSMIC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::MTCC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) enable = true;

    }

  } else {

    isData = false; enable = true;
    edm::LogWarning("EBTimingTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  // channel status
  edm::ESHandle<EcalChannelStatus> pChannelStatus;
  c.get<EcalChannelStatusRcd>().get(pChannelStatus);
  const EcalChannelStatus* chStatus = pChannelStatus.product();

  edm::Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EBTimingTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EBDetId id = hitItr->id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      if ( isData ) {

        if ( ! ( runType[ism-1] == EcalDCCHeaderBlock::COSMIC ||
                 runType[ism-1] == EcalDCCHeaderBlock::MTCC ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
                 runType[ism-1] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) ) continue;

      }

      MonitorElement* meTime = 0;
      MonitorElement* meTimeMap = 0;
      MonitorElement* meTimeAmpli = 0;

      meTime = meTime_[ism-1];
      meTimeMap = meTimeMap_[ism-1];
      meTimeAmpli = meTimeAmpli_[ism-1];

      float xval = hitItr->energy();
      float yval = hitItr->time();

      uint32_t flag = hitItr->recoFlag();
      uint32_t sev = EcalSeverityLevelAlgo::severityLevel(id, *hits, *chStatus );

      if ( (flag == EcalRecHit::kGood || flag == EcalRecHit::kOutOfTime) && sev != EcalSeverityLevelAlgo::kWeird ) {
        if ( meTimeAmpli ) meTimeAmpli->Fill(xval, yval);
        if ( meTimeAmpliSummary_ ) meTimeAmpliSummary_->Fill(xval, yval);

        if ( xval > 0.480 ) {
          if ( meTime ) meTime->Fill(yval);
          if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval+50.);
          if ( meTimeSummary1D_ ) meTimeSummary1D_->Fill(yval);

          float xebeta = id.ieta() - 0.5 * id.zside();
          float xebphi = id.iphi() - 0.5;
          if ( meTimeSummaryMap_ ) meTimeSummaryMap_->Fill(xebphi, xebeta, yval+50.);
          if ( meTimeSummaryMapProjEta_ ) meTimeSummaryMapProjEta_->Fill(xebeta, yval);
          if ( meTimeSummaryMapProjPhi_ ) meTimeSummaryMapProjPhi_->Fill(xebphi, yval);
        }

      }
    }

  } else {

    edm::LogWarning("EBTimingTask") << EcalRecHitCollection_ << " not available";

  }

}

