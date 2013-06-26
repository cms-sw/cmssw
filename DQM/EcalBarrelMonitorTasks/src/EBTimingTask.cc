/*
 * \file EBTimingTask.cc
 *
 * $Date: 2012/04/27 13:46:03 $
 * $Revision: 1.77 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBTimingTask.h"

EBTimingTask::EBTimingTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  energyThreshold_ = ps.getUntrackedParameter<double>("energyTreshold",1.0);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  L1GtEvmReadoutRecord_ = ps.getParameter<edm::InputTag>("L1GtEvmReadoutRecord");

  useBeamStatus_ = ps.getUntrackedParameter<bool>("useBeamStatus", false);

  for (int i = 0; i < 36; i++) {
    meTime_[i] = 0;
    meTimeMap_[i] = 0;
    meTimeAmpli_[i] = 0;
  }

  meTimeAmpliSummary_ = 0;
  meTimeSummary1D_ = 0;
  meTimeSummaryMap_ = 0;

  stableBeamsDeclared_ = false;

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

  stableBeamsDeclared_ = false;

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

}

void EBTimingTask::setup(void){

  init_ = true;

  std::string name;

  // for timing vs amplitude plots
  const int nbinsE = 25;
  const float minlogE = -0.5;
  const float maxlogE = 2.;
  float binEdgesE[nbinsE + 1];
  for(int i = 0; i <= nbinsE; i++)
    binEdgesE[i] = std::pow((float)10., minlogE + (maxlogE - minlogE) / nbinsE * i);

  const int nbinsT = 200;
  const float minT = -50.;
  const float maxT = 50.;
  float binEdgesT[nbinsT + 1];
  for(int i = 0; i <= nbinsT; i++)
    binEdgesT[i] = minT + (maxT - minT) / nbinsT * i;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBTimingTask");

    for (int i = 0; i < 36; i++) {
      name = "EBTMT timing 1D " + Numbers::sEB(i+1);
      meTime_[i] = dqmStore_->book1D(name, name, 50, -25., 25.);
      meTime_[i]->setAxisTitle("time (ns)", 1);
      dqmStore_->tag(meTime_[i], i+1);

      name = "EBTMT timing " + Numbers::sEB(i+1);
      meTimeMap_[i] = dqmStore_->bookProfile2D(name, name, 85, 0., 85., 20, 0., 20., -20.+shiftProf2D_, 20.+shiftProf2D_, "s");
      meTimeMap_[i]->setAxisTitle("ieta", 1);
      meTimeMap_[i]->setAxisTitle("iphi", 2);
      meTimeMap_[i]->setAxisTitle("time (ns)", 3);
      dqmStore_->tag(meTimeMap_[i], i+1);

      name = "EBTMT timing vs amplitude " + Numbers::sEB(i+1);
      meTimeAmpli_[i] = dqmStore_->book2D(name, name, nbinsE, binEdgesE, nbinsT, binEdgesT);
      meTimeAmpli_[i]->setAxisTitle("energy (GeV)", 1);
      meTimeAmpli_[i]->setAxisTitle("time (ns)", 2);
      dqmStore_->tag(meTimeAmpli_[i], i+1);
    }

    name = "EBTMT timing vs amplitude summary";
    meTimeAmpliSummary_ = dqmStore_->book2D(name, name, nbinsE, binEdgesE, nbinsT, binEdgesT);
    meTimeAmpliSummary_->setAxisTitle("energy (GeV)", 1);
    meTimeAmpliSummary_->setAxisTitle("time (ns)", 2);

    name = "EBTMT timing 1D summary";
    meTimeSummary1D_ = dqmStore_->book1D(name, name, 50, -25., 25.);
    meTimeSummary1D_->setAxisTitle("time (ns)", 1);

    name = "EBTMT timing map";
    meTimeSummaryMap_ = dqmStore_->bookProfile2D(name, name, 72, 0., 360., 34, -85, 85, -20.+shiftProf2D_, 20.+shiftProf2D_, "s");
    meTimeSummaryMap_->setAxisTitle("jphi", 1);
    meTimeSummaryMap_->setAxisTitle("jeta", 2);
    meTimeSummaryMap_->setAxisTitle("time (ns)", 3);

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

  }

  init_ = false;

}

void EBTimingTask::endJob(void){

  edm::LogInfo("EBTimingTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBTimingTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  const unsigned STABLE_BEAMS = 11;

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

  // resetting plots when stable beam is declared
  if( useBeamStatus_ && !stableBeamsDeclared_ ) {
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtRecord;
    if( e.getByLabel(L1GtEvmReadoutRecord_, gtRecord) ) {

      unsigned lhcBeamMode = gtRecord->gtfeWord().beamMode();

      if( lhcBeamMode == STABLE_BEAMS ){

	reset();

	stableBeamsDeclared_ = true;

      }
    }
  }

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv;
  c.get<EcalSeverityLevelAlgoRcd>().get(sevlv);

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

      uint32_t sev = sevlv->severityLevel(id, *hits);

      if ( (flag == EcalRecHit::kGood || flag == EcalRecHit::kOutOfTime) && sev != EcalSeverityLevel::kWeird ) {
        if ( meTimeAmpli ) meTimeAmpli->Fill(xval, yval);
        if ( meTimeAmpliSummary_ ) meTimeAmpliSummary_->Fill(xval, yval);

        if ( xval > energyThreshold_ ) {
          if ( meTime ) meTime->Fill(yval);
          if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval+shiftProf2D_);
          if ( meTimeSummary1D_ ) meTimeSummary1D_->Fill(yval);

          float xebeta = id.ieta() - 0.5 * id.zside();
          float xebphi = id.iphi() - 0.5;
          if ( meTimeSummaryMap_ ) meTimeSummaryMap_->Fill(xebphi, xebeta, yval+shiftProf2D_);
        }

      }
    }

  } else {

    edm::LogWarning("EBTimingTask") << EcalRecHitCollection_ << " not available";

  }

}

