/*
 * \file EETimingTask.cc
 *
 * $Date: 2010/02/12 21:57:31 $
 * $Revision: 1.56 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EETimingTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EETimingTask::EETimingTask(const ParameterSet& ps){

  init_ = false;

  initGeometry_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  for (int i = 0; i < 18; i++) {
    meTime_[i] = 0;
    meTimeMap_[i] = 0;
    meTimeAmpli_[i] = 0;
  }

  for (int i = 0; i < 2; i++) {
    meTimeAmpliSummary_[i] = 0;
    meTimeSummary1D_[i] = 0;
    meTimeSummaryMap_[i] = 0;
    meTimeSummaryMapProjEta_[i] = 0;
    meTimeSummaryMapProjPhi_[i] = 0;
  }

  meTimeDelta_ = 0;
  meDTimeVsDEnergy_ = 0;

}

EETimingTask::~EETimingTask(){

}

void EETimingTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");
    dqmStore_->rmdir(prefixME_ + "/EETimingTask");
  }

}

void EETimingTask::beginRun(const Run& r, const EventSetup& c) {

  if( !initGeometry_ ) { 
    // ideal
    Numbers::initGeometry(c, true);
    // calo geometry
    c.get<CaloGeometryRecord>().get(pGeometry_);
    initGeometry_ = true;
  }

  if ( ! mergeRuns_ ) this->reset();

}

void EETimingTask::endRun(const Run& r, const EventSetup& c) {

}

void EETimingTask::reset(void) {

  for (int i = 0; i < 18; i++) {
    if ( meTime_[i] ) meTime_[i]->Reset();
    if ( meTimeMap_[i] ) meTimeMap_[i]->Reset();
    if ( meTimeAmpli_[i] ) meTimeAmpli_[i]->Reset();
  }

  for (int i = 0; i < 2; i++) {
    if ( meTimeAmpliSummary_[i] ) meTimeAmpliSummary_[i]->Reset();
    if ( meTimeSummary1D_[i] ) meTimeSummary1D_[i]->Reset();
    if ( meTimeSummaryMap_[i] ) meTimeSummaryMap_[i]->Reset();
    if ( meTimeSummaryMapProjEta_[i] )  meTimeSummaryMapProjEta_[i]->Reset();
    if ( meTimeSummaryMapProjPhi_[i] )  meTimeSummaryMapProjPhi_[i]->Reset();    
  }

  if ( meTimeDelta_ ) meTimeDelta_->Reset();
  if ( meDTimeVsDEnergy_ ) meTimeDelta_->Reset();

}

void EETimingTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EETMT timing 1D %s", Numbers::sEE(i+1).c_str());
      meTime_[i] = dqmStore_->book1D(histo, histo, 50, -50., 50.);
      meTime_[i]->setAxisTitle("time (ns)", 1);
      dqmStore_->tag(meTime_[i], i+1);

      sprintf(histo, "EETMT timing %s", Numbers::sEE(i+1).c_str());
      meTimeMap_[i] = dqmStore_->bookProfile2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., 25., 75., "s");
      meTimeMap_[i]->setAxisTitle("jx", 1);
      meTimeMap_[i]->setAxisTitle("jy", 2);
      meTimeMap_[i]->setAxisTitle("time (ns)", 3);
      dqmStore_->tag(meTimeMap_[i], i+1);

      sprintf(histo, "EETMT timing vs amplitude %s", Numbers::sEE(i+1).c_str());
      meTimeAmpli_[i] = dqmStore_->book2D(histo, histo, 100, 0., 10., 50, -50., 50.);
      meTimeAmpli_[i]->setAxisTitle("energy (GeV)", 1);
      meTimeAmpli_[i]->setAxisTitle("time (ns)", 2);
      dqmStore_->tag(meTimeAmpli_[i], i+1);
    }
    
    sprintf(histo, "EETMT timing vs amplitude summary EE -");
    meTimeAmpliSummary_[0] = dqmStore_->book2D(histo, histo, 100, 0., 10., 50, -50., 50.);
    meTimeAmpliSummary_[0]->setAxisTitle("energy (GeV)", 1);
    meTimeAmpliSummary_[0]->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EETMT timing vs amplitude summary EE +");
    meTimeAmpliSummary_[1] = dqmStore_->book2D(histo, histo, 100, 0., 10., 50, -50., 50.);
    meTimeAmpliSummary_[1]->setAxisTitle("energy (GeV)", 1);
    meTimeAmpliSummary_[1]->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EETMT timing 1D summary EE -");
    meTimeSummary1D_[0] = dqmStore_->book1D(histo, histo, 50, -50., 50.);
    meTimeSummary1D_[0]->setAxisTitle("time (ns)", 1);

    sprintf(histo, "EETMT timing 1D summary EE +");
    meTimeSummary1D_[1] = dqmStore_->book1D(histo, histo, 50, -50., 50.);
    meTimeSummary1D_[1]->setAxisTitle("time (ns)", 1);

    sprintf(histo, "EETMT timing map EE -");
    meTimeSummaryMap_[0] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 100., 20, 0., 100., 50, 25., 75., "s");
    meTimeSummaryMap_[0]->setAxisTitle("jx", 1);
    meTimeSummaryMap_[0]->setAxisTitle("jy", 2);
    meTimeSummaryMap_[0]->setAxisTitle("time (ns)", 3);

    sprintf(histo, "EETMT timing map EE +");
    meTimeSummaryMap_[1] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 100., 20, 0., 100., 50, 25., 75., "s");
    meTimeSummaryMap_[1]->setAxisTitle("jx", 1);
    meTimeSummaryMap_[1]->setAxisTitle("jy", 2);
    meTimeSummaryMap_[1]->setAxisTitle("time (ns)", 3);

    sprintf(histo, "EETMT timing projection eta EE -");
    meTimeSummaryMapProjEta_[0] = dqmStore_->bookProfile(histo, histo, 20, -3.0, -1.479, 50, -50., 50., "s");
    meTimeSummaryMapProjEta_[0]->setAxisTitle("eta", 1);
    meTimeSummaryMapProjEta_[0]->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EETMT timing projection eta EE +");
    meTimeSummaryMapProjEta_[1] = dqmStore_->bookProfile(histo, histo, 20, 1.479, 3.0, 50, -50., 50., "s");
    meTimeSummaryMapProjEta_[1]->setAxisTitle("phi", 1);
    meTimeSummaryMapProjEta_[1]->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EETMT timing projection phi EE -");
    meTimeSummaryMapProjPhi_[0] = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 50, -50., 50., "s");
    meTimeSummaryMapProjPhi_[0]->setAxisTitle("phi", 1);
    meTimeSummaryMapProjPhi_[0]->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EETMT timing projection phi EE +");
    meTimeSummaryMapProjPhi_[1] = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 50, -50., 50., "s");
    meTimeSummaryMapProjPhi_[1]->setAxisTitle("phi", 1);
    meTimeSummaryMapProjPhi_[1]->setAxisTitle("time (ns)", 2);

    sprintf(histo, "EETMT timing EE+ - EE-");
    meTimeDelta_ = dqmStore_->book1D(histo, histo, 100, -3., 3.);
    meTimeDelta_->setAxisTitle("time (ns)", 1);

    sprintf(histo, "EETMT timing min vs Et min");
    meDTimeVsDEnergy_ = dqmStore_->book2D(histo, histo, 30, 0., 3., 30, -50., 50.);
    meDTimeVsDEnergy_->setAxisTitle("min(E_{T}^{EE+},E_{T}^{EE-}) (GeV)", 1);
    meDTimeVsDEnergy_->setAxisTitle("min(time^{EE+},time^{EE-}) (ns)", 2);

  }

}

void EETimingTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EETimingTask");

    for ( int i = 0; i < 18; i++ ) {
      if ( meTime_[i] ) dqmStore_->removeElement( meTime_[i]->getName() );
      meTime_[i] = 0;

      if ( meTimeMap_[i] ) dqmStore_->removeElement( meTimeMap_[i]->getName() );
      meTimeMap_[i] = 0;

      if ( meTimeAmpli_[i] ) dqmStore_->removeElement( meTimeAmpli_[i]->getName() );
      meTimeAmpli_[i] = 0;
    }

    for (int i = 0; i < 2; i++) {
      if ( meTimeAmpliSummary_[i] ) dqmStore_->removeElement( meTimeAmpliSummary_[i]->getName() );
      meTimeAmpliSummary_[i] = 0;

      if ( meTimeSummary1D_[i] ) dqmStore_->removeElement( meTimeSummary1D_[i]->getName() );
      meTimeSummary1D_[i] = 0;

      if ( meTimeSummaryMap_[i] ) dqmStore_->removeElement( meTimeSummaryMap_[i]->getName() );
      meTimeSummaryMap_[i] = 0;
      
      if ( meTimeSummaryMapProjEta_[i] ) dqmStore_->removeElement( meTimeSummaryMapProjEta_[i]->getName() );
      meTimeSummaryMapProjEta_[i] = 0;

      if ( meTimeSummaryMapProjPhi_[i] ) dqmStore_->removeElement( meTimeSummaryMapProjPhi_[i]->getName() );
      meTimeSummaryMapProjPhi_[i] = 0;
    }

    if ( meTimeDelta_ ) dqmStore_->removeElement( meTimeDelta_->getName() );
    meTimeDelta_ = 0;

    if ( meDTimeVsDEnergy_ ) dqmStore_->removeElement( meDTimeVsDEnergy_->getName() );
    meDTimeVsDEnergy_ = 0;

  }

  init_ = false;

}

void EETimingTask::endJob(void){

  LogInfo("EETimingTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EETimingTask::analyze(const Event& e, const EventSetup& c){

  bool isData = true;
  bool enable = false;
  int runType[18];
  for (int i=0; i<18; i++) runType[i] = -1;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *dcchItr, EcalEndcap );

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
    LogWarning("EETimingTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  // channel status
  edm::ESHandle<EcalChannelStatus> pChannelStatus;
  c.get<EcalChannelStatusRcd>().get(pChannelStatus);
  const EcalChannelStatus *chStatus = pChannelStatus.product();

  float sumTime_hithr[2] = {0.,0.};
  float wsumTime_lowthr[2] = {0.,0};
  float wsum[2] = {0.,0.};
  int n_hithr[2] = {0,0};
  int n_lowthr[2] = {0,0};

  Handle<EcalRecHitCollection> hits;

  if ( e.getByLabel(EcalRecHitCollection_, hits) ) {

    int neh = hits->size();
    LogDebug("EETimingTask") << "event " << ievt_ << " hits collection size " << neh;

    for ( EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

      EEDetId id = hitItr->id();

      int ix = id.ix();
      int iy = id.iy();
      int iz = ( id.positiveZ() ) ? 1 : 0;
      
      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

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

      float theta = pGeometry_->getGeometry(id)->getPosition().theta();
      float eta = pGeometry_->getGeometry(id)->getPosition().eta();
      float phi = pGeometry_->getGeometry(id)->getPosition().phi();
      float et = hitItr->energy() * fabs(sin(theta));

      if ( flag == EcalRecHit::kGood && sev == EcalSeverityLevelAlgo::kGood ) {
        if ( meTimeAmpli ) meTimeAmpli->Fill(xval, yval);
        if ( meTimeAmpliSummary_[iz] ) meTimeAmpliSummary_[iz]->Fill(xval, yval);

        if ( et > 0.300 ) {

          wsum[iz] += et;
          wsumTime_lowthr[iz] += hitItr->time() * et;
          n_lowthr[iz]++;

          if ( et > 0.600 ) {
            if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval+50.);

            // exclude the noisiest region around the hole from 1D
            if ( fabs(ix-50) >= 5 && fabs(ix-50) <= 10 && fabs(iy-50) >= 5 && fabs(iy-50) <= 10 ) {
              if ( meTime ) meTime->Fill(yval);
              if ( meTimeSummary1D_[iz] ) meTimeSummary1D_[iz]->Fill(yval);
              sumTime_hithr[iz] += yval;
              n_hithr[iz]++;
            }

          }

          if ( meTimeSummaryMap_[iz] ) meTimeSummaryMap_[iz]->Fill(xix, xiy, yval+50.);
          if ( meTimeSummaryMapProjEta_[iz] ) meTimeSummaryMapProjEta_[iz]->Fill(eta, yval);
          if ( meTimeSummaryMapProjPhi_[iz] ) meTimeSummaryMapProjPhi_[iz]->Fill(phi, yval);
        
        }

      }
    }

    float mean_hithr[2], mean_lowthr[2];
    for ( int i=0; i<2; i++ ) {
      if ( n_hithr[i] > 0 ) mean_hithr[i] = sumTime_hithr[i] / n_hithr[i];
      if ( wsum[i] > 0 ) mean_lowthr[i] = wsumTime_lowthr[i] / wsum[i];
    }

    if ( meTimeDelta_ && n_hithr[0] > 0 && n_lowthr[0] > 1 && n_hithr[1] > 0 && n_lowthr[1] > 1 ) meTimeDelta_->Fill( mean_hithr[1] - mean_hithr[0] );
    if ( meDTimeVsDEnergy_ && n_lowthr[0] > 0 && n_lowthr[1] > 0 ) {
      float mintime = TMath::Min(wsumTime_lowthr[0], wsumTime_lowthr[1]);
      float minEt = TMath::Min(wsum[0], wsum[1]);
      meDTimeVsDEnergy_->Fill(minEt,mintime);
    }

  } else {

    LogWarning("EETimingTask") << EcalRecHitCollection_ << " not available";

  }

}

