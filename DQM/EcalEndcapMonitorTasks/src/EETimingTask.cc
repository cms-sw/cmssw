/*
 * \file EETimingTask.cc
 *
 * $Date: 2011/10/30 15:01:28 $
 * $Revision: 1.82 $
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
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EETimingTask.h"

EETimingTask::EETimingTask(const edm::ParameterSet& ps){

  init_ = false;

  initCaloGeometry_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  energyThreshold_ = ps.getUntrackedParameter<double>("energyThreshold", 3.);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  L1GtEvmReadoutRecord_ = ps.getParameter<edm::InputTag>("L1GtEvmReadoutRecord");

  useBeamStatus_ = ps.getUntrackedParameter<bool>("useBeamStatus", false);

  for (int i = 0; i < 18; i++) {
    meTime_[i] = 0;
    meTimeMap_[i] = 0;
    meTimeAmpli_[i] = 0;
  }

  for (int i = 0; i < 2; i++) {
    meTimeAmpliSummary_[i] = 0;
    meTimeSummary1D_[i] = 0;
    meTimeSummaryMap_[i] = 0;
  }

  meTimeDelta_ = 0;
  meTimeDelta2D_ = 0;

  stableBeamsDeclared_ = false;

  ievt_ = 0;
}

EETimingTask::~EETimingTask(){

}

void EETimingTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Timing");
    dqmStore_->rmdir(prefixME_ + "/Timing");
  }

}

void EETimingTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, true);

  if( !initCaloGeometry_ ) {
    c.get<CaloGeometryRecord>().get(pGeometry_);
    initCaloGeometry_ = true;
  }

  if ( ! mergeRuns_ ) this->reset();

  stableBeamsDeclared_ = false;

}

void EETimingTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

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
  }

  if ( meTimeDelta_ ) meTimeDelta_->Reset();
  if ( meTimeDelta2D_ ) meTimeDelta2D_->Reset();

}

void EETimingTask::setup(void){

  init_ = true;

  std::string name;
  std::string subdet[2] = {"EE-", "EE+"};

  //for timing vs amplitude plots
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
    dqmStore_->setCurrentFolder(prefixME_ + "/Timing");

    for(int iSubdet(0); iSubdet < 2; iSubdet++){
      name = "TimingTask timing vs amplitude all " + subdet[iSubdet];
      meTimeAmpliSummary_[iSubdet] = dqmStore_->book2D(name, name, nbinsE, binEdgesE, nbinsT, binEdgesT);
      meTimeAmpliSummary_[iSubdet]->setAxisTitle("energy (GeV)", 1);
      meTimeAmpliSummary_[iSubdet]->setAxisTitle("time (ns)", 2);

      name = "TimingTask timing all 1D " + subdet[iSubdet];
      meTimeSummary1D_[iSubdet] = dqmStore_->book1D(name, name, 50, -25., 25.);
      meTimeSummary1D_[iSubdet]->setAxisTitle("time (ns)", 1);

      name = "TimingTask timing " + subdet[iSubdet];
      meTimeSummaryMap_[iSubdet] = dqmStore_->bookProfile2D(name, name, 20, 0., 100., 20, 0., 100., -7.+shiftProf2D, 7.+shiftProf2D, "s");
      meTimeSummaryMap_[iSubdet]->setAxisTitle("ix'", 1);
      meTimeSummaryMap_[iSubdet]->setAxisTitle("101-iy'", 2);
      meTimeSummaryMap_[iSubdet]->setAxisTitle("time (ns)", 3);
    }

    name = "TimingTask timing EE+ - EE-";
    meTimeDelta_ = dqmStore_->book1D(name, name, 100, -3., 3.);
    meTimeDelta_->setAxisTitle("time (ns)", 1);

    name = "TimingTask timing EE+ vs EE-";
    meTimeDelta2D_ = dqmStore_->book2D(name, name, 25, -25., 25., 25, -25., 25.);
    meTimeDelta2D_->setAxisTitle("EE+ average time (ns)", 1);
    meTimeDelta2D_->setAxisTitle("EE- average time (ns)", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/Timing/Distribution");
    for (int i = 0; i < 18; i++) {
      name = "TimingTask timing 1D " + Numbers::sEE(i+1);
      meTime_[i] = dqmStore_->book1D(name, name, 50, -25., 25.);
      meTime_[i]->setAxisTitle("time (ns)", 1);
      dqmStore_->tag(meTime_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/Timing/Profile");
    for (int i = 0; i < 18; i++) {
      name = "TimingTask timing " + Numbers::sEE(i+1);
      meTimeMap_[i] = dqmStore_->bookProfile2D(name, name, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50., -25.+shiftProf2D, 25.+shiftProf2D, "s");
      meTimeMap_[i]->setAxisTitle("ix", 1);
      if ( i+1 >= 1 && i+1 <= 9 ) meTimeMap_[i]->setAxisTitle("101-ix", 1);
      meTimeMap_[i]->setAxisTitle("iy", 2);
      meTimeMap_[i]->setAxisTitle("time (ns)", 3);
      dqmStore_->tag(meTimeMap_[i], i+1);
    }

    dqmStore_->setCurrentFolder(prefixME_ + "/Timing/VsAmplitude");
    for (int i = 0; i < 18; i++) {
      name = "TimingTask timing vs amplitude " + Numbers::sEE(i+1);
      meTimeAmpli_[i] = dqmStore_->book2D(name, name, nbinsE, binEdgesE, nbinsT, binEdgesT);
      meTimeAmpli_[i]->setAxisTitle("energy (GeV)", 1);
      meTimeAmpli_[i]->setAxisTitle("time (ns)", 2);
      dqmStore_->tag(meTimeAmpli_[i], i+1);
    }

  }

}

void EETimingTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/Timing");

    for ( int i = 0; i < 18; i++ ) {
      if ( meTime_[i] ) dqmStore_->removeElement( meTime_[i]->getFullname() );
      meTime_[i] = 0;

      if ( meTimeMap_[i] ) dqmStore_->removeElement( meTimeMap_[i]->getFullname() );
      meTimeMap_[i] = 0;

      if ( meTimeAmpli_[i] ) dqmStore_->removeElement( meTimeAmpli_[i]->getFullname() );
      meTimeAmpli_[i] = 0;
    }

    for (int i = 0; i < 2; i++) {
      if ( meTimeAmpliSummary_[i] ) dqmStore_->removeElement( meTimeAmpliSummary_[i]->getFullname() );
      meTimeAmpliSummary_[i] = 0;

      if ( meTimeSummary1D_[i] ) dqmStore_->removeElement( meTimeSummary1D_[i]->getFullname() );
      meTimeSummary1D_[i] = 0;

      if ( meTimeSummaryMap_[i] ) dqmStore_->removeElement( meTimeSummaryMap_[i]->getFullname() );
      meTimeSummaryMap_[i] = 0;

    }

    if ( meTimeDelta_ ) dqmStore_->removeElement( meTimeDelta_->getFullname() );
    meTimeDelta_ = 0;

    if ( meTimeDelta2D_ ) dqmStore_->removeElement( meTimeDelta2D_->getFullname() );
    meTimeDelta2D_ = 0;

  }

  init_ = false;

}

void EETimingTask::endJob(void){

  edm::LogInfo("EETimingTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EETimingTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  const unsigned STABLE_BEAMS = 11;

  bool isData = true;
  bool enable = false;
  int runType[18];
  for (int i=0; i<18; i++) runType[i] = -1;

  edm::Handle<EcalRawDataCollection> dcchs;

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
    edm::LogWarning("EETimingTask") << EcalRawDataCollection_ << " not available";

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

  float sumTime_hithr[2] = {0.,0.};
  int n_hithr[2] = {0,0};

  edm::Handle<EcalRecHitCollection> hits;

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

      // it's no use to use severitylevel to detect spikes (SeverityLevelAlgo simply uses RecHit flag for spikes)
      uint32_t mask = 0xffffffff ^ ((0x1 << EcalRecHit::kGood) | (0x1 << EcalRecHit::kOutOfTime));

      float energyThreshold = std::abs( Numbers::eta(id) ) < 2.4 ? energyThreshold_ : 2.*energyThreshold_;

      // allow only kGood or kOutOfTime hits
      if( !hitItr->checkFlagMask(mask) ){
        if ( meTimeAmpli ) meTimeAmpli->Fill(xval, yval);
        if ( meTimeAmpliSummary_[iz] ) meTimeAmpliSummary_[iz]->Fill(xval, yval);
        if ( hitItr->energy() > energyThreshold ) {
          if ( meTimeMap ) meTimeMap->Fill(xix, xiy, yval+shiftProf2D);
          if ( meTime ) meTime->Fill(yval);
          if ( meTimeSummary1D_[iz] ) meTimeSummary1D_[iz]->Fill(yval);

          if ( meTimeSummaryMap_[iz] ) meTimeSummaryMap_[iz]->Fill(id.ix()-0.5, xiy, yval+shiftProf2D);

          sumTime_hithr[iz] += yval;
          n_hithr[iz]++;
        }
      } // good rh for timing
    } // loop over rh

    if (n_hithr[0] > 0 && n_hithr[1] > 0 ) {
      if ( meTimeDelta_ ) meTimeDelta_->Fill( sumTime_hithr[1]/n_hithr[1] - sumTime_hithr[0]/n_hithr[0] );
      if ( meTimeDelta2D_ ) meTimeDelta2D_->Fill( sumTime_hithr[1]/n_hithr[1], sumTime_hithr[0]/n_hithr[0] );
    }

  } else {

    edm::LogWarning("EETimingTask") << EcalRecHitCollection_ << " not available";

  }

}

