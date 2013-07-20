/*
 * \file EBDataCertificationTask.cc
 *
 * $Date: 2012/04/27 13:46:02 $
 * $Revision: 1.33 $
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <algorithm>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQM/EcalCommon/interface/Numbers.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBDataCertificationTask.h"

EBDataCertificationTask::EBDataCertificationTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEBDataCertificationSummary_ = 0;
  meEBDataCertificationSummaryMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDataCertification_[i] = 0;
  }

  hDQM_ = 0;
  hDAQ_ = 0;
  hDCS_ = 0;
  hIntegrityByLumi_ = 0;
  hFrontendByLumi_ = 0;
  hSynchronizationByLumi_ = 0;

}

EBDataCertificationTask::~EBDataCertificationTask() {

}

void EBDataCertificationTask::beginJob(void){

  std::string name;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    name = "CertificationSummary";
    meEBDataCertificationSummary_ = dqmStore_->bookFloat(name);
    meEBDataCertificationSummary_->Fill(-1.0);

    name = "CertificationSummaryMap";
    meEBDataCertificationSummaryMap_ = dqmStore_->book2D(name, name, 72, 0., 72., 34, 0., 34.);
    meEBDataCertificationSummaryMap_->setAxisTitle("jphi", 1);
    meEBDataCertificationSummaryMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 36; i++) {
      name = "EcalBarrel_" + Numbers::sEB(i+1);
      meEBDataCertification_[i] = dqmStore_->bookFloat(name);
      meEBDataCertification_[i]->Fill(-1.0);
    }

  }

}

void EBDataCertificationTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDataCertificationTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

}

void EBDataCertificationTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->reset();

  MonitorElement* me;

  // evaluate the DQM quality of observables checked by lumi
  float DQMVal[36];
  for (int i = 0; i < 36; i++) {
    DQMVal[i] = -1.;
  }

  me = dqmStore_->get(prefixME_ + "/EBIntegrityTask/EBIT weighted integrity errors by lumi");
  hIntegrityByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hIntegrityByLumi_ );

  me = dqmStore_->get(prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT weighted frontend errors by lumi");
  hFrontendByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hFrontendByLumi_ );

  me = dqmStore_->get(prefixME_ + "/EBRawDataTask/EBRDT FE synchronization errors by lumi");
  hSynchronizationByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hSynchronizationByLumi_ );

  if( hIntegrityByLumi_ && hFrontendByLumi_ && hSynchronizationByLumi_ ) {

    float integrityErrSum = 0.;
    float integrityQual = 1.0;
    float frontendErrSum = 0.;
    float frontendQual = 1.0;
    float synchronizationErrSum = 0.;
    float synchronizationQual = 1.0;

    for ( int i=0; i<36; i++) {
      float ismIntegrityQual = 1.0;
      if( hIntegrityByLumi_->GetBinContent(0) > 0 ) {
        float errors = hIntegrityByLumi_->GetBinContent(i+1);
        ismIntegrityQual = 1.0 - errors/hIntegrityByLumi_->GetBinContent(0);
        integrityErrSum += errors;
      }
      float ismFrontendQual = 1.0;
      if( hFrontendByLumi_->GetBinContent(0) > 0 ) {
        float errors = hFrontendByLumi_->GetBinContent(i+1);
        ismFrontendQual = 1.0 - errors/hFrontendByLumi_->GetBinContent(0);
        frontendErrSum += errors;
      }
      float ismSynchronizationQual = 1.0;
      if( hSynchronizationByLumi_->GetBinContent(0) > 0 ) {
        float errors = hSynchronizationByLumi_->GetBinContent(i+1);
        ismSynchronizationQual = 1.0 - errors/hSynchronizationByLumi_->GetBinContent(0);
        synchronizationErrSum += errors;
      }
      float minVal= std::min(ismIntegrityQual,ismFrontendQual);
      DQMVal[i] = std::min(minVal,ismSynchronizationQual);
    }

    if( hIntegrityByLumi_->GetBinContent(0) > 0 ) integrityQual = 1.0 - integrityErrSum/hIntegrityByLumi_->GetBinContent(0)/36.;
    if( hFrontendByLumi_->GetBinContent(0) > 0 ) frontendQual = 1.0 - frontendErrSum/hFrontendByLumi_->GetBinContent(0)/36.;
    if( hSynchronizationByLumi_->GetBinContent(0) > 0 ) synchronizationQual = 1.0 - synchronizationErrSum/hSynchronizationByLumi_->GetBinContent(0)/36.;
    float minVal = std::min(integrityQual,frontendQual);
    float totDQMVal = std::min(minVal,synchronizationQual);

    me = dqmStore_->get((prefixME_ + "/EventInfo/reportSummary"));
    if( me ) me->Fill(totDQMVal);
    for ( int i=0; i<36; i++) {
      me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryContents/EcalBarrel_" + Numbers::sEB(i+1) ) ;
      if( me ) me->Fill(DQMVal[i]);

      me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
      if( me ) {
        for ( int iett = 0; iett < 34; iett++ ) {
          for ( int iptt = 0; iptt < 72; iptt++ ) {
            int ism = ( iett<17 ) ? iptt/4+1 : 18+iptt/4+1;
            if( i == (ism-1) ) me->setBinContent(iptt+1, iett+1, DQMVal[ism-1]);
          }
        }
      }
    }

  }

  // now combine reduced DQM with DCS and DAQ
  me = dqmStore_->get(prefixME_ + "/EventInfo/DAQSummaryMap");
  hDAQ_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDAQ_ );

  me = dqmStore_->get(prefixME_ + "/EventInfo/DCSSummaryMap");
  hDCS_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDCS_ );

  float sumCert = 0.;
  float sumCertEB[36];
  int nValidChannels = 0;
  int nValidChannelsEB[36];

  for (int i = 0; i < 36; i++) {
    sumCertEB[i] = 0.;
    nValidChannelsEB[i] = 0;
  }

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {

      int ism = ( iett<17 ) ? iptt/4+1 : 18+iptt/4+1;

      float xvalDQM = DQMVal[ism-1];

      float xvalDAQ, xvalDCS;
      xvalDAQ = xvalDCS = -1.;
      float xcert = -1.;

      if ( hDAQ_ ) xvalDAQ = hDAQ_->GetBinContent( iptt+1, iett+1 );
      if ( hDCS_ ) xvalDCS = hDCS_->GetBinContent( iptt+1, iett+1 );

      if ( xvalDQM == -1  || ( xvalDAQ == -1 && xvalDCS == -1 ) ) {
        // problems: DQM empty or DAQ and DCS not available
        xcert = 0.0;
      } else {
        // do not consider the white value of DAQ and DCS (problems with DB)
        xcert = std::abs(xvalDQM) * std::abs(xvalDAQ) * std::abs(xvalDCS);
      }

      if ( meEBDataCertificationSummaryMap_ ) meEBDataCertificationSummaryMap_->setBinContent( iptt+1, iett+1, xcert );

      sumCertEB[ism-1] += xcert;
      nValidChannelsEB[ism-1]++;

      sumCert += xcert;
      nValidChannels++;

    }
  }

  if( meEBDataCertificationSummary_ ) {
    if( nValidChannels>0 ) meEBDataCertificationSummary_->Fill( sumCert/nValidChannels );
    else meEBDataCertificationSummary_->Fill( 0.0 );
  }

  for (int i = 0; i < 36; i++) {
    if( meEBDataCertification_[i] ) {
      if( nValidChannelsEB[i]>0 ) meEBDataCertification_[i]->Fill( sumCertEB[i]/nValidChannelsEB[i] );
      else meEBDataCertification_[i]->Fill( 0.0 );
    }
  }

}

void EBDataCertificationTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EBDataCertificationTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  this->reset();

  MonitorElement* me;

  me = dqmStore_->get(prefixME_ + "/EventInfo/reportSummaryMap");
  hDQM_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDQM_ );

  me = dqmStore_->get(prefixME_ + "/EventInfo/DAQSummaryMap");
  hDAQ_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDAQ_ );

  me = dqmStore_->get(prefixME_ + "/EventInfo/DCSSummaryMap");
  hDCS_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDCS_ );

  float sumCert = 0.;
  float sumCertEB[36];
  int nValidChannels = 0;
  int nValidChannelsEB[36];

  for (int i = 0; i < 36; i++) {
    sumCertEB[i] = 0.;
    nValidChannelsEB[i] = 0;
  }

  for ( int iett = 0; iett < 34; iett++ ) {
    for ( int iptt = 0; iptt < 72; iptt++ ) {

      float xvalDQM, xvalDAQ, xvalDCS;
      xvalDQM = xvalDAQ = xvalDCS = -1.;
      float xcert = -1;

      if ( hDQM_ ) xvalDQM = hDQM_->GetBinContent( iptt+1, iett+1 );
      if ( hDAQ_ ) xvalDAQ = hDAQ_->GetBinContent( iptt+1, iett+1 );
      if ( hDCS_ ) xvalDCS = hDCS_->GetBinContent( iptt+1, iett+1 );

      if ( xvalDQM == -1 || ( xvalDAQ == -1 && xvalDCS == -1 ) ) {
        // problems: DQM empty or DAQ and DCS not available
        xcert = 0.0;
      } else {
        // do not consider the white value of DAQ and DCS (problems with DB)
        xcert = std::abs(xvalDQM) * std::abs(xvalDAQ) * std::abs(xvalDCS);
      }

      if ( meEBDataCertificationSummaryMap_ ) meEBDataCertificationSummaryMap_->setBinContent( iptt+1, iett+1, xcert );

      int ism = ( iett<17 ) ? iptt/4+1 : 18+iptt/4+1;

      sumCertEB[ism-1] += xcert;
      nValidChannelsEB[ism-1]++;

      sumCert += xcert;
      nValidChannels++;

    }
  }

  if( meEBDataCertificationSummary_ ) {
    if( nValidChannels>0 ) {
      meEBDataCertificationSummary_->Fill( sumCert/nValidChannels );
    } else {
      meEBDataCertificationSummary_->Fill( 0.0 );
    }
  }

  for (int i = 0; i < 36; i++) {
    if( meEBDataCertification_[i] ) {
      if( nValidChannelsEB[i]>0 ) {
        meEBDataCertification_[i]->Fill( sumCertEB[i]/nValidChannelsEB[i] );
      } else {
        meEBDataCertification_[i]->Fill( 0.0 );
      }
    }
  }

}

void EBDataCertificationTask::reset(void) {

  if ( meEBDataCertificationSummary_ ) meEBDataCertificationSummary_->Reset();

  for (int i = 0; i < 36; i++) {
    if ( meEBDataCertification_[i] ) meEBDataCertification_[i]->Reset();
  }

  if ( meEBDataCertificationSummaryMap_ ) meEBDataCertificationSummaryMap_->Reset();

}


void EBDataCertificationTask::cleanup(void){

  if ( cloneME_ ) {
    if( hDQM_ ) delete hDQM_;
    if( hDAQ_ ) delete hDAQ_;
    if( hDCS_ ) delete hDCS_;
    if( hIntegrityByLumi_ ) delete hIntegrityByLumi_;
    if( hFrontendByLumi_ ) delete hFrontendByLumi_;
    if( hSynchronizationByLumi_ ) delete hSynchronizationByLumi_;
  }
  hDQM_ = 0;
  hDAQ_ = 0;
  hDCS_ = 0;
  hIntegrityByLumi_ = 0;
  hFrontendByLumi_ = 0;
  hSynchronizationByLumi_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    if ( meEBDataCertificationSummary_ ) dqmStore_->removeElement( meEBDataCertificationSummary_->getName() );
    if ( meEBDataCertificationSummaryMap_ ) dqmStore_->removeElement( meEBDataCertificationSummaryMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");
    for (int i = 0; i < 36; i++) {
      if ( meEBDataCertification_[i] ) dqmStore_->removeElement( meEBDataCertification_[i]->getName() );
    }
  }

}

void EBDataCertificationTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
