#include <iostream>
#include <algorithm>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <DQM/EcalCommon/interface/Numbers.h>
#include "DQM/EcalCommon/interface/UtilsClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBDataCertificationTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EBDataCertificationTask::EBDataCertificationTask(const ParameterSet& ps) {

  dqmStore_ = Service<DQMStore>().operator->();

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEBDataCertificationSummary_ = 0;
  meEBDataCertificationSummaryMap_ = 0;
  for (int i = 0; i < 36; i++) {
    meEBDataCertification_[i] = 0;
  }
  meDataQualityByLumi_ = 0;
  meDCSQualityByLumi_ = 0;

  hDQM_ = 0;
  hDAQ_ = 0;
  hDCS_ = 0;
  hIntegrityByLumi_ = 0;
  hFrontendByLumi_ = 0;
  hDCSByLumi_ = 0;

}

EBDataCertificationTask::~EBDataCertificationTask() {

}

void EBDataCertificationTask::beginJob(void){

  char histo[200];
  
  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    
    sprintf(histo, "CertificationSummary");
    meEBDataCertificationSummary_ = dqmStore_->bookFloat(histo);
    meEBDataCertificationSummary_->Fill(0.0);

    sprintf(histo, "CertificationSummaryMap");
    meEBDataCertificationSummaryMap_ = dqmStore_->book2D(histo,histo, 72, 0., 72., 34, 0., 34.);
    meEBDataCertificationSummaryMap_->setAxisTitle("jphi", 1);
    meEBDataCertificationSummaryMap_->setAxisTitle("jeta", 2);
    
    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EcalBarrel_%s", Numbers::sEB(i+1).c_str());
      meEBDataCertification_[i] = dqmStore_->bookFloat(histo);
      meEBDataCertification_[i]->Fill(0.0);
    }

    sprintf(histo, "data quality by lumi");
    meDataQualityByLumi_ = dqmStore_->bookFloat(histo);
    meDataQualityByLumi_->setLumiFlag();
    meDataQualityByLumi_->Fill(-1.0);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    sprintf(histo, "DCS quality by lumi");
    meDCSQualityByLumi_ = dqmStore_->bookFloat(histo);
    meDCSQualityByLumi_->setLumiFlag();
    meDCSQualityByLumi_->Fill(-1.0);

  }

}

void EBDataCertificationTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EBDataCertificationTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

}

void EBDataCertificationTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->reset();

  char histo[200];
  
  MonitorElement* me;

  sprintf(histo, (prefixME_ + "/EventInfo/reportSummaryMap").c_str());
  me = dqmStore_->get(histo);
  hDQM_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDQM_ );

  sprintf(histo, (prefixME_ + "/EventInfo/DAQSummaryMap").c_str());
  me = dqmStore_->get(histo);
  hDAQ_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDAQ_ );

  sprintf(histo, (prefixME_ + "/EventInfo/DCSSummaryMap").c_str());
  me = dqmStore_->get(histo);
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

      // all white means problems: DAQ and DCS not available and DQM empty
      if ( xvalDQM == -1 && xvalDAQ == -1 && xvalDCS == -1) xcert = 0.0;
      else {
        // do not consider the white value of DAQ and DCS (problems with DB)
        xcert = fabs(xvalDQM) * fabs(xvalDAQ) * fabs(xvalDCS);
      }

      if ( meEBDataCertificationSummaryMap_ ) meEBDataCertificationSummaryMap_->setBinContent( iptt+1, iett+1, xcert );

      int i = ( iett<17 ) ? iptt/4 : 18+iptt/4;

      sumCertEB[i] += xcert;
      nValidChannelsEB[i]++;

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

  // evaluate the quality of observables checked by lumi
  sprintf(histo, (prefixME_ + "/EBIntegrityTask/EBIT weighted integrity errors by lumi").c_str());
  me = dqmStore_->get(histo);
  hIntegrityByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hIntegrityByLumi_ );

  sprintf(histo, (prefixME_ + "/EBStatusFlagsTask/FEStatus/EBSFT weighted frontend errors by lumi").c_str());
  me = dqmStore_->get(histo);
  hFrontendByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hFrontendByLumi_ );

  sprintf(histo, (prefixME_ + "/EventInfo/EB weighted DCS errors by lumi").c_str());
  me = dqmStore_->get(histo);
  hDCSByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hDCSByLumi_ );

  float integrityErrSum, frontendErrSum, dcsErrSum;
  integrityErrSum = frontendErrSum = dcsErrSum = 0.;
  for ( int i=0; i<36; i++) {
    if( hIntegrityByLumi_ ) integrityErrSum += hIntegrityByLumi_->GetBinContent(i+1);
    if( hFrontendByLumi_ ) frontendErrSum += hFrontendByLumi_->GetBinContent(i+1);
    if ( hDCSByLumi_ ) dcsErrSum += hDCSByLumi_->GetBinContent(i+1);
  }

  float integrityQual = 1.0;
  if ( hIntegrityByLumi_ ) integrityQual = 1.0 - integrityErrSum / 36. / hIntegrityByLumi_->GetBinContent(0);
  float frontendQual = 1.0;
  if ( hFrontendByLumi_ ) frontendQual = 1.0 - frontendErrSum / 36. / hFrontendByLumi_->GetBinContent(0);
  float dcsQual = 1.0;
  if ( hDCSByLumi_ ) dcsQual = 1.0 - dcsErrSum / 36.;

  if( meDataQualityByLumi_ ) meDataQualityByLumi_->Fill(min(integrityQual,frontendQual));
  if( meDCSQualityByLumi_ ) meDCSQualityByLumi_->Fill(dcsQual);

}

void EBDataCertificationTask::reset(void) {

  if ( meEBDataCertificationSummary_ ) meEBDataCertificationSummary_->Reset();

  for (int i = 0; i < 36; i++) {
    if ( meEBDataCertification_[i] ) meEBDataCertification_[i]->Reset();
  }

  if ( meEBDataCertificationSummaryMap_ ) meEBDataCertificationSummaryMap_->Reset();
  
  if ( meDataQualityByLumi_ ) meDataQualityByLumi_->Reset();
  if ( meDCSQualityByLumi_ ) meDCSQualityByLumi_->Reset();
  
}


void EBDataCertificationTask::cleanup(void){

  if ( cloneME_ ) {
    if( hDQM_ ) delete hDQM_;
    if( hDAQ_ ) delete hDAQ_;
    if( hDCS_ ) delete hDCS_;
    if( hIntegrityByLumi_ ) delete hIntegrityByLumi_;
    if( hFrontendByLumi_ ) delete hFrontendByLumi_;
    if( hDCSByLumi_ ) delete hDCSByLumi_;
  }
  hDQM_ = 0;
  hDAQ_ = 0;
  hDCS_ = 0;
  hIntegrityByLumi_ = 0;
  hFrontendByLumi_ = 0;
  hDCSByLumi_  = 0;

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");
    if ( meEBDataCertificationSummary_ ) dqmStore_->removeElement( meEBDataCertificationSummary_->getName() );
    if ( meEBDataCertificationSummaryMap_ ) dqmStore_->removeElement( meEBDataCertificationSummaryMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");
    for (int i = 0; i < 36; i++) {
      if ( meEBDataCertification_[i] ) dqmStore_->removeElement( meEBDataCertification_[i]->getName() );
    }
    if ( meDataQualityByLumi_ ) dqmStore_->removeElement( meDataQualityByLumi_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents"); 
    if ( meDCSQualityByLumi_ ) dqmStore_->removeElement( meDCSQualityByLumi_->getName() );

  }

}

void EBDataCertificationTask::analyze(const Event& e, const EventSetup& c){ 

}
