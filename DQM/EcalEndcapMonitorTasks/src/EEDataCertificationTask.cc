/*
 * \file EEDataCertificationTask.cc
 *
 * $Date: 2012/04/27 13:46:14 $
 * $Revision: 1.31 $
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <algorithm>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEDataCertificationTask.h"

EEDataCertificationTask::EEDataCertificationTask(const edm::ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEEDataCertificationSummary_ = 0;
  meEEDataCertificationSummaryMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDataCertification_[i] = 0;
  }

  hDQM_ = 0;
  hDAQ_ = 0;
  hIntegrityByLumi_ = 0;
  hFrontendByLumi_ = 0;
  hSynchronizationByLumi_ = 0;

}

EEDataCertificationTask::~EEDataCertificationTask() {

}

void EEDataCertificationTask::beginJob(void){

  if ( dqmStore_ ) {

    std::string name;

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    meEEDataCertificationSummary_ = dqmStore_->bookFloat( "CertificationSummary" );
    meEEDataCertificationSummary_->Fill(-1.0);

    name = "CertificationSummaryMap";
    meEEDataCertificationSummaryMap_ = dqmStore_->book2D(name, name, 40, 0., 200., 20, 0., 100.);
    meEEDataCertificationSummaryMap_->setAxisTitle("ix / ix+100", 1);
    meEEDataCertificationSummaryMap_->setAxisTitle("iy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");

    for (int i = 0; i < 18; i++) {
      meEEDataCertification_[i] = dqmStore_->bookFloat( "EcalEndcap_" + Numbers::sEE(i+1) );
      meEEDataCertification_[i]->Fill(-1.0);
    }

  }

}

void EEDataCertificationTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDataCertificationTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

}

void EEDataCertificationTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  this->reset();

  MonitorElement* me;

  // evaluate the DQM quality of observables checked by lumi
  float DQMVal[18];
  for (int i = 0; i < 18; i++) {
    DQMVal[i] = -1.;
  }

  edm::ESHandle< EcalElectronicsMapping > handle;
  iSetup.get< EcalMappingRcd >().get(handle);
  const EcalElectronicsMapping *map = handle.product();
  if( ! map ) edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";

  me = dqmStore_->get( prefixME_ + "/EEIntegrityTask/EEIT weighted integrity errors by lumi" );
  hIntegrityByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hIntegrityByLumi_ );

  me = dqmStore_->get( prefixME_ + "/EEStatusFlagsTask/FEStatus/EESFT weighted frontend errors by lumi" );
  hFrontendByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hFrontendByLumi_ );

  me = dqmStore_->get( prefixME_ + "/EERawDataTask/EERDT FE synchronization errors by lumi" );
  hSynchronizationByLumi_ = UtilsClient::getHisto<TH1F*>( me, cloneME_, hSynchronizationByLumi_ );

  if( hIntegrityByLumi_ && hFrontendByLumi_ && hSynchronizationByLumi_ && map) {

    float integrityErrSum = 0.;
    float integrityQual = 1.0;
    float frontendErrSum = 0.;
    float frontendQual = 1.0;
    float synchronizationErrSum = 0.;
    float synchronizationQual = 1.0;

    for ( int i=0; i<18; i++) {
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

    if( hIntegrityByLumi_->GetBinContent(0) > 0 ) integrityQual = 1.0 - integrityErrSum/hIntegrityByLumi_->GetBinContent(0)/18.;
    if( hFrontendByLumi_->GetBinContent(0) > 0 ) frontendQual = 1.0 - frontendErrSum/hFrontendByLumi_->GetBinContent(0)/18.;
    if( hSynchronizationByLumi_->GetBinContent(0) > 0 ) synchronizationQual = 1.0 - synchronizationErrSum/hSynchronizationByLumi_->GetBinContent(0)/36.;
    float minVal = std::min(integrityQual,frontendQual);
    float totDQMVal = std::min(minVal,synchronizationQual);

    me = dqmStore_->get( prefixME_ + "/EventInfo/reportSummary" );
    if( me ) me->Fill(totDQMVal);

    for ( int i=0; i<18; i++) {
      me = dqmStore_->get( prefixME_ + "/EventInfo/reportSummaryContents/EcalEndcap_" + Numbers::sEE(i+1) );
      if( me ) me->Fill(DQMVal[i]);

      me = dqmStore_->get( prefixME_ + "/EventInfo/reportSummaryMap" );
      if( me ) {
	for(int t=1 ; t<=nTowerMax_ ; t++){
	  if(! map->dccTowerConstituents(DccId_[i], t).size() ) continue;
	  std::vector<EcalScDetId> scs = map->getEcalScDetId(DccId_[i], t, false);
	  for(unsigned u=0 ; u<scs.size() ; u++){
	    int jx = scs[u].ix() + (scs[u].zside()<0 ? 0 : 20);
	    int jy = scs[u].iy();
	    me->setBinContent(jx,jy, DQMVal[i]);
          }
        }
      }
    }

  }

  // now combine reduced DQM with DCS and DAQ
  me = dqmStore_->get( prefixME_ + "/EventInfo/DAQSummaryMap" );
  hDAQ_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDAQ_ );

  me = dqmStore_->get( prefixME_ + "/EventInfo/DCSSummaryMap" );
  hDCS_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDCS_ );

  float sumCert = 0.;
  float sumCertEE[18];
  int nValidChannels = 0;
  int nValidChannelsEE[18];

  if ( meEEDataCertificationSummaryMap_ ){
    for(int ix=1 ; ix<=meEEDataCertificationSummaryMap_->getNbinsX() ; ix++){
      for(int iy=1 ; iy<=meEEDataCertificationSummaryMap_->getNbinsY() ; iy++){
	meEEDataCertificationSummaryMap_->setBinContent( ix, iy, -1.0 );
      }
    }
  }

  for (int i = 0; i < 18; i++) {
    sumCertEE[i] = 0;
    nValidChannelsEE[i] = 0;

    for(int t=1 ; t<=nTowerMax_ ; t++){

      std::vector<DetId> crystals = map->dccTowerConstituents(DccId_[i], t);
      if(!crystals.size()) continue; // getEcalScDetId throws an exception when no crystal is found

      std::vector<EcalScDetId> scs = map->getEcalScDetId(DccId_[i], t, false);
      for(unsigned u=0 ; u<scs.size() ; u++){

	int jx = scs[u].ix() + (scs[u].zside()<0 ? 0 : 20);
	int jy = scs[u].iy();

	float xvalDQM = DQMVal[i];

	float xvalDAQ, xvalDCS;
	xvalDAQ = xvalDCS = -1.;
	float xcert = -1.;

	if ( hDAQ_ ) xvalDAQ = hDAQ_->GetBinContent( jx, jy );
	if ( hDCS_ ) xvalDCS = hDCS_->GetBinContent( jx, jy );

	if ( xvalDQM == -1 || ( xvalDAQ == -1 && xvalDCS == -1 ) ) {
	  // problems: DQM empty or DAQ and DCS not available
	  xcert = 0.0;
	} else {
	  // do not consider the white value of DAQ and DCS (problems with DB)
	  xcert = std::abs(xvalDQM) * std::abs(xvalDAQ) * std::abs(xvalDCS);
	}

	if ( meEEDataCertificationSummaryMap_ ) meEEDataCertificationSummaryMap_->setBinContent( jx, jy, xcert );

	int ncrystals = 0;

	for(std::vector<DetId>::const_iterator it=crystals.begin() ; it!=crystals.end() ; ++it){
	  EEDetId id(*it);
	  if( id.zside() == scs[u].zside() && (id.ix()-1)/5+1 == scs[u].ix() && (id.iy()-1)/5+1 == scs[u].iy() ) ncrystals++;
	}

	sumCertEE[i] += xcert * ncrystals;
	nValidChannelsEE[i] += ncrystals;

	sumCert += xcert * ncrystals;
	nValidChannels += ncrystals;

      }
    }

    if( meEEDataCertification_[i] ) {
      if( nValidChannelsEE[i]>0 ) meEEDataCertification_[i]->Fill( sumCertEE[i]/nValidChannelsEE[i] );
      else meEEDataCertification_[i]->Fill( 0.0 );
    }
  }

  if( meEEDataCertificationSummary_ ) {
    if( nValidChannels>0 ) meEEDataCertificationSummary_->Fill( sumCert/nValidChannels );
    else meEEDataCertificationSummary_->Fill( 0.0 );
  }

}

void EEDataCertificationTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EEDataCertificationTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  this->reset();

  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  const EcalElectronicsMapping *map = handle.product();
  if( ! map ){
    edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";
    return;
  }

  MonitorElement* me;

  me = dqmStore_->get( prefixME_ + "/EventInfo/reportSummaryMap" );
  hDQM_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDQM_ );

  me = dqmStore_->get( prefixME_ + "/EventInfo/DAQSummaryMap" );
  hDAQ_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDAQ_ );

  me = dqmStore_->get( prefixME_ + "/EventInfo/DCSSummaryMap" );
  hDCS_ = UtilsClient::getHisto<TH2F*>( me, cloneME_, hDCS_ );

  float sumCert = 0.;
  float sumCertEE[18];
  int nValidChannels = 0;
  int nValidChannelsEE[18];

  if ( meEEDataCertificationSummaryMap_ ){
    for(int ix=1 ; ix<=meEEDataCertificationSummaryMap_->getNbinsX() ; ix++){
      for(int iy=1 ; iy<=meEEDataCertificationSummaryMap_->getNbinsY() ; iy++){
	meEEDataCertificationSummaryMap_->setBinContent( ix, iy, -1.0 );
      }
    }
  }

  for (int i = 0; i < 18; i++) {
    sumCertEE[i] = 0;
    nValidChannelsEE[i] = 0;

    for(int t=1 ; t<=nTowerMax_ ; t++){

      std::vector<DetId> crystals = map->dccTowerConstituents(DccId_[i], t);
      if(!crystals.size()) continue;

      std::vector<EcalScDetId> scs = map->getEcalScDetId(DccId_[i], t, false);
      for(unsigned u=0 ; u<scs.size() ; u++){

	int jx = scs[u].ix() + (scs[u].zside()<0 ? 0 : 20);
	int jy = scs[u].iy();

	float xvalDQM, xvalDAQ, xvalDCS;
	xvalDQM = xvalDAQ = xvalDCS = -1.;
	float xcert = -1.;

	if ( hDQM_ ) xvalDQM = hDQM_->GetBinContent( jx, jy );
	if ( hDAQ_ ) xvalDAQ = hDAQ_->GetBinContent( jx, jy );
	if ( hDCS_ ) xvalDCS = hDCS_->GetBinContent( jx, jy );

	if ( xvalDQM == -1 || ( xvalDAQ == -1 && xvalDCS == -1 ) ) {
	  // problems: DQM empty or DAQ and DCS not available
	  xcert = 0.0;
	} else {
	  // do not consider the white value of DAQ and DCS (problems with DB)
	  xcert = std::abs(xvalDQM) * std::abs(xvalDAQ) * std::abs(xvalDCS);
	}

	if ( meEEDataCertificationSummaryMap_ ) meEEDataCertificationSummaryMap_->setBinContent( jx, jy, xcert );

	int ncrystals = 0;

	for(std::vector<DetId>::const_iterator it=crystals.begin() ; it!=crystals.end() ; ++it){
	  EEDetId id(*it);
	  if( id.zside() == scs[u].zside() && (id.ix()-1)/5+1 == scs[u].ix() && (id.iy()-1)/5+1 == scs[u].iy() ) ncrystals++;
	}

	sumCertEE[i] += xcert * ncrystals;
	nValidChannelsEE[i] += ncrystals;

	sumCert += xcert * ncrystals;
	nValidChannels += ncrystals;

      }
    }

    if( meEEDataCertification_[i] ) {
      if( nValidChannelsEE[i]>0 ) meEEDataCertification_[i]->Fill( sumCertEE[i]/nValidChannelsEE[i] );
      else meEEDataCertification_[i]->Fill( 0.0 );
    }
  }

  if( meEEDataCertificationSummary_ ) {
    if( nValidChannels>0 ) meEEDataCertificationSummary_->Fill( sumCert/nValidChannels );
    else meEEDataCertificationSummary_->Fill( 0.0 );
  }

}

void EEDataCertificationTask::reset(void) {

  if ( meEEDataCertificationSummary_ ) meEEDataCertificationSummary_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDataCertification_[i] ) meEEDataCertification_[i]->Reset();
  }

  if ( meEEDataCertificationSummaryMap_ ) meEEDataCertificationSummaryMap_->Reset();

  if ( meEEDataCertificationSummaryMap_ ) meEEDataCertificationSummaryMap_->Reset();

}


void EEDataCertificationTask::cleanup(void){

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
    if ( meEEDataCertificationSummary_ ) dqmStore_->removeElement( meEEDataCertificationSummary_->getName() );
    if ( meEEDataCertificationSummaryMap_ ) dqmStore_->removeElement( meEEDataCertificationSummaryMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/CertificationContents");
    for (int i = 0; i < 18; i++) {
      if ( meEEDataCertification_[i] ) dqmStore_->removeElement( meEEDataCertification_[i]->getName() );
    }
  }

}

void EEDataCertificationTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
