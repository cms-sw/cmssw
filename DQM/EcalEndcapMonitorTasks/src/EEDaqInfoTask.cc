/*
 * \file EEDaqInfoTask.cc
 *
 * $Date: 2012/04/27 13:46:14 $
 * $Revision: 1.18 $
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEDaqInfoTask.h"

EEDaqInfoTask::EEDaqInfoTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEEDaqFraction_ = 0;
  meEEDaqActiveMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDaqActive_[i] = 0;
  }

}

EEDaqInfoTask::~EEDaqInfoTask() {

}

void EEDaqInfoTask::beginJob(void){

  if ( dqmStore_ ) {

    std::string name;

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    meEEDaqFraction_ = dqmStore_->bookFloat( "DAQSummary" );
    meEEDaqFraction_->Fill(0.0);

    name = "DAQSummaryMap";
    meEEDaqActiveMap_ = dqmStore_->book2D(name, name, 40, 0., 200., 20, 0., 100.);
    meEEDaqActiveMap_->setAxisTitle("ix / ix+100", 1);
    meEEDaqActiveMap_->setAxisTitle("iy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQContents");

    for (int i = 0; i < 18; i++) {
      meEEDaqActive_[i] = dqmStore_->bookFloat( "EcalEndcap_" + Numbers::sEE(i+1) );
      meEEDaqActive_[i]->Fill(0.0);
    }

  }

}

void EEDaqInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDaqInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  // information is by run, so fill the same for the run and for every lumi section
  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyLumi[itx][ity] = 1;
    }
  }

  if ( !iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDAQTowerStatusRcd>() ) ) {
    edm::LogWarning("EEDaqInfoTask") << "EcalDAQTowerStatus record not found";
    return;
  }

  edm::ESHandle<EcalDAQTowerStatus> pDAQStatus;
  iSetup.get<EcalDAQTowerStatusRcd>().get(pDAQStatus);
  if ( !pDAQStatus.isValid() ) {
    edm::LogWarning("EEDaqInfoTask") << "EcalDAQTowerStatus record not valid";
    return;
  }
  const EcalDAQTowerStatus* daqStatus = pDAQStatus.product();

  edm::ESHandle< EcalElectronicsMapping > pElecMapping;
  iSetup.get< EcalMappingRcd >().get(pElecMapping);
  if( !pElecMapping.isValid() ) {
    edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";
    return;
  }
  const EcalElectronicsMapping *map = pElecMapping.product();

  std::vector<DetId> crystals;
  std::vector<EcalScDetId> scs;

  for(unsigned i=0 ; i<sizeof(DccId_)/sizeof(int) ; i++){
    for(int t=1 ; t<=nTowerMax_ ; t++){

      crystals = map->dccTowerConstituents(DccId_[i], t);
      if(!crystals.size()) continue;

      scs = map->getEcalScDetId(DccId_[i], t, false);

      for(unsigned u=0 ; u<scs.size() ; u++){

	uint16_t dbStatus = 0; // 0 = good
	EcalDAQTowerStatus::const_iterator daqStatusIt = daqStatus->find( scs[u].rawId() );
	if ( daqStatusIt != daqStatus->end() ) dbStatus = daqStatusIt->getStatusCode();
	
	if ( dbStatus > 0 ) {
	  int jx = scs[u].ix() - 1 + (scs[u].zside()<0 ? 0 : 20);
	  int jy = scs[u].iy() - 1;
	  readyRun[jx][jy] = 0;
	  readyLumi[jx][jy] = 0;
	}
      }
    }
  }

}

void EEDaqInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  edm::ESHandle< EcalElectronicsMapping > handle;
  iSetup.get< EcalMappingRcd >().get(handle);
  const EcalElectronicsMapping *map = handle.product();
  if( ! map ) edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";
  else this->fillMonitorElements(readyLumi, map);

}

void EEDaqInfoTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyRun[itx][ity] = 1;
    }
  }

}

void EEDaqInfoTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  const EcalElectronicsMapping *map = handle.product();
  if( ! map ) edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";
  else this->fillMonitorElements(readyRun, map);

}

void EEDaqInfoTask::reset(void) {

  if ( meEEDaqFraction_ ) meEEDaqFraction_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDaqActive_[i] ) meEEDaqActive_[i]->Reset();
  }

  if ( meEEDaqActiveMap_ ) meEEDaqActiveMap_->Reset();

}


void EEDaqInfoTask::cleanup(void){

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    if ( meEEDaqFraction_ ) dqmStore_->removeElement( meEEDaqFraction_->getName() );

    if ( meEEDaqActiveMap_ ) dqmStore_->removeElement( meEEDaqActiveMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DAQContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDaqActive_[i] ) dqmStore_->removeElement( meEEDaqActive_[i]->getName() );
    }

  }

}

void EEDaqInfoTask::fillMonitorElements(int ready[40][20], const EcalElectronicsMapping *map) {

  float readySum[18];
  int nValidChannels[18];
  for ( int ism = 0; ism < 18; ism++ ) {
    readySum[ism] = 0;
    nValidChannels[ism] = 0;
  }
  float readySumTot = 0.;
  int nValidChannelsTot = 0;

  if(meEEDaqActiveMap_){
    for(int ix=1 ; ix<=meEEDaqActiveMap_->getNbinsX() ; ix++){
      for(int iy=1 ; iy<=meEEDaqActiveMap_->getNbinsY() ; iy++){
	meEEDaqActiveMap_->setBinContent( ix, iy, -1.0 );
      }
    }
  }

  std::vector<DetId> crystals;
  std::vector<EcalScDetId> scs;

  for ( unsigned iDcc = 0; iDcc < sizeof(DccId_)/sizeof(int); iDcc++) {
    for ( int t = 1; t<=nTowerMax_; t++ ) {

      crystals = map->dccTowerConstituents(DccId_[iDcc], t);
      if(!crystals.size()) continue;

      scs = map->getEcalScDetId(DccId_[iDcc], t, false);

      for(unsigned u=0 ; u<scs.size() ; u++){ // most of the time one DCC tower = one SC

	int jx = scs[u].ix() - 1 + (scs[u].zside()<0 ? 0 : 20);
	int jy = scs[u].iy() - 1;

	if(meEEDaqActiveMap_) meEEDaqActiveMap_->setBinContent( jx+1, jy+1, ready[jx][jy] );

	int ncrystals = 0;

	for(std::vector<DetId>::const_iterator it=crystals.begin() ; it!=crystals.end() ; ++it){
	  EEDetId id(*it);
	  if( id.zside() == scs[u].zside() && (id.ix()-1)/5+1 == scs[u].ix() && (id.iy()-1)/5+1 == scs[u].iy() ) ncrystals++;
	}

	if(ready[jx][jy]) {
	  readySum[iDcc] += ncrystals;
	  readySumTot += ncrystals;
	}

	nValidChannels[iDcc] += ncrystals;
	nValidChannelsTot += ncrystals;

      }
    }
    if( meEEDaqActive_[iDcc] ) meEEDaqActive_[iDcc]->Fill( readySum[iDcc]/float(nValidChannels[iDcc]) );
  }

  if( meEEDaqFraction_ ) meEEDaqFraction_->Fill( readySumTot/float(nValidChannelsTot) );

}

void EEDaqInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
