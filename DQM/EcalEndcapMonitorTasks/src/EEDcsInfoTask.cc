/*
 * \file EEDcsInfoTask.cc
 *
 * $Date: 2012/04/27 13:46:14 $
 * $Revision: 1.23 $
 * \author E. Di Marco
 *
*/

#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEDcsInfoTask.h"

EEDcsInfoTask::EEDcsInfoTask(const edm::ParameterSet& ps) {

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  meEEDcsFraction_ = 0;
  meEEDcsActiveMap_ = 0;
  for (int i = 0; i < 18; i++) {
    meEEDcsActive_[i] = 0;
  }

}

EEDcsInfoTask::~EEDcsInfoTask() {

}

void EEDcsInfoTask::beginJob(void){

  if ( dqmStore_ ) {

    std::string name;

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    meEEDcsFraction_ = dqmStore_->bookFloat( "DCSSummary" );
    meEEDcsFraction_->Fill(0.0);

    name = "DCSSummaryMap";
    meEEDcsActiveMap_ = dqmStore_->book2D(name, name, 40, 0., 200., 20, 0., 100.);
    meEEDcsActiveMap_->setAxisTitle("ix / ix+100", 1);
    meEEDcsActiveMap_->setAxisTitle("iy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 18; i++) {
      meEEDcsActive_[i] = dqmStore_->bookFloat( "EcalEndcap_" + Numbers::sEE(i+1) );
      meEEDcsActive_[i]->Fill(-1.0);
    }

  }

}

void EEDcsInfoTask::endJob(void) {

  if ( enableCleanup_ ) this->cleanup();

}

void EEDcsInfoTask::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){

  // information is by run, so fill the same for the run and for every lumi section
  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyLumi[itx][ity] = 1;
    }
  }

  if ( !iSetup.find( edm::eventsetup::EventSetupRecordKey::makeKey<EcalDCSTowerStatusRcd>() ) ) {
    edm::LogWarning("EEDcsInfoTask") << "EcalDCSTowerStatus record not found";
    return;
  }

  edm::ESHandle<EcalDCSTowerStatus> pDCSStatus;
  iSetup.get<EcalDCSTowerStatusRcd>().get(pDCSStatus);
  if ( !pDCSStatus.isValid() ) {
    edm::LogWarning("EEDcsInfoTask") << "EcalDCSTowerStatus record not valid";
    return;
  }
  const EcalDCSTowerStatus* dcsStatus = pDCSStatus.product();

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
	EcalDCSTowerStatus::const_iterator dcsStatusIt = dcsStatus->find( scs[u].rawId() );
	if ( dcsStatusIt != dcsStatus->end() ) dbStatus = dcsStatusIt->getStatusCode();
	
	if ( dbStatus > 0 ) {
	  int jx = scs[u].ix() - 1 + (scs[u].zside()<0 ? 0 : 20);
	  int jy = scs[u].iy() - 1;
	  readyRun[jx][jy] = 0;
	  readyLumi[jx][jy] = 0;
	}
      }
    }
  }

//   for(int iz=-1; iz<=1; iz+=2) {
//     for(int itx=0 ; itx<20; itx++) {
//       for(int ity=0 ; ity<20; ity++) {
//         if (EcalScDetId::validDetId(itx+1,ity+1,iz )){

//           EcalScDetId eeid(itx+1,ity+1,iz);

//           uint16_t dbStatus = 0; // 0 = good
//           EcalDCSTowerStatus::const_iterator dcsStatusIt = dcsStatus->find( eeid.rawId() );
//           if ( dcsStatusIt != dcsStatus->end() ) dbStatus = dcsStatusIt->getStatusCode();

//           if ( dbStatus > 0 ) {
//             int offsetSC = (iz > 0) ? 0 : 20;
//             readyRun[offsetSC+itx][ity] = 0;
//             readyLumi[offsetSC+itx][ity] = 0;
//           }

//         }
//       }
//     }
//   }

}

void EEDcsInfoTask::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup) {

  edm::ESHandle< EcalElectronicsMapping > handle;
  iSetup.get< EcalMappingRcd >().get(handle);
  const EcalElectronicsMapping *map = handle.product();
  if( ! map ) edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";
  else this->fillMonitorElements(readyLumi, map);

}

void EEDcsInfoTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

  for ( int itx = 0; itx < 40; itx++ ) {
    for ( int ity = 0; ity < 20; ity++ ) {
      readyRun[itx][ity] = 1;
    }
  }

}

void EEDcsInfoTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  const EcalElectronicsMapping *map = handle.product();
  if( ! map ) edm::LogWarning("EEDaqInfoTask") << "EcalElectronicsMapping not available";
  else this->fillMonitorElements(readyRun, map);

}

void EEDcsInfoTask::reset(void) {

  if ( meEEDcsFraction_ ) meEEDcsFraction_->Reset();

  for (int i = 0; i < 18; i++) {
    if ( meEEDcsActive_[i] ) meEEDcsActive_[i]->Reset();
  }

  if ( meEEDcsActiveMap_ ) meEEDcsActiveMap_->Reset();

}


void EEDcsInfoTask::cleanup(void){

  if ( dqmStore_ ) {

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo");

    if ( meEEDcsFraction_ ) dqmStore_->removeElement( meEEDcsFraction_->getName() );

    if ( meEEDcsActiveMap_ ) dqmStore_->removeElement( meEEDcsActiveMap_->getName() );

    dqmStore_->setCurrentFolder(prefixME_ + "/EventInfo/DCSContents");

    for (int i = 0; i < 18; i++) {
      if ( meEEDcsActive_[i] ) dqmStore_->removeElement( meEEDcsActive_[i]->getName() );
    }

  }

}

void EEDcsInfoTask::fillMonitorElements(int ready[40][20], const EcalElectronicsMapping *map) {

  float readySum[18];
  int nValidChannels[18];
  for ( int ism = 0; ism < 18; ism++ ) {
    readySum[ism] = 0;
    nValidChannels[ism] = 0;
  }
  float readySumTot = 0.;
  int nValidChannelsTot = 0;

  if(meEEDcsActiveMap_){
    for(int ix=1 ; ix<=meEEDcsActiveMap_->getNbinsX() ; ix++){
      for(int iy=1 ; iy<=meEEDcsActiveMap_->getNbinsY() ; iy++){
	meEEDcsActiveMap_->setBinContent( ix, iy, -1.0 );
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

	if(meEEDcsActiveMap_) meEEDcsActiveMap_->setBinContent( jx+1, jy+1, ready[jx][jy] );

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

    if( meEEDcsActive_[iDcc] ) meEEDcsActive_[iDcc]->Fill( readySum[iDcc]/float(nValidChannels[iDcc]) );
  }

  if( meEEDcsFraction_ ) meEEDcsFraction_->Fill( readySumTot/float(nValidChannelsTot) );

}

void EEDcsInfoTask::analyze(const edm::Event& e, const edm::EventSetup& c){

}
