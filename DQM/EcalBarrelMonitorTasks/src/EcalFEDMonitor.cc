#include <iostream>
#include <fstream>

#include "../interface/EcalFEDMonitor.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

EcalFEDMonitor::EcalFEDMonitor(const edm::ParameterSet& _ps) :
  initialized_(false),
  folderName_(_ps.getUntrackedParameter<std::string>("folderName")),
  FEDRawDataTag_(_ps.getUntrackedParameter<edm::InputTag>("FEDRawDataTag")),
  gainErrorsTag_(_ps.getUntrackedParameter<edm::InputTag>("gainErrorsTag")),
  chIdErrorsTag_(_ps.getUntrackedParameter<edm::InputTag>("chIdErrorsTag")),
  gainSwitchErrorsTag_(_ps.getUntrackedParameter<edm::InputTag>("gainSwitchErrorsTag")),
  towerIdErrorsTag_(_ps.getUntrackedParameter<edm::InputTag>("towerIdErrorsTag")),
  blockSizeErrorsTag_(_ps.getUntrackedParameter<edm::InputTag>("blockSizeErrorsTag")),
  MEs_(nMEs, 0)
{
}

void
EcalFEDMonitor::beginRun(const edm::Run& _run, const edm::EventSetup& _es)
{
  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  ecaldqm::setElectronicsMap(elecMapHandle.product());
}

void
EcalFEDMonitor::initialize()
{
  std::string name;

  DQMStore& dqmStore(*edm::Service<DQMStore>());

  dqmStore.setCurrentFolder("EcalBarrel/" + folderName_);

  name = "FEDEntries";
  MEs_[kEBOccupancy] = dqmStore.book1D(name, name, 36, 610, 646);

  name = "FEDFatal";
  MEs_[kEBFatal] = dqmStore.book1D(name, name, 36, 610, 646);

  name = "FEDNonFatal";
  MEs_[kEBNonFatal] = dqmStore.book1D(name, name, 36, 610, 646);

  dqmStore.setCurrentFolder("EcalEndcap/" + folderName_);

  name = "FEDEntries";
  MEs_[kEEOccupancy] = dqmStore.book1D(name, name, 54, 601, 655);

  name = "FEDFatal";
  MEs_[kEEFatal] = dqmStore.book1D(name, name, 54, 601, 655);

  name = "FEDNonFatal";
  MEs_[kEENonFatal] = dqmStore.book1D(name, name, 54, 601, 655);

  initialized_ = true;
}

void
EcalFEDMonitor::analyze(const edm::Event& _evt, const edm::EventSetup&)
{
  if(!initialized_) initialize();
  if(!initialized_) return;

  edm::Handle<FEDRawDataCollection> fedHndl;
  if(_evt.getByLabel(FEDRawDataTag_, fedHndl)){
    for(unsigned fedId(601); fedId <= 654; fedId++){
      unsigned occupancy(-1);
      //      unsigned fatal(-1);
      if(fedId < 610 || fedId > 645){
        occupancy = kEEOccupancy;
	//        fatal = kEEFatal;
      }
      else{
        occupancy = kEBOccupancy;
	//        fatal = kEBFatal;
      }

      const FEDRawData& fedData(fedHndl->FEDData(fedId));
      unsigned length(fedData.size() / sizeof(uint64_t));

      if(length > 1){ // FED header is one 64 bit word
        MEs_[occupancy]->Fill(fedId + 0.5);

//  	const uint64_t* pData(reinterpret_cast<const uint64_t*>(fedData.data()));
//  	bool crcError(((pData[length - 1] >> 2) & 0x1) == 0x1);

// 	if(crcError) MEs_[fatal]->Fill(fedId + 0.5);
      }
    }
  }

  edm::Handle<EBDetIdCollection> ebHndl;
  edm::Handle<EEDetIdCollection> eeHndl;
  edm::Handle<EcalElectronicsIdCollection> eleHndl;

  if(_evt.getByLabel(gainErrorsTag_, ebHndl) && _evt.getByLabel(gainErrorsTag_, eeHndl)){
    EBDetIdCollection::const_iterator ebEnd(ebHndl->end());
    for(EBDetIdCollection::const_iterator ebItr(ebHndl->begin()); ebItr != ebEnd; ++ebItr){
      unsigned iDCC(ecaldqm::dccId(*ebItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEBNonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }

    EEDetIdCollection::const_iterator eeEnd(eeHndl->end());
    for(EEDetIdCollection::const_iterator eeItr(eeHndl->begin()); eeItr != eeEnd; ++eeItr){
      unsigned iDCC(ecaldqm::dccId(*eeItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEENonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }

  if(_evt.getByLabel(chIdErrorsTag_, ebHndl) && _evt.getByLabel(chIdErrorsTag_, eeHndl)){
    EBDetIdCollection::const_iterator ebEnd(ebHndl->end());
    for(EBDetIdCollection::const_iterator ebItr(ebHndl->begin()); ebItr != ebEnd; ++ebItr){
      unsigned iDCC(ecaldqm::dccId(*ebItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEBNonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }

    EEDetIdCollection::const_iterator eeEnd(eeHndl->end());
    for(EEDetIdCollection::const_iterator eeItr(eeHndl->begin()); eeItr != eeEnd; ++eeItr){
      unsigned iDCC(ecaldqm::dccId(*eeItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEENonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }

  if(_evt.getByLabel(gainSwitchErrorsTag_, ebHndl) && _evt.getByLabel(gainSwitchErrorsTag_, eeHndl)){
    EBDetIdCollection::const_iterator ebEnd(ebHndl->end());
    for(EBDetIdCollection::const_iterator ebItr(ebHndl->begin()); ebItr != ebEnd; ++ebItr){
      unsigned iDCC(ecaldqm::dccId(*ebItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEBNonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }

    EEDetIdCollection::const_iterator eeEnd(eeHndl->end());
    for(EEDetIdCollection::const_iterator eeItr(eeHndl->begin()); eeItr != eeEnd; ++eeItr){
      unsigned iDCC(ecaldqm::dccId(*eeItr) - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      MEs_[kEENonFatal]->Fill(iDCC + 601.5, 1. / normalization);
    }
  }

  if(_evt.getByLabel(towerIdErrorsTag_, eleHndl)){
    EcalElectronicsIdCollection::const_iterator eleEnd(eleHndl->end());
    for(EcalElectronicsIdCollection::const_iterator eleItr(eleHndl->begin()); eleItr != eleEnd; ++eleItr){
      unsigned iDCC(eleItr->dccId() - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      unsigned nonfatal(-1);
      if(iDCC <= ecaldqm::kEEmHigh || iDCC >= ecaldqm::kEEpLow)
        nonfatal = kEENonFatal;
      else
        nonfatal = kEBNonFatal;

      MEs_[nonfatal]->Fill(iDCC + 601.5, 25. / normalization);
    }
  }

  if(_evt.getByLabel(blockSizeErrorsTag_, eleHndl)){
    EcalElectronicsIdCollection::const_iterator eleEnd(eleHndl->end());
    for(EcalElectronicsIdCollection::const_iterator eleItr(eleHndl->begin()); eleItr != eleEnd; ++eleItr){
      unsigned iDCC(eleItr->dccId() - 1);

      double normalization(ecaldqm::nCrystals(iDCC + 1));
      if(normalization < 1.) continue;

      unsigned nonfatal(-1);
      if(iDCC <= ecaldqm::kEEmHigh || iDCC >= ecaldqm::kEEpLow)
        nonfatal = kEENonFatal;
      else
        nonfatal = kEBNonFatal;

      MEs_[nonfatal]->Fill(iDCC + 601.5, 25. / normalization);
    }
  }
}

DEFINE_FWK_MODULE(EcalFEDMonitor);
