// -*- C++ -*-
//
// Package:    SiStripMonitorRawData
// Class:      SiStripMonitorRawData
// 
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Suchandra Dutta
//         Created:  Fri June  1 17:00:00 CET 2007
// $Id: SiStripMonitorRawData.cc,v 1.0 2007/06/01 17:00:00 CET dutta Exp $
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorRawData.h"


// std
#include <cstdlib>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>

SiStripMonitorRawData::SiStripMonitorRawData(const edm::ParameterSet& iConfig):
  dbe_(edm::Service<DaqMonitorBEInterface>().operator->()),
  conf_(iConfig)
{

}


SiStripMonitorRawData::~SiStripMonitorRawData()
{
}


void SiStripMonitorRawData::beginJob(const edm::EventSetup& es){
   // retrieve parameters from configuration file

  dbe_->setCurrentFolder("Track/GlobalParameters");
  
  BadFedNumber = dbe_->book1D("FaultyFedNumberAndChannel","Faulty Fed Id and Channel and Numbers", 60000, 0.5, 600.5);
  BadFedNumber->setAxisTitle("Fed Id and Channel numbers",1);

  
 //getting det id from the det cabling    
  es.get<SiStripDetCablingRcd>().get( detcabling );

  detcabling->addActiveDetectorsRawIds(SelectedDetIds);
}


// ------------ method called to produce the data  ------------
void SiStripMonitorRawData::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::cout << "Run " << iEvent.id().run() << " Event " << iEvent.id().event() << std::endl;

  
  iSetup.get<SiStripDetCablingRcd>().get( detcabling );

  // retrieve producer name of input StripDigiCollection
  std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
  // get DigiCollection object from Event
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > digi_collection;
  std::string digiType = "VirginRaw";
  //you have a collection as there are all the digis for the event for every detector
  iEvent.getByLabel(digiProducer, digiType, digi_collection);

  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), 
                               iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){
     std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = digi_collection->find( (*idetid) );
    if (digis->data.size() == 0 || 
        digis->data.size() > 768 || 
        digis == digi_collection->end() ) {
      std::vector<FedChannelConnection> fed_conns = detcabling->getConnections((*idetid));
      for (unsigned int  k = 0; k < fed_conns.size() ; k++) {
        float fed_id = fed_conns[k].fedId() + 0.01*fed_conns[k].fedCh();
        BadFedNumber->Fill(fed_id);
      }
      continue;
    }
  }
}


void SiStripMonitorRawData::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if (outputMEsInRootFile) {    
    dbe_->showDirStructure();
    dbe_->save(outputFileName);
  }

}

