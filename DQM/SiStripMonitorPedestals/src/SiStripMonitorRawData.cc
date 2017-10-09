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
//
//

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorRawData.h"

// std
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <algorithm>

SiStripMonitorRawData::SiStripMonitorRawData(edm::ParameterSet const& iConfig):
  BadFedNumber(0),
  dqmStore_(edm::Service<DQMStore>().operator->()),
  conf_(iConfig),
  m_cacheID_(0)


{
  // retrieve producer name of input StripDigiCollection
  std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
  std::string digiType = "VirginRaw";
  digiToken_ = consumes<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(digiProducer,digiType) );

  edm::LogInfo("SiStripMonitorRawData") <<"SiStripMonitorRawData  " 
					  << " Constructing....... ";     
}


SiStripMonitorRawData::~SiStripMonitorRawData()
{
  edm::LogInfo("SiStripMonitorRawData") <<"SiStripMonitorRawData  " 
					  << " Destructing....... ";     
}
//
// -- Begin Job
//
void SiStripMonitorRawData::beginJob() {
}
//
// -- BeginRun
//

void SiStripMonitorRawData::bookHistograms(DQMStore::IBooker & ibooker , const edm::Run & run, const edm::EventSetup & eSetup)
{
  unsigned long long cacheID = eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();

  if (BadFedNumber) BadFedNumber->Reset();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;
    eSetup.get<SiStripDetCablingRcd>().get( detcabling );
    SelectedDetIds.clear();
    detcabling->addActiveDetectorsRawIds(SelectedDetIds);

    edm::LogInfo("SiStripMonitorRawData") <<"SiStripMonitorRawData::bookHistograms: "
                                          << " Creating MEs for new Cabling ";
    ibooker.setCurrentFolder("Track/GlobalParameter");
    if (!BadFedNumber) {
      BadFedNumber = ibooker.book1D("FaultyFedNumberAndChannel","Faulty Fed Id and Channel and Numbers", 60000, 0.5, 600.5);
      BadFedNumber->setAxisTitle("Fed Id and Channel numbers",1);
    }
  }
}

// ------------ method called to produce the data  ------------
void SiStripMonitorRawData::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
{

   edm::LogInfo("SiStripMonitorRawData") <<"SiStripMonitorRawData::analyze: Run "<< 
                              iEvent.id().run()  << " Event " << iEvent.id().event();

  
  iSetup.get<SiStripDetCablingRcd>().get( detcabling );

  // get DigiCollection object from Event
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > digi_collection;
  iEvent.getByToken(digiToken_, digi_collection);

  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), 
                               iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){
    std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = digi_collection->find( (*idetid) );
    if (digis == digi_collection->end() ||
        digis->data.size() == 0 || 
        digis->data.size() > 768 )  {
      std::vector<const FedChannelConnection *> fed_conns = detcabling->getConnections((*idetid));
      for (unsigned int  k = 0; k < fed_conns.size() ; k++) {
	if(fed_conns[k] && fed_conns[k]->isConnected()) {
	  float fed_id = fed_conns[k]->fedId() + 0.01*fed_conns[k]->fedCh();
	  BadFedNumber->Fill(fed_id);
	}
      }
      continue;
    }
  }
}
//
// -- End Run
//    
void SiStripMonitorRawData::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if (outputMEsInRootFile) {    
    //dqmStore_->showDirStructure();
    dqmStore_->save(outputFileName);
  }
}
//
// -- End Job
//
void SiStripMonitorRawData::endJob(void){
  edm::LogInfo("SiStripMonitorRawData") <<"SiStripMonitorRawData::EndJob: " 
					  << " Finishing!! ";        
}
DEFINE_FWK_MODULE(SiStripMonitorRawData);

