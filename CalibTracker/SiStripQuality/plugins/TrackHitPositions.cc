// -*- C++ -*-
//
// Package:    TrackHitPositions
// Class:      TrackHitPositions
//
/**\class TrackHitPositions TrackHitPositions.h CalibTracker/SiStripQuality/plugins/TrackHitPositions.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 12:11:10 CEST 2007
// $Id: TrackHitPositions.cc,v 1.15 2009/11/30 11:23:27 giordano Exp $
//
//
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripQuality/plugins/TrackHitPositions.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <sys/time.h>


TrackHitPositions::TrackHitPositions( const edm::ParameterSet& iConfig ):
  m_cacheID_(0),
  dataLabel_(iConfig.getUntrackedParameter<std::string>("dataLabel","")),
  TkMapFileName_(iConfig.getUntrackedParameter<std::string>("TkMapFileName","")),
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  saveTkHistoMap_(iConfig.getUntrackedParameter<bool>("SaveTkHistoMap",true)),
  tkMap(0),
  ptCut_(iConfig.getParameter<double>("PtCut"))
  //, tkMapFullIOVs(0)
{
  reader = new SiStripDetInfoFileReader(fp_.fullPath());

  // tkMapFullIOVs = new TrackerMap( "BadComponents" );
  tkhisto=0;
  if (TkMapFileName_!=""){
    tkhisto = new TkHistoMap("BadComp","BadComp",-1.); //here the baseline (the value of the empty,not assigned bins) is put to -1 (default is zero)
  }
}

void TrackHitPositions::endJob(){
}

void TrackHitPositions::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();

  std::stringstream ss;

  if (m_cacheID_ == cacheID)
    return;

  m_cacheID_ = cacheID;

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(dataLabel_,SiStripQuality_);


  // Loop on tracks and produce tracker maps with the position of the hits
  edm::Handle<reco::TrackCollection> tracks;
  e.getByLabel("generalTracks", tracks);
  unsigned int trackNumber = 0;
  std::vector<reco::Track>::const_iterator ittrk = tracks->begin();
  for( ; ittrk != tracks->end(); ++ittrk, ++trackNumber ) {

    // std::cout << "Track Pt = " << ittrk->pt() << std::endl;

    if( ittrk->pt() < ptCut_ ) continue;
    std::cout << "Selected track with charge = " << ittrk->charge() << " and Pt = " << ittrk->pt() << std::endl;

    // Create a new trackerMap for this track
    std::stringstream trackNumStr;
    trackNumStr << trackNumber;
    if( tkMap ) delete tkMap;
    std::string mapName("TrackHitsMap_" + trackNumStr.str());
    tkMap = new TrackerMap( mapName );
    initializeMap(SiStripQuality_, tkMap);

    // Loop on hits
    for (trackingRecHit_iterator ith = ittrk->recHitsBegin(); ith != ittrk->recHitsEnd(); ++ith) {
      const TrackingRecHit * hit = ith->get(); // ith is an iterator on edm::Ref to rechit

      DetId detid = hit->geographicalId();

      //check that the hit is a real hit and not a constraint
      if(hit->isValid() && hit==0 && detid.rawId()==0) continue;

      // fill the tracker map with the hit position
      std::cout << "Track number " << trackNumber << " hit on DetId " << detid.rawId() << std::endl;
      tkMap->fillc(detid,0x0);
    }
    std::string fileName = mapName + ".pdf";
    tkMap->save(true,0,0,fileName.c_str());
    fileName.erase(fileName.begin()+fileName.find("."),fileName.end());
    tkMap->print(true,0,0,fileName.c_str());
  }
}

void TrackHitPositions::initializeMap(const edm::ESHandle<SiStripQuality> & SiStripQuality_, TrackerMap * tkMap)
{
  std::stringstream ss;
  std::vector<uint32_t> detids=reader->getAllDetIds();
  std::vector<uint32_t>::const_iterator idet=detids.begin();
  for(;idet!=detids.end();++idet){
    ss << "detid " << (*idet) << " IsModuleUsable " << SiStripQuality_->IsModuleUsable((*idet)) << "\n";
    // if (SiStripQuality_->IsModuleUsable((*idet))) {
    //  tkMap->fillc(*idet,0x00ff00);
    // }
  }
  LogDebug("TrackHitPositions") << ss.str() << std::endl;
}
