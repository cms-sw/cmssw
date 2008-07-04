#ifndef SiPixelMonitorRecHits_SiPixelRecHitSource_h
#define SiPixelMonitorRecHits_SiPixelRecHitSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorRecHits
// Class  :     SiPixelRecHitSource
// 
/**

 Description: header file for Pixel Monitor Rec Hits

 Usage:
    see description

*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelRecHitSource.h,v 1.4 2008/06/23 15:52:05 merkelp Exp $
//
// Updated by: Keith Rose
// for use in SiPixelMonitorRecHits
// Updated by: Lukas Wehrli
// for pixel offline DQM 


#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorRecHit/interface/SiPixelRecHitModule.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/EDProduct.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/cstdint.hpp>

 class SiPixelRecHitSource : public edm::EDAnalyzer {
    public:
       explicit SiPixelRecHitSource(const edm::ParameterSet& conf);
       ~SiPixelRecHitSource();

//       typedef edm::DetSet<PixelRecHit>::const_iterator    RecHitIterator;
       
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob(edm::EventSetup const&) ;
       virtual void endJob() ;

       virtual void buildStructure(edm::EventSetup const&);
       virtual void bookMEs();

    private:
       edm::ParameterSet conf_;
       edm::InputTag src_;
       bool saveFile;
       bool isPIB;
       bool slowDown;
       int eventNo;
       DQMStore* theDMBE;
       std::map<uint32_t,SiPixelRecHitModule*> thePixelStructure;
       std::map<uint32_t,int> rechit_count;
       bool modOn; 
       //barrel:
       bool ladOn, layOn, phiOn;
       //forward:
       bool ringOn, bladeOn, diskOn; 

 };

#endif
