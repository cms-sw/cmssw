#ifndef SiPixelMonitorCluster_SiPixelClusterSource_h
#define SiPixelMonitorCluster_SiPixelClusterSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorCluster
// Class  :     SiPixelClusterSource
// 
/*

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:  
// $Id: SiPixelClusterSource.h,v 1.6 2008/04/24 07:21:33 andrewdc Exp $
//
// Updated by: Lukas Wehrli
// for pixel offline DQM 

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterModule.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/EDProduct.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/cstdint.hpp>


 class SiPixelClusterSource : public edm::EDAnalyzer {
    public:
       explicit SiPixelClusterSource(const edm::ParameterSet& conf);
       ~SiPixelClusterSource();

       typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;
       
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob(edm::EventSetup const&) ;
       virtual void endJob() ;

       virtual void buildStructure(edm::EventSetup const&);
       virtual void bookMEs();

    private:
       edm::ParameterSet conf_;
       edm::InputTag src_;
       int eventNo;
       DQMStore* theDMBE;
       std::map<uint32_t,SiPixelClusterModule*> thePixelStructure;
       bool modOn; 
       //barrel:
       bool ladOn, layOn, phiOn;
       //forward:
       bool ringOn, bladeOn, diskOn; 
 };

#endif
