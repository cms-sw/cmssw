// -*- C++ -*-
//
// Package:    ElectronPixelSeed
// Class:      ElectronPixelSeedProducer
// 
/**\class ElectronPixelSeedAnalyzer RecoEgamma/ElectronTrackSeedProducers/src/ElectronPixelSeedAnalyzer.cc

 Description: rereading of electron seeds for verification

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedAnalyzer.cc,v 1.1 2006/06/02 15:32:45 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedAnalyzer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <iostream>
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"

using namespace reco;
 
ElectronPixelSeedAnalyzer::ElectronPixelSeedAnalyzer(const edm::ParameterSet& iConfig)
{
  histfile_ = new TFile("histos.root","UPDATE");
}

ElectronPixelSeedAnalyzer::~ElectronPixelSeedAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void ElectronPixelSeedAnalyzer::beginJob(edm::EventSetup const&iSetup){
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD); 
  iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
  histpt_ = new TH1F("pt","pt of seed ",100,0.,100.);
  histnrseeds_ = new TH1I("nrs","Nr seeds/evt",50,0.,50.);
}     

void
ElectronPixelSeedAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
  // rereads the seeds for test purposes
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef recHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;

  edm::Handle<ElectronPixelSeedCollection> elSeeds;
  e.getByType(elSeeds); 
  std::cout <<"\n\n =================> Treating event "<<e.id()<<" Number of seeds "<<elSeeds.product()->size()<<std::endl;
  int is=0;
  for( ElectronPixelSeedCollection::const_iterator MyS= (*elSeeds).begin(); MyS != (*elSeeds).end(); ++MyS) {
    std::cout <<"\nSeed nr "<<is++<<": "<<std::endl;
    range r=(*MyS).recHits();
    const GeomDet *det = pDD->idToDet(((*r.first)).geographicalId());
    std::cout<<" Number of RecHits= "<<(*MyS).nHits()<<std::endl;
    for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) {
      std::cout <<" SiPixelHit   local x,y,z "<<(*rhits).localPosition()<<" det "<<(*rhits).geographicalId().det()<<" subdet "<<(*rhits).geographicalId().subdetId()<<std::endl;
      std::cout <<" SiPixelHit   global  "<<det->toGlobal((*rhits).localPosition())<<std::endl;
      //       const DetLayer *ilayer=theGeometricSearchTracker->detLayer((*rhits).geographicalId());

      //       if ((*rhits).geographicalId().subdetId()==1) {
      // 	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(ilayer);
      // 	if (bdetl) {
      // 	  std::cout <<" PixelHit layer radius "<<bdetl->specificSurface().radius()<<std::endl;
      // 	}
      // 	else printf("Could not downcast to barrel!!\n");
      //       }else {
      // 	const ForwardDetLayer *fdetl = dynamic_cast<const ForwardDetLayer *>(ilayer);
      // 	if (!fdetl)
      // 	  printf("Could not downcast to forward!!\n");
      //       }
    }

    std::cout<<" ElectronPixelSeed charge: "<<(*MyS).getCharge()<<std::endl;

    TrajectoryStateOnSurface t= transformer_.transientState((*MyS).startingState(), &(det->surface()), &(*theMagField));

    std::cout<<" TSOS, position: "<<t.globalPosition()<<std::endl;
    std::cout<<" momentum: "<<t.globalMomentum()<<std::endl;
    const SuperCluster *theClus=(*MyS).superCluster();
    std::cout<<" SuperCluster energy: "<<theClus->energy()<<", position: "<<theClus->position()<<std::endl;
  histpt_->Fill(TMath::Sqrt(t.globalMomentum().x()*t.globalMomentum().x() + t.globalMomentum().y()*t.globalMomentum().y()));
  }
  // get input clusters 
  edm::Handle<SuperClusterCollection> clusters;
  e.getByType(clusters);
  if (clusters.product()->size()>0)      histnrseeds_->Fill(elSeeds.product()->size());
}


