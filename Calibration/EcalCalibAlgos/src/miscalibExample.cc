// -*- C++ -*-
//
// Package:    miscalibExample
// Class:      miscalibExample
// 
/**\class miscalibExample miscalibExample.cc Calibration/EcalCalibAlgos/src/miscalibExample.cc

 Description: Perform single electron calibration (tested on TB data only).

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lorenzo AGOSTINO
//         Created:  Tue Jul 18 12:17:01 CEST 2006
// $Id: miscalibExample.cc,v 1.2 2006/09/11 12:44:58 malgeri Exp $
//
//


// system include files
#include <memory>

// user include files
#include "Calibration/EcalCalibAlgos/interface/miscalibExample.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>



// class decleration
//
/*
class miscalibExample : public edm::EDAnalyzer {
   public:
      explicit miscalibExample(const edm::ParameterSet&);
      ~miscalibExample();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
   private:


      // ----------member data ---------------------------
      std::string rootfile_;
      std::string correctedHybridSuperClusterProducer_;
      std::string correctedHybridSuperClusterCollection_;
      std::string BarrelHitsCollection_;
      std::string ecalHitsProducer_ ;
      int read_events;

      TH1F* scEnergy;
};

*/
miscalibExample::miscalibExample(const edm::ParameterSet& iConfig)
{

   rootfile_                  = iConfig.getUntrackedParameter<std::string>("rootfile","ecalSimpleTBanalysis.root");
   correctedHybridSuperClusterProducer_ = iConfig.getParameter<std::string>("correctedHybridSuperClusterProducer");
   correctedHybridSuperClusterCollection_ = iConfig.getParameter<std::string>("correctedHybridSuperClusterCollection");

}


miscalibExample::~miscalibExample()
{
 

}

//========================================================================
void
miscalibExample::beginJob(edm::EventSetup const& iSetup) {
//========================================================================

  // Book histograms 
  scEnergy = new TH1F("scEnergy","SuperCluster energy", 300, 0., 60.);

 
 }

//========================================================================
void
miscalibExample::endJob() {
//========================================================================

   std::cout << "************* STATISTICS **************" << std::endl;
   std::cout << "Read Events: " << read_events << std::endl;


/////////////////////////////////////////////////////////////////////////////

  TFile f(rootfile_.c_str(),"RECREATE");

  scEnergy->Write(); 
  f.Close();
 
}



//=================================================================================
void
miscalibExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
//=================================================================================
   using namespace edm;
   using namespace std;

   read_events++;

 // Get hybrid super clusters after energy correction
 
  Handle<reco::SuperClusterCollection> pCorrectedHybridSuperClusters;
  try {
    iEvent.getByLabel(correctedHybridSuperClusterProducer_, correctedHybridSuperClusterCollection_, pCorrectedHybridSuperClusters);
  } catch ( cms::Exception& ex ) {
    LogError("EgammaSimpleAnalyzer") << "Error! can't get collection with label " << correctedHybridSuperClusterCollection_.c_str() ;
  }
  const reco::SuperClusterCollection* correctedHybridSuperClusters = pCorrectedHybridSuperClusters.product();


  reco::SuperClusterCollection::const_iterator superClusterIt;
  for(superClusterIt=correctedHybridSuperClusters->begin(); superClusterIt!=correctedHybridSuperClusters->end(); superClusterIt++ )
  {

  scEnergy->Fill(superClusterIt->energy());

  }

}

//define this as a plug-in
//DEFINE_FWK_MODULE(miscalibExample)
