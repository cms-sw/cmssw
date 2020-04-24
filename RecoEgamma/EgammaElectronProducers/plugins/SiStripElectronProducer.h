#ifndef EgammaElectronProducers_SiStripElectronProducer_h
#define EgammaElectronProducers_SiStripElectronProducer_h
// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     SiStripElectronProducer
// 
/**\class SiStripElectronProducer SiStripElectronProducer.h RecoEgamma/EgammaElectronProducers/interface/SiStripElectronProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Fri May 26 16:11:37 EDT 2006
//

// system include files

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

// forward declarations

class SiStripElectronProducer : public edm::stream::EDProducer<> {
   public:
      explicit SiStripElectronProducer(const edm::ParameterSet&);
      ~SiStripElectronProducer() override;


      void produce(edm::Event&, const edm::EventSetup&) override;
   private:
      // ----------member data ---------------------------
      std::string siHitProducer_;
      std::string siRphiHitCollection_;
      std::string siStereoHitCollection_;
      std::string siMatchedHitCollection_;
      std::string superClusterProducer_;
      std::string superClusterCollection_;
      std::string siStripElectronsLabel_;
      std::string trackCandidatesLabel_;

      edm::EDGetTokenT<SiStripRecHit2DCollection> rphi_sistrips2dtag_;
      edm::EDGetTokenT<SiStripRecHit2DCollection> stereo_sistrips2dtag_;
      edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> matched_sistrips2dtag_;
      edm::EDGetTokenT<reco::SuperClusterCollection> superClustertag_;
      
      

      SiStripElectronAlgo* algo_p;
};

#endif
