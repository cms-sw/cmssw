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
// $Id: SiStripElectronProducer.h,v 1.1 2007/04/20 14:54:21 uberthon Exp $
//

// system include files

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronAlgo.h"

// forward declarations

class SiStripElectronProducer : public edm::EDProducer {
   public:
      explicit SiStripElectronProducer(const edm::ParameterSet&);
      ~SiStripElectronProducer();


      virtual void produce(edm::Event&, const edm::EventSetup&);
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

      SiStripElectronAlgo* algo_p;
};

#endif
