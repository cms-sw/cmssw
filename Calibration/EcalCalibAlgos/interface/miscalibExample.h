#ifndef Calibration_EcalCalibAlgos_miscalibExample_h
#define Calibration_EcalCalibAlgos_miscalibExample_h

/**\class miscalibExample

 Description: Analyzer to fetch collection of objects from event and make simple plots

 Implementation:
     \\\author: Lorenzo Agostino, September 2006
*/
//
// $Id: miscalibExample.h,v 1.2 2010/01/04 15:07:17 ferriff Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include "TH1.h"
#include "TFile.h"
//
// class declaration
//



class miscalibExample : public edm::EDAnalyzer {
   public:
      explicit miscalibExample(const edm::ParameterSet&);
      ~miscalibExample();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob();
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

#endif
