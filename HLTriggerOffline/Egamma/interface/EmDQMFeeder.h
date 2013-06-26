//
// Package:    EmDQMFeeder
// Class:      EmDQMFeeder
// 
/**\class EmDQMFeeder EmDQMFeeder.h HLTriggerOffline/Egamma/interface/EmDQMFeeder.h

 Description: Reads the trigger menu and calls EmDQM with generated parameter sets for each Egamma path

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis,40 4-B24,+41227671567,
//         Created:  Tue Mar 15 12:24:11 CET 2011
// $Id: EmDQMFeeder.h,v 1.3 2011/05/26 08:58:51 treis Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTriggerOffline/Egamma/interface/EmDQM.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

//
// class declaration
//
class EmDQMFeeder : public edm::EDAnalyzer {
   public:
      explicit EmDQMFeeder(const edm::ParameterSet&);
      ~EmDQMFeeder();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------

      const edm::ParameterSet& iConfig;

      //std::string processName_; // process name of (HLT) process for which to get HLT configuration
      edm::InputTag triggerObject_;
      /// The instance of the HLTConfigProvider as a data member
      HLTConfigProvider hltConfig_;

      std::vector<std::vector<std::string> > findEgammaPaths();
      std::vector<std::string> getFilterModules(const std::string&);
      double getPrimaryEtCut(const std::string&);

      edm::ParameterSet makePSetForL1SeedFilter(const std::string&);
      edm::ParameterSet makePSetForL1SeedToSuperClusterMatchFilter(const std::string&);
      edm::ParameterSet makePSetForEtFilter(const std::string&);
      edm::ParameterSet makePSetForOneOEMinusOneOPFilter(const std::string&);
      edm::ParameterSet makePSetForPixelMatchFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaGenericFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaGenericQuadraticFilter(const std::string&);
      edm::ParameterSet makePSetForElectronGenericFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaDoubleEtDeltaPhiFilter(const std::string&);

      std::vector<EmDQM*> emDQMmodules;

      static const unsigned TYPE_SINGLE_ELE = 0;
      static const unsigned TYPE_DOUBLE_ELE = 1;
      static const unsigned TYPE_SINGLE_PHOTON = 2;
      static const unsigned TYPE_DOUBLE_PHOTON = 3;
      static const unsigned TYPE_TRIPLE_ELE = 4;

      unsigned verbosity_;
      static const unsigned OUTPUT_SILENT = 0;
      static const unsigned OUTPUT_ERRORS = 1;
      static const unsigned OUTPUT_WARNINGS = 2;
      static const unsigned OUTPUT_ALL = 3;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//


