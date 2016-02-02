#ifndef HiJetBackground_HiFJRhoAnalyzer_h
#define HiJetBackground_HiFJRhoAnalyzer_h

// system include files
#include <memory>
#include <sstream>
#include <string>
#include <vector>

//root
#include "TTree.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class HiFJRhoAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HiFJRhoAnalyzer(const edm::ParameterSet&);
      ~HiFJRhoAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      //input
      edm::EDGetTokenT<std::vector<double>>                  etaToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhoToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhomToken_;
      
       //output
      TTree *tree_;
      edm::Service<TFileService> fs_;

      struct RHO {
        std::vector<double> etaMin;
        std::vector<double> etaMax;
        std::vector<double> rho;
        std::vector<double> rhom;
      };
      
      RHO rhoObj_;
};

#endif
