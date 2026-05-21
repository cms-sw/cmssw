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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class HiFJRhoAnalyzer : public edm::one::EDAnalyzer<> {
   public:
      explicit HiFJRhoAnalyzer(const edm::ParameterSet&);
      ~HiFJRhoAnalyzer() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void beginJob() override;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override;

   private:
      // ----------member data ---------------------------
      //input
      edm::EDGetTokenT<std::vector<double>>                  etaToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhoToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhomToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhoCorrToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhomCorrToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhoCorr1BinToken_;
      edm::EDGetTokenT<std::vector<double>>                  rhomCorr1BinToken_;
	  
      edm::EDGetTokenT<std::vector<double>>                  rhoGridToken_;
      edm::EDGetTokenT<std::vector<double>>                  meanRhoGridToken_;
      edm::EDGetTokenT<std::vector<double>>                  etaMinRhoGridToken_;
      edm::EDGetTokenT<std::vector<double>>                  etaMaxRhoGridToken_;
      
      edm::EDGetTokenT<std::vector<double>>                  ptJetsToken_;
      edm::EDGetTokenT<std::vector<double>>                  areaJetsToken_;
      edm::EDGetTokenT<std::vector<double>>                  etaJetsToken_;

      edm::EDGetTokenT<std::vector<double>>                  rhoFlowFitParamsToken_;
      edm::EDGetTokenT<std::vector<int>>                     nTowToken_;
      edm::EDGetTokenT<std::vector<double>>                  towExcludePtToken_;
      edm::EDGetTokenT<std::vector<double>>                  towExcludePhiToken_;
      edm::EDGetTokenT<std::vector<double>>                  towExcludeEtaToken_;

      bool useModulatedRho_;

      //output
      TTree *tree_;
      edm::Service<TFileService> fs_;

      struct RHO {
        std::vector<double> etaMin;
        std::vector<double> etaMax;
        std::vector<double> rho;
        std::vector<double> rhom;
        std::vector<double> rhoCorr;
        std::vector<double> rhomCorr;
        std::vector<double> rhoCorr1Bin;
        std::vector<double> rhomCorr1Bin;
		
        std::vector<double> rhoGrid;
        std::vector<double> meanRhoGrid;
        std::vector<double> etaMinRhoGrid;
        std::vector<double> etaMaxRhoGrid;
		
        std::vector<double>ptJets;
        std::vector<double>areaJets;
        std::vector<double>etaJets;

        std::vector<double> rhoFlowFitParams;
        std::vector<int> nTow;
        std::vector<double> towExcludePt;
        std::vector<double> towExcludePhi;
        std::vector<double> towExcludeEta;
      };
      
      RHO rhoObj_;
};

#endif
