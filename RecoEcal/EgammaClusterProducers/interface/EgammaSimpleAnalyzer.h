#ifndef RecoEcal_EgammaClusterProducers_EgammaSimpleAnalyzer_h
#define RecoEcal_EgammaClusterProducers_EgammaSimpleAnalyzer_h
/**\class EgammaSimpleAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EgammaSimpleAnalyzer.h,v 1.3 2006/05/12 16:20:54 rahatlou Exp $
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
class TFile;

//
// class declaration
//

class EgammaSimpleAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EgammaSimpleAnalyzer( const edm::ParameterSet& );
      ~EgammaSimpleAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
 private:

      std::string outputFile_; // output file

      std::string islandBasicClusterCollection_;
      std::string islandBasicClusterProducer_;

      std::string islandSuperClusterCollection_;
      std::string islandSuperClusterProducer_;

      std::string correctedIslandSuperClusterCollection_;
      std::string correctedIslandSuperClusterProducer_;


      std::string hybridSuperClusterCollection_;
      std::string hybridSuperClusterProducer_;

      std::string correctedHybridSuperClusterCollection_;
      std::string correctedHybridSuperClusterProducer_;

      // root file to store histograms
      TFile*  rootFile_;

      // min and max of energy histograms
      double xMinHist_;
      double xMaxHist_;
      int    nbinHist_;

      // data members for histograms to be filled
      TH1F* h1_islandBCEnergy_;
      TH1F* h1_islandSCEnergy_;
      TH1F* h1_corrIslandSCEnergy_;
      TH1F* h1_hybridSCEnergy_;
      TH1F* h1_corrHybridSCEnergy_;
      TH1F* h1_corrHybridSCEta_;
      TH1F* h1_corrHybridSCPhi_;
};
#endif
