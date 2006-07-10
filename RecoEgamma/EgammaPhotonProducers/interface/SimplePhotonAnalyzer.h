#ifndef RecoEcal_EgammaPhotonProducers_SimplePhotonAnalyzer_h
#define RecoEcal_EgammaPhotonProducers_SimplePhotonAnalyzer_h
/**\class SimplePhotonAnalyzer
 **
 ** Description: Get Photon collection from the event and make very basic histos
 ** $Date: $
 ** $Revision: $
 ** \author Nancy Marinelli, U. of Notre Dame, US
 **
 **/
//
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


class SimplePhotonAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SimplePhotonAnalyzer( const edm::ParameterSet& );
      ~SimplePhotonAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
 private:

      std::string outputFile_; 
      TFile*  rootFile_;

      std::string photonCollectionProducer_;       
      std::string photonCorrCollectionProducer_;       
      std::string uncorrectedPhotonCollection_;

      
      
      std::string correctedPhotonCollection_;




      double xMinHist_;
      double xMaxHist_;
      int    nbinHist_;


      TH1F* h1_scE_;
      TH1F* h1_scEta_;
      TH1F* h1_scPhi_;


      TH1F* h1_corrPho_scE_;
      TH1F* h1_corrPho_scEta_;
      TH1F* h1_corrPho_scPhi_;



};
#endif
