#ifndef RecoEcal_Examples_SimplePhotonAnalyzer_h
#define RecoEcal_Examples_SimplePhotonAnalyzer_h
/**\class SimplePhotonAnalyzer
 **
 ** Description: Get Photon collection from the event and make very basic histos
 ** $Date: 2008/02/25 12:32:02 $
 ** $Revision: 1.5 $
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

      float  etaTransformation( float a, float b);

      std::string mcProducer_;
      std::string mcCollection_;
      std::string photonCollectionProducer_;       
      std::string photonCollection_;       


      std::string vertexProducer_;

      TH1F* h1_scE_;
      TH1F* h1_scEt_;
      TH1F* h1_scEta_;
      TH1F* h1_scPhi_;
      TH1F* h1_deltaEtaSC_;
      TH1F* h1_deltaPhiSC_ ;
 

      TH1F* h1_e5x5_unconvBarrel_;
      TH1F* h1_e5x5_unconvEndcap_;
      TH1F* h1_ePho_convBarrel_;
      TH1F* h1_ePho_convEndcap_;


      TH1F* h1_recEoverTrueEBarrel_ ;
      TH1F* h1_recEoverTrueEEndcap_ ;
      TH1F* h1_recESCoverTrueEBarrel_ ;
      TH1F* h1_recESCoverTrueEEndcap_ ;
      TH1F* h1_deltaEta_;
      TH1F* h1_deltaPhi_ ;
    

      TH1F* h1_pho_E_;
      TH1F* h1_pho_Eta_;
      TH1F* h1_pho_Phi_;
      TH1F* h1_pho_R9Barrel_;
      TH1F* h1_pho_R9Endcap_;
    


};
#endif
