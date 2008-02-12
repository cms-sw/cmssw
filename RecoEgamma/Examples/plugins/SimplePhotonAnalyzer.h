#ifndef RecoEcal_Examples_SimplePhotonAnalyzer_h
#define RecoEcal_Examples_SimplePhotonAnalyzer_h
/**\class SimplePhotonAnalyzer
 **
 ** Description: Get Photon collection from the event and make very basic histos
 ** $Date: 2008/01/20 17:37:42 $
 ** $Revision: 1.2 $
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


      std::string mcProducer_;
      std::string mcCollection_;
      std::string photonCollectionProducer_;       
      std::string photonCollection_;       


      std::string vertexProducer_;

      TH1F* h1_scE_;
      TH1F* h1_scEt_;
      TH1F* h1_scEta_;
      TH1F* h1_scPhi_;


      TH1F* h1_phoE_;
      TH1F* h1_phoEta_;
      TH1F* h1_phoPhi_;

      
      TH1F* h1_recEoverTrueE_ ;
      TH1F* h1_deltaEta_;
      TH1F* h1_deltaPhi_ ;
      



      TH1F* h1_corrPho_E_;
      TH1F* h1_corrPho_Eta_;
      TH1F* h1_corrPho_Phi_;
      TH1F* h1_corrPho_R9_;



};
#endif
