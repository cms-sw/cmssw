#ifndef RecoEcal_Examples_SimplePhotonAnalyzer_h
#define RecoEcal_Examples_SimplePhotonAnalyzer_h
/**\class SimplePhotonAnalyzer
 **
 ** Description: Get Photon collection from the event and make very basic histos
 ** $Date: 2009/11/26 19:37:02 $
 ** $Revision: 1.11 $
 ** \author Nancy Marinelli, U. of Notre Dame, US
 **
 **/
//
//


// system include files
#include <memory>

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include "TH1.h"
#include "TProfile.h"
class TFile;


class SimplePhotonAnalyzer : public edm::EDAnalyzer {
   public:
      explicit SimplePhotonAnalyzer( const edm::ParameterSet& );
      ~SimplePhotonAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob();
      virtual void endJob();
 private:

      float  etaTransformation( float a, float b);

      std::string mcProducer_;
      std::string mcCollection_;
      std::string photonCollectionProducer_;
      std::string photonCollection_;

      edm::InputTag barrelEcalHits_;
      edm::InputTag endcapEcalHits_;

      edm::ESHandle<CaloTopology> theCaloTopo_;

      std::string vertexProducer_;
      float sample_;


      TProfile* effEta_;
      TProfile* effPhi_;
      
      TH1F* h1_scEta_;
      TH1F* h1_deltaEtaSC_;
      TH1F* h1_pho_E_;
      TH1F* h1_pho_Et_;
      TH1F* h1_pho_Eta_;
      TH1F* h1_pho_Phi_;
      TH1F* h1_pho_R9Barrel_;
      TH1F* h1_pho_R9Endcap_;
      TH1F* h1_pho_sigmaIetaIetaBarrel_;
      TH1F* h1_pho_sigmaIetaIetaEndcap_;
      TH1F* h1_pho_hOverEBarrel_;
      TH1F* h1_pho_hOverEEndcap_;
      TH1F* h1_pho_ecalIsoBarrel_;
      TH1F* h1_pho_ecalIsoEndcap_;
      TH1F* h1_pho_hcalIsoBarrel_;
      TH1F* h1_pho_hcalIsoEndcap_;
      TH1F* h1_pho_trkIsoBarrel_;
      TH1F* h1_pho_trkIsoEndcap_;



      TH1F* h1_recEoverTrueEBarrel_ ;
      TH1F* h1_recEoverTrueEEndcap_ ;
      TH1F* h1_deltaEta_;
      TH1F* h1_deltaPhi_ ;





};
#endif
