#ifndef RecoEcal_Examples_SimplePhotonAnalyzer_h
#define RecoEcal_Examples_SimplePhotonAnalyzer_h
/**\class SimplePhotonAnalyzer
 **
 ** Description: Get Photon collection from the event and make very basic histos
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
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


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
      std::string valueMapPFCandPhoton_;
      edm::InputTag pfEgammaCandidates_;
      edm::InputTag barrelEcalHits_;
      edm::InputTag endcapEcalHits_;

      edm::ESHandle<CaloTopology> theCaloTopo_;

      std::string vertexProducer_;
      float sample_;

      DQMStore *dbe_;

      
      MonitorElement* h1_scEta_;
      MonitorElement* h1_deltaEtaSC_;
      MonitorElement* h1_pho_E_;
      MonitorElement* h1_pho_Et_;
      MonitorElement* h1_pho_Eta_;
      MonitorElement* h1_pho_Phi_;
      MonitorElement* h1_pho_R9Barrel_;
      MonitorElement* h1_pho_R9Endcap_;
      MonitorElement* h1_pho_sigmaIetaIetaBarrel_;
      MonitorElement* h1_pho_sigmaIetaIetaEndcap_;
      MonitorElement* h1_pho_hOverEBarrel_;
      MonitorElement* h1_pho_hOverEEndcap_;
      MonitorElement* h1_pho_ecalIsoBarrel_;
      MonitorElement* h1_pho_ecalIsoEndcap_;
      MonitorElement* h1_pho_hcalIsoBarrel_;
      MonitorElement* h1_pho_hcalIsoEndcap_;
      MonitorElement* h1_pho_trkIsoBarrel_;
      MonitorElement* h1_pho_trkIsoEndcap_;



      MonitorElement* h1_recEoverTrueEBarrel_ ;
      MonitorElement* h1_recEoverTrueEEndcap_ ;
      MonitorElement* h1_deltaEta_;
      MonitorElement* h1_deltaPhi_ ;





};
#endif
