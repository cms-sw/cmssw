#ifndef GammaJetAnalysis_h
#define GammaJetAnalysis_h
// system include files
#include <memory>
#include <string>
#include <iostream>
#include <map>

// user include files
// #include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
/* #include "FWCore/Framework/interface/Event.h" */
/* #include "FWCore/Framework/interface/MakerMacros.h" */
/* #include "FWCore/Framework/interface/ESHandle.h" */
/* #include "FWCore/Framework/interface/EventSetup.h" */
/* #include "FWCore/ParameterSet/interface/ParameterSet.h" */

/* #include "DataFormats/Common/interface/Ref.h" */
/* #include "DataFormats/DetId/interface/DetId.h" */

/* #include "Geometry/Records/interface/IdealGeometryRecord.h" */
/* #include "Geometry/CaloGeometry/interface/CaloGeometry.h" */
/* #include "Geometry/Vector/interface/GlobalPoint.h" */
/* #include "DataFormats/CaloTowers/interface/CaloTowerCollection.h" */
/* #include "DataFormats/CaloTowers/interface/CaloTowerDetId.h" */
/* #include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h" */
/* #include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h" */
/* #include "DataFormats/JetReco/interface/CaloJetCollection.h" */
/* #include "DataFormats/JetReco/interface/CaloJet.h" */

#include <fstream>

using namespace std;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TTree;
class CaloGeometry;

//
// class decleration
//
namespace cms
{
class GammaJetAnalysis : public edm::EDAnalyzer {
   public:
      explicit GammaJetAnalysis(const edm::ParameterSet&);
      ~GammaJetAnalysis();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup& ) ;
      virtual void endJob() ;

   private:
  // ----------member data ---------------------------
  // names of modules, producing object collections
     
     edm::InputTag hbheLabel_;
     edm::InputTag hoLabel_;
     edm::InputTag hfLabel_;
     std::vector<edm::InputTag> ecalLabels_;
     std::vector<edm::InputTag> mInputCalo;
     std::vector<edm::InputTag> mInputGen;
     std::string myName;
  // Ecal reconstruction
      std::string islandBarrelBasicClusterCollection_;
      std::string islandBarrelBasicClusterProducer_;
      std::string islandBarrelBasicClusterShapes_;

      std::string islandBarrelSuperClusterCollection_;
      std::string islandBarrelSuperClusterProducer_;

      std::string correctedIslandBarrelSuperClusterCollection_;
      std::string correctedIslandBarrelSuperClusterProducer_;

      std::string islandEndcapBasicClusterCollection_;
      std::string islandEndcapBasicClusterProducer_;
      std::string islandEndcapBasicClusterShapes_;

      std::string islandEndcapSuperClusterCollection_;
      std::string islandEndcapSuperClusterProducer_;

      std::string correctedIslandEndcapSuperClusterCollection_;
      std::string correctedIslandEndcapSuperClusterProducer_;

      std::string hybridSuperClusterCollection_;
      std::string hybridSuperClusterProducer_;

      std::string correctedHybridSuperClusterCollection_;
      std::string correctedHybridSuperClusterProducer_;
   
      double CutOnEgammaEnergy_;
     
  // stuff for histogramms
  //  output file name with histograms
     string fOutputFileName ;
     bool allowMissingInputs_;
  //
     TFile*      hOutputFile ;
     TTree * myTree;
     ofstream *myout_part;   
     ofstream *myout_hcal;
     ofstream *myout_ecal;     
     ofstream *myout_jet;   
     ofstream *myout_photon;
  //  
      int NumRecoJets,NumGenJets,NumRecoGamma,NumRecoTrack,NumRecoHcal,NumPart;
      int run,event;
      float JetRecoEt[10],JetRecoEta[10],JetRecoPhi[10],JetRecoType[10];
      float JetGenEt[10],JetGenEta[10],JetGenPhi[10],JetGenType[10];
      float TrackRecoEt[10],TrackRecoEta[10],TrackRecoPhi[10];
      int EcalClusDet[20];
      float GammaRecoEt[20],GammaRecoEta[20],GammaRecoPhi[20],GammaIsoEcal[9][20],GammaIsoHcal[9][20];
      float HcalDet[8000],HcalRecoEt[8000],HcalRecoEta[8000],HcalRecoPhi[8000];
      int Status[4000],Code[4000],Mother1[4000];
      float partpx[4000],partpy[4000],partpz[4000],parte[4000],partm[4000],partvx[4000];
      float partvy[4000],partvz[4000],partvt[4000];
      float risol[3];
      float ecut[3][3];
      
// Calo geometry
  const CaloGeometry* geo;

};
}
#endif
