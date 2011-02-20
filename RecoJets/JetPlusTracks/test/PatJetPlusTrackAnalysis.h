#ifndef RecoJetsJetPlusTracksPatJetPlusTrackAnalysis_h
#define RecoJetsJetPlusTracksPatJetPlusTrackAnalysis_h
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <memory>
#include <map>
#include "DataFormats/HeavyIonEvent/interface/CentralityProvider.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}
#include "TFile.h"
#include "TTree.h"

namespace cms
{

class PatJetPlusTrackAnalysis : public edm::EDAnalyzer
{
public:  

  PatJetPlusTrackAnalysis(const edm::ParameterSet& fParameters);

  virtual ~PatJetPlusTrackAnalysis();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob() ;
      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endJob() ;
   
private:
// Histograms/Tree
     std::string fOutputFileName ;
     bool allowMissingInputs_;
     TFile*      hOutputFile ;
     TTree * myTree;
     int NumRecoJetsCaloTower, NumRecoJetsCaloTower2, NumRecoJetsJPTCorrected, NumRecoJetsJPTCorrected2, NumRecoJetsRecHit, NumGenJets, NumGenJets2;
     int NvtxEv,Ntrkv,VertNDF;
     int centrality_bin;
     float VertChi2;
     float JetRecoEtCaloTower[10],JetRecoEtaCaloTower[10],JetRecoPhiCaloTower[10];
     float JetRecoEtCaloTower2[10],JetRecoEtaCaloTower2[10],JetRecoPhiCaloTower2[10];
     float JetRecoEtJPTCorrected[10],JetRecoEtCaloJetInit[10],JetRecoEtZSPCorrected[10],
           JetRecoEtaJPTCorrected[10],JetRecoPhiJPTCorrected[10],JetRecoEtaCaloJetInit[10],JetRecoPhiCaloJetInit[10];
     float JetRecoInitMN90a[10],JetRecoInitMN90Hits[10],JetRecoJPTSumETrack[10];
     float JetRecoEtJPTCorrected2[10],JetRecoEtCaloJetInit2[10],JetRecoEtZSPCorrected2[10],
           JetRecoEtaJPTCorrected2[10],JetRecoPhiJPTCorrected2[10],JetRecoEtaCaloJetInit2[10],JetRecoPhiCaloJetInit2[10];
     float JetRecoInitMN90a2[10],JetRecoInitMN90Hits2[10],JetRecoJPTSumETrack2[10];   
     float JetRecoEmf[10], JetRecofHPD[10], JetRecofRBX[10];
     float JetRecoEmf2[10], JetRecofHPD2[10], JetRecofRBX2[10];
     float JetRecoInitEmf[10], JetRecoInitfHPD[10], JetRecoInitfRBX[10];
     float JetRecoInitEmf2[10], JetRecoInitfHPD2[10], JetRecoInitfRBX2[10];
     float EnergyCaloTowerEtaPlus[42], EnergyCaloTowerEtaMinus[42];
     	   
     float JetRecoGenRecType[10],JetRecoGenPartonType[10];
     float JetRecoEtRecHit05[10],JetRecoEtRecHit07[10],EcalEmpty[10],HcalEmpty[10];
     float JetGenEt[10],JetGenEta[10],JetGenPhi[10],JetGenCode[10];
     float JetGenEt2[10],JetGenEta2[10],JetGenPhi2[10],JetGenCode2[10];
     
     int JetRecoJPTTrackMultInVertInCalo[10],JetRecoJPTTrackMultInVertInCalo2[10];
     int JetRecoJPTTrackMultInVertOutCalo[10],JetRecoJPTTrackMultInVertOutCalo2[10];
     int JetRecoJPTTrackMultOutVertInCalo[10],JetRecoJPTTrackMultOutVertInCalo2[10];
     
     int Code[2],Charge[2],NumPart;
     float partpx[2],partpy[2],partpz[2],parte[2],partm[2];
     int run, event;
     int data;

  double mCone05,mCone07;
  edm::InputTag mInputJetsCaloTower;
  edm::InputTag mInputJetsCaloTower2;
  edm::InputTag mInputJetsCorrected; 
  edm::InputTag mInputJetsCorrected2;
  edm::InputTag mInputJetsGen;
  edm::InputTag mInputJetsGen2;
  edm::InputTag mJetsIDName;
  edm::InputTag mJetsIDName2;
  
  std::vector<edm::InputTag> ecalLabels_;
  edm::InputTag ebrechit;
  edm::InputTag eerechit;
  edm::InputTag hbhelabel_;
  edm::InputTag holabel_;
  const CaloGeometry* geo;
  const CentralityBins* cbins_;
  CentralityProvider * centrality_;

};
}
#endif
