#ifndef JetPlusTrackAnalysis_h
#define JetPlusTrackAnalysis_h

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <memory>
#include <map>

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}
#include "TFile.h"
#include "TTree.h"

///
/// jet energy corrections from MCjet calibration
///
namespace cms
{

class JetPlusTrackAnalysis : public edm::EDAnalyzer
{
public:  

  JetPlusTrackAnalysis(const edm::ParameterSet& fParameters);

  virtual ~JetPlusTrackAnalysis();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob() ;
      virtual void endJob() ;
   
private:
// Histograms/Tree
     std::string fOutputFileName ;
     bool allowMissingInputs_;
     TFile*      hOutputFile ;
     TTree * myTree;
     int NumRecoJetsCaloTower, NumRecoJetsCorrected, NumRecoJetsZSPCorrected,NumRecoJetsRecHit, NumGenJets;
     float JetRecoEtCaloTower[10],JetRecoEtaCaloTower[10],JetRecoPhiCaloTower[10];
     float JetRecoEtCorrected[10],JetRecoEtaCorrected[10],JetRecoPhiCorrected[10];
     float JetRecoEtZSPCorrected[10],JetRecoEtaZSPCorrected[10],JetRecoPhiZSPCorrected[10];
     float JetRecoGenRecType[10],JetRecoGenPartonType[10];
     float JetRecoEtRecHit[10],EcalEmpty[10],HcalEmpty[10];
     float JetGenEt[10],JetGenEta[10],JetGenPhi[10],JetGenCode[10];
     int Code[4000],Charge[4000],NumPart,NumRecoTrack;
     float partpx[4000],partpy[4000],partpz[4000],parte[4000],partm[4000];
     float TrackRecoEt[5000],TrackRecoEta[5000],TrackRecoPhi[5000]; 
     int run, event;

  double mCone;
  edm::InputTag mInputJetsCaloTower;
  edm::InputTag mInputJetsCorrected; 
  edm::InputTag mInputJetsZSPCorrected;
  edm::InputTag mInputJetsGen;
  edm::InputTag mInputCaloTower;
  edm::InputTag mInputPVfCTF;
  std::string m_inputTrackLabel;
  std::vector<edm::InputTag> ecalLabels_;
  edm::InputTag ebrechit;
  edm::InputTag eerechit;
  edm::InputTag calotower;
  edm::InputTag hbhelabel_;
  edm::InputTag holabel_;
  edm::InputTag dtrecseg;
  edm::InputTag cscseg; 
  const CaloGeometry* geo;
};
}
#endif
