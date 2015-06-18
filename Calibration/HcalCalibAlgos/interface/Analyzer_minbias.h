#ifndef PhysicsToolsAnalysisAnalyzerMinBias_h
#define PhysicsToolsAnalysisAnalyzerMinBias_h

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>
#include <map>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// class declaration
namespace cms{
  class Analyzer_minbias : public edm::EDAnalyzer {
  public:
    explicit Analyzer_minbias(const edm::ParameterSet&);
    ~Analyzer_minbias();
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void beginJob() ;
    virtual void endJob() ;
    virtual void beginRun( const edm::Run& r, const edm::EventSetup& iSetup);
    virtual void endRun( const edm::Run& r, const edm::EventSetup& iSetup);
    
  private:
    
    // ----------member data ---------------------------
    std::string fOutputFileName ;
    std::string hcalfile_;
    std::ofstream *myout_hcal;
    TFile*      hOutputFile ;
    TTree*      myTree;

    // Root tree members
    double rnnum, rnnumber;
    int mydet, mysubd, depth, iphi, ieta, cells, trigbit;
    float phi,eta;
    float mom0_MB,mom1_MB,mom2_MB,mom3_MB,mom4_MB,occup;
    float mom0_Noise,mom1_Noise,mom2_Noise,mom3_Noise,mom4_Noise;
    float mom0_Diff,mom1_Diff,mom2_Diff,mom3_Diff,mom4_Diff;
    struct myInfo{
      double theMB0, theMB1, theMB2, theMB3, theMB4;
      double theNS0, theNS1, theNS2, theNS3, theNS4;
      double theDif0, theDif1, theDif2, runcheck;
      void MyInfo() {
	theMB0 = theMB1 = theMB2 = theMB3 = theMB4 = 0;
	theNS0 = theNS1 = theNS2 = theNS3 = theNS4 = 0;
	theDif0 = theDif1 = theDif2 = runcheck = 0;
      }
    };
    std::map<std::pair<int,HcalDetId>,myInfo> myMap;
    edm::EDGetTokenT<HBHERecHitCollection>  tok_hbherecoMB_, tok_hbherecoNoise_;
    edm::EDGetTokenT<HFRecHitCollection>    tok_hfrecoMB_,   tok_hfrecoNoise_;
    edm::EDGetTokenT<HORecHitCollection>    tok_horecoMB_,   tok_horecoNoise_;
    bool theRecalib;
    edm::EDGetTokenT<HBHERecHitCollection>  tok_hbheNormal_;
    edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> tok_hltL1GtMap_;
  };
}
#endif
