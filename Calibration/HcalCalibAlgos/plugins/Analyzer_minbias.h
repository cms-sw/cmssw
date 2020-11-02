#ifndef PhysicsToolsAnalysisAnalyzerMinBias_h
#define PhysicsToolsAnalysisAnalyzerMinBias_h

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
//#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>
#include <map>
//#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapHcal.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//
// class declaration
//
namespace cms {
  class Analyzer_minbias : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    explicit Analyzer_minbias(const edm::ParameterSet&);
    ~Analyzer_minbias() override;

    void beginJob() override;
    void analyze(edm::Event const&, edm::EventSetup const&) override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void endRun(edm::Run const&, edm::EventSetup const&) override;
    void endJob() override;

  private:
    // ----------member data ---------------------------
    std::string fOutputFileName;
    std::string hcalfile_;
    std::ofstream* myout_hcal;

    edm::EDGetTokenT<FEDRawDataCollection> tok_data_;

    // names of modules, producing object collections
    edm::Service<TFileService> fs;

    TFile* hOutputFile;
    TTree* myTree;
    TH1F* hCalo1[73][43];
    TH1F* hCalo2[73][43];
    TH1F* hCalo1mom2[73][43];
    TH1F* hCalo2mom2[73][43];
    TH1F* hbheNoiseE;
    TH1F* hbheSignalE;
    TH1F* hfNoiseE;
    TH1F* hfSignalE;

    TH2F* hHBHEsize_vs_run;
    TH2F* hHFsize_vs_run;
    // Root tree members
    int nevent_run;
    int mydet, mysubd, depth, iphi, ieta;
    float phi, eta;
    float mom0_MB, mom1_MB, mom2_MB, mom3_MB, mom4_MB, occup;
    float mom0_Noise, mom1_Noise, mom2_Noise, mom3_Noise, mom4_Noise;
    float mom0_Diff, mom1_Diff, mom2_Diff, mom3_Diff, mom4_Diff;

    // Noise subtraction

    double meannoise_pl[73][43], meannoise_min[73][43];
    double noise_pl[73][43], noise_min[73][43];

    // counters

    double nevent;
    double theMBFillDetMapPl0[5][5][73][43];
    double theMBFillDetMapPl1[5][5][73][43];
    double theMBFillDetMapPl2[5][5][73][43];
    double theMBFillDetMapPl4[5][5][73][43];

    double theMBFillDetMapMin0[5][5][73][43];
    double theMBFillDetMapMin1[5][5][73][43];
    double theMBFillDetMapMin2[5][5][73][43];
    double theMBFillDetMapMin4[5][5][73][43];

    double theNSFillDetMapPl0[5][5][73][43];
    double theNSFillDetMapPl1[5][5][73][43];
    double theNSFillDetMapPl2[5][5][73][43];
    double theNSFillDetMapPl4[5][5][73][43];

    double theNSFillDetMapMin0[5][5][73][43];
    double theNSFillDetMapMin1[5][5][73][43];
    double theNSFillDetMapMin2[5][5][73][43];
    double theNSFillDetMapMin4[5][5][73][43];

    double theDFFillDetMapPl0[5][5][73][43];
    double theDFFillDetMapPl1[5][5][73][43];
    double theDFFillDetMapPl2[5][5][73][43];
    double theDFFillDetMapMin0[5][5][73][43];
    double theDFFillDetMapMin1[5][5][73][43];
    double theDFFillDetMapMin2[5][5][73][43];

    edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

    edm::EDGetTokenT<HBHERecHitCollection> tok_hbheNoise_;
    edm::EDGetTokenT<HORecHitCollection> tok_hoNoise_;
    edm::EDGetTokenT<HFRecHitCollection> tok_hfNoise_;

    //
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> tok_gtRec_;
    edm::EDGetTokenT<HBHERecHitCollection> tok_hbheNorm_;

    edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_respCorr_;

    bool theRecalib;
  };
}  // namespace cms
#endif
