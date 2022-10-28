// system include files
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

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
    std::string fOutputFileName_;
    bool theRecalib_;
    std::string hcalfile_;
    std::ofstream* myout_hcal;

    // names of modules, producing object collections
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

    const edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    const edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    const edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

    const edm::EDGetTokenT<FEDRawDataCollection> tok_data_;

    const edm::EDGetTokenT<HBHERecHitCollection> tok_hbheNoise_;
    const edm::EDGetTokenT<HORecHitCollection> tok_hoNoise_;
    const edm::EDGetTokenT<HFRecHitCollection> tok_hfNoise_;

    //
    const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> tok_gtRec_;
    const edm::EDGetTokenT<HBHERecHitCollection> tok_hbheNorm_;

    const edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_respCorr_;
    const edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> tok_l1gt_;
  };
}  // namespace cms

//
// constructors and destructor
//
namespace cms {
  Analyzer_minbias::Analyzer_minbias(const edm::ParameterSet& iConfig)
      : fOutputFileName_(iConfig.getUntrackedParameter<std::string>("HistOutFile")),
        theRecalib_(iConfig.getParameter<bool>("Recalib")),
        tok_hbhe_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputMB"))),
        tok_ho_(consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputMB"))),
        tok_hf_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputMB"))),
        tok_data_(consumes<FEDRawDataCollection>(edm::InputTag(iConfig.getParameter<std::string>("InputLabel")))),
        tok_hbheNoise_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInputNoise"))),
        tok_hoNoise_(consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputNoise"))),
        tok_hfNoise_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInputNoise"))),
        tok_gtRec_(consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag("gtDigisAlCaMB"))),
        tok_hbheNorm_(consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"))),
        tok_respCorr_(esConsumes<HcalRespCorrs, HcalRespCorrsRcd>()),
        tok_l1gt_(esConsumes<L1GtTriggerMenu, L1GtTriggerMenuRcd>()) {
    usesResource(TFileService::kSharedResource);
    // get name of output file with histogramms
    // get names of modules, producing object collections
    // some of the label names are hardcodded..
    //
    for (int i = 0; i < 73; i++) {
      for (int j = 0; j < 43; j++) {
        noise_min[i][j] = 0.;
        noise_pl[i][j] = 0.;
      }
    }
  }

  Analyzer_minbias::~Analyzer_minbias() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
  }

  void Analyzer_minbias::beginRun(const edm::Run&, const edm::EventSetup&) { nevent_run = 0; }
  void Analyzer_minbias::endRun(const edm::Run& r, const edm::EventSetup&) {
    edm::LogVerbatim("AnalyzerMB") << " Runnumber " << r.run() << " Nevents  " << nevent_run;
  }

  void Analyzer_minbias::beginJob() {
    edm::Service<TFileService> fs;
    myTree = fs->make<TTree>("RecJet", "RecJet Tree");
    myTree->Branch("mydet", &mydet, "mydet/I");
    myTree->Branch("mysubd", &mysubd, "mysubd/I");
    myTree->Branch("depth", &depth, "depth/I");
    myTree->Branch("ieta", &ieta, "ieta/I");
    myTree->Branch("iphi", &iphi, "iphi/I");
    myTree->Branch("eta", &eta, "eta/F");
    myTree->Branch("phi", &phi, "phi/F");

    myTree->Branch("mom0_MB", &mom0_MB, "mom0_MB/F");
    myTree->Branch("mom1_MB", &mom1_MB, "mom1_MB/F");
    myTree->Branch("mom2_MB", &mom2_MB, "mom2_MB/F");
    myTree->Branch("mom4_MB", &mom4_MB, "mom4_MB/F");

    myTree->Branch("mom0_Noise", &mom0_Noise, "mom0_Noise/F");
    myTree->Branch("mom1_Noise", &mom1_Noise, "mom1_Noise/F");
    myTree->Branch("mom2_Noise", &mom2_Noise, "mom2_Noise/F");
    myTree->Branch("mom4_Noise", &mom4_Noise, "mom4_Noise/F");

    myTree->Branch("mom0_Diff", &mom0_Diff, "mom0_Diff/F");
    myTree->Branch("mom1_Diff", &mom1_Diff, "mom1_Diff/F");
    myTree->Branch("mom2_Diff", &mom2_Diff, "mom2_Diff/F");

    myTree->Branch("occup", &occup, "occup/F");

    edm::LogVerbatim("AnalyzerMB") << " Before ordering Histos ";

    char str0[32];
    char str1[32];

    char str10[32];
    char str11[32];

    int k = 0;
    nevent = 0;
    // Size of collections

    hHBHEsize_vs_run =
        fs->make<TH2F>("hHBHEsize_vs_run", "hHBHEsize_vs_run", 500, 111500., 112000., 6101, -100.5, 6000.5);
    hHFsize_vs_run = fs->make<TH2F>("hHFsize_vs_run", "hHFsize_vs_run", 500, 111500., 112000., 6101, -100.5, 6000.5);

    for (int i = 1; i < 73; i++) {
      for (int j = 1; j < 43; j++) {
        meannoise_pl[i][j] = 0.;
        meannoise_min[i][j] = 0.;

        k = i * 1000 + j;
        sprintf(str0, "mpl%d", k);
        sprintf(str1, "mmin%d", k);

        sprintf(str10, "vpl%d", k);
        sprintf(str11, "vmin%d", k);
        if (j < 30) {
          // first order moment
          hCalo1[i][j] = fs->make<TH1F>(str0, "h0", 320, -10., 10.);
          hCalo2[i][j] = fs->make<TH1F>(str1, "h1", 320, -10., 10.);

          // second order moment
          hCalo1mom2[i][j] = fs->make<TH1F>(str10, "h10", 320, 0., 20.);
          hCalo2mom2[i][j] = fs->make<TH1F>(str11, "h11", 320, 0., 20.);
        } else {
          // HF
          // first order moment
          if (j < 40) {
            hCalo1[i][j] = fs->make<TH1F>(str0, "h0", 320, -10., 10.);
            hCalo2[i][j] = fs->make<TH1F>(str1, "h1", 320, -10., 10.);
            //
            // second order moment
            hCalo1mom2[i][j] = fs->make<TH1F>(str10, "h10", 320, 0., 40.);
            hCalo2mom2[i][j] = fs->make<TH1F>(str11, "h11", 320, 0., 40.);
          } else {
            hCalo1[i][j] = fs->make<TH1F>(str0, "h0", 320, -10., 10.);
            hCalo2[i][j] = fs->make<TH1F>(str1, "h1", 320, -10., 10.);

            // second order moment
            hCalo1mom2[i][j] = fs->make<TH1F>(str10, "h10", 320, 0., 120.);
            hCalo2mom2[i][j] = fs->make<TH1F>(str11, "h11", 320, 0., 120.);
          }
        }  // HE/HF boundary
      }    // j
    }      // i

    hbheNoiseE = fs->make<TH1F>("hbheNoiseE", "hbheNoiseE", 320, -10., 10.);
    hfNoiseE = fs->make<TH1F>("hfNoiseE", "hfNoiseE", 320, -10., 10.);
    hbheSignalE = fs->make<TH1F>("hbheSignalE", "hbheSignalE", 320, -10., 10.);
    hfSignalE = fs->make<TH1F>("hfSignalE", "hfSignalE", 320, -10., 10.);

    edm::LogVerbatim("AnalyzerMB") << " After ordering Histos ";

    std::string ccc = "noise_0.dat";

    myout_hcal = new std::ofstream(ccc.c_str());
    if (!myout_hcal)
      edm::LogVerbatim("AnalyzerMB") << " Output file not open!!! ";

    //
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        for (int k = 0; k < 73; k++) {
          for (int l = 0; l < 43; l++) {
            theMBFillDetMapPl0[i][j][k][l] = 0.;
            theMBFillDetMapPl1[i][j][k][l] = 0.;
            theMBFillDetMapPl2[i][j][k][l] = 0.;
            theMBFillDetMapPl4[i][j][k][l] = 0.;

            theMBFillDetMapMin0[i][j][k][l] = 0.;
            theMBFillDetMapMin1[i][j][k][l] = 0.;
            theMBFillDetMapMin2[i][j][k][l] = 0.;
            theMBFillDetMapMin4[i][j][k][l] = 0.;

            theNSFillDetMapPl0[i][j][k][l] = 0.;
            theNSFillDetMapPl1[i][j][k][l] = 0.;
            theNSFillDetMapPl2[i][j][k][l] = 0.;
            theNSFillDetMapPl4[i][j][k][l] = 0.;

            theNSFillDetMapMin0[i][j][k][l] = 0.;
            theNSFillDetMapMin1[i][j][k][l] = 0.;
            theNSFillDetMapMin2[i][j][k][l] = 0.;
            theNSFillDetMapMin4[i][j][k][l] = 0.;

            theDFFillDetMapPl0[i][j][k][l] = 0.;
            theDFFillDetMapPl1[i][j][k][l] = 0.;
            theDFFillDetMapPl2[i][j][k][l] = 0.;
            theDFFillDetMapMin0[i][j][k][l] = 0.;
            theDFFillDetMapMin1[i][j][k][l] = 0.;
            theDFFillDetMapMin2[i][j][k][l] = 0.;
          }
        }
      }
    }

    return;
  }
  //
  //  EndJob
  //
  void Analyzer_minbias::endJob() {
    int ii = 0;

    for (int i = 1; i < 5; i++) {
      for (int j = 1; j < 5; j++) {
        for (int k = 1; k < 73; k++) {
          for (int l = 1; l < 43; l++) {
            if (theMBFillDetMapPl0[i][j][k][l] > 0) {
              mom0_MB = theMBFillDetMapPl0[i][j][k][l];
              mom1_MB = theMBFillDetMapPl1[i][j][k][l];
              mom2_MB = theMBFillDetMapPl2[i][j][k][l];
              mom4_MB = theMBFillDetMapPl4[i][j][k][l];
              mom0_Noise = theNSFillDetMapPl0[i][j][k][l];
              mom1_Noise = theNSFillDetMapPl1[i][j][k][l];
              mom2_Noise = theNSFillDetMapPl2[i][j][k][l];
              mom4_Noise = theNSFillDetMapPl4[i][j][k][l];
              mom0_Diff = theDFFillDetMapPl0[i][j][k][l];
              mom1_Diff = theDFFillDetMapPl1[i][j][k][l];
              mom2_Diff = theDFFillDetMapPl2[i][j][k][l];

              mysubd = i;
              depth = j;
              ieta = l;
              iphi = k;
              edm::LogVerbatim("AnalyzerMB") << " Result Plus= " << mysubd << " " << ieta << " " << iphi << " mom0  "
                                             << mom0_MB << " mom1 " << mom1_MB << " mom2 " << mom2_MB;
              myTree->Fill();
              ii++;
            }  // Pl > 0

            if (theMBFillDetMapMin0[i][j][k][l] > 0) {
              mom0_MB = theMBFillDetMapMin0[i][j][k][l];
              mom1_MB = theMBFillDetMapMin1[i][j][k][l];
              mom2_MB = theMBFillDetMapMin2[i][j][k][l];
              mom4_MB = theMBFillDetMapMin4[i][j][k][l];
              mom0_Noise = theNSFillDetMapMin0[i][j][k][l];
              mom1_Noise = theNSFillDetMapMin1[i][j][k][l];
              mom2_Noise = theNSFillDetMapMin2[i][j][k][l];
              mom4_Noise = theNSFillDetMapMin4[i][j][k][l];
              mom0_Diff = theDFFillDetMapMin0[i][j][k][l];
              mom1_Diff = theDFFillDetMapMin1[i][j][k][l];
              mom2_Diff = theDFFillDetMapMin2[i][j][k][l];

              mysubd = i;
              depth = j;
              ieta = -1 * l;
              iphi = k;
              edm::LogVerbatim("AnalyzerMB") << " Result Minus= " << mysubd << " " << ieta << " " << iphi << " mom0  "
                                             << mom0_MB << " mom1 " << mom1_MB << " mom2 " << mom2_MB;
              myTree->Fill();
              ii++;

            }  // Min>0
          }    // ieta
        }      // iphi
      }        // depth
    }          //subd

    edm::LogVerbatim("AnalyzerMB") << " Number of cells " << ii;

    for (int i = 1; i < 73; i++) {
      for (int j = 1; j < 43; j++) {
        hCalo1[i][j]->Write();
        hCalo2[i][j]->Write();
        hCalo1mom2[i][j]->Write();
        hCalo2mom2[i][j]->Write();
      }
    }

    edm::LogVerbatim("AnalyzerMB") << " File is closed ";

    return;
  }

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void Analyzer_minbias::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::LogVerbatim("AnalyzerMB") << " Start Analyzer_minbias::analyze " << nevent;
    nevent++;
    nevent_run++;

    float rnnum = (float)iEvent.run();

    std::vector<edm::StableProvenance const*> theProvenance;
    iEvent.getAllStableProvenance(theProvenance);

    for (auto const& provenance : theProvenance) {
      edm::LogVerbatim("AnalyzerMB") << " Print all process/modulelabel/product names " << provenance->processName()
                                     << " , " << provenance->moduleLabel() << " , "
                                     << provenance->productInstanceName();
    }
    const HcalRespCorrs* myRecalib = nullptr;
    if (theRecalib_) {
      myRecalib = &iSetup.getData(tok_respCorr_);
    }  // theRecalib_

    // Noise part for HB HE

    double tmpNSFillDetMapPl1[5][5][73][43];
    double tmpNSFillDetMapMin1[5][5][73][43];

    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        for (int k = 0; k < 73; k++) {
          for (int l = 0; l < 43; l++) {
            tmpNSFillDetMapPl1[i][j][k][l] = 0.;
            tmpNSFillDetMapMin1[i][j][k][l] = 0.;
          }
        }
      }
    }

    const edm::Handle<HBHERecHitCollection> hbheNormal = iEvent.getHandle(tok_hbheNorm_);
    if (!hbheNormal.isValid()) {
      edm::LogWarning("AnalyzerMB") << " hbheNormal failed ";
    } else {
      edm::LogVerbatim("AnalyzerMB") << " The size of the normal collection " << hbheNormal->size();
    }

    const edm::Handle<HBHERecHitCollection> hbheNS = iEvent.getHandle(tok_hbheNoise_);

    if (!hbheNS.isValid()) {
      edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe"
                                    << " product! No HBHE MS ";
      return;
    }

    const HBHERecHitCollection HithbheNS = *(hbheNS.product());
    edm::LogVerbatim("AnalyzerMB") << " HBHE NS size of collection " << HithbheNS.size();
    hHBHEsize_vs_run->Fill(rnnum, (float)HithbheNS.size());

    if (HithbheNS.size() != 5184) {
      edm::LogWarning("AnalyzerMB") << " HBHE problem " << rnnum << " " << HithbheNS.size();
    }
    const edm::Handle<HBHERecHitCollection> hbheMB = iEvent.getHandle(tok_hbhe_);

    if (!hbheMB.isValid()) {
      edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hbhe"
                                    << " product! No HBHE MB";
    }

    const HBHERecHitCollection HithbheMB = *(hbheMB.product());
    edm::LogVerbatim("AnalyzerMB") << " HBHE MB size of collection " << HithbheMB.size();
    if (HithbheMB.size() != 5184) {
      edm::LogWarning("AnalyzerMB") << " HBHE problem " << rnnum << " " << HithbheMB.size();
    }

    const edm::Handle<HFRecHitCollection> hfNS = iEvent.getHandle(tok_hfNoise_);

    if (!hfNS.isValid()) {
      edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hf"
                                    << " product! No HF NS ";
    }

    const HFRecHitCollection HithfNS = *(hfNS.product());
    edm::LogVerbatim("AnalyzerMB") << " HFE NS size of collection " << HithfNS.size();
    hHFsize_vs_run->Fill(rnnum, (float)HithfNS.size());
    if (HithfNS.size() != 1728) {
      edm::LogWarning("AnalyzerMB") << " HF problem " << rnnum << " " << HithfNS.size();
    }

    const edm::Handle<HFRecHitCollection> hfMB = iEvent.getHandle(tok_hf_);

    if (!hfMB.isValid()) {
      edm::LogWarning("AnalyzerMB") << "HcalCalibAlgos: Error! can't get hf"
                                    << " product! No HF MB";
    }

    const HFRecHitCollection HithfMB = *(hfMB.product());
    edm::LogVerbatim("AnalyzerMB") << " HF MB size of collection " << HithfMB.size();
    if (HithfMB.size() != 1728) {
      edm::LogWarning("AnalyzerMB") << " HF problem " << rnnum << " " << HithfMB.size();
    }

    for (HBHERecHitCollection::const_iterator hbheItr = HithbheNS.begin(); hbheItr != HithbheNS.end(); hbheItr++) {
      // Recalibration of energy
      float icalconst = 1.;
      DetId mydetid = hbheItr->id().rawId();
      if (theRecalib_)
        icalconst = myRecalib->getValues(mydetid)->getValue();

      HBHERecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());

      double energyhit = aHit.energy();

      DetId id = (*hbheItr).detid();
      HcalDetId hid = HcalDetId(id);

      int mysu = ((hid).rawId() >> 25) & 0x7;
      if (hid.ieta() > 0) {
        theNSFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] + 1.;
        theNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + energyhit;
        theNSFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(energyhit, 2);
        theNSFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(energyhit, 4);

        tmpNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] = energyhit;

      } else {
        theNSFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + 1.;
        theNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + energyhit;
        theNSFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 2);
        theNSFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 4);

        tmpNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()] = energyhit;
      }

      if (hid.depth() == 1) {
        hbheNoiseE->Fill(energyhit);

        if (energyhit < -2.)
          edm::LogVerbatim("AnalyzerMB") << " Run " << rnnum << " ieta,iphi " << hid.ieta() << " " << hid.iphi()
                                         << energyhit;

      }  // depth=1

    }  // HBHE_NS

    // Signal part for HB HE

    for (HBHERecHitCollection::const_iterator hbheItr = HithbheMB.begin(); hbheItr != HithbheMB.end(); hbheItr++) {
      // Recalibration of energy
      float icalconst = 1.;
      DetId mydetid = hbheItr->id().rawId();
      if (theRecalib_)
        icalconst = myRecalib->getValues(mydetid)->getValue();

      HBHERecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());

      double energyhit = aHit.energy();

      DetId id = (*hbheItr).detid();
      HcalDetId hid = HcalDetId(id);

      int mysu = ((hid).rawId() >> 25) & 0x7;
      if (hid.ieta() > 0) {
        theMBFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] += 1.;
        theMBFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] += energyhit;
        theMBFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] += pow(energyhit, 2);
        theMBFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] += pow(energyhit, 4);
        float mydiff = energyhit - tmpNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()];

        theDFFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + mydiff;
        theDFFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(mydiff, 2);
      } else {
        theMBFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + 1.;
        theMBFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + energyhit;
        theMBFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 2);
        theMBFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 4);

        float mydiff = energyhit - tmpNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()];
        theDFFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + mydiff;
        theDFFillDetMapMin2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapMin2[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(mydiff, 2);
      }

      if (hid.depth() == 1) {
        hbheSignalE->Fill(energyhit);

        if (hid.ieta() > 0) {
          hCalo1[hid.iphi()][hid.ieta()]->Fill(energyhit);
          hCalo1mom2[hid.iphi()][hid.ieta()]->Fill(pow(energyhit, 2));
        } else {
          hCalo2[hid.iphi()][abs(hid.ieta())]->Fill(energyhit);
          hCalo2mom2[hid.iphi()][abs(hid.ieta())]->Fill(pow(energyhit, 2));
        }  // eta><0

      }  // depth=1

    }  // HBHE_MB

    // HF

    for (HFRecHitCollection::const_iterator hbheItr = HithfNS.begin(); hbheItr != HithfNS.end(); hbheItr++) {
      // Recalibration of energy
      float icalconst = 1.;
      DetId mydetid = hbheItr->id().rawId();
      if (theRecalib_)
        icalconst = myRecalib->getValues(mydetid)->getValue();

      HFRecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());

      double energyhit = aHit.energy();
      //
      // Remove PMT hits
      //
      DetId id = (*hbheItr).detid();
      HcalDetId hid = HcalDetId(id);

      if (fabs(energyhit) > 40.)
        continue;

      int mysu = hid.subdetId();
      if (hid.ieta() > 0) {
        theNSFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] + 1.;
        theNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + energyhit;
        theNSFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(energyhit, 2);
        theNSFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theNSFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(energyhit, 4);

        tmpNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] = energyhit;

      } else {
        theNSFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + 1.;
        theNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + energyhit;
        theNSFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 2);
        theNSFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theNSFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 4);

        tmpNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()] = energyhit;
      }

      if (hid.depth() == 1) {
        hfNoiseE->Fill(energyhit);

      }  // depth=1

    }  // HBHE_NS

    // Signal part for HB HE

    for (HFRecHitCollection::const_iterator hbheItr = HithfMB.begin(); hbheItr != HithfMB.end(); hbheItr++) {
      // Recalibration of energy
      float icalconst = 1.;
      DetId mydetid = hbheItr->id().rawId();
      if (theRecalib_)
        icalconst = myRecalib->getValues(mydetid)->getValue();

      HFRecHit aHit(hbheItr->id(), hbheItr->energy() * icalconst, hbheItr->time());

      double energyhit = aHit.energy();
      //
      // Remove PMT hits
      //
      if (fabs(energyhit) > 40.)
        continue;

      DetId id = (*hbheItr).detid();
      HcalDetId hid = HcalDetId(id);

      int mysu = ((hid).rawId() >> 25) & 0x7;
      if (hid.ieta() > 0) {
        theMBFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theMBFillDetMapPl0[mysu][hid.depth()][hid.iphi()][hid.ieta()] + 1.;
        theMBFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theMBFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + energyhit;
        theMBFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theMBFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(energyhit, 2);
        theMBFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theMBFillDetMapPl4[mysu][hid.depth()][hid.iphi()][hid.ieta()] + pow(energyhit, 4);

        theDFFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + energyhit -
            tmpNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()];
        theDFFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapPl2[mysu][hid.depth()][hid.iphi()][hid.ieta()] +
            pow((energyhit - tmpNSFillDetMapPl1[mysu][hid.depth()][hid.iphi()][hid.ieta()]), 2);
      } else {
        theMBFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin0[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + 1.;
        theMBFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin1[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + energyhit;
        theMBFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin2[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 2);
        theMBFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] =
            theMBFillDetMapMin4[mysu][hid.depth()][hid.iphi()][abs(hid.ieta())] + pow(energyhit, 4);

        theDFFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()] + energyhit -
            tmpNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()];
        theDFFillDetMapMin2[mysu][hid.depth()][hid.iphi()][hid.ieta()] =
            theDFFillDetMapMin2[mysu][hid.depth()][hid.iphi()][hid.ieta()] +
            pow((energyhit - tmpNSFillDetMapMin1[mysu][hid.depth()][hid.iphi()][hid.ieta()]), 2);
      }

      if (hid.depth() == 1) {
        hfSignalE->Fill(energyhit);

        if (hid.ieta() > 0) {
          hCalo1[hid.iphi()][hid.ieta()]->Fill(energyhit);
          hCalo1mom2[hid.iphi()][hid.ieta()]->Fill(pow(energyhit, 2));
        } else {
          hCalo2[hid.iphi()][abs(hid.ieta())]->Fill(energyhit);
          hCalo2mom2[hid.iphi()][abs(hid.ieta())]->Fill(pow(energyhit, 2));
        }  // eta><0

      }  // depth=1

    }  // HF_MB

    edm::LogVerbatim("AnalyzerMB") << " Event is finished ";
  }
}  // namespace cms

#include "FWCore/Framework/interface/MakerMacros.h"

using cms::Analyzer_minbias;

DEFINE_FWK_MODULE(Analyzer_minbias);
