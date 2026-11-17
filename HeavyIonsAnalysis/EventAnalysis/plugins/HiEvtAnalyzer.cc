#include <string>
#include <utility>
#include "FWCore/Utilities/interface/Exception.h"

// system include files
#include <memory>
#include <vector>
#include <algorithm>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "DataFormats/HeavyIonEvent/interface/HFFilterInfo.h"  //this line is needed to access the HF Filters
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include <HepMC/PdfInfo.h>

#include "TTree.h"

// Add this after your includes
constexpr double vcent[201] = {
  0,0.91197,1.82394,2.73591,3.64788,4.55985,5.47182,6.38379,7.29576,8.20773,9.1197,10.0317,10.6667,10.7669,11.1084,11.4511,11.7967,12.1453,12.4985,12.8586,13.2279,
  13.6047,13.9872,14.378,14.7734,15.1775,15.5875,16.0048,16.4344,16.8729,17.3249,17.7878,18.2572,18.7342,19.2173,19.7139,20.221,20.7368,21.2654,21.7955,22.3333,22.8819,
  23.446,24.0097,24.5881,25.179,25.7856,26.3914,27.0037,27.6247,28.2619,28.9063,29.5473,30.2005,30.8754,31.5574,32.2518,32.9558,33.6616,34.3925,35.1279,35.8711,
  36.618,37.3905,38.1709,38.9568,39.76,40.5757,41.3886,42.2135,43.0613,43.921,44.8002,45.6699,46.5569,47.4536,48.3673,49.2971,50.2437,51.1962,52.1656,53.1554,54.1508,
  55.1587,56.1668,57.1942,58.2508,59.3168,60.406,61.5058,62.6151,63.7513,64.8938,66.0488,67.2111,68.4189,69.6168,70.8422,72.0903,73.3433,74.6371,75.9344,77.2465,78.5776,
  79.9278,81.3019,82.7002,84.105,85.5389,86.9959,88.4836,89.9969,91.5244,93.0648,94.6373,96.2243,97.8198,99.454,101.105,102.798,104.466,106.186,107.925,109.692,
  111.487,113.265,115.092,116.936,118.856,120.819,122.768,124.744,126.732,128.739,130.809,132.907,135.033,137.173,139.34,141.543,143.777,146.028,148.298,150.674,
  153.07,155.464,157.912,160.369,162.846,165.358,167.907,170.503,173.094,175.76,178.495,181.231,183.991,186.784,189.674,192.575,195.507,198.523,201.592,204.653,207.796,
  210.983,214.245,217.554,220.881,224.256,227.741,231.249,234.786,238.444,242.096,245.842,249.627,253.523,257.577,261.697,265.88,270.16,274.601,279.162,283.888,
  288.745,293.767,298.968,304.526,310.308,316.408,322.893,329.956,337.457,345.746,354.995,365.685,378.512,395.16,420.836,526.238
};

constexpr double vcentlow[201] = {
  0,0.664956,1.32991,1.99487,2.65983,3.32478,3.98974,4.65469,5.31965,5.98461,6.64956,7.31452,7.97948,8.64443,9.30939,9.97434,10.6393,10.6667,11.0065,11.3561,11.7097,
  12.0652,12.426,12.7928,13.1695,13.5542,13.9451,14.3443,14.7483,15.1617,15.5812,16.0084,16.4483,16.8977,17.3614,17.8356,18.3171,18.8068,19.3012,19.8138,20.3333,
  20.8647,21.4078,21.9517,22.5062,23.0709,23.6494,24.2309,24.8299,25.4392,26.0639,26.6814,27.3178,27.963,28.6149,29.2726,29.9356,30.6168,31.3187,32.022,32.7412,
  33.4611,34.2037,34.955,35.7135,36.4749,37.261,38.0614,38.8636,39.6841,40.5192,41.3506,42.1943,43.0617,43.9417,44.8425,45.7338,46.6394,47.5622,48.4985,49.4524,
  50.4201,51.3994,52.3978,53.4095,54.4421,55.4623,56.5029,57.5694,58.657,59.7544,60.8762,62.0036,63.1486,64.3134,65.4901,66.6694,67.8993,69.1269,70.3681,71.6404,
  72.9151,74.234,75.5526,76.8901,78.2487,79.6284,81.0321,82.4579,83.8925,85.3538,86.8458,88.3651,89.913,91.478,93.053,94.6621,96.286,97.9208,99.5965,101.291,
  103.024,104.736,106.499,108.278,110.105,111.937,113.768,115.638,117.54,119.519,121.551,123.533,125.564,127.631,129.684,131.827,133.982,136.185,138.377,140.645,
  142.908,145.202,147.523,149.903,152.335,154.814,157.286,159.812,162.344,164.9,167.509,170.146,172.804,175.518,178.327,181.121,183.948,186.807,189.767,192.737,
  195.735,198.836,201.973,205.108,208.356,211.617,214.97,218.376,221.795,225.254,228.836,232.441,236.147,239.864,243.598,247.518,251.393,255.476,259.646,263.896,
  268.232,272.698,277.342,282.075,287.006,292.079,297.328,302.945,308.803,314.958,321.457,328.592,336.221,344.523,353.866,364.599,377.476,394.22,420.051,525.849
};

constexpr double vcentup[201] = {
  0,1.59161,3.18322,4.77483,6.36644,7.95805,9.54966,10.6667,10.7535,11.0863,11.4198,11.7564,12.0952,12.4387,12.7879,13.146,13.5116,13.8831,14.2616,14.6437,15.036,
  15.4326,15.837,16.2505,16.6715,17.1074,17.554,18.0063,18.4684,18.9363,19.4095,19.8995,20.3953,20.9015,21.4188,21.9364,22.4634,22.9983,23.5502,24.1013,24.6667,
  25.2431,25.8359,26.4255,27.0228,27.6278,28.2487,28.8758,29.5,30.1351,30.7898,31.4562,32.1294,32.8146,33.5004,34.208,34.9226,35.6441,36.3668,37.1135,37.8764,
  38.6334,39.4058,40.2017,40.9935,41.7885,42.6001,43.4346,44.2785,45.1374,45.9918,46.8504,47.7328,48.6271,49.5365,50.4583,51.3906,52.3402,53.3043,54.2815,55.2567,
  56.2415,57.2449,58.2755,59.3138,60.3745,61.4469,62.5264,63.6316,64.745,65.8642,66.9913,68.1678,69.3339,70.5219,71.7335,72.9467,74.2026,75.4589,76.7306,78.0211,
  79.3336,80.6587,82.009,83.3735,84.7555,86.1608,87.5938,89.0515,90.543,92.0151,93.5346,95.0844,96.626,98.1764,99.7758,101.391,103.041,104.669,106.348,108.043,
  109.767,111.516,113.246,115.027,116.821,118.686,120.59,122.503,124.412,126.342,128.314,130.3,132.343,134.394,136.488,138.575,140.742,142.897,145.085,147.282,
  149.539,151.859,154.21,156.532,158.955,161.367,163.779,166.227,168.729,171.243,173.797,176.415,179.084,181.76,184.454,187.175,190,192.824,195.678,198.63,
  201.617,204.597,207.651,210.764,213.942,217.146,220.394,223.675,227.036,230.457,233.889,237.448,241.011,244.592,248.322,252.034,255.935,259.926,263.958,268.086,
  272.314,276.722,281.237,285.907,290.662,295.631,300.827,306.298,312.015,318.019,324.491,331.502,338.857,347.048,356.285,366.891,379.693,396.132,421.674,526.775
};
//
// class declaration
//
#define NHFLEAD 3

class HiEvtAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HiEvtAnalyzer(const edm::ParameterSet&);
  ~HiEvtAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::Centrality> CentralityTag_;
  edm::EDGetTokenT<int> CentralityBinTag_;

  edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidateTag_;

  edm::EDGetTokenT<reco::EvtPlaneCollection> EvtPlaneTag_;
  edm::EDGetTokenT<reco::EvtPlaneCollection> EvtPlaneFlatTag_;

  edm::EDGetTokenT<edm::GenHIEvent> HiMCTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> VertexTag_;

  edm::EDGetTokenT<reco::HFFilterInfo> HFfilters_;

  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> puInfoToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genInfoToken_;
  edm::EDGetTokenT<LHEEventProduct> generatorlheToken_;

  bool doEvtPlane_;
  bool doEvtPlaneFlat_;
  bool doCentrality_;
  bool doPFHF_;

  bool doMC_;
  bool doHiMC_;
  bool doHFfilters_;
  bool useHepMC_;
  bool doVertex_;

  int evtPlaneLevel_;

  edm::Service<TFileService> fs_;

  TTree* thi_;
  std::vector<std::string> branchesToKeep_;

  float* hiEvtPlane;
  int nEvtPlanes;
  int HltEvtCnt;
  int hiBin, hiBin_up, hiBin_down;;
  int hiNpix, hiNpixelTracks, hiNtracks, hiNtracksPtCut, hiNtracksEtaCut, hiNtracksEtaPtCut;
  int hiNpixPlus, hiNpixMinus, hiNpixelTracksPlus, hiNpixelTracksMinus;
  float hiHF, hiHFplus, hiHFminus, hiHFplusEta4, hiHFminusEta4, hiHFhit, hiHFhitPlus, hiHFhitMinus;
  float hiHFECut, hiHFECutPlus, hiHFECutMinus;
  float hiEB, hiET, hiEE, hiEEplus, hiEEminus;
  float hiZDC, hiZDCplus, hiZDCminus;

  float hiHF_pf, hiHFE_pf, hiHF_pfha, hiHF_pfem;
  float hiHFPlus_pf, hiHFEPlus_pf, hiHFPlus_pfha, hiHFPlus_pfem;
  float hiHFMinus_pf, hiHFEMinus_pf, hiHFMinus_pfha, hiHFMinus_pfem;
  float hiHF_pfle[NHFLEAD], hiHFPlus_pfle[NHFLEAD], hiHFMinus_pfle[NHFLEAD];
  int nCountsHF_pf, nCountsHFPlus_pf, nCountsHFMinus_pf;

  float fNpart;
  float fNcoll;
  float fNhard;
  float fPhi0;
  float fb;

  int fNcharged;
  int fNchargedMR;
  float fMeanPt;
  float fMeanPtMR;
  float fEtMR;
  int fNchargedPtCut;
  int fNchargedPtCutMR;

  int proc_id;
  float pthat;
  float weight;
  float alphaQCD;
  float alphaQED;
  float qScale;
  int nMEPartons;
  int nMEPartonsFiltered;
  std::pair<int, int> pdfID;
  std::pair<float, float> pdfX;
  std::pair<float, float> pdfXpdf;

  std::vector<float> ttbar_w;  //weights for systematics

  std::vector<int> npus;     //number of pileup interactions
  std::vector<float> tnpus;  //true number of interactions

  int numMinHFTower2, numMinHFTower3, numMinHFTower4, numMinHFTower5;

  float vx, vy, vz;

  unsigned long long event;
  unsigned int run;
  unsigned int lumi;

  void inspfle(float hfe, float pfle[NHFLEAD]) {
    if (hfe <= pfle[NHFLEAD-1]) { return; }
    auto* end = pfle + NHFLEAD;
    auto* insert_pos = std::lower_bound(pfle, end, hfe, std::greater<float>{});
    if (insert_pos != end) {
        std::move_backward(insert_pos, end - 1, end);
        *insert_pos = hfe;
    }
  }
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiEvtAnalyzer::HiEvtAnalyzer(const edm::ParameterSet& iConfig)
    : CentralityTag_(consumes<reco::Centrality>(iConfig.getParameter<edm::InputTag>("CentralitySrc"))),
      CentralityBinTag_(consumes<int>(iConfig.getParameter<edm::InputTag>("CentralityBinSrc"))),
      pfCandidateTag_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidateSrc"))),
      EvtPlaneTag_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlane"))),
      EvtPlaneFlatTag_(consumes<reco::EvtPlaneCollection>(iConfig.getParameter<edm::InputTag>("EvtPlaneFlat"))),
      HiMCTag_(consumes<edm::GenHIEvent>(iConfig.getParameter<edm::InputTag>("HiMC"))),
      VertexTag_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("Vertex"))),
      HFfilters_(consumes<reco::HFFilterInfo>(iConfig.getParameter<edm::InputTag>("HFfilters"))),
      puInfoToken_(consumes<std::vector<PileupSummaryInfo>>(edm::InputTag("addPileupInfo"))),
      genInfoToken_(consumes<GenEventInfoProduct>(edm::InputTag("generator"))),
      generatorlheToken_(consumes<LHEEventProduct>(edm::InputTag("externalLHEProducer", ""))),
      doEvtPlane_(iConfig.getParameter<bool>("doEvtPlane")),
      doEvtPlaneFlat_(iConfig.getParameter<bool>("doEvtPlaneFlat")),
      doCentrality_(iConfig.getParameter<bool>("doCentrality")),
      doMC_(iConfig.getParameter<bool>("doMC")),
      doHiMC_(iConfig.getParameter<bool>("doHiMC")),
      doHFfilters_(iConfig.getParameter<bool>("doHFfilters")),
      useHepMC_(iConfig.getParameter<bool>("useHepMC")),
      doVertex_(iConfig.getParameter<bool>("doVertex")),
      evtPlaneLevel_(iConfig.getParameter<int>("evtPlaneLevel")) {
  branchesToKeep_ = iConfig.getUntrackedParameter<std::vector<std::string>>("branchesToKeep", {});
  doPFHF_ = iConfig.getUntrackedParameter<bool>("doPFHF", doCentrality_);
}

HiEvtAnalyzer::~HiEvtAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void HiEvtAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //cleanup previous event
  npus.clear();
  tnpus.clear();
  ttbar_w.clear();

  using namespace edm;

  // Run info
  event = iEvent.id().event();
  run = iEvent.id().run();
  lumi = iEvent.id().luminosityBlock();

  if (doHiMC_) {
    edm::Handle<edm::GenHIEvent> mchievt;
    if (iEvent.getByToken(HiMCTag_, mchievt)) {
      fb = mchievt->b();
      fNpart = mchievt->Npart();
      fNcoll = mchievt->Ncoll();
      fNhard = mchievt->Nhard();
      fPhi0 = mchievt->evtPlane();
      fNcharged = mchievt->Ncharged();
      fNchargedMR = mchievt->NchargedMR();
      fMeanPt = mchievt->MeanPt();
      fMeanPtMR = mchievt->MeanPtMR();
      fEtMR = mchievt->EtMR();
      fNchargedPtCut = mchievt->NchargedPtCut();
      fNchargedPtCutMR = mchievt->NchargedPtCutMR();
    }
  }

  if (doMC_) {
    if (useHepMC_) {
      edm::Handle<edm::HepMCProduct> hepmcevt;
      iEvent.getByLabel("generator", hepmcevt);
      proc_id = hepmcevt->GetEvent()->signal_process_id();
      weight = hepmcevt->GetEvent()->weights()[0];
      alphaQCD = hepmcevt->GetEvent()->alphaQCD();
      alphaQED = hepmcevt->GetEvent()->alphaQED();
      qScale = hepmcevt->GetEvent()->event_scale();
      const HepMC::PdfInfo* hepPDF = hepmcevt->GetEvent()->pdf_info();
      if (hepPDF) {
        pdfID = std::make_pair(hepPDF->id1(), hepPDF->id2());
        pdfX = std::make_pair(hepPDF->x1(), hepPDF->x2());
        pdfXpdf = std::make_pair(hepPDF->pdf1(), hepPDF->pdf2());
      }
    } else {
      edm::Handle<GenEventInfoProduct> genInfo;
      if (iEvent.getByToken(genInfoToken_, genInfo)) {
        proc_id = genInfo->signalProcessID();
        if (genInfo->hasBinningValues())
          pthat = genInfo->binningValues()[0];
        weight = genInfo->weight();
        nMEPartons = genInfo->nMEPartons();
        nMEPartonsFiltered = genInfo->nMEPartonsFiltered();
        alphaQCD = genInfo->alphaQCD();
        alphaQED = genInfo->alphaQED();
        qScale = genInfo->qScale();

        if (genInfo->hasPDF()) {
          pdfID = genInfo->pdf()->id;
          pdfX.first = genInfo->pdf()->x.first;
          pdfX.second = genInfo->pdf()->x.second;
          pdfXpdf.first = genInfo->pdf()->xPDF.first;
          pdfXpdf.second = genInfo->pdf()->xPDF.second;
        }
      }

      //alternative weights for systematics
      edm::Handle<LHEEventProduct> evet;
      iEvent.getByToken(generatorlheToken_, evet);
      if (evet.isValid() && genInfo.isValid()) {
        double asdd = evet->originalXWGTUP();
        for (unsigned int i = 0; i < evet->weights().size(); i++) {
          double asdde = evet->weights()[i].wgt;
          ttbar_w.push_back(genInfo->weight() * asdde / asdd);
        }
      }
    }

    // MC PILEUP INFORMATION
    edm::Handle<std::vector<PileupSummaryInfo>> puInfos;
    if (iEvent.getByToken(puInfoToken_, puInfos)) {
      for (const auto& pu : *puInfos) {
        npus.push_back(pu.getPU_NumInteractions());
        tnpus.push_back(pu.getTrueNumInteractions());
      }
    }
  }

  if (doCentrality_) {
    edm::Handle<reco::Centrality> centrality;
    iEvent.getByToken(CentralityTag_, centrality);

    hiNpix = centrality->multiplicityPixel();
    hiNpixPlus = centrality->multiplicityPixelPlus();
    hiNpixMinus = centrality->multiplicityPixelMinus();
    hiNpixelTracks = centrality->NpixelTracks();
    hiNpixelTracksPlus = centrality->NpixelTracksPlus();
    hiNpixelTracksMinus = centrality->NpixelTracksMinus();
    hiNtracks = centrality->Ntracks();
    hiNtracksPtCut = centrality->NtracksPtCut();
    hiNtracksEtaCut = centrality->NtracksEtaCut();
    hiNtracksEtaPtCut = centrality->NtracksEtaPtCut();

    hiHF = centrality->EtHFtowerSum();
    hiHFplus = centrality->EtHFtowerSumPlus();
    hiHFminus = centrality->EtHFtowerSumMinus();
    hiHFECut = centrality->EtHFtowerSumECut();
    hiHFECutPlus = centrality->EtHFtowerSumECutPlus();
    hiHFECutMinus = centrality->EtHFtowerSumECutMinus();
    hiHFplusEta4 = centrality->EtHFtruncatedPlus();
    hiHFminusEta4 = centrality->EtHFtruncatedMinus();
    hiHFhit = centrality->EtHFhitSum();
    hiHFhitPlus = centrality->EtHFhitSumPlus();
    hiHFhitMinus = centrality->EtHFhitSumMinus();

    hiZDC = centrality->zdcSum();
    hiZDCplus = centrality->zdcSumPlus();
    hiZDCminus = centrality->zdcSumMinus();

    hiEEplus = centrality->EtEESumPlus();
    hiEEminus = centrality->EtEESumMinus();
    hiEE = centrality->EtEESum();
    hiEB = centrality->EtEBSum();
    hiET = centrality->EtMidRapiditySum();
    
  }

  if (doCentrality_ || doPFHF_) {
    edm::Handle<pat::PackedCandidateCollection> pfCandidates;
    iEvent.getByToken(pfCandidateTag_, pfCandidates);

    hiHF_pf = 0; hiHFE_pf = 0; hiHF_pfha = 0; hiHF_pfem = 0;
    for (auto& le : hiHF_pfle) le = 0;
    hiHFPlus_pf = 0; hiHFEPlus_pf = 0; hiHFPlus_pfha = 0; hiHFPlus_pfem = 0;
    for (auto& le : hiHFPlus_pfle) le = 0;
    hiHFMinus_pf = 0; hiHFEMinus_pf = 0; hiHFMinus_pfha = 0; hiHFMinus_pfem = 0;
    for (auto& le : hiHFMinus_pfle) le = 0;
    nCountsHF_pf = 0; nCountsHFPlus_pf = 0; nCountsHFMinus_pf = 0;

    for (const auto& pfcand : *pfCandidates) {
      if (pfcand.pdgId() != 1 && pfcand.pdgId() != 2) continue;
      if (pfcand.et() < 0.0) continue;
      const bool eta_plus = (pfcand.eta() > 3.0) && (pfcand.eta() < 6.0);
      const bool eta_minus = (pfcand.eta() < -3.0) && (pfcand.eta() > -6.0);
      if (!eta_plus && !eta_minus) continue;
      const auto hfe = pfcand.energy();
      const auto hfet = pfcand.et();
      const auto hfid = pfcand.pdgId();

      hiHF_pf += hfet;
      hiHFE_pf += hfe;
      if(hfid == 1) hiHF_pfha += hfet;
      if(hfid == 2) hiHF_pfem += hfet;
      nCountsHF_pf++;
      inspfle(hfe, hiHF_pfle);

      if (eta_plus) {
        hiHFPlus_pf += hfet;
        hiHFEPlus_pf += hfe;
        if(hfid == 1) hiHFPlus_pfha += hfet;
        if(hfid == 2) hiHFPlus_pfem += hfet;
        nCountsHFPlus_pf++;
        inspfle(hfe, hiHFPlus_pfle);
      } // if (eta_plus) {
      if (eta_minus) {
        hiHFMinus_pf += hfet;
        hiHFEMinus_pf += hfe;
        if(hfid == 1) hiHFMinus_pfha += hfet;
        if(hfid == 2) hiHFMinus_pfem += hfet;
        nCountsHFMinus_pf++;
        inspfle(hfe, hiHFMinus_pfle);
      } // if(eta_minus) {
    } // for (const auto& pfcand : *pfCandidates) {

  }

  if (doCentrality_) {
    // --- NEW PLACEMENT FOR HIBIN CALCULATIONS ---
    // Now that hiHF_pf is fully accumulated for the event, calculate the bins.
    
    auto getHiBin = [](const double* array, double hf) -> int {
      auto it = std::upper_bound(array, array + 201, hf);
      int bin_index = std::distance(array, it) - 1;
      
      if (bin_index >= 0 && bin_index < 200) {
        return 199 - bin_index;
      } else if (hf >= array[200]) {
        return 0;   // Overflow -> most central
      } else {
        return 199; // Underflow -> most peripheral
      }
    };

    // Calculate nominal, up, and down bins using hiHF_pf
    hiBin      = getHiBin(vcent, hiHF_pf);
    hiBin_up   = getHiBin(vcentup, hiHF_pf);
    hiBin_down = getHiBin(vcentlow, hiHF_pf);
  }

  nEvtPlanes = 0;
  edm::Handle<reco::EvtPlaneCollection> evtPlanes;

  if (doEvtPlane_) {
    iEvent.getByToken(EvtPlaneTag_, evtPlanes);
    if (evtPlanes.isValid()) {
      nEvtPlanes += evtPlanes->size();
      for (unsigned int i = 0; i < evtPlanes->size(); ++i) {
        hiEvtPlane[i] = (*evtPlanes)[i].angle(evtPlaneLevel_);
      }
    }
  }

  if (doEvtPlaneFlat_) {
    iEvent.getByToken(EvtPlaneFlatTag_, evtPlanes);
    if (evtPlanes.isValid()) {
      for (unsigned int i = 0; i < evtPlanes->size(); ++i) {
        hiEvtPlane[nEvtPlanes + i] = (*evtPlanes)[i].angle();
      }
      nEvtPlanes += evtPlanes->size();
    }
  }

  if (doVertex_) {
    edm::Handle<std::vector<reco::Vertex>> vertex;
    iEvent.getByToken(VertexTag_, vertex);
    vx = vertex->begin()->x();
    vy = vertex->begin()->y();
    vz = vertex->begin()->z();
  }

  // Option to disable HF filters for ppref
  if(doHFfilters_){
    edm::Handle<reco::HFFilterInfo> HFfilter;
    iEvent.getByToken(HFfilters_, HFfilter);

    numMinHFTower2 = HFfilter->numMinHFTowers2;
    numMinHFTower3 = HFfilter->numMinHFTowers3;
    numMinHFTower4 = HFfilter->numMinHFTowers4;
    numMinHFTower5 = HFfilter->numMinHFTowers5;
  } else {
    numMinHFTower2 = 0;
    numMinHFTower3 = 0;
    numMinHFTower4 = 0;
    numMinHFTower5 = 0;
  }

  thi_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void HiEvtAnalyzer::beginJob() {
  thi_ = fs_->make<TTree>("HiTree", "");
  const auto bookBranch = [this](const char* name, auto&&... args) {
    if (branchesToKeep_.empty() ||
        std::find(branchesToKeep_.begin(), branchesToKeep_.end(), name) != branchesToKeep_.end())
      thi_->Branch(name, std::forward<decltype(args)>(args)...);
  };


  HltEvtCnt = 0;
  const int kMaxEvtPlanes = 1000;

  fNpart = -1;
  fNcoll = -1;
  fNhard = -1;
  fPhi0 = -1;
  fb = -1;
  fNcharged = -1;
  fNchargedMR = -1;
  fMeanPt = -1;
  fMeanPtMR = -1;

  fEtMR = -1;
  fNchargedPtCut = -1;
  fNchargedPtCutMR = -1;

  proc_id = -1;
  pthat = -1.;
  weight = -1.;
  alphaQCD = -1.;
  alphaQED = -1.;
  qScale = -1.;
  //  npu      =   1;

  nEvtPlanes = 0;
  hiBin = -1;
  hiBin_up = -1;
  hiBin_down = -1;

  hiEvtPlane = new float[kMaxEvtPlanes];

  vx = -100;
  vy = -100;
  vz = -100;

  numMinHFTower2 = -1;
  numMinHFTower3 = -1;
  numMinHFTower4 = -1;
  numMinHFTower5 = -1;

  // Run info
  bookBranch("run", &run, "run/i");
  bookBranch("evt", &event, "evt/l");
  bookBranch("lumi", &lumi, "lumi/i");

  // Vertex
  bookBranch("vx", &vx, "vx/F");
  bookBranch("vy", &vy, "vy/F");
  bookBranch("vz", &vz, "vz/F");

  //Event observables
  if (doHiMC_) {
    bookBranch("Npart", &fNpart, "Npart/F");
    bookBranch("Ncoll", &fNcoll, "Ncoll/F");
    bookBranch("Nhard", &fNhard, "Nhard/F");
    bookBranch("phi0", &fPhi0, "NPhi0/F");
    bookBranch("b", &fb, "b/F");
    bookBranch("Ncharged", &fNcharged, "Ncharged/I");
    bookBranch("NchargedMR", &fNchargedMR, "NchargedMR/I");
    bookBranch("MeanPt", &fMeanPt, "MeanPt/F");
    bookBranch("MeanPtMR", &fMeanPtMR, "MeanPtMR/F");
    bookBranch("EtMR", &fEtMR, "EtMR/F");
    bookBranch("NchargedPtCut", &fNchargedPtCut, "NchargedPtCut/I");
    bookBranch("NchargedPtCutMR", &fNchargedPtCutMR, "NchargedPtCutMR/I");
  }
  if (doMC_) {
    bookBranch("ProcessID", &proc_id, "ProcessID/I");
    bookBranch("pthat", &pthat, "pthat/F");
    bookBranch("weight", &weight, "weight/F");
    bookBranch("alphaQCD", &alphaQCD, "alphaQCD/F");
    bookBranch("alphaQED", &alphaQED, "alphaQED/F");
    bookBranch("qScale", &qScale, "qScale/F");
    bookBranch("nMEPartons", &nMEPartons, "nMEPartons/I");
    bookBranch("nMEPartonsFiltered", &nMEPartonsFiltered, "nMEPartonsFiltered/I");
    bookBranch("pdfID", &pdfID);
    bookBranch("pdfX", &pdfX);
    bookBranch("pdfXpdf", &pdfXpdf);
    bookBranch("ttbar_w", &ttbar_w);
    bookBranch("npus", &npus);
    bookBranch("tnpus", &tnpus);
  }

  bookBranch("hiBin", &hiBin, "hiBin/I");
  bookBranch("hiBin_up", &hiBin_up, "hiBin_up/I");
  bookBranch("hiBin_down", &hiBin_down, "hiBin_down/I");
  /*
  bookBranch("hiHF", &hiHF, "hiHF/F");
  bookBranch("hiHFplus", &hiHFplus, "hiHFplus/F");
  bookBranch("hiHFminus", &hiHFminus, "hiHFminus/F");
  bookBranch("hiHFECut", &hiHFECut, "hiHFECut/F");
  bookBranch("hiHFECutPlus", &hiHFECutPlus, "hiHFECutPlus/F");
  bookBranch("hiHFECutMinus", &hiHFECutMinus, "hiHFECutMinus/F");
  bookBranch("hiHFplusEta4", &hiHFplusEta4, "hiHFplusEta4/F");
  bookBranch("hiHFminusEta4", &hiHFminusEta4, "hiHFminusEta4/F");

  bookBranch("hiZDC", &hiZDC, "hiZDC/F");
  bookBranch("hiZDCplus", &hiZDCplus, "hiZDCplus/F");
  bookBranch("hiZDCminus", &hiZDCminus, "hiZDCminus/F");

  bookBranch("hiHFhit", &hiHFhit, "hiHFhit/F");
  bookBranch("hiHFhitPlus", &hiHFhitPlus, "hiHFhitPlus/F");
  bookBranch("hiHFhitMinus", &hiHFhitMinus, "hiHFhitMinus/F");

  bookBranch("hiET", &hiET, "hiET/F");
  bookBranch("hiEE", &hiEE, "hiEE/F");
  bookBranch("hiEB", &hiEB, "hiEB/F");
  bookBranch("hiEEplus", &hiEEplus, "hiEEplus/F");
  bookBranch("hiEEminus", &hiEEminus, "hiEEminus/F");
  bookBranch("hiNpix", &hiNpix, "hiNpix/I");
  bookBranch("hiNpixPlus", &hiNpixPlus, "hiNpixPlus/I");
  bookBranch("hiNpixMinus", &hiNpixMinus, "hiNpixMinus/I");
  bookBranch("hiNpixelTracks", &hiNpixelTracks, "hiNpixelTracks/I");
  bookBranch("hiNpixelTracksPlus", &hiNpixelTracksPlus, "hiNpixelTracksPlus/I");
  bookBranch("hiNpixelTracksMinus", &hiNpixelTracksMinus, "hiNpixelTracksMinus/I");
  bookBranch("hiNtracks", &hiNtracks, "hiNtracks/I");
  bookBranch("hiNtracksPtCut", &hiNtracksPtCut, "hiNtracksPtCut/I");
  bookBranch("hiNtracksEtaCut", &hiNtracksEtaCut, "hiNtracksEtaCut/I");
  bookBranch("hiNtracksEtaPtCut", &hiNtracksEtaPtCut, "hiNtracksEtaPtCut/I");
  */
  bookBranch("hiHF_pf", &hiHF_pf, "hiHF_pf/F");
  /*
  bookBranch("hiHFE_pf", &hiHFE_pf, "hiHFE_pf/F");

  bookBranch("hiHF_pfha", &hiHF_pfha, "hiHF_pfha/F");
  bookBranch("hiHF_pfem", &hiHF_pfem, "hiHF_pfem/F");
  */
  bookBranch("hiHFPlus_pf", &hiHFPlus_pf, "hiHFPlus_pf/F");
  /*
  bookBranch("hiHFEPlus_pf", &hiHFEPlus_pf, "hiHFEPlus_pf/F");
  bookBranch("hiHFPlus_pfha", &hiHFPlus_pfha, "hiHFPlus_pfha/F");
  bookBranch("hiHFPlus_pfem", &hiHFPlus_pfem, "hiHFPlus_pfem/F");
  */
  bookBranch("hiHFMinus_pf", &hiHFMinus_pf, "hiHFMinus_pf/F");
  /*
  bookBranch("hiHFEMinus_pf", &hiHFEMinus_pf, "hiHFEMinus_pf/F");
  bookBranch("hiHFMinus_pfha", &hiHFMinus_pfha, "hiHFMinus_pfha/F");
  bookBranch("hiHFMinus_pfem", &hiHFMinus_pfem, "hiHFMinus_pfem/F");

  for (int i=0; i<NHFLEAD; i++) {
    bookBranch(Form("hiHF_pfle%d", i+1), &(hiHF_pfle[i]), Form("hiHF_pfle%d/F", i+1));
    bookBranch(Form("hiHFPlus_pfle%d", i+1), &(hiHFPlus_pfle[i]), Form("hiHFPlus_pfle%d/F", i+1));
    bookBranch(Form("hiHFMinus_pfle%d", i+1), &(hiHFMinus_pfle[i]), Form("hiHFMinus_pfle%d/F", i+1));
  }
  bookBranch("nCountsHF_pf", &nCountsHF_pf, "nCountsHF_pf/I");
  bookBranch("nCountsHFPlus_pf", &nCountsHFPlus_pf, "nCountsHFPlus_pf/I");
  bookBranch("nCountsHFMinus_pf", &nCountsHFMinus_pf, "nCountsHFMinus_pf/I");

  // Event plane
  if (doEvtPlane_) {
    bookBranch("hiNevtPlane", &nEvtPlanes, "hiNevtPlane/I");
    bookBranch("hiEvtPlanes", hiEvtPlane, "hiEvtPlanes[hiNevtPlane]/F");
  }

  bookBranch("numMinHFTower2", &numMinHFTower2, "numMinHFTower2/I");
  bookBranch("numMinHFTower3", &numMinHFTower3, "numMinHFTower3/I");
  bookBranch("numMinHFTower4", &numMinHFTower4, "numMinHFTower4/I");
  bookBranch("numMinHFTower5", &numMinHFTower5, "numMinHFTower5/I");
  */
  for (const auto& name : branchesToKeep_)
    if (!thi_->GetBranch(name.c_str()))
      throw cms::Exception("Configuration") << "branchesToKeep requested an unavailable branch: " << name;

}

// ------------ method called once each job just after ending the event loop  ------------
void HiEvtAnalyzer::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HiEvtAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiEvtAnalyzer);
