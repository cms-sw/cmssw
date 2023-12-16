// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripApvGainInspector
//
/*
 *\class SiStripApvGainInspector SiStripApvGainInspector.cc CalibTracker/SiStripChannelGain/plugins/SiStripApvGainInspector.cc

 Description: This module allows redo the per-APV gain fits with different PDFs (landau, landau + gaus convolution, etc.) starting from the Charge vs APV index plot produced in the SiStrip G2 APV gain PCL workflow. It is possible to inspect the 1D charge distributions for certain APVs after fitting by means of specifying them via the parameter selectedModules.

 Implementation: largely based off CalibTracker/SiStripChannelGain/src/SiStripGainsPCLHarvester.cc

*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 05 Jun 2018 15:46:15 GMT
//
//

// system include files
#include <cmath> /* log */
#include <memory>

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondTools/SiStrip/interface/SiStripMiscalibrateHelper.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

// ROOT includes
#include "TStyle.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2S.h"
#include "TProfile.h"
#include "TF1.h"

//
// class declaration
//
class SiStripApvGainInspector : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripApvGainInspector(const edm::ParameterSet&);
  ~SiStripApvGainInspector() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void checkBookAPVColls(const edm::EventSetup& es);
  void checkAndRetrieveTopology(const edm::EventSetup& setup);
  bool isGoodLandauFit(double* FitResults);
  void getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange = 50, double HighRange = 5400);
  void getPeakOfLanGau(TH1* InputHisto, double* FitResults, double LowRange = 50, double HighRange = 5400);
  void doFakeFit(TH1* InputHisto, double* FitResults);
  void getPeakOfLandauAroundMax(TH1* InputHisto, double* FitResults, double LowRange = 100, double HighRange = 100);
  static double langaufun(Double_t* x, Double_t* par);
  void storeOnTree(TFileService* tfs);
  void makeNicePlotStyle(TH1F* plot);
  std::unique_ptr<SiStripApvGain> getNewObject();
  std::map<std::string, TH1*> bookQualityMonitor(const TFileDirectory& dir);
  void fillQualityMonitor();

  void inline fill1D(std::map<std::string, TH1*>& h, const std::string& s, double x) {
    if (h.count(s) == 0) {
      edm::LogWarning("SiStripApvGainInspector") << "Trying to fill non-existing Histogram named " << s << std::endl;
      return;
    }
    h[s]->Fill(x);
  }

  void inline fill2D(std::map<std::string, TH1*>& h, const std::string& s, double x, double y) {
    if (h.count(s) == 0) {
      edm::LogWarning("SiStripApvGainInspector") << "Trying to fill non-existing Histogram named " << s << std::endl;
      return;
    }
    h[s]->Fill(x, y);
  }

  // ----------member data ---------------------------
  enum fitMode { landau = 1, landauAroundMax = 2, landauGauss = 3, fake = 4 };

  const std::vector<std::string> fitModeStrings = {
      "",  // Enum values start from 1, so index 0 is empty or can be used as "invalid"
      "landau",
      "landauAroundMax",
      "landauGauss",
      "fake"};

  inline bool isValidMode(int mode) const {
    return mode == landau || mode == landauAroundMax || mode == landauGauss || mode == fake;
  }

  const edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  TFileService* tfs;

  // map the APV ids to the charge plots
  std::map<std::pair<unsigned char, uint32_t>, TH1F*> histoMap_;

  edm::ESHandle<TrackerGeometry> tkGeom_;
  const TrackerGeometry* bareTkGeomPtr_;  // ugly hack to fill APV colls only once, but checks
  const TrackerTopology* tTopo_;

  int NStripAPVs;
  int NPixelDets;

  unsigned int GOOD;
  unsigned int BAD;
  unsigned int MASKED;

  std::vector<std::shared_ptr<stAPVGain>> APVsCollOrdered;
  std::unordered_map<unsigned int, std::shared_ptr<stAPVGain>> APVsColl;

  const TH2F* Charge_Vs_Index;
  TFile* fin;
  fitMode fitMode_;  // Declare the enum variable
  const std::string filename_;
  double minNrEntries;
  std::vector<unsigned int> wantedmods;

  std::unique_ptr<TrackerMap> ratio_map;
  std::unique_ptr<TrackerMap> old_payload_map;
  std::unique_ptr<TrackerMap> new_payload_map;
  std::unique_ptr<TrackerMap> mpv_map;
  std::unique_ptr<TrackerMap> mpv_err_map;
  std::unique_ptr<TrackerMap> entries_map;
  std::unique_ptr<TrackerMap> fitChi2_map;

  std::map<std::string, TH1*> hControl;
};

//
// constructors and destructor
//
SiStripApvGainInspector::SiStripApvGainInspector(const edm::ParameterSet& iConfig)
    : gainToken_(esConsumes()),
      qualityToken_(esConsumes()),
      tkGeomToken_(esConsumes()),
      tTopoToken_(esConsumes()),
      bareTkGeomPtr_(nullptr),
      tTopo_(nullptr),
      GOOD(0),
      BAD(0),
      filename_(iConfig.getUntrackedParameter<std::string>("inputFile")),
      minNrEntries(iConfig.getUntrackedParameter<double>("minNrEntries", 20)),
      wantedmods(iConfig.getUntrackedParameter<std::vector<unsigned int>>("selectedModules")) {
  usesResource(TFileService::kSharedResource);
  usesResource(cond::service::PoolDBOutputService::kSharedResource);

  sort(wantedmods.begin(), wantedmods.end());

  edm::LogInfo("SelectedModules") << "Selected module list";
  for (std::vector<unsigned int>::const_iterator mod = wantedmods.begin(); mod != wantedmods.end(); mod++) {
    edm::LogVerbatim("SelectedModules") << *mod;
  }

  int modeValue = iConfig.getParameter<int>("fitMode");
  if (!isValidMode(modeValue)) {
    throw std::invalid_argument("Invalid value provided for 'fitMode'");
  } else {
    edm::LogPrint("SiStripApvGainInspector") << "Chosen fitting mode: " << fitModeStrings[modeValue];
  }

  fitMode_ = static_cast<fitMode>(modeValue);

  //now do what ever initialization is needed
  fin = TFile::Open(filename_.c_str(), "READ");
  Charge_Vs_Index = (TH2F*)fin->Get("DQMData/Run 999999/AlCaReco/Run summary/SiStripGainsAAG/Charge_Vs_Index_AagBunch");

  ratio_map = std::make_unique<TrackerMap>("ratio");
  ratio_map->setTitle("Average by module of the G2 Gain payload ratio (new/old)");
  ratio_map->setPalette(1);

  new_payload_map = std::make_unique<TrackerMap>("new_payload");
  new_payload_map->setTitle("Tracker Map of Updated G2 Gain payload averaged by module");
  new_payload_map->setPalette(1);

  old_payload_map = std::make_unique<TrackerMap>("old_payload");
  old_payload_map->setTitle("Tracker Map of Starting G2 Gain Payload averaged by module");
  old_payload_map->setPalette(1);

  // fit quality maps

  mpv_map = std::make_unique<TrackerMap>("MPV");
  mpv_map->setTitle("Landau Fit MPV average value per module [ADC counts/mm]");
  mpv_map->setPalette(1);

  mpv_err_map = std::make_unique<TrackerMap>("MPVerr");
  mpv_err_map->setTitle("Landau Fit MPV average error per module [ADC counts/mm]");
  mpv_err_map->setPalette(1);

  entries_map = std::make_unique<TrackerMap>("Entries");
  entries_map->setTitle("log_{10}(entries) average per module");
  entries_map->setPalette(1);

  fitChi2_map = std::make_unique<TrackerMap>("FitChi2");
  fitChi2_map->setTitle("log_{10}(Fit #chi^{2}/ndf) average per module");
  fitChi2_map->setPalette(1);
}

// do anything here that needs to be done at desctruction time
// (e.g. close files, deallocate resources etc.)
SiStripApvGainInspector::~SiStripApvGainInspector() {
  fin->Close();
  delete fin;
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripApvGainInspector::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  this->checkBookAPVColls(iSetup);  // check whether APV colls are booked and do so if not yet done
  this->checkAndRetrieveTopology(iSetup);

  edm::ESHandle<SiStripGain> gainHandle = iSetup.getHandle(gainToken_);
  if (!gainHandle.isValid()) {
    edm::LogError("SiStripApvGainInspector") << "gainHandle is not valid\n";
    exit(0);
  }

  edm::ESHandle<SiStripQuality> SiStripQuality_ = iSetup.getHandle(qualityToken_);

  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    std::shared_ptr<stAPVGain> APV = APVsCollOrdered[a];

    if (APV->SubDet == PixelSubdetector::PixelBarrel || APV->SubDet == PixelSubdetector::PixelEndcap)
      continue;

    APV->isMasked = SiStripQuality_->IsApvBad(APV->DetId, APV->APVId);

    if (gainHandle->getNumberOfTags() != 2) {
      edm::LogError("SiStripApvGainInspector") << "NUMBER OF GAIN TAG IS EXPECTED TO BE 2\n";
      fflush(stdout);
      exit(0);
    };
    float newPreviousGain = gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 1), 1);
    if (APV->PreviousGain != 1 and newPreviousGain != APV->PreviousGain)
      edm::LogWarning("SiStripApvGainInspector") << "WARNING: ParticleGain in the global tag changed\n";
    APV->PreviousGain = newPreviousGain;

    float newPreviousGainTick = gainHandle->getApvGain(APV->APVId, gainHandle->getRange(APV->DetId, 0), 0);
    if (APV->PreviousGainTick != 1 and newPreviousGainTick != APV->PreviousGainTick) {
      edm::LogWarning("SiStripApvGainInspector")
          << "WARNING: TickMarkGain in the global tag changed\n"
          << std::endl
          << " APV->SubDet: " << APV->SubDet << " APV->APVId:" << APV->APVId << std::endl
          << " APV->PreviousGainTick: " << APV->PreviousGainTick << " newPreviousGainTick: " << newPreviousGainTick
          << std::endl;
    }
    APV->PreviousGainTick = newPreviousGainTick;
  }

  unsigned int I = 0;
  TH1F* Proj = nullptr;
  double FitResults[6];
  double MPVmean = 300;

  if (Charge_Vs_Index == nullptr) {
    edm::LogError("SiStripGainsPCLHarvester") << "Harvesting: could not find input histogram " << std::endl;
    return;
  }

  printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
  printf("Fitting Charge Distribution  :");
  int TreeStep = APVsColl.size() / 50;

  for (auto it = APVsColl.begin(); it != APVsColl.end(); it++, I++) {
    if (I % TreeStep == 0) {
      printf(".");
      fflush(stdout);
    }
    std::shared_ptr<stAPVGain> APV = it->second;
    if (APV->Bin < 0)
      APV->Bin = Charge_Vs_Index->GetXaxis()->FindBin(APV->Index);

    Proj = (TH1F*)(Charge_Vs_Index->ProjectionY(
        "", Charge_Vs_Index->GetXaxis()->FindBin(APV->Index), Charge_Vs_Index->GetXaxis()->FindBin(APV->Index), "e"));
    if (!Proj)
      continue;

    switch (fitMode_) {
      case landau:
        getPeakOfLandau(Proj, FitResults);
        break;
      case landauAroundMax:
        getPeakOfLandauAroundMax(Proj, FitResults);
        break;
      case landauGauss:
        getPeakOfLanGau(Proj, FitResults);
        break;
      case fake:
        doFakeFit(Proj, FitResults);
        break;
      default:
        throw std::invalid_argument("Invalid value provided for 'fitMode'");
    }

    APV->FitMPV = FitResults[0];
    APV->FitMPVErr = FitResults[1];
    APV->FitWidth = FitResults[2];
    APV->FitWidthErr = FitResults[3];
    APV->FitChi2 = FitResults[4];
    APV->FitNorm = FitResults[5];
    APV->NEntries = Proj->GetEntries();

    for (const auto& mod : wantedmods) {
      if (mod == APV->DetId) {
        edm::LogInfo("ModuleFound") << " module " << mod << " found! Storing... " << std::endl;
        histoMap_[std::make_pair(APV->APVId, APV->DetId)] = (TH1F*)Proj->Clone(Form("hClone_%s", Proj->GetName()));
      }
    }

    if (isGoodLandauFit(FitResults)) {
      APV->Gain = APV->FitMPV / MPVmean;
      if (APV->SubDet > 2)
        GOOD++;
    } else {
      APV->Gain = APV->PreviousGain;
      if (APV->SubDet > 2)
        BAD++;
    }
    if (APV->Gain <= 0)
      APV->Gain = 1;

    delete Proj;
  }
  printf("\n");
}

//********************************************************************************//
// ------------ method called once each job just before starting event loop  ------------
void SiStripApvGainInspector::checkBookAPVColls(const edm::EventSetup& es) {
  tkGeom_ = es.getHandle(tkGeomToken_);
  const TrackerGeometry* newBareTkGeomPtr = &(*tkGeom_);
  if (newBareTkGeomPtr == bareTkGeomPtr_)
    return;  // already filled APVColls, nothing changed

  if (!bareTkGeomPtr_) {  // pointer not yet set: called the first time => fill the APVColls
    auto const& Det = newBareTkGeomPtr->dets();

    unsigned int Index = 0;

    for (unsigned int i = 0; i < Det.size(); i++) {
      DetId Detid = Det[i]->geographicalId();
      int SubDet = Detid.subdetId();

      if (SubDet == StripSubdetector::TIB || SubDet == StripSubdetector::TID || SubDet == StripSubdetector::TOB ||
          SubDet == StripSubdetector::TEC) {
        auto DetUnit = dynamic_cast<const StripGeomDetUnit*>(Det[i]);
        if (!DetUnit)
          continue;

        const StripTopology& Topo = DetUnit->specificTopology();
        unsigned int NAPV = Topo.nstrips() / 128;

        for (unsigned int j = 0; j < NAPV; j++) {
          auto APV = std::make_shared<stAPVGain>();
          APV->Index = Index;
          APV->Bin = -1;
          APV->DetId = Detid.rawId();
          APV->APVId = j;
          APV->SubDet = SubDet;
          APV->FitMPV = -1;
          APV->FitMPVErr = -1;
          APV->FitWidth = -1;
          APV->FitWidthErr = -1;
          APV->FitChi2 = -1;
          APV->FitNorm = -1;
          APV->Gain = -1;
          APV->PreviousGain = 1;
          APV->PreviousGainTick = 1;
          APV->x = DetUnit->position().basicVector().x();
          APV->y = DetUnit->position().basicVector().y();
          APV->z = DetUnit->position().basicVector().z();
          APV->Eta = DetUnit->position().basicVector().eta();
          APV->Phi = DetUnit->position().basicVector().phi();
          APV->R = DetUnit->position().basicVector().transverse();
          APV->Thickness = DetUnit->surface().bounds().thickness();
          APV->NEntries = 0;
          APV->isMasked = false;

          APVsCollOrdered.push_back(APV);
          APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
          Index++;
          NStripAPVs++;
        }  // loop on APVs
      }    // if is Strips
    }      // loop on dets

    for (unsigned int i = 0; i < Det.size();
         i++) {  //Make two loop such that the Pixel information is added at the end --> make transition simpler
      DetId Detid = Det[i]->geographicalId();
      int SubDet = Detid.subdetId();
      if (SubDet == PixelSubdetector::PixelBarrel || SubDet == PixelSubdetector::PixelEndcap) {
        auto DetUnit = dynamic_cast<const PixelGeomDetUnit*>(Det[i]);
        if (!DetUnit)
          continue;

        const PixelTopology& Topo = DetUnit->specificTopology();
        unsigned int NROCRow = Topo.nrows() / (80.);
        unsigned int NROCCol = Topo.ncolumns() / (52.);

        for (unsigned int j = 0; j < NROCRow; j++) {
          for (unsigned int i = 0; i < NROCCol; i++) {
            auto APV = std::make_shared<stAPVGain>();
            APV->Index = Index;
            APV->Bin = -1;
            APV->DetId = Detid.rawId();
            APV->APVId = (j << 3 | i);
            APV->SubDet = SubDet;
            APV->FitMPV = -1;
            APV->FitMPVErr = -1;
            APV->FitWidth = -1;
            APV->FitWidthErr = -1;
            APV->FitChi2 = -1;
            APV->Gain = -1;
            APV->PreviousGain = 1;
            APV->PreviousGainTick = 1;
            APV->x = DetUnit->position().basicVector().x();
            APV->y = DetUnit->position().basicVector().y();
            APV->z = DetUnit->position().basicVector().z();
            APV->Eta = DetUnit->position().basicVector().eta();
            APV->Phi = DetUnit->position().basicVector().phi();
            APV->R = DetUnit->position().basicVector().transverse();
            APV->Thickness = DetUnit->surface().bounds().thickness();
            APV->isMasked = false;  //SiPixelQuality_->IsModuleBad(Detid.rawId());
            APV->NEntries = 0;

            APVsCollOrdered.push_back(APV);
            APVsColl[(APV->DetId << 4) | APV->APVId] = APV;
            Index++;
            NPixelDets++;

          }  // loop on ROC cols
        }    // loop on ROC rows
      }      // if Pixel
    }        // loop on Dets
  }          //if (!bareTkGeomPtr_) ...
  bareTkGeomPtr_ = newBareTkGeomPtr;
}

void SiStripApvGainInspector::storeOnTree(TFileService* tfs) {
  unsigned int tree_Index;
  unsigned int tree_Bin;
  unsigned int tree_DetId;
  unsigned char tree_APVId;
  unsigned char tree_SubDet;
  float tree_x;
  float tree_y;
  float tree_z;
  float tree_Eta;
  float tree_R;
  float tree_Phi;
  float tree_Thickness;
  float tree_FitMPV;
  float tree_FitMPVErr;
  float tree_FitWidth;
  float tree_FitWidthErr;
  float tree_FitChi2NDF;
  float tree_FitNorm;
  double tree_Gain;
  double tree_PrevGain;
  double tree_PrevGainTick;
  double tree_NEntries;
  bool tree_isMasked;

  TTree* MyTree;
  MyTree = tfs->make<TTree>("APVGain", "APVGain");
  MyTree->Branch("Index", &tree_Index, "Index/i");
  MyTree->Branch("Bin", &tree_Bin, "Bin/i");
  MyTree->Branch("DetId", &tree_DetId, "DetId/i");
  MyTree->Branch("APVId", &tree_APVId, "APVId/b");
  MyTree->Branch("SubDet", &tree_SubDet, "SubDet/b");
  MyTree->Branch("x", &tree_x, "x/F");
  MyTree->Branch("y", &tree_y, "y/F");
  MyTree->Branch("z", &tree_z, "z/F");
  MyTree->Branch("Eta", &tree_Eta, "Eta/F");
  MyTree->Branch("R", &tree_R, "R/F");
  MyTree->Branch("Phi", &tree_Phi, "Phi/F");
  MyTree->Branch("Thickness", &tree_Thickness, "Thickness/F");
  MyTree->Branch("FitMPV", &tree_FitMPV, "FitMPV/F");
  MyTree->Branch("FitMPVErr", &tree_FitMPVErr, "FitMPVErr/F");
  MyTree->Branch("FitWidth", &tree_FitWidth, "FitWidth/F");
  MyTree->Branch("FitWidthErr", &tree_FitWidthErr, "FitWidthErr/F");
  MyTree->Branch("FitChi2NDF", &tree_FitChi2NDF, "FitChi2NDF/F");
  MyTree->Branch("FitNorm", &tree_FitNorm, "FitNorm/F");
  MyTree->Branch("Gain", &tree_Gain, "Gain/D");
  MyTree->Branch("PrevGain", &tree_PrevGain, "PrevGain/D");
  MyTree->Branch("PrevGainTick", &tree_PrevGainTick, "PrevGainTick/D");
  MyTree->Branch("NEntries", &tree_NEntries, "NEntries/D");
  MyTree->Branch("isMasked", &tree_isMasked, "isMasked/O");

  uint32_t cachedId(0);
  SiStripMiscalibrate::Entry gain_ratio;
  SiStripMiscalibrate::Entry o_gain;
  SiStripMiscalibrate::Entry n_gain;
  SiStripMiscalibrate::Entry mpv;
  SiStripMiscalibrate::Entry mpv_err;
  SiStripMiscalibrate::Entry entries;
  SiStripMiscalibrate::Entry fitChi2;

  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    std::shared_ptr<stAPVGain> APV = APVsCollOrdered[a];
    if (APV == nullptr)
      continue;
    //     printf(      "%i | %i | PreviousGain = %7.5f NewGain = %7.5f (#clusters=%8.0f)\n", APV->DetId,APV->APVId,APV->PreviousGain,APV->Gain, APV->NEntries);
    //fprintf(Gains,"%i | %i | PreviousGain = %7.5f(tick) x %7.5f(particle) NewGain (particle) = %7.5f (#clusters=%8.0f)\n", APV->DetId,APV->APVId,APV->PreviousGainTick, APV->PreviousGain,APV->Gain, APV->NEntries);

    // do not fill the Pixel
    if (APV->SubDet == PixelSubdetector::PixelBarrel || APV->SubDet == PixelSubdetector::PixelEndcap)
      continue;

    tree_Index = APV->Index;
    tree_Bin = Charge_Vs_Index->GetXaxis()->FindBin(APV->Index);
    tree_DetId = APV->DetId;
    tree_APVId = APV->APVId;
    tree_SubDet = APV->SubDet;
    tree_x = APV->x;
    tree_y = APV->y;
    tree_z = APV->z;
    tree_Eta = APV->Eta;
    tree_R = APV->R;
    tree_Phi = APV->Phi;
    tree_Thickness = APV->Thickness;
    tree_FitMPV = APV->FitMPV;
    tree_FitMPVErr = APV->FitMPVErr;
    tree_FitWidth = APV->FitWidth;
    tree_FitWidthErr = APV->FitWidthErr;
    tree_FitChi2NDF = APV->FitChi2;
    tree_FitNorm = APV->FitNorm;
    tree_Gain = APV->Gain;
    tree_PrevGain = APV->PreviousGain;
    tree_PrevGainTick = APV->PreviousGainTick;
    tree_NEntries = APV->NEntries;
    tree_isMasked = APV->isMasked;

    // flush the counters
    if (cachedId != 0 && tree_DetId != cachedId) {
      //ratio_map->fill(cachedId,gain_ratio.mean());
      ratio_map->fill(cachedId, o_gain.mean() / n_gain.mean());
      old_payload_map->fill(cachedId, o_gain.mean());
      new_payload_map->fill(cachedId, n_gain.mean());

      if (entries.mean() > 0) {
        mpv_map->fill(cachedId, mpv.mean());
        mpv_err_map->fill(cachedId, mpv_err.mean());
        entries_map->fill(cachedId, log10(entries.mean()));
        if (fitChi2.mean() > 0) {
          fitChi2_map->fill(cachedId, log10(fitChi2.mean()));
        } else {
          fitChi2_map->fill(cachedId, -1);
        }
      }

      gain_ratio.reset();
      o_gain.reset();
      n_gain.reset();

      mpv.reset();
      mpv_err.reset();
      entries.reset();
      fitChi2.reset();
    }

    cachedId = tree_DetId;
    gain_ratio.add(tree_PrevGain / tree_Gain);
    o_gain.add(tree_PrevGain);
    n_gain.add(tree_Gain);
    mpv.add(tree_FitMPV);
    mpv_err.add(tree_FitMPVErr);
    entries.add(tree_NEntries);
    fitChi2.add(tree_FitChi2NDF);

    if (tree_DetId == 402673324) {
      printf("%i | %i : %f --> %f  (%f)\n", tree_DetId, tree_APVId, tree_PrevGain, tree_Gain, tree_NEntries);
    }

    MyTree->Fill();
  }
}

//********************************************************************************//
void SiStripApvGainInspector::checkAndRetrieveTopology(const edm::EventSetup& setup) {
  if (!tTopo_) {
    edm::ESHandle<TrackerTopology> TopoHandle = setup.getHandle(tTopoToken_);
    tTopo_ = TopoHandle.product();
  }
}

//********************************************************************************//
void SiStripApvGainInspector::getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange) {
  FitResults[0] = -0.5;  //MPV
  FitResults[1] = 0;     //MPV error
  FitResults[2] = -0.5;  //Width
  FitResults[3] = 0;     //Width error
  FitResults[4] = -0.5;  //Fit Chi2/NDF
  FitResults[5] = 0;     //Normalization

  if (InputHisto->GetEntries() < minNrEntries)
    return;

  // perform fit with standard landau
  TF1 MyLandau("MyLandau", "landau", LowRange, HighRange);
  MyLandau.SetParameter(1, 300);
  InputHisto->Fit(&MyLandau, "QR WW");

  // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
  FitResults[0] = MyLandau.GetParameter(1);                     //MPV
  FitResults[1] = MyLandau.GetParError(1);                      //MPV error
  FitResults[2] = MyLandau.GetParameter(2);                     //Width
  FitResults[3] = MyLandau.GetParError(2);                      //Width error
  FitResults[4] = MyLandau.GetChisquare() / MyLandau.GetNDF();  //Fit Chi2/NDF
  FitResults[5] = MyLandau.GetParameter(0);
}

void SiStripApvGainInspector::doFakeFit(TH1* InputHisto, double* FitResults) {
  FitResults[0] = -0.5;  //MPV
  FitResults[1] = 0;     //MPV error
  FitResults[2] = -0.5;  //Width
  FitResults[3] = 0;     //Width error
  FitResults[4] = -0.5;  //Fit Chi2/NDF
  FitResults[5] = 0;     //Normalization
}

//********************************************************************************//
double SiStripApvGainInspector::langaufun(Double_t* x, Double_t* par)
//********************************************************************************//
{
  //Fit parameters:
  //par[0]=Width (scale) parameter of Landau density
  //par[1]=Most Probable (MP, location) parameter of Landau density
  //par[2]=Total area (integral -inf to inf, normalization constant)
  //par[3]=Width (sigma) of convoluted Gaussian function
  //
  //In the Landau distribution (represented by the CERNLIB approximation),
  //the maximum is located at x=-0.22278298 with the location parameter=0.
  //This shift is corrected within this function, so that the actual
  //maximum is identical to the MP parameter.

  // Numeric constants
  Double_t invsq2pi = 0.3989422804014;  // (2 pi)^(-1/2)
  Double_t mpshift = -0.22278298;       // Landau maximum location

  // Control constants
  Double_t np = 100.0;  // number of convolution steps
  Double_t sc = 5.0;    // convolution extends to +-sc Gaussian sigmas

  // Variables
  Double_t xx;
  Double_t mpc;
  Double_t fland;
  Double_t sum = 0.0;
  Double_t xlow, xupp;
  Double_t step;
  Double_t i;

  // MP shift correction
  mpc = par[1] - mpshift * par[0];

  // Range of convolution integral
  xlow = x[0] - sc * par[3];
  xupp = x[0] + sc * par[3];

  step = (xupp - xlow) / np;

  // Convolution integral of Landau and Gaussian by sum
  for (i = 1.0; i <= np / 2; i++) {
    xx = xlow + (i - .5) * step;
    fland = TMath::Landau(xx, mpc, par[0]) / par[0];
    sum += fland * TMath::Gaus(x[0], xx, par[3]);

    xx = xupp - (i - .5) * step;
    fland = TMath::Landau(xx, mpc, par[0]) / par[0];
    sum += fland * TMath::Gaus(x[0], xx, par[3]);
  }

  return (par[2] * step * sum * invsq2pi / par[3]);
}

//********************************************************************************//
void SiStripApvGainInspector::getPeakOfLanGau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange) {
  FitResults[0] = -0.5;  //MPV
  FitResults[1] = 0;     //MPV error
  FitResults[2] = -0.5;  //Width
  FitResults[3] = 0;     //Width error
  FitResults[4] = -0.5;  //Fit Chi2/NDF
  FitResults[5] = 0;     //Normalization

  if (InputHisto->GetEntries() < minNrEntries)
    return;

  // perform fit with standard landau
  TF1 MyLandau("MyLandau", "landau", LowRange, HighRange);
  MyLandau.SetParameter(1, 300);
  InputHisto->Fit(&MyLandau, "QR WW");

  double startvalues[4] = {100, 300, 10000, 100};
  double parlimitslo[4] = {0, 250, 10, 0};
  double parlimitshi[4] = {200, 350, 1000000, 200};

  TF1 MyLangau("MyLanGau", langaufun, LowRange, HighRange, 4);

  MyLangau.SetParameters(startvalues);
  MyLangau.SetParNames("Width", "MP", "Area", "GSigma");

  for (unsigned int i = 0; i < 4; i++) {
    MyLangau.SetParLimits(i, parlimitslo[i], parlimitshi[i]);
  }

  InputHisto->Fit("MyLanGau", "QRB0");  // fit within specified range, use ParLimits, do not plot

  // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
  FitResults[0] = MyLangau.GetParameter(1);                     //MPV
  FitResults[1] = MyLangau.GetParError(1);                      //MPV error
  FitResults[2] = MyLangau.GetParameter(0);                     //Width
  FitResults[3] = MyLangau.GetParError(0);                      //Width error
  FitResults[4] = MyLangau.GetChisquare() / MyLangau.GetNDF();  //Fit Chi2/NDF
  FitResults[5] = MyLangau.GetParameter(3);
}

//********************************************************************************//
void SiStripApvGainInspector::getPeakOfLandauAroundMax(TH1* InputHisto,
                                                       double* FitResults,
                                                       double LowRange,
                                                       double HighRange) {
  FitResults[0] = -0.5;  //MPV
  FitResults[1] = 0;     //MPV error
  FitResults[2] = -0.5;  //Width
  FitResults[3] = 0;     //Width error
  FitResults[4] = -0.5;  //Fit Chi2/NDF
  FitResults[5] = 0;     //Normalization

  if (InputHisto->GetEntries() < minNrEntries)
    return;

  int maxbin = InputHisto->GetMaximumBin();
  int maxbin2 = -9999.;

  if (InputHisto->GetBinContent(maxbin - 1) > InputHisto->GetBinContent(maxbin + 1)) {
    maxbin2 = maxbin - 1;
  } else {
    maxbin2 = maxbin + 1;
  }

  float maxbincenter = (InputHisto->GetBinCenter(maxbin) + InputHisto->GetBinCenter(maxbin2)) / 2;

  TF1 MyLandau("MyLandau", "[2]*TMath::Landau(x,[0],[1],0)", maxbincenter - LowRange, maxbincenter + HighRange);
  // TF1 MyLandau("MyLandau", "landau", LowRange, HighRange);
  // MyLandau.SetParameter(1, 300);
  InputHisto->Fit(&MyLandau, "QR WW");

  MyLandau.SetParameter(0, maxbincenter);
  MyLandau.SetParameter(1, maxbincenter / 10.);
  MyLandau.SetParameter(2, InputHisto->GetMaximum());

  float mpv = MyLandau.GetParameter(1);
  MyLandau.SetParameter(1, mpv);
  //InputHisto->Rebin(3);
  InputHisto->Fit(&MyLandau, "QOR", "", mpv - 50, mpv + 100);

  InputHisto->Fit(&MyLandau, "QOR", "", maxbincenter - LowRange, maxbincenter + HighRange);
  InputHisto->Fit(&MyLandau, "QOR", "", maxbincenter - LowRange, maxbincenter + HighRange);

  // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
  FitResults[0] = MyLandau.GetParameter(1);                     //MPV
  FitResults[1] = MyLandau.GetParError(1);                      //MPV error
  FitResults[2] = MyLandau.GetParameter(2);                     //Width
  FitResults[3] = MyLandau.GetParError(2);                      //Width error
  FitResults[4] = MyLandau.GetChisquare() / MyLandau.GetNDF();  //Fit Chi2/NDF
  FitResults[5] = MyLandau.GetParameter(0);
}

//********************************************************************************//
bool SiStripApvGainInspector::isGoodLandauFit(double* FitResults) {
  if (FitResults[0] <= 0)
    return false;
  //   if(FitResults[1] > MaxMPVError   )return false;
  //   if(FitResults[4] > MaxChi2OverNDF)return false;
  return true;
}

/*--------------------------------------------------------------------*/
void SiStripApvGainInspector::makeNicePlotStyle(TH1F* plot)
/*--------------------------------------------------------------------*/
{
  plot->GetXaxis()->CenterTitle(true);
  plot->GetYaxis()->CenterTitle(true);
  plot->GetXaxis()->SetTitleFont(42);
  plot->GetYaxis()->SetTitleFont(42);
  plot->GetXaxis()->SetTitleSize(0.05);
  plot->GetYaxis()->SetTitleSize(0.05);
  plot->GetXaxis()->SetTitleOffset(0.9);
  plot->GetYaxis()->SetTitleOffset(1.3);
  plot->GetXaxis()->SetLabelFont(42);
  plot->GetYaxis()->SetLabelFont(42);
  plot->GetYaxis()->SetLabelSize(.05);
  plot->GetXaxis()->SetLabelSize(.05);
}

//********************************************************************************//
std::unique_ptr<SiStripApvGain> SiStripApvGainInspector::getNewObject() {
  std::unique_ptr<SiStripApvGain> obj = std::make_unique<SiStripApvGain>();

  std::vector<float> theSiStripVector;
  unsigned int PreviousDetId = 0;
  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    std::shared_ptr<stAPVGain> APV = APVsCollOrdered[a];
    if (APV == nullptr) {
      printf("Bug\n");
      continue;
    }
    if (APV->SubDet <= 2)
      continue;
    if (APV->DetId != PreviousDetId) {
      if (!theSiStripVector.empty()) {
        SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
        if (!obj->put(PreviousDetId, range))
          printf("Bug to put detId = %i\n", PreviousDetId);
      }
      theSiStripVector.clear();
      PreviousDetId = APV->DetId;
    }
    theSiStripVector.push_back(APV->Gain);

    LogDebug("SiStripGainsPCLHarvester") << " DetId: " << APV->DetId << " APV:   " << APV->APVId
                                         << " Gain:  " << APV->Gain << std::endl;
  }
  if (!theSiStripVector.empty()) {
    SiStripApvGain::Range range(theSiStripVector.begin(), theSiStripVector.end());
    if (!obj->put(PreviousDetId, range))
      printf("Bug to put detId = %i\n", PreviousDetId);
  }

  return obj;
}

// ------------ method called once each job just before starting event loop  ------------
void SiStripApvGainInspector::beginJob() {
  TFileDirectory control_dir = tfs->mkdir("Control");
  //DA.cd();
  hControl = this->bookQualityMonitor(control_dir);
}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripApvGainInspector::endJob() {
  edm::LogVerbatim("SelectedModules") << "Selected APVs:" << histoMap_.size() << std::endl;
  for (const auto& plot : histoMap_) {
    TCanvas* c1 = new TCanvas(Form("c1_%i_%i", plot.first.second, plot.first.first),
                              Form("c1_%i_%i", plot.first.second, plot.first.first),
                              800,
                              600);
    // Define common things for the different fits

    gStyle->SetOptFit(1011);
    c1->Clear();

    c1->SetLeftMargin(0.15);
    c1->SetRightMargin(0.10);
    plot.second->SetTitle(Form("Cluster Charge (%i,%i)", plot.first.second, plot.first.first));
    plot.second->GetXaxis()->SetTitle("Normalized Cluster Charge [ADC counts/mm]");
    plot.second->GetYaxis()->SetTitle("On-track clusters");
    plot.second->GetXaxis()->SetRangeUser(0., 1000.);

    this->makeNicePlotStyle(plot.second);
    plot.second->Draw();
    edm::LogVerbatim("SelectedModules") << " DetId: " << plot.first.second << " (" << plot.first.first << ")"
                                        << std::endl;
    ;

    c1->Print(Form("c1_%i_%i.png", plot.first.second, plot.first.first));
    c1->Print(Form("c1_%i_%i.pdf", plot.first.second, plot.first.first));
  }

  tfs = edm::Service<TFileService>().operator->();
  storeOnTree(tfs);

  auto range = SiStripMiscalibrate::getTruncatedRange(ratio_map.get());

  ratio_map->save(true, range.first, range.second, "G2_gain_ratio_map.pdf");
  ratio_map->save(true, range.first, range.second, "G2_gain_ratio_map.png");

  range = SiStripMiscalibrate::getTruncatedRange(old_payload_map.get());

  old_payload_map->save(true, range.first, range.second, "starting_G2_gain_payload_map.pdf");
  old_payload_map->save(true, range.first, range.second, "starting_G2_gain_payload_map.png");

  range = SiStripMiscalibrate::getTruncatedRange(new_payload_map.get());

  new_payload_map->save(true, range.first, range.second, "new_G2_gain_payload_map.pdf");
  new_payload_map->save(true, range.first, range.second, "new_G2_gain_payload_map.png");

  mpv_map->save(true, 250, 350., "mpv_map.pdf");
  mpv_map->save(true, 250, 350., "mpv_map.png");

  mpv_err_map->save(true, 0., 3., "mpv_err_map.pdf");
  mpv_err_map->save(true, 0., 3., "mpv_err_map.png");

  entries_map->save(true, 0, 0, "entries_map.pdf");
  entries_map->save(true, 0, 0, "entries_map.png");

  fitChi2_map->save(true, 0., 0., "fitChi2_map.pdf");
  fitChi2_map->save(true, 0., 0., "fitChi2_map.png");

  fillQualityMonitor();

  std::unique_ptr<SiStripApvGain> theAPVGains = this->getNewObject();

  // write out the APVGains record
  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (poolDbService.isAvailable())
    poolDbService->writeOneIOV(theAPVGains.get(), poolDbService->currentTime(), "SiStripApvGainRcd");
  else
    throw std::runtime_error("PoolDBService required.");
}

std::map<std::string, TH1*> SiStripApvGainInspector::bookQualityMonitor(const TFileDirectory& dir) {
  int MPVbin = 300;
  float MPVmin = 0.;
  float MPVmax = 600.;

  TH1F::SetDefaultSumw2(kTRUE);
  std::map<std::string, TH1*> h;

  h["MPV_Vs_EtaTIB"] = dir.make<TH2F>("MPVvsEtaTIB", "MPV vs Eta TIB", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_EtaTID"] = dir.make<TH2F>("MPVvsEtaTID", "MPV vs Eta TID", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_EtaTOB"] = dir.make<TH2F>("MPVvsEtaTOB", "MPV vs Eta TOB", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_EtaTEC"] = dir.make<TH2F>("MPVvsEtaTEC", "MPV vs Eta TEC", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_EtaTECthin"] = dir.make<TH2F>("MPVvsEtaTEC1", "MPV vs Eta TEC-thin", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_EtaTECthick"] =
      dir.make<TH2F>("MPVvsEtaTEC2", "MPV vs Eta TEC-thick", 50, -3.0, 3.0, MPVbin, MPVmin, MPVmax);

  h["MPV_Vs_PhiTIB"] = dir.make<TH2F>("MPVvsPhiTIB", "MPV vs Phi TIB", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_PhiTID"] = dir.make<TH2F>("MPVvsPhiTID", "MPV vs Phi TID", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_PhiTOB"] = dir.make<TH2F>("MPVvsPhiTOB", "MPV vs Phi TOB", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_PhiTEC"] = dir.make<TH2F>("MPVvsPhiTEC", "MPV vs Phi TEC", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_PhiTECthin"] =
      dir.make<TH2F>("MPVvsPhiTEC1", "MPV vs Phi TEC-thin ", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);
  h["MPV_Vs_PhiTECthick"] =
      dir.make<TH2F>("MPVvsPhiTEC2", "MPV vs Phi TEC-thick", 50, -3.4, 3.4, MPVbin, MPVmin, MPVmax);

  h["NoMPVfit"] = dir.make<TH2F>("NoMPVfit", "Modules with bad Landau Fit", 350, -350, 350, 240, 0, 120);
  h["NoMPVmasked"] = dir.make<TH2F>("NoMPVmasked", "Masked Modules", 350, -350, 350, 240, 0, 120);

  h["Gains"] = dir.make<TH1F>("Gains", "Gains", 300, 0, 2);
  h["MPVs"] = dir.make<TH1F>("MPVs", "MPVs", MPVbin, MPVmin, MPVmax);
  h["MPVs320"] = dir.make<TH1F>("MPV_320", "MPV 320 thickness", MPVbin, MPVmin, MPVmax);
  h["MPVs500"] = dir.make<TH1F>("MPV_500", "MPV 500 thickness", MPVbin, MPVmin, MPVmax);
  h["MPVsTIB"] = dir.make<TH1F>("MPV_TIB", "MPV TIB", MPVbin, MPVmin, MPVmax);
  h["MPVsTID"] = dir.make<TH1F>("MPV_TID", "MPV TID", MPVbin, MPVmin, MPVmax);
  h["MPVsTIDP"] = dir.make<TH1F>("MPV_TIDP", "MPV TIDP", MPVbin, MPVmin, MPVmax);
  h["MPVsTIDM"] = dir.make<TH1F>("MPV_TIDM", "MPV TIDM", MPVbin, MPVmin, MPVmax);
  h["MPVsTOB"] = dir.make<TH1F>("MPV_TOB", "MPV TOB", MPVbin, MPVmin, MPVmax);
  h["MPVsTEC"] = dir.make<TH1F>("MPV_TEC", "MPV TEC", MPVbin, MPVmin, MPVmax);
  h["MPVsTECP"] = dir.make<TH1F>("MPV_TECP", "MPV TECP", MPVbin, MPVmin, MPVmax);
  h["MPVsTECM"] = dir.make<TH1F>("MPV_TECM", "MPV TECM", MPVbin, MPVmin, MPVmax);
  h["MPVsTECthin"] = dir.make<TH1F>("MPV_TEC1", "MPV TEC thin", MPVbin, MPVmin, MPVmax);
  h["MPVsTECthick"] = dir.make<TH1F>("MPV_TEC2", "MPV TEC thick", MPVbin, MPVmin, MPVmax);
  h["MPVsTECP1"] = dir.make<TH1F>("MPV_TECP1", "MPV TECP thin ", MPVbin, MPVmin, MPVmax);
  h["MPVsTECP2"] = dir.make<TH1F>("MPV_TECP2", "MPV TECP thick", MPVbin, MPVmin, MPVmax);
  h["MPVsTECM1"] = dir.make<TH1F>("MPV_TECM1", "MPV TECM thin", MPVbin, MPVmin, MPVmax);
  h["MPVsTECM2"] = dir.make<TH1F>("MPV_TECM2", "MPV TECM thick", MPVbin, MPVmin, MPVmax);

  h["MPVError"] = dir.make<TH1F>("MPVError", "MPV Error", 150, 0, 150);
  h["MPVErrorVsMPV"] = dir.make<TH2F>("MPVErrorVsMPV", "MPV Error vs MPV", 300, 0, 600, 150, 0, 150);
  h["MPVErrorVsEta"] = dir.make<TH2F>("MPVErrorVsEta", "MPV Error vs Eta", 50, -3.0, 3.0, 150, 0, 150);
  h["MPVErrorVsPhi"] = dir.make<TH2F>("MPVErrorVsPhi", "MPV Error vs Phi", 50, -3.4, 3.4, 150, 0, 150);
  h["MPVErrorVsN"] = dir.make<TH2F>("MPVErrorVsN", "MPV Error vs N", 500, 0, 1000, 150, 0, 150);

  h["DiffWRTPrevGainTIB"] = dir.make<TH1F>("DiffWRTPrevGainTIB", "Diff w.r.t. PrevGain TIB", 250, 0.5, 1.5);
  h["DiffWRTPrevGainTID"] = dir.make<TH1F>("DiffWRTPrevGainTID", "Diff w.r.t. PrevGain TID", 250, 0.5, 1.5);
  h["DiffWRTPrevGainTOB"] = dir.make<TH1F>("DiffWRTPrevGainTOB", "Diff w.r.t. PrevGain TOB", 250, 0.5, 1.5);
  h["DiffWRTPrevGainTEC"] = dir.make<TH1F>("DiffWRTPrevGainTEC", "Diff w.r.t. PrevGain TEC", 250, 0.5, 1.5);

  h["GainVsPrevGainTIB"] = dir.make<TH2F>("GainVsPrevGainTIB", "Gain vs PrevGain TIB", 100, 0, 2, 100, 0, 2);
  h["GainVsPrevGainTID"] = dir.make<TH2F>("GainVsPrevGainTID", "Gain vs PrevGain TID", 100, 0, 2, 100, 0, 2);
  h["GainVsPrevGainTOB"] = dir.make<TH2F>("GainVsPrevGainTOB", "Gain vs PrevGain TOB", 100, 0, 2, 100, 0, 2);
  h["GainVsPrevGainTEC"] = dir.make<TH2F>("GainVsPrevGainTEC", "Gain vs PrevGain TEC", 100, 0, 2, 100, 0, 2);

  return h;
}

void SiStripApvGainInspector::fillQualityMonitor() {
  for (unsigned int a = 0; a < APVsCollOrdered.size(); a++) {
    std::shared_ptr<stAPVGain> APV = APVsCollOrdered[a];
    if (APV == nullptr)
      continue;

    //unsigned int Index = APV->Index;
    //unsigned int DetId = APV->DetId;
    unsigned int SubDet = APV->SubDet;
    float z = APV->z;
    float Eta = APV->Eta;
    float R = APV->R;
    float Phi = APV->Phi;
    float Thickness = APV->Thickness;
    double FitMPV = APV->FitMPV;
    double FitMPVErr = APV->FitMPVErr;
    double Gain = APV->Gain;
    double NEntries = APV->NEntries;
    double PreviousGain = APV->PreviousGain;

    if (SubDet < 3)
      continue;  // avoid to loop over Pixel det id

    if (FitMPV <= 0.) {  // No fit of MPV
      if (APV->isMasked)
        fill2D(hControl, "NoMPVmasked", z, R);
      else
        fill2D(hControl, "NoMPVfit", z, R);
    } else {  // Fit of MPV
      if (FitMPV > 0.)
        fill1D(hControl, "Gains", Gain);

      fill1D(hControl, "MPVs", FitMPV);
      if (Thickness < 0.04)
        fill1D(hControl, "MPVs320", FitMPV);
      if (Thickness > 0.04)
        fill1D(hControl, "MPVs500", FitMPV);

      fill1D(hControl, "MPVError", FitMPVErr);
      fill2D(hControl, "MPVErrorVsMPV", FitMPV, FitMPVErr);
      fill2D(hControl, "MPVErrorVsEta", Eta, FitMPVErr);
      fill2D(hControl, "MPVErrorVsPhi", Phi, FitMPVErr);
      fill2D(hControl, "MPVErrorVsN", NEntries, FitMPVErr);

      if (SubDet == 3) {
        fill2D(hControl, "MPV_Vs_EtaTIB", Eta, FitMPV);
        fill2D(hControl, "MPV_Vs_PhiTIB", Phi, FitMPV);
        fill1D(hControl, "MPVsTIB", FitMPV);

      } else if (SubDet == 4) {
        fill2D(hControl, "MPV_Vs_EtaTID", Eta, FitMPV);
        fill2D(hControl, "MPV_Vs_PhiTID", Phi, FitMPV);
        fill1D(hControl, "MPVsTID", FitMPV);
        if (Eta < 0.)
          fill1D(hControl, "MPVsTIDM", FitMPV);
        if (Eta > 0.)
          fill1D(hControl, "MPVsTIDP", FitMPV);

      } else if (SubDet == 5) {
        fill2D(hControl, "MPV_Vs_EtaTOB", Eta, FitMPV);
        fill2D(hControl, "MPV_Vs_PhiTOB", Phi, FitMPV);
        fill1D(hControl, "MPVsTOB", FitMPV);

      } else if (SubDet == 6) {
        fill2D(hControl, "MPV_Vs_EtaTEC", Eta, FitMPV);
        fill2D(hControl, "MPV_Vs_PhiTEC", Phi, FitMPV);
        fill1D(hControl, "MPVsTEC", FitMPV);
        if (Eta < 0.)
          fill1D(hControl, "MPVsTECM", FitMPV);
        if (Eta > 0.)
          fill1D(hControl, "MPVsTECP", FitMPV);
        if (Thickness < 0.04) {
          fill2D(hControl, "MPV_Vs_EtaTECthin", Eta, FitMPV);
          fill2D(hControl, "MPV_Vs_PhiTECthin", Phi, FitMPV);
          fill1D(hControl, "MPVsTECthin", FitMPV);
          if (Eta > 0.)
            fill1D(hControl, "MPVsTECP1", FitMPV);
          if (Eta < 0.)
            fill1D(hControl, "MPVsTECM1", FitMPV);
        }
        if (Thickness > 0.04) {
          fill2D(hControl, "MPV_Vs_EtaTECthick", Eta, FitMPV);
          fill2D(hControl, "MPV_Vs_PhiTECthick", Phi, FitMPV);
          fill1D(hControl, "MPVsTECthick", FitMPV);
          if (Eta > 0.)
            fill1D(hControl, "MPVsTECP2", FitMPV);
          if (Eta < 0.)
            fill1D(hControl, "MPVsTECM2", FitMPV);
        }
      }
    }

    if (SubDet == 3 && PreviousGain != 0.)
      fill1D(hControl, "DiffWRTPrevGainTIB", Gain / PreviousGain);
    else if (SubDet == 4 && PreviousGain != 0.)
      fill1D(hControl, "DiffWRTPrevGainTID", Gain / PreviousGain);
    else if (SubDet == 5 && PreviousGain != 0.)
      fill1D(hControl, "DiffWRTPrevGainTOB", Gain / PreviousGain);
    else if (SubDet == 6 && PreviousGain != 0.)
      fill1D(hControl, "DiffWRTPrevGainTEC", Gain / PreviousGain);

    if (SubDet == 3)
      fill2D(hControl, "GainVsPrevGainTIB", PreviousGain, Gain);
    else if (SubDet == 4)
      fill2D(hControl, "GainVsPrevGainTID", PreviousGain, Gain);
    else if (SubDet == 5)
      fill2D(hControl, "GainVsPrevGainTOB", PreviousGain, Gain);
    else if (SubDet == 6)
      fill2D(hControl, "GainVsPrevGainTEC", PreviousGain, Gain);

  }  // loop on the APV collections
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripApvGainInspector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("inputFile", {});
  desc.addUntracked<double>("minNrEntries", 20);
  desc.add<int>("fitMode", 2)
      ->setComment("fit mode. Available options: 1: landau\n 2: landau around max\n 3:landau&gaus convo\n 4: fake");
  desc.addUntracked<std::vector<unsigned int>>("selectedModules", {});
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripApvGainInspector);
