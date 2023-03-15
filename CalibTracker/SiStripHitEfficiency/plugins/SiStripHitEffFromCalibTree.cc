//Original Author:  Christopher Edelmaier
//        Created:  Feb. 11, 2010

// system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEfficiencyHelpers.h"
#include "CalibTracker/SiStripHitEfficiency/interface/TrajectoryAtInvalidHit.h"
#include "CalibTracker/SiStripHitEfficiency/plugins/HitEff.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfo.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

// ROOT includes
#include "TCanvas.h"
#include "TEfficiency.h"
#include "TF1.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraphAsymmErrors.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLeaf.h"
#include "TLegend.h"
#include "TObjString.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TTree.h"

// custom made printout
#define LOGPRINT edm::LogPrint("SiStripHitEffFromCalibTree")

using namespace edm;
using namespace reco;
using namespace std;

struct hit {
  double x;
  double y;
  double z;
  unsigned int id;
};

class SiStripHitEffFromCalibTree : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripHitEffFromCalibTree(const edm::ParameterSet&);
  ~SiStripHitEffFromCalibTree() override = default;

private:
  // overridden from ConditionDBWriter
  void algoAnalyze(const edm::Event& e, const edm::EventSetup& c) override;
  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  // native methods
  void setBadComponents(int i,
                        int component,
                        SiStripQuality::BadComponent& BC,
                        std::stringstream ssV[4][19],
                        int NBadComponent[4][19][4]);
  void makeTKMap(bool autoTagging);
  void makeHotColdMaps();
  void makeSQLite();
  void totalStatistics();
  void makeSummary();
  void makeSummaryVsBx();
  void computeEff(vector<TH1F*>& vhfound, vector<TH1F*>& vhtotal, string name);
  void makeSummaryVsLumi();
  void makeSummaryVsCM();
  TString getLayerSideName(Long_t k);

  // to be used everywhere
  static constexpr int siStripLayers_ = 22;
  static constexpr double nBxInAnOrbit_ = 3565;

  edm::Service<TFileService> fs;
  SiStripDetInfo detInfo_;
  edm::FileInPath fileInPath_;
  SiStripQuality* quality_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  TTree* calibTree_;
  vector<string> calibTreeFileNames_;
  float threshold_;
  unsigned int nModsMin_;
  string badModulesFile_;
  bool autoIneffModTagging_;
  unsigned int clusterMatchingMethod_;
  float resXSig_;
  float clusterTrajDist_;
  float stripsApvEdge_;
  bool useOnlyHighPurityTracks_;
  unsigned int bunchX_;
  unsigned int spaceBetweenTrains_;
  bool useCM_;
  bool showEndcapSides_;
  bool showRings_;
  bool showTOB6TEC9_;
  bool showOnlyGoodModules_;
  float tkMapMin_;
  float effPlotMin_;
  TString title_;

  unsigned int nTEClayers;

  TH1F* bxHisto;
  TH1F* instLumiHisto;
  TH1F* PUHisto;

  // for association of informations of the hitEff tree and the event infos tree
  map<pair<unsigned int, unsigned int>, array<double, 3> > eventInfos;

  vector<hit> hits[23];
  vector<TH2F*> HotColdMaps;
  map<unsigned int, pair<unsigned int, unsigned int> > modCounter[23];
  TrackerMap* tkmap;
  TrackerMap* tkmapbad;
  TrackerMap* tkmapeff;
  TrackerMap* tkmapnum;
  TrackerMap* tkmapden;
  long layerfound[23];
  long layertotal[23];
  map<unsigned int, vector<int> > layerfound_perBx;
  map<unsigned int, vector<int> > layertotal_perBx;
  vector<TH1F*> layerfound_vsLumi;
  vector<TH1F*> layertotal_vsLumi;
  vector<TH1F*> layerfound_vsPU;
  vector<TH1F*> layertotal_vsPU;
  vector<TH1F*> layerfound_vsCM;
  vector<TH1F*> layertotal_vsCM;
  vector<TH1F*> layerfound_vsBX;
  vector<TH1F*> layertotal_vsBX;
  int goodlayertotal[35];
  int goodlayerfound[35];
  int alllayertotal[35];
  int alllayerfound[35];
  map<unsigned int, double> BadModules;
};

SiStripHitEffFromCalibTree::SiStripHitEffFromCalibTree(const edm::ParameterSet& conf)
    : ConditionDBWriter<SiStripBadStrip>(conf),
      fileInPath_(SiStripDetInfoFileReader::kDefaultFile),
      tkGeomToken_(esConsumes()),
      tTopoToken_(esConsumes()) {
  usesResource(TFileService::kSharedResource);
  calibTreeFileNames_ = conf.getUntrackedParameter<vector<std::string> >("CalibTreeFilenames");
  threshold_ = conf.getParameter<double>("Threshold");
  nModsMin_ = conf.getParameter<int>("nModsMin");
  badModulesFile_ = conf.getUntrackedParameter<std::string>("BadModulesFile", "");
  autoIneffModTagging_ = conf.getUntrackedParameter<bool>("AutoIneffModTagging", false);
  clusterMatchingMethod_ = conf.getUntrackedParameter<int>("ClusterMatchingMethod", 0);
  resXSig_ = conf.getUntrackedParameter<double>("ResXSig", -1);
  clusterTrajDist_ = conf.getUntrackedParameter<double>("ClusterTrajDist", 64.0);
  stripsApvEdge_ = conf.getUntrackedParameter<double>("StripsApvEdge", 10.0);
  useOnlyHighPurityTracks_ = conf.getUntrackedParameter<bool>("UseOnlyHighPurityTracks", true);
  bunchX_ = conf.getUntrackedParameter<int>("BunchCrossing", 0);
  spaceBetweenTrains_ = conf.getUntrackedParameter<int>("SpaceBetweenTrains", 25);
  useCM_ = conf.getUntrackedParameter<bool>("UseCommonMode", false);
  showEndcapSides_ = conf.getUntrackedParameter<bool>("ShowEndcapSides", true);
  showRings_ = conf.getUntrackedParameter<bool>("ShowRings", false);
  showTOB6TEC9_ = conf.getUntrackedParameter<bool>("ShowTOB6TEC9", false);
  showOnlyGoodModules_ = conf.getUntrackedParameter<bool>("ShowOnlyGoodModules", false);
  tkMapMin_ = conf.getUntrackedParameter<double>("TkMapMin", 0.9);
  effPlotMin_ = conf.getUntrackedParameter<double>("EffPlotMin", 0.9);
  title_ = conf.getParameter<std::string>("Title");
  detInfo_ = SiStripDetInfoFileReader::read(fileInPath_.fullPath());

  nTEClayers = 9;  // number of wheels
  if (showRings_)
    nTEClayers = 7;  // number of rings

  quality_ = new SiStripQuality(detInfo_);
}

void SiStripHitEffFromCalibTree::algoAnalyze(const edm::Event& e, const edm::EventSetup& c) {
  const auto& tkgeom = c.getData(tkGeomToken_);
  const auto& tTopo = c.getData(tTopoToken_);

  // read bad modules to mask
  ifstream badModules_file;
  set<uint32_t> badModules_list;
  if (!badModulesFile_.empty()) {
    badModules_file.open(badModulesFile_.c_str());
    uint32_t badmodule_detid;
    int mods, fiber1, fiber2, fiber3;
    if (badModules_file.is_open()) {
      string line;
      while (getline(badModules_file, line)) {
        if (badModules_file.eof())
          continue;
        stringstream ss(line);
        ss >> badmodule_detid >> mods >> fiber1 >> fiber2 >> fiber3;
        if (badmodule_detid != 0 && mods == 1 && (fiber1 == 1 || fiber2 == 1 || fiber3 == 1))
          badModules_list.insert(badmodule_detid);
      }
      badModules_file.close();
    }
  }
  if (!badModules_list.empty())
    LOGPRINT << "Remove additionnal bad modules from the analysis: ";
  set<uint32_t>::iterator itBadMod;
  for (const auto& badMod : badModules_list) {
    LOGPRINT << " " << badMod;
  }

  // initialze counters and histos

  bxHisto = fs->make<TH1F>("bx", "bx", 3600, 0, 3600);
  instLumiHisto = fs->make<TH1F>("instLumi", "inst. lumi.", 250, 0, 25000);
  PUHisto = fs->make<TH1F>("PU", "PU", 200, 0, 200);

  for (int l = 0; l < 35; l++) {
    goodlayertotal[l] = 0;
    goodlayerfound[l] = 0;
    alllayertotal[l] = 0;
    alllayerfound[l] = 0;
  }

  TH1F* resolutionPlots[23];
  for (Long_t ilayer = 0; ilayer < 23; ilayer++) {
    std::string lyrName = ::layerName(ilayer, showRings_, nTEClayers);

    resolutionPlots[ilayer] = fs->make<TH1F>(Form("resol_layer_%i", (int)(ilayer)), lyrName.c_str(), 125, -125, 125);
    resolutionPlots[ilayer]->GetXaxis()->SetTitle("trajX-clusX [strip unit]");

    layerfound_vsLumi.push_back(
        fs->make<TH1F>(Form("layerfound_vsLumi_layer_%i", (int)(ilayer)), lyrName.c_str(), 100, 0, 25000));
    layertotal_vsLumi.push_back(
        fs->make<TH1F>(Form("layertotal_vsLumi_layer_%i", (int)(ilayer)), lyrName.c_str(), 100, 0, 25000));
    layerfound_vsPU.push_back(
        fs->make<TH1F>(Form("layerfound_vsPU_layer_%i", (int)(ilayer)), lyrName.c_str(), 45, 0, 90));
    layertotal_vsPU.push_back(
        fs->make<TH1F>(Form("layertotal_vsPU_layer_%i", (int)(ilayer)), lyrName.c_str(), 45, 0, 90));

    layerfound_vsBX.push_back(fs->make<TH1F>(
        Form("foundVsBx_layer%i", (int)ilayer), Form("layer %i", (int)ilayer), nBxInAnOrbit_, 0, nBxInAnOrbit_));
    layertotal_vsBX.push_back(fs->make<TH1F>(
        Form("totalVsBx_layer%i", (int)ilayer), Form("layer %i", (int)ilayer), nBxInAnOrbit_, 0, nBxInAnOrbit_));

    if (useCM_) {
      layerfound_vsCM.push_back(
          fs->make<TH1F>(Form("layerfound_vsCM_layer_%i", (int)(ilayer)), lyrName.c_str(), 20, 0, 400));
      layertotal_vsCM.push_back(
          fs->make<TH1F>(Form("layertotal_vsCM_layer_%i", (int)(ilayer)), lyrName.c_str(), 20, 0, 400));
    }
    layertotal[ilayer] = 0;
    layerfound[ilayer] = 0;
  }

  if (!autoIneffModTagging_)
    LOGPRINT << "A module is bad if efficiency < " << threshold_ << " and has at least " << nModsMin_ << " nModsMin.";
  else
    LOGPRINT << "A module is bad if the upper limit on the efficiency is < to the avg in the layer - " << threshold_
             << " and has at least " << nModsMin_ << " nModsMin.";

  unsigned int run, evt, bx{0};
  double instLumi, PU;

  //Open the ROOT Calib Tree
  for (const auto& calibTreeFileName : calibTreeFileNames_) {
    LOGPRINT << "Loading file: " << calibTreeFileName;
    TFile* CalibTreeFile = TFile::Open(calibTreeFileName.c_str(), "READ");

    // Get event infos
    bool foundEventInfos = false;
    try {
      CalibTreeFile->cd("eventInfo");
    } catch (exception& e) {
      LOGPRINT << "No event infos tree";
    }
    TTree* EventTree = (TTree*)(gDirectory->Get("tree"));

    TLeaf* runLf;
    TLeaf* evtLf;
    TLeaf* BunchLf;
    TLeaf* InstLumiLf;
    TLeaf* PULf;
    if (EventTree) {
      LOGPRINT << "Found event infos tree";

      runLf = EventTree->GetLeaf("run");
      evtLf = EventTree->GetLeaf("event");

      BunchLf = EventTree->GetLeaf("bx");
      InstLumiLf = EventTree->GetLeaf("instLumi");
      PULf = EventTree->GetLeaf("PU");

      int nevt = EventTree->GetEntries();
      if (nevt)
        foundEventInfos = true;

      for (int j = 0; j < nevt; j++) {
        EventTree->GetEntry(j);
        run = runLf->GetValue();
        evt = evtLf->GetValue();
        bx = BunchLf->GetValue();
        instLumi = InstLumiLf->GetValue();
        PU = PULf->GetValue();

        bxHisto->Fill(bx);
        instLumiHisto->Fill(instLumi);
        PUHisto->Fill(PU);

        eventInfos[make_pair(run, evt)] = array<double, 3>{{(double)bx, instLumi, PU}};
      }
    }

    // Get hit infos
    CalibTreeFile->cd("anEff");
    calibTree_ = (TTree*)(gDirectory->Get("traj"));

    runLf = calibTree_->GetLeaf("run");
    evtLf = calibTree_->GetLeaf("event");
    TLeaf* BadLf = calibTree_->GetLeaf("ModIsBad");
    TLeaf* sistripLf = calibTree_->GetLeaf("SiStripQualBad");
    TLeaf* idLf = calibTree_->GetLeaf("Id");
    TLeaf* acceptLf = calibTree_->GetLeaf("withinAcceptance");
    TLeaf* layerLf = calibTree_->GetLeaf("layer");
    //TLeaf* nHitsLf = calibTree_->GetLeaf("nHits");
    TLeaf* highPurityLf = calibTree_->GetLeaf("highPurity");
    TLeaf* xLf = calibTree_->GetLeaf("TrajGlbX");
    TLeaf* yLf = calibTree_->GetLeaf("TrajGlbY");
    TLeaf* zLf = calibTree_->GetLeaf("TrajGlbZ");
    TLeaf* ResXSigLf = calibTree_->GetLeaf("ResXSig");
    TLeaf* TrajLocXLf = calibTree_->GetLeaf("TrajLocX");
    TLeaf* TrajLocYLf = calibTree_->GetLeaf("TrajLocY");
    TLeaf* ClusterLocXLf = calibTree_->GetLeaf("ClusterLocX");
    BunchLf = calibTree_->GetLeaf("bunchx");
    InstLumiLf = calibTree_->GetLeaf("instLumi");
    PULf = calibTree_->GetLeaf("PU");
    TLeaf* CMLf = nullptr;
    if (useCM_)
      CMLf = calibTree_->GetLeaf("commonMode");

    int nevents = calibTree_->GetEntries();
    LOGPRINT << "Successfully loaded analyze function with " << nevents << " events!\n";

    map<pair<unsigned int, unsigned int>, array<double, 3> >::iterator itEventInfos;

    //Loop through all of the events
    for (int j = 0; j < nevents; j++) {
      calibTree_->GetEntry(j);
      run = (unsigned int)runLf->GetValue();
      evt = (unsigned int)evtLf->GetValue();
      unsigned int isBad = (unsigned int)BadLf->GetValue();
      unsigned int quality = (unsigned int)sistripLf->GetValue();
      unsigned int id = (unsigned int)idLf->GetValue();
      unsigned int accept = (unsigned int)acceptLf->GetValue();
      unsigned int layer_wheel = (unsigned int)layerLf->GetValue();
      unsigned int layer = layer_wheel;
      if (showRings_ && layer > 10) {  // use rings instead of wheels
        if (layer < 14)
          layer = 10 + ((id >> 9) & 0x3);  //TID   3 disks and also 3 rings -> use the same container
        else
          layer = 13 + ((id >> 5) & 0x7);  //TEC
      }
      //unsigned int nHits = (unsigned int)nHitsLf->GetValue();
      bool highPurity = (bool)highPurityLf->GetValue();
      double x = xLf->GetValue();
      double y = yLf->GetValue();
      double z = zLf->GetValue();
      double resxsig = ResXSigLf->GetValue();
      double TrajLocX = TrajLocXLf->GetValue();
      double TrajLocY = TrajLocYLf->GetValue();
      double ClusterLocX = ClusterLocXLf->GetValue();
      double TrajLocXMid;
      double stripTrajMid;
      double stripCluster;
      bool badquality = false;

      instLumi = 0;
      PU = 0;

      // if no special tree with event infos, they may be stored in the hit eff tree
      if (!foundEventInfos) {
        bx = (unsigned int)BunchLf->GetValue();
        if (InstLumiLf != nullptr)
          instLumi = InstLumiLf->GetValue();  // branch not filled by default
        if (PULf != nullptr)
          PU = PULf->GetValue();  // branch not filled by default
      }
      int CM = -100;
      if (useCM_)
        CM = CMLf->GetValue();

      // Get infos from eventInfos if they exist
      if (foundEventInfos) {
        itEventInfos = eventInfos.find(make_pair(run, evt));
        if (itEventInfos != eventInfos.end()) {
          bx = itEventInfos->second[0];
          instLumi = itEventInfos->second[1];
          PU = itEventInfos->second[2];
        }
      }

      //We have two things we want to do, both an XY color plot, and the efficiency measurement
      //First, ignore anything that isn't in acceptance and isn't good quality

      if (bunchX_ > 0 && bunchX_ != bx)
        continue;

      //if(quality == 1 || accept != 1 || nHits < 8) continue;
      if (accept != 1)
        continue;
      if (useOnlyHighPurityTracks_ && !highPurity)
        continue;
      if (quality == 1)
        badquality = true;

      // don't compute efficiencies in modules from TOB6 and TEC9
      if (!showTOB6TEC9_ && (layer_wheel == 10 || layer_wheel == siStripLayers_))
        continue;

      // don't use bad modules given in the bad module list
      itBadMod = badModules_list.find(id);
      if (itBadMod != badModules_list.end())
        continue;

      //Now that we have a good event, we need to look at if we expected it or not, and the location
      //if we didn't
      //Fill the missing hit information first
      bool badflag = false;

      // By default uses the old matching method
      if (resXSig_ < 0) {
        if (isBad == 1)
          badflag = true;  // isBad set to false in the tree when resxsig<999.0
      } else {
        if (isBad == 1 || resxsig > resXSig_)
          badflag = true;
      }

      // Conversion of positions in strip unit
      int nstrips = -9;
      float Pitch = -9.0;

      if (resxsig == 1000.0) {  // special treatment, no GeomDetUnit associated in some cases when no cluster found
        Pitch = 0.0205;         // maximum
        nstrips = 768;          // maximum
        stripTrajMid = TrajLocX / Pitch + nstrips / 2.0;
        stripCluster = ClusterLocX / Pitch + nstrips / 2.0;
      } else {
        DetId ClusterDetId(id);
        const StripGeomDetUnit* stripdet = (const StripGeomDetUnit*)tkgeom.idToDetUnit(ClusterDetId);
        const StripTopology& Topo = stripdet->specificTopology();
        nstrips = Topo.nstrips();
        Pitch = stripdet->surface().bounds().width() / Topo.nstrips();
        stripTrajMid = TrajLocX / Pitch + nstrips / 2.0;  //layer01->10
        stripCluster = ClusterLocX / Pitch + nstrips / 2.0;

        // For trapezoidal modules: extrapolation of x trajectory position to the y middle of the module
        //  for correct comparison with cluster position
        float hbedge = 0;
        float htedge = 0;
        float hapoth = 0;
        if (layer >= 11) {
          const BoundPlane& plane = stripdet->surface();
          const TrapezoidalPlaneBounds* trapezoidalBounds(
              dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
          std::array<const float, 4> const& parameters = (*trapezoidalBounds).parameters();
          hbedge = parameters[0];
          htedge = parameters[1];
          hapoth = parameters[3];
          TrajLocXMid = TrajLocX / (1 + (htedge - hbedge) * TrajLocY / (htedge + hbedge) /
                                            hapoth);  // radialy extrapolated x loc position at middle
          stripTrajMid = TrajLocXMid / Pitch + nstrips / 2.0;
        }
      }

      if (!badquality && layer < 23) {
        if (resxsig != 1000.0)
          resolutionPlots[layer]->Fill(stripTrajMid - stripCluster);
        else
          resolutionPlots[layer]->Fill(1000);
      }

      // New matching methods
      int tapv = -9;
      int capv = -9;
      float stripInAPV = 64.;

      if (clusterMatchingMethod_ >= 1) {
        badflag = false;          // reset
        if (resxsig == 1000.0) {  // default value when no cluster found in the module
          badflag = true;         // consider the module inefficient in this case
        } else {
          if (clusterMatchingMethod_ == 2 ||
              clusterMatchingMethod_ == 4) {  // check the distance between cluster and trajectory position
            if (abs(stripCluster - stripTrajMid) > clusterTrajDist_)
              badflag = true;
          }
          if (clusterMatchingMethod_ == 3 ||
              clusterMatchingMethod_ ==
                  4) {  // cluster and traj have to be in the same APV (don't take edges into accounts)
            tapv = (int)stripTrajMid / 128;
            capv = (int)stripCluster / 128;
            stripInAPV = stripTrajMid - tapv * 128;

            if (stripInAPV < stripsApvEdge_ || stripInAPV > 128 - stripsApvEdge_)
              continue;
            if (tapv != capv)
              badflag = true;
          }
        }
      }

      if (badflag && !badquality) {
        hit temphit;
        temphit.x = x;
        temphit.y = y;
        temphit.z = z;
        temphit.id = id;
        hits[layer].push_back(temphit);
      }
      pair<unsigned int, unsigned int> newgoodpair(1, 1);
      pair<unsigned int, unsigned int> newbadpair(1, 0);
      //First, figure out if the module already exists in the map of maps
      map<unsigned int, pair<unsigned int, unsigned int> >::iterator it = modCounter[layer].find(id);
      if (!badquality) {
        if (it == modCounter[layer].end()) {
          if (badflag)
            modCounter[layer][id] = newbadpair;
          else
            modCounter[layer][id] = newgoodpair;
        } else {
          ((*it).second.first)++;
          if (!badflag)
            ((*it).second.second)++;
        }

        if (layerfound_perBx.find(bx) == layerfound_perBx.end()) {
          layerfound_perBx[bx] = vector<int>(23, 0);
          layertotal_perBx[bx] = vector<int>(23, 0);
        }
        if (!badflag)
          layerfound_perBx[bx][layer]++;
        layertotal_perBx[bx][layer]++;

        if (!badflag)
          layerfound_vsLumi[layer]->Fill(instLumi);
        layertotal_vsLumi[layer]->Fill(instLumi);
        if (!badflag)
          layerfound_vsPU[layer]->Fill(PU);
        layertotal_vsPU[layer]->Fill(PU);

        if (useCM_) {
          if (!badflag)
            layerfound_vsCM[layer]->Fill(CM);
          layertotal_vsCM[layer]->Fill(CM);
        }

        //Have to do the decoding for which side to go on (ugh)
        if (layer <= 10) {
          if (!badflag)
            goodlayerfound[layer]++;
          goodlayertotal[layer]++;
        } else if (layer > 10 && layer < 14) {
          if (((id >> 13) & 0x3) == 1) {
            if (!badflag)
              goodlayerfound[layer]++;
            goodlayertotal[layer]++;
          } else if (((id >> 13) & 0x3) == 2) {
            if (!badflag)
              goodlayerfound[layer + 3]++;
            goodlayertotal[layer + 3]++;
          }
        } else if (layer > 13 && layer <= siStripLayers_) {
          if (((id >> 18) & 0x3) == 1) {
            if (!badflag)
              goodlayerfound[layer + 3]++;
            goodlayertotal[layer + 3]++;
          } else if (((id >> 18) & 0x3) == 2) {
            if (!badflag)
              goodlayerfound[layer + 3 + nTEClayers]++;
            goodlayertotal[layer + 3 + nTEClayers]++;
          }
        }
      }
      //Do the one where we don't exclude bad modules!
      if (layer <= 10) {
        if (!badflag)
          alllayerfound[layer]++;
        alllayertotal[layer]++;
      } else if (layer > 10 && layer < 14) {
        if (((id >> 13) & 0x3) == 1) {
          if (!badflag)
            alllayerfound[layer]++;
          alllayertotal[layer]++;
        } else if (((id >> 13) & 0x3) == 2) {
          if (!badflag)
            alllayerfound[layer + 3]++;
          alllayertotal[layer + 3]++;
        }
      } else if (layer > 13 && layer <= siStripLayers_) {
        if (((id >> 18) & 0x3) == 1) {
          if (!badflag)
            alllayerfound[layer + 3]++;
          alllayertotal[layer + 3]++;
        } else if (((id >> 18) & 0x3) == 2) {
          if (!badflag)
            alllayerfound[layer + 3 + nTEClayers]++;
          alllayertotal[layer + 3 + nTEClayers]++;
        }
      }
      //At this point, both of our maps are loaded with the correct information
    }
  }  // go to next CalibTreeFile

  makeHotColdMaps();
  makeTKMap(autoIneffModTagging_);
  makeSQLite();
  totalStatistics();
  makeSummary();
  makeSummaryVsBx();
  makeSummaryVsLumi();
  if (useCM_)
    makeSummaryVsCM();

  ////////////////////////////////////////////////////////////////////////
  //try to write out what's in the quality record
  /////////////////////////////////////////////////////////////////////////////
  int NTkBadComponent[4];  //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int NBadComponent[4][19][4];
  //legend: NBadComponent[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
  //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
  //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  std::stringstream ssV[4][19];

  for (int i = 0; i < 4; ++i) {
    NTkBadComponent[i] = 0;
    for (int j = 0; j < 19; ++j) {
      ssV[i][j].str("");
      for (int k = 0; k < 4; ++k)
        NBadComponent[i][j][k] = 0;
    }
  }

  std::vector<SiStripQuality::BadComponent> BC = quality_->getBadComponentList();

  for (size_t i = 0; i < BC.size(); ++i) {
    //&&&&&&&&&&&&&
    //Full Tk
    //&&&&&&&&&&&&&

    if (BC[i].BadModule)
      NTkBadComponent[0]++;
    if (BC[i].BadFibers)
      NTkBadComponent[1] += ((BC[i].BadFibers >> 2) & 0x1) + ((BC[i].BadFibers >> 1) & 0x1) + ((BC[i].BadFibers) & 0x1);
    if (BC[i].BadApvs)
      NTkBadComponent[2] += ((BC[i].BadApvs >> 5) & 0x1) + ((BC[i].BadApvs >> 4) & 0x1) + ((BC[i].BadApvs >> 3) & 0x1) +
                            ((BC[i].BadApvs >> 2) & 0x1) + ((BC[i].BadApvs >> 1) & 0x1) + ((BC[i].BadApvs) & 0x1);

    //&&&&&&&&&&&&&&&&&
    //Single SubSystem
    //&&&&&&&&&&&&&&&&&

    int component;
    SiStripDetId a(BC[i].detid);
    if (a.subdetId() == SiStripDetId::TIB) {
      //&&&&&&&&&&&&&&&&&
      //TIB
      //&&&&&&&&&&&&&&&&&

      component = tTopo.tibLayer(BC[i].detid);
      setBadComponents(0, component, BC[i], ssV, NBadComponent);

    } else if (a.subdetId() == SiStripDetId::TID) {
      //&&&&&&&&&&&&&&&&&
      //TID
      //&&&&&&&&&&&&&&&&&

      component = tTopo.tidSide(BC[i].detid) == 2 ? tTopo.tidWheel(BC[i].detid) : tTopo.tidWheel(BC[i].detid) + 3;
      setBadComponents(1, component, BC[i], ssV, NBadComponent);

    } else if (a.subdetId() == SiStripDetId::TOB) {
      //&&&&&&&&&&&&&&&&&
      //TOB
      //&&&&&&&&&&&&&&&&&

      component = tTopo.tobLayer(BC[i].detid);
      setBadComponents(2, component, BC[i], ssV, NBadComponent);

    } else if (a.subdetId() == SiStripDetId::TEC) {
      //&&&&&&&&&&&&&&&&&
      //TEC
      //&&&&&&&&&&&&&&&&&

      component = tTopo.tecSide(BC[i].detid) == 2 ? tTopo.tecWheel(BC[i].detid) : tTopo.tecWheel(BC[i].detid) + 9;
      setBadComponents(3, component, BC[i], ssV, NBadComponent);
    }
  }

  //&&&&&&&&&&&&&&&&&&
  // Single Strip Info
  //&&&&&&&&&&&&&&&&&&
  float percentage = 0;

  SiStripQuality::RegistryIterator rbegin = quality_->getRegistryVectorBegin();
  SiStripQuality::RegistryIterator rend = quality_->getRegistryVectorEnd();

  for (SiStripBadStrip::RegistryIterator rp = rbegin; rp != rend; ++rp) {
    unsigned int detid = rp->detid;

    int subdet = -999;
    int component = -999;
    SiStripDetId a(detid);
    if (a.subdetId() == 3) {
      subdet = 0;
      component = tTopo.tibLayer(detid);
    } else if (a.subdetId() == 4) {
      subdet = 1;
      component = tTopo.tidSide(detid) == 2 ? tTopo.tidWheel(detid) : tTopo.tidWheel(detid) + 3;
    } else if (a.subdetId() == 5) {
      subdet = 2;
      component = tTopo.tobLayer(detid);
    } else if (a.subdetId() == 6) {
      subdet = 3;
      component = tTopo.tecSide(detid) == 2 ? tTopo.tecWheel(detid) : tTopo.tecWheel(detid) + 9;
    }

    SiStripQuality::Range sqrange =
        SiStripQuality::Range(quality_->getDataVectorBegin() + rp->ibegin, quality_->getDataVectorBegin() + rp->iend);

    percentage = 0;
    for (int it = 0; it < sqrange.second - sqrange.first; it++) {
      unsigned int range = quality_->decode(*(sqrange.first + it)).range;
      NTkBadComponent[3] += range;
      NBadComponent[subdet][0][3] += range;
      NBadComponent[subdet][component][3] += range;
      percentage += range;
    }
    if (percentage != 0)
      percentage /= 128. * detInfo_.getNumberOfApvsAndStripLength(detid).first;
    if (percentage > 1)
      edm::LogError("SiStripQualityStatistics") << "PROBLEM detid " << detid << " value " << percentage << std::endl;
  }
  //&&&&&&&&&&&&&&&&&&
  // printout
  //&&&&&&&&&&&&&&&&&&
  std::ostringstream ss;

  ss << "\n-----------------\nNew IOV starting from run " << e.id().run() << " event " << e.id().event()
     << " lumiBlock " << e.luminosityBlock() << " time " << e.time().value() << "\n-----------------\n";
  ss << "\n-----------------\nGlobal Info\n-----------------";
  ss << "\nBadComponent \t	Modules \tFibers "
        "\tApvs\tStrips\n----------------------------------------------------------------";
  ss << "\nTracker:\t\t" << NTkBadComponent[0] << "\t" << NTkBadComponent[1] << "\t" << NTkBadComponent[2] << "\t"
     << NTkBadComponent[3];
  ss << "\nTIB:\t\t\t" << NBadComponent[0][0][0] << "\t" << NBadComponent[0][0][1] << "\t" << NBadComponent[0][0][2]
     << "\t" << NBadComponent[0][0][3];
  ss << "\nTID:\t\t\t" << NBadComponent[1][0][0] << "\t" << NBadComponent[1][0][1] << "\t" << NBadComponent[1][0][2]
     << "\t" << NBadComponent[1][0][3];
  ss << "\nTOB:\t\t\t" << NBadComponent[2][0][0] << "\t" << NBadComponent[2][0][1] << "\t" << NBadComponent[2][0][2]
     << "\t" << NBadComponent[2][0][3];
  ss << "\nTEC:\t\t\t" << NBadComponent[3][0][0] << "\t" << NBadComponent[3][0][1] << "\t" << NBadComponent[3][0][2]
     << "\t" << NBadComponent[3][0][3];
  ss << "\n";

  for (int i = 1; i < 5; ++i)
    ss << "\nTIB Layer " << i << " :\t\t" << NBadComponent[0][i][0] << "\t" << NBadComponent[0][i][1] << "\t"
       << NBadComponent[0][i][2] << "\t" << NBadComponent[0][i][3];
  ss << "\n";
  for (int i = 1; i < 4; ++i)
    ss << "\nTID+ Disk " << i << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
       << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
  for (int i = 4; i < 7; ++i)
    ss << "\nTID- Disk " << i - 3 << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
       << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
  ss << "\n";
  for (int i = 1; i < 7; ++i)
    ss << "\nTOB Layer " << i << " :\t\t" << NBadComponent[2][i][0] << "\t" << NBadComponent[2][i][1] << "\t"
       << NBadComponent[2][i][2] << "\t" << NBadComponent[2][i][3];
  ss << "\n";
  for (int i = 1; i < 10; ++i)
    ss << "\nTEC+ Disk " << i << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
       << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
  for (int i = 10; i < 19; ++i)
    ss << "\nTEC- Disk " << i - 9 << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
       << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
  ss << "\n";

  ss << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers "
        "Apvs\n----------------------------------------------------------------";
  for (int i = 1; i < 5; ++i)
    ss << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  ss << "\n";
  for (int i = 1; i < 4; ++i)
    ss << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i = 4; i < 7; ++i)
    ss << "\nTID- Disk " << i - 3 << " :" << ssV[1][i].str();
  ss << "\n";
  for (int i = 1; i < 7; ++i)
    ss << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  ss << "\n";
  for (int i = 1; i < 10; ++i)
    ss << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i = 10; i < 19; ++i)
    ss << "\nTEC- Disk " << i - 9 << " :" << ssV[3][i].str();

  LOGPRINT << ss.str();

  // store also bad modules in log file
  ofstream badModules;
  badModules.open("BadModules.log");
  badModules << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers "
                "Apvs\n----------------------------------------------------------------";
  for (int i = 1; i < 5; ++i)
    badModules << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  badModules << "\n";
  for (int i = 1; i < 4; ++i)
    badModules << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i = 4; i < 7; ++i)
    badModules << "\nTID- Disk " << i - 3 << " :" << ssV[1][i].str();
  badModules << "\n";
  for (int i = 1; i < 7; ++i)
    badModules << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  badModules << "\n";
  for (int i = 1; i < 10; ++i)
    badModules << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i = 10; i < 19; ++i)
    badModules << "\nTEC- Disk " << i - 9 << " :" << ssV[3][i].str();
  badModules.close();
}

void SiStripHitEffFromCalibTree::makeHotColdMaps() {
  LOGPRINT << "Entering hot cold map generation!\n";
  TStyle* gStyle = new TStyle("gStyle", "myStyle");
  gStyle->cd();
  gStyle->SetPalette(1);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetOptStat(0);
  //Here we make the hot/cold color maps that we love so very much
  //Already have access to the data as a private variable
  //Create all of the histograms in the TFileService
  TH2F* temph2;
  for (Long_t maplayer = 1; maplayer <= siStripLayers_; maplayer++) {
    //Initialize all of the histograms
    if (maplayer > 0 && maplayer <= 4) {
      //We are in the TIB
      temph2 = fs->make<TH2F>(Form("%s%i", "TIB", (int)(maplayer)), "TIB", 100, -1, 361, 100, -100, 100);
      temph2->GetXaxis()->SetTitle("Phi");
      temph2->GetXaxis()->SetBinLabel(1, TString("360"));
      temph2->GetXaxis()->SetBinLabel(50, TString("180"));
      temph2->GetXaxis()->SetBinLabel(100, TString("0"));
      temph2->GetYaxis()->SetTitle("Global Z");
      temph2->SetOption("colz");
      HotColdMaps.push_back(temph2);
    } else if (maplayer > 4 && maplayer <= 10) {
      //We are in the TOB
      temph2 = fs->make<TH2F>(Form("%s%i", "TOB", (int)(maplayer - 4)), "TOB", 100, -1, 361, 100, -120, 120);
      temph2->GetXaxis()->SetTitle("Phi");
      temph2->GetXaxis()->SetBinLabel(1, TString("360"));
      temph2->GetXaxis()->SetBinLabel(50, TString("180"));
      temph2->GetXaxis()->SetBinLabel(100, TString("0"));
      temph2->GetYaxis()->SetTitle("Global Z");
      temph2->SetOption("colz");
      HotColdMaps.push_back(temph2);
    } else if (maplayer > 10 && maplayer <= 13) {
      //We are in the TID
      //Split by +/-
      temph2 = fs->make<TH2F>(Form("%s%i", "TID-", (int)(maplayer - 10)), "TID-", 100, -100, 100, 100, -100, 100);
      temph2->GetXaxis()->SetTitle("Global Y");
      temph2->GetXaxis()->SetBinLabel(1, TString("+Y"));
      temph2->GetXaxis()->SetBinLabel(50, TString("0"));
      temph2->GetXaxis()->SetBinLabel(100, TString("-Y"));
      temph2->GetYaxis()->SetTitle("Global X");
      temph2->GetYaxis()->SetBinLabel(1, TString("-X"));
      temph2->GetYaxis()->SetBinLabel(50, TString("0"));
      temph2->GetYaxis()->SetBinLabel(100, TString("+X"));
      temph2->SetOption("colz");
      HotColdMaps.push_back(temph2);
      temph2 = fs->make<TH2F>(Form("%s%i", "TID+", (int)(maplayer - 10)), "TID+", 100, -100, 100, 100, -100, 100);
      temph2->GetXaxis()->SetTitle("Global Y");
      temph2->GetXaxis()->SetBinLabel(1, TString("+Y"));
      temph2->GetXaxis()->SetBinLabel(50, TString("0"));
      temph2->GetXaxis()->SetBinLabel(100, TString("-Y"));
      temph2->GetYaxis()->SetTitle("Global X");
      temph2->GetYaxis()->SetBinLabel(1, TString("-X"));
      temph2->GetYaxis()->SetBinLabel(50, TString("0"));
      temph2->GetYaxis()->SetBinLabel(100, TString("+X"));
      temph2->SetOption("colz");
      HotColdMaps.push_back(temph2);
    } else if (maplayer > 13) {
      //We are in the TEC
      //Split by +/-
      temph2 = fs->make<TH2F>(Form("%s%i", "TEC-", (int)(maplayer - 13)), "TEC-", 100, -120, 120, 100, -120, 120);
      temph2->GetXaxis()->SetTitle("Global Y");
      temph2->GetXaxis()->SetBinLabel(1, TString("+Y"));
      temph2->GetXaxis()->SetBinLabel(50, TString("0"));
      temph2->GetXaxis()->SetBinLabel(100, TString("-Y"));
      temph2->GetYaxis()->SetTitle("Global X");
      temph2->GetYaxis()->SetBinLabel(1, TString("-X"));
      temph2->GetYaxis()->SetBinLabel(50, TString("0"));
      temph2->GetYaxis()->SetBinLabel(100, TString("+X"));
      temph2->SetOption("colz");
      HotColdMaps.push_back(temph2);
      temph2 = fs->make<TH2F>(Form("%s%i", "TEC+", (int)(maplayer - 13)), "TEC+", 100, -120, 120, 100, -120, 120);
      temph2->GetXaxis()->SetTitle("Global Y");
      temph2->GetXaxis()->SetBinLabel(1, TString("+Y"));
      temph2->GetXaxis()->SetBinLabel(50, TString("0"));
      temph2->GetXaxis()->SetBinLabel(100, TString("-Y"));
      temph2->GetYaxis()->SetTitle("Global X");
      temph2->GetYaxis()->SetBinLabel(1, TString("-X"));
      temph2->GetYaxis()->SetBinLabel(50, TString("0"));
      temph2->GetYaxis()->SetBinLabel(100, TString("+X"));
      temph2->SetOption("colz");
      HotColdMaps.push_back(temph2);
    }
  }
  for (Long_t mylayer = 1; mylayer <= siStripLayers_; mylayer++) {
    //Determine what kind of plot we want to write out
    //Loop through the entirety of each layer
    //Create an array of the histograms
    vector<hit>::const_iterator iter;
    for (iter = hits[mylayer].begin(); iter != hits[mylayer].end(); iter++) {
      //Looping over the particular layer
      //Fill by 360-x to get the proper location to compare with TKMaps of phi
      //Also global xy is messed up
      if (mylayer > 0 && mylayer <= 4) {
        //We are in the TIB
        float phi = ::calcPhi(iter->x, iter->y);
        HotColdMaps[mylayer - 1]->Fill(360. - phi, iter->z, 1.);
      } else if (mylayer > 4 && mylayer <= 10) {
        //We are in the TOB
        float phi = ::calcPhi(iter->x, iter->y);
        HotColdMaps[mylayer - 1]->Fill(360. - phi, iter->z, 1.);
      } else if (mylayer > 10 && mylayer <= 13) {
        //We are in the TID
        //There are 2 different maps here
        int side = (((iter->id) >> 13) & 0x3);
        if (side == 1)
          HotColdMaps[(mylayer - 1) + (mylayer - 11)]->Fill(-iter->y, iter->x, 1.);
        else if (side == 2)
          HotColdMaps[(mylayer - 1) + (mylayer - 10)]->Fill(-iter->y, iter->x, 1.);
        //if(side == 1) HotColdMaps[(mylayer - 1) + (mylayer - 11)]->Fill(iter->x,iter->y,1.);
        //else if(side == 2) HotColdMaps[(mylayer - 1) + (mylayer - 10)]->Fill(iter->x,iter->y,1.);
      } else if (mylayer > 13) {
        //We are in the TEC
        //There are 2 different maps here
        int side = (((iter->id) >> 18) & 0x3);
        if (side == 1)
          HotColdMaps[(mylayer + 2) + (mylayer - 14)]->Fill(-iter->y, iter->x, 1.);
        else if (side == 2)
          HotColdMaps[(mylayer + 2) + (mylayer - 13)]->Fill(-iter->y, iter->x, 1.);
        //if(side == 1) HotColdMaps[(mylayer + 2) + (mylayer - 14)]->Fill(iter->x,iter->y,1.);
        //else if(side == 2) HotColdMaps[(mylayer + 2) + (mylayer - 13)]->Fill(iter->x,iter->y,1.);
      }
    }
  }
  LOGPRINT << "Finished HotCold Map Generation\n";
}

void SiStripHitEffFromCalibTree::makeTKMap(bool autoTagging = false) {
  LOGPRINT << "Entering TKMap generation!\n";
  tkmap = new TrackerMap("  Detector Inefficiency  ");
  tkmapbad = new TrackerMap("  Inefficient Modules  ");
  tkmapeff = new TrackerMap(title_.Data());
  tkmapnum = new TrackerMap(" Detector numerator   ");
  tkmapden = new TrackerMap(" Detector denominator ");

  double myeff, mynum, myden, myeff_up;
  double layer_min_eff = 0;

  for (Long_t i = 1; i <= siStripLayers_; i++) {
    //Loop over every layer, extracting the information from
    //the map of the efficiencies
    layertotal[i] = 0;
    layerfound[i] = 0;
    TH1F* hEffInLayer =
        fs->make<TH1F>(Form("eff_layer%i", int(i)), Form("Module efficiency in layer %i", int(i)), 201, 0, 1.005);

    for (const auto& ih : modCounter[i]) {
      //We should be in the layer in question, and looping over all of the modules in said layer
      //Generate the list for the TKmap, and the bad module list
      mynum = (double)((ih.second).second);
      myden = (double)((ih.second).first);
      if (myden > 0)
        myeff = mynum / myden;
      else
        myeff = 0;
      hEffInLayer->Fill(myeff);

      if (!autoTagging) {
        if ((myden >= nModsMin_) && (myeff < threshold_)) {
          //We have a bad module, put it in the list!
          BadModules[ih.first] = myeff;
          tkmapbad->fillc(ih.first, 255, 0, 0);
          LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers) << ")  module " << ih.first
                   << " efficiency: " << myeff << " , " << mynum << "/" << myden;
        } else {
          //Fill the bad list with empty results for every module
          tkmapbad->fillc(ih.first, 255, 255, 255);
        }
        if (myeff < threshold_)
          LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers) << ")  module " << ih.first
                   << " efficiency: " << myeff << " , " << mynum << "/" << myden;
        if (myden < nModsMin_) {
          LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers) << ")  module " << ih.first
                   << " is under occupancy at " << myden;
        }
      }

      //Put any module into the TKMap
      tkmap->fill(ih.first, 1. - myeff);
      tkmapeff->fill(ih.first, myeff);
      tkmapnum->fill(ih.first, mynum);
      tkmapden->fill(ih.first, myden);

      //Add the number of hits in the layer
      layertotal[i] += long(myden);
      layerfound[i] += long(mynum);
    }

    if (autoTagging) {
      //Compute threshold to use for each layer
      hEffInLayer->GetXaxis()->SetRange(3, hEffInLayer->GetNbinsX() + 1);  // Remove from the avg modules below 1%
      layer_min_eff =
          hEffInLayer->GetMean() - 2.5 * hEffInLayer->GetRMS();  // uses RMS in case the distribution is wide
      if (threshold_ > 2.5 * hEffInLayer->GetRMS())
        layer_min_eff = hEffInLayer->GetMean() - threshold_;  // otherwise uses the parameter 'threshold'
      LOGPRINT << "Layer " << i << " threshold for bad modules: <" << layer_min_eff
               << "  (layer mean: " << hEffInLayer->GetMean() << " rms: " << hEffInLayer->GetRMS() << ")";

      hEffInLayer->GetXaxis()->SetRange(1, hEffInLayer->GetNbinsX() + 1);

      for (const auto& ih : modCounter[i]) {
        // Second loop over modules to tag inefficient ones
        mynum = (double)((ih.second).second);
        myden = (double)((ih.second).first);
        if (myden > 0)
          myeff = mynum / myden;
        else
          myeff = 0;
        // upper limit on the efficiency
        myeff_up = TEfficiency::Bayesian(myden, mynum, .99, 1, 1, true);
        if ((myden >= nModsMin_) && (myeff_up < layer_min_eff)) {
          //We have a bad module, put it in the list!
          BadModules[ih.first] = myeff;
          tkmapbad->fillc(ih.first, 255, 0, 0);
        } else {
          //Fill the bad list with empty results for every module
          tkmapbad->fillc(ih.first, 255, 255, 255);
        }
        if (myeff_up < layer_min_eff + 0.08)  // printing message also for modules slighly above (8%) the limit
          LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers) << ")  module " << ih.first
                   << " efficiency: " << myeff << " , " << mynum << "/" << myden << " , upper limit: " << myeff_up;
        if (myden < nModsMin_) {
          LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers) << ")  module " << ih.first
                   << " layer " << i << " is under occupancy at " << myden;
        }
      }
    }
  }
  tkmap->save(true, 0, 0, "SiStripHitEffTKMap.png");
  tkmapbad->save(true, 0, 0, "SiStripHitEffTKMapBad.png");
  tkmapeff->save(true, tkMapMin_, 1., "SiStripHitEffTKMapEff.png");
  tkmapnum->save(true, 0, 0, "SiStripHitEffTKMapNum.png");
  tkmapden->save(true, 0, 0, "SiStripHitEffTKMapDen.png");
  LOGPRINT << "Finished TKMap Generation\n";
}

void SiStripHitEffFromCalibTree::makeSQLite() {
  //Generate the SQLite file for use in the Database of the bad modules!
  LOGPRINT << "Entering SQLite file generation!\n";
  std::vector<unsigned int> BadStripList;
  unsigned short NStrips;
  unsigned int id1;
  std::unique_ptr<SiStripQuality> pQuality = std::make_unique<SiStripQuality>(detInfo_);
  //This is the list of the bad strips, use to mask out entire APVs
  //Now simply go through the bad hit list and mask out things that
  //are bad!
  for (const auto& it : BadModules) {
    //We need to figure out how many strips are in this particular module
    //To Mask correctly!
    NStrips = detInfo_.getNumberOfApvsAndStripLength(it.first).first * 128;
    LOGPRINT << "Number of strips module " << it.first << " is " << NStrips;
    BadStripList.push_back(pQuality->encode(0, NStrips, 0));
    //Now compact into a single bad module
    id1 = (unsigned int)it.first;
    LOGPRINT << "ID1 shoudl match list of modules above " << id1;
    quality_->compact(id1, BadStripList);
    SiStripQuality::Range range(BadStripList.begin(), BadStripList.end());
    quality_->put(id1, range);
    BadStripList.clear();
  }
  //Fill all the bad components now
  quality_->fillBadComponents();
}

void SiStripHitEffFromCalibTree::totalStatistics() {
  //Calculate the statistics by layer
  int totalfound = 0;
  int totaltotal = 0;
  double layereff;
  int subdetfound[5];
  int subdettotal[5];

  for (Long_t i = 1; i < 5; i++) {
    subdetfound[i] = 0;
    subdettotal[i] = 0;
  }

  for (Long_t i = 1; i <= siStripLayers_; i++) {
    layereff = double(layerfound[i]) / double(layertotal[i]);
    LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers) << ") has total efficiency " << layereff
             << " " << layerfound[i] << "/" << layertotal[i];
    totalfound += layerfound[i];
    totaltotal += layertotal[i];
    if (i < 5) {
      subdetfound[1] += layerfound[i];
      subdettotal[1] += layertotal[i];
    }
    if (i >= 5 && i < 11) {
      subdetfound[2] += layerfound[i];
      subdettotal[2] += layertotal[i];
    }
    if (i >= 11 && i < 14) {
      subdetfound[3] += layerfound[i];
      subdettotal[3] += layertotal[i];
    }
    if (i >= 14) {
      subdetfound[4] += layerfound[i];
      subdettotal[4] += layertotal[i];
    }
  }

  LOGPRINT << "The total efficiency is " << double(totalfound) / double(totaltotal);
  LOGPRINT << "      TIB: " << double(subdetfound[1]) / subdettotal[1] << " " << subdetfound[1] << "/"
           << subdettotal[1];
  LOGPRINT << "      TOB: " << double(subdetfound[2]) / subdettotal[2] << " " << subdetfound[2] << "/"
           << subdettotal[2];
  LOGPRINT << "      TID: " << double(subdetfound[3]) / subdettotal[3] << " " << subdetfound[3] << "/"
           << subdettotal[3];
  LOGPRINT << "      TEC: " << double(subdetfound[4]) / subdettotal[4] << " " << subdetfound[4] << "/"
           << subdettotal[4];
}

void SiStripHitEffFromCalibTree::makeSummary() {
  //setTDRStyle();

  int nLayers = 34;
  if (showRings_)
    nLayers = 30;
  if (!showEndcapSides_) {
    if (!showRings_)
      nLayers = siStripLayers_;
    else
      nLayers = 20;
  }

  TH1F* found = fs->make<TH1F>("found", "found", nLayers + 1, 0, nLayers + 1);
  TH1F* all = fs->make<TH1F>("all", "all", nLayers + 1, 0, nLayers + 1);
  TH1F* found2 = fs->make<TH1F>("found2", "found2", nLayers + 1, 0, nLayers + 1);
  TH1F* all2 = fs->make<TH1F>("all2", "all2", nLayers + 1, 0, nLayers + 1);
  // first bin only to keep real data off the y axis so set to -1
  found->SetBinContent(0, -1);
  all->SetBinContent(0, 1);

  // new ROOT version: TGraph::Divide don't handle null or negative values
  for (Long_t i = 1; i < nLayers + 2; ++i) {
    found->SetBinContent(i, 1e-6);
    all->SetBinContent(i, 1);
    found2->SetBinContent(i, 1e-6);
    all2->SetBinContent(i, 1);
  }

  TCanvas* c7 = new TCanvas("c7", " test ", 10, 10, 800, 600);
  c7->SetFillColor(0);
  c7->SetGrid();

  int nLayers_max = nLayers + 1;  // barrel+endcap
  if (!showEndcapSides_)
    nLayers_max = 11;  // barrel
  for (Long_t i = 1; i < nLayers_max; ++i) {
    LOGPRINT << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i]
             << "    B = " << goodlayertotal[i];
    if (goodlayertotal[i] > 5) {
      found->SetBinContent(i, goodlayerfound[i]);
      all->SetBinContent(i, goodlayertotal[i]);
    }

    LOGPRINT << "Filling all modules layer " << i << ":  S = " << alllayerfound[i] << "    B = " << alllayertotal[i];
    if (alllayertotal[i] > 5) {
      found2->SetBinContent(i, alllayerfound[i]);
      all2->SetBinContent(i, alllayertotal[i]);
    }
  }

  // endcap - merging sides
  if (!showEndcapSides_) {
    for (Long_t i = 11; i < 14; ++i) {  // TID disks
      LOGPRINT << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i] + goodlayerfound[i + 3]
               << "    B = " << goodlayertotal[i] + goodlayertotal[i + 3];
      if (goodlayertotal[i] + goodlayertotal[i + 3] > 5) {
        found->SetBinContent(i, goodlayerfound[i] + goodlayerfound[i + 3]);
        all->SetBinContent(i, goodlayertotal[i] + goodlayertotal[i + 3]);
      }
      LOGPRINT << "Filling all modules layer " << i << ":  S = " << alllayerfound[i] + alllayerfound[i + 3]
               << "    B = " << alllayertotal[i] + alllayertotal[i + 3];
      if (alllayertotal[i] + alllayertotal[i + 3] > 5) {
        found2->SetBinContent(i, alllayerfound[i] + alllayerfound[i + 3]);
        all2->SetBinContent(i, alllayertotal[i] + alllayertotal[i + 3]);
      }
    }
    for (Long_t i = 17; i < 17 + nTEClayers; ++i) {  // TEC disks
      LOGPRINT << "Fill only good modules layer " << i - 3
               << ":  S = " << goodlayerfound[i] + goodlayerfound[i + nTEClayers]
               << "    B = " << goodlayertotal[i] + goodlayertotal[i + nTEClayers];
      if (goodlayertotal[i] + goodlayertotal[i + nTEClayers] > 5) {
        found->SetBinContent(i - 3, goodlayerfound[i] + goodlayerfound[i + nTEClayers]);
        all->SetBinContent(i - 3, goodlayertotal[i] + goodlayertotal[i + nTEClayers]);
      }
      LOGPRINT << "Filling all modules layer " << i - 3 << ":  S = " << alllayerfound[i] + alllayerfound[i + nTEClayers]
               << "    B = " << alllayertotal[i] + alllayertotal[i + nTEClayers];
      if (alllayertotal[i] + alllayertotal[i + nTEClayers] > 5) {
        found2->SetBinContent(i - 3, alllayerfound[i] + alllayerfound[i + nTEClayers]);
        all2->SetBinContent(i - 3, alllayertotal[i] + alllayertotal[i + nTEClayers]);
      }
    }
  }

  found->Sumw2();
  all->Sumw2();

  found2->Sumw2();
  all2->Sumw2();

  TGraphAsymmErrors* gr = fs->make<TGraphAsymmErrors>(nLayers + 1);
  gr->SetName("eff_good");
  gr->BayesDivide(found, all);

  TGraphAsymmErrors* gr2 = fs->make<TGraphAsymmErrors>(nLayers + 1);
  gr2->SetName("eff_all");
  gr2->BayesDivide(found2, all2);

  for (int j = 0; j < nLayers + 1; j++) {
    gr->SetPointError(j, 0., 0., gr->GetErrorYlow(j), gr->GetErrorYhigh(j));
    gr2->SetPointError(j, 0., 0., gr2->GetErrorYlow(j), gr2->GetErrorYhigh(j));
  }

  gr->GetXaxis()->SetLimits(0, nLayers);
  gr->SetMarkerColor(2);
  gr->SetMarkerSize(1.2);
  gr->SetLineColor(2);
  gr->SetLineWidth(4);
  gr->SetMarkerStyle(20);
  gr->SetMinimum(effPlotMin_);
  gr->SetMaximum(1.001);
  gr->GetYaxis()->SetTitle("Efficiency");
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleBorderSize(0);
  gr->SetTitle(title_);

  gr2->GetXaxis()->SetLimits(0, nLayers);
  gr2->SetMarkerColor(1);
  gr2->SetMarkerSize(1.2);
  gr2->SetLineColor(1);
  gr2->SetLineWidth(4);
  gr2->SetMarkerStyle(21);
  gr2->SetMinimum(effPlotMin_);
  gr2->SetMaximum(1.001);
  gr2->GetYaxis()->SetTitle("Efficiency");
  gr2->SetTitle(title_);

  for (Long_t k = 1; k < nLayers + 1; k++) {
    TString label;
    if (showEndcapSides_)
      label = getLayerSideName(k);
    else
      label = ::layerName(k, showRings_, nTEClayers);
    if (!showTOB6TEC9_) {
      if (k == 10)
        label = "";
      if (!showRings_ && k == nLayers)
        label = "";
      if (!showRings_ && showEndcapSides_ && k == 25)
        label = "";
    }
    if (!showRings_) {
      if (showEndcapSides_) {
        gr->GetXaxis()->SetBinLabel(((k + 1) * 100 + 2) / (nLayers)-4, label);
        gr2->GetXaxis()->SetBinLabel(((k + 1) * 100 + 2) / (nLayers)-4, label);
      } else {
        gr->GetXaxis()->SetBinLabel((k + 1) * 100 / (nLayers)-6, label);
        gr2->GetXaxis()->SetBinLabel((k + 1) * 100 / (nLayers)-6, label);
      }
    } else {
      if (showEndcapSides_) {
        gr->GetXaxis()->SetBinLabel((k + 1) * 100 / (nLayers)-4, label);
        gr2->GetXaxis()->SetBinLabel((k + 1) * 100 / (nLayers)-4, label);
      } else {
        gr->GetXaxis()->SetBinLabel((k + 1) * 100 / (nLayers)-7, label);
        gr2->GetXaxis()->SetBinLabel((k + 1) * 100 / (nLayers)-7, label);
      }
    }
  }

  gr->Draw("AP");
  gr->GetXaxis()->SetNdivisions(36);

  c7->cd();
  TPad* overlay = new TPad("overlay", "", 0, 0, 1, 1);
  overlay->SetFillStyle(4000);
  overlay->SetFillColor(0);
  overlay->SetFrameFillStyle(4000);
  overlay->Draw("same");
  overlay->cd();
  if (!showOnlyGoodModules_)
    gr2->Draw("AP");

  TLegend* leg = new TLegend(0.70, 0.27, 0.88, 0.40);
  leg->AddEntry(gr, "Good Modules", "p");
  if (!showOnlyGoodModules_)
    leg->AddEntry(gr2, "All Modules", "p");
  leg->SetTextSize(0.020);
  leg->SetFillColor(0);
  leg->Draw("same");

  c7->SaveAs("Summary.png");
}

void SiStripHitEffFromCalibTree::makeSummaryVsBx() {
  LOGPRINT << "Computing efficiency vs bx";

  unsigned int nLayers = siStripLayers_;
  if (showRings_)
    nLayers = 20;

  for (unsigned int ilayer = 1; ilayer < nLayers; ilayer++) {
    for (unsigned int ibx = 0; ibx <= nBxInAnOrbit_; ibx++) {
      layerfound_vsBX[ilayer]->SetBinContent(ibx, 1e-6);
      layertotal_vsBX[ilayer]->SetBinContent(ibx, 1);
    }

    for (const auto& iterMapvsBx : layerfound_perBx) {
      layerfound_vsBX[ilayer]->SetBinContent(iterMapvsBx.first, iterMapvsBx.second[ilayer]);
    }
    for (const auto& iterMapvsBx : layertotal_perBx) {
      if (iterMapvsBx.second[ilayer] > 0) {
        layertotal_vsBX[ilayer]->SetBinContent(iterMapvsBx.first, iterMapvsBx.second[ilayer]);
      }
    }

    layerfound_vsBX[ilayer]->Sumw2();
    layertotal_vsBX[ilayer]->Sumw2();

    TGraphAsymmErrors* geff = fs->make<TGraphAsymmErrors>(nBxInAnOrbit_ - 1);
    geff->SetName(Form("effVsBx_layer%i", ilayer));

    geff->SetTitle(fmt::format("Hit Efficiency vs bx - {}", ::layerName(ilayer, showRings_, nTEClayers)).c_str());
    geff->BayesDivide(layerfound_vsBX[ilayer], layertotal_vsBX[ilayer]);

    //Average over trains
    TGraphAsymmErrors* geff_avg = fs->make<TGraphAsymmErrors>();
    geff_avg->SetName(Form("effVsBxAvg_layer%i", ilayer));
    geff_avg->SetTitle(fmt::format("Hit Efficiency vs bx - {}", ::layerName(ilayer, showRings_, nTEClayers)).c_str());
    geff_avg->SetMarkerStyle(20);
    int ibx = 0;
    int previous_bx = -80;
    int delta_bx = 0;
    int nbx = 0;
    int found = 0;
    int total = 0;
    double sum_bx = 0;
    int ipt = 0;
    float low, up, eff;
    int firstbx = 0;
    for (const auto& iterMapvsBx : layertotal_perBx) {
      ibx = iterMapvsBx.first;
      delta_bx = ibx - previous_bx;
      // consider a new train
      if (delta_bx > (int)spaceBetweenTrains_ && nbx > 0 && total > 0) {
        eff = found / (float)total;
        //LOGPRINT<<"new train "<<ipt<<" "<<sum_bx/nbx<<" "<<eff<<endl;
        geff_avg->SetPoint(ipt, sum_bx / nbx, eff);
        low = TEfficiency::Bayesian(total, found, .683, 1, 1, false);
        up = TEfficiency::Bayesian(total, found, .683, 1, 1, true);
        geff_avg->SetPointError(ipt, sum_bx / nbx - firstbx, previous_bx - sum_bx / nbx, eff - low, up - eff);
        ipt++;
        sum_bx = 0;
        found = 0;
        total = 0;
        nbx = 0;
        firstbx = ibx;
      }
      sum_bx += ibx;
      found += layerfound_vsBX[ilayer]->GetBinContent(ibx);
      total += layertotal_vsBX[ilayer]->GetBinContent(ibx);
      nbx++;

      previous_bx = ibx;
    }
    //last train
    eff = found / (float)total;
    //LOGPRINT<<"new train "<<ipt<<" "<<sum_bx/nbx<<" "<<eff<<endl;
    geff_avg->SetPoint(ipt, sum_bx / nbx, eff);
    low = TEfficiency::Bayesian(total, found, .683, 1, 1, false);
    up = TEfficiency::Bayesian(total, found, .683, 1, 1, true);
    geff_avg->SetPointError(ipt, sum_bx / nbx - firstbx, previous_bx - sum_bx / nbx, eff - low, up - eff);
  }
}

void SiStripHitEffFromCalibTree::computeEff(vector<TH1F*>& vhfound, vector<TH1F*>& vhtotal, string name) {
  unsigned int nLayers = siStripLayers_;
  if (showRings_)
    nLayers = 20;

  TH1F* hfound;
  TH1F* htotal;

  for (unsigned int ilayer = 1; ilayer < nLayers; ilayer++) {
    hfound = vhfound[ilayer];
    htotal = vhtotal[ilayer];

    hfound->Sumw2();
    htotal->Sumw2();

    // new ROOT version: TGraph::Divide don't handle null or negative values
    for (Long_t i = 0; i < hfound->GetNbinsX() + 1; ++i) {
      if (hfound->GetBinContent(i) == 0)
        hfound->SetBinContent(i, 1e-6);
      if (htotal->GetBinContent(i) == 0)
        htotal->SetBinContent(i, 1);
    }

    TGraphAsymmErrors* geff = fs->make<TGraphAsymmErrors>(hfound->GetNbinsX());
    geff->SetName(Form("%s_layer%i", name.c_str(), ilayer));
    geff->BayesDivide(hfound, htotal);
    if (name == "effVsLumi")
      geff->SetTitle(
          fmt::format("Hit Efficiency vs inst. lumi. - {}", ::layerName(ilayer, showRings_, nTEClayers)).c_str());
    if (name == "effVsPU")
      geff->SetTitle(fmt::format("Hit Efficiency vs pileup - {}", ::layerName(ilayer, showRings_, nTEClayers)).c_str());
    if (name == "effVsCM")
      geff->SetTitle(
          fmt::format("Hit Efficiency vs common Mode - {}", ::layerName(ilayer, showRings_, nTEClayers)).c_str());
    geff->SetMarkerStyle(20);
  }
}

void SiStripHitEffFromCalibTree::makeSummaryVsLumi() {
  LOGPRINT << "Computing efficiency vs lumi";

  if (instLumiHisto->GetEntries())  // from infos per event
    LOGPRINT << "Avg conditions (avg+/-rms):   lumi :" << instLumiHisto->GetMean() << "+/-" << instLumiHisto->GetRMS()
             << "   pu: " << PUHisto->GetMean() << "+/-" << PUHisto->GetRMS();

  else {  // from infos per hit

    unsigned int nLayers = siStripLayers_;
    if (showRings_)
      nLayers = 20;
    unsigned int nLayersForAvg = 0;
    float layerLumi = 0;
    float layerPU = 0;
    float avgLumi = 0;
    float avgPU = 0;

    LOGPRINT << "Lumi summary:  (avg over trajectory measurements)";
    for (unsigned int ilayer = 1; ilayer < nLayers; ilayer++) {
      layerLumi = layertotal_vsLumi[ilayer]->GetMean();
      layerPU = layertotal_vsPU[ilayer]->GetMean();
      //LOGPRINT<<" layer "<<ilayer<<"  lumi: "<<layerLumi<<"  pu: "<<layerPU<<endl;
      if (layerLumi != 0 && layerPU != 0) {
        avgLumi += layerLumi;
        avgPU += layerPU;
        nLayersForAvg++;
      }
    }
    avgLumi /= nLayersForAvg;
    avgPU /= nLayersForAvg;
    LOGPRINT << "Avg conditions:   lumi :" << avgLumi << "  pu: " << avgPU;
  }

  computeEff(layerfound_vsLumi, layertotal_vsLumi, "effVsLumi");
  computeEff(layerfound_vsPU, layertotal_vsPU, "effVsPU");
}

void SiStripHitEffFromCalibTree::makeSummaryVsCM() {
  LOGPRINT << "Computing efficiency vs CM";
  computeEff(layerfound_vsCM, layertotal_vsCM, "effVsCM");
}

TString SiStripHitEffFromCalibTree::getLayerSideName(Long_t k) {
  TString layername = "";
  TString ringlabel = "D";
  if (showRings_)
    ringlabel = "R";
  if (k > 0 && k < 5) {
    layername = TString("TIB L") + k;
  } else if (k > 4 && k < 11) {
    layername = TString("TOB L") + (k - 4);
  } else if (k > 10 && k < 14) {
    layername = TString("TID- ") + ringlabel + (k - 10);
  } else if (k > 13 && k < 17) {
    layername = TString("TID+ ") + ringlabel + (k - 13);
  } else if (k > 16 && k < 17 + nTEClayers) {
    layername = TString("TEC- ") + ringlabel + (k - 16);
  } else if (k > 16 + nTEClayers) {
    layername = TString("TEC+ ") + ringlabel + (k - 16 - nTEClayers);
  }

  return layername;
}

std::unique_ptr<SiStripBadStrip> SiStripHitEffFromCalibTree::getNewObject() {
  //Need this for a Condition DB Writer
  //Initialize a return variable
  auto obj = std::make_unique<SiStripBadStrip>();

  SiStripBadStrip::RegistryIterator rIter = quality_->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rIterEnd = quality_->getRegistryVectorEnd();

  for (; rIter != rIterEnd; ++rIter) {
    SiStripBadStrip::Range range(quality_->getDataVectorBegin() + rIter->ibegin,
                                 quality_->getDataVectorBegin() + rIter->iend);
    if (!obj->put(rIter->detid, range))
      edm::LogError("SiStripHitEffFromCalibTree")
          << "[SiStripHitEffFromCalibTree::getNewObject] detid already exists" << std::endl;
  }

  return obj;
}

void SiStripHitEffFromCalibTree::setBadComponents(
    int i, int component, SiStripQuality::BadComponent& BC, std::stringstream ssV[4][19], int NBadComponent[4][19][4]) {
  int napv = detInfo_.getNumberOfApvsAndStripLength(BC.detid).first;

  ssV[i][component] << "\n\t\t " << BC.detid << " \t " << BC.BadModule << " \t " << ((BC.BadFibers) & 0x1) << " ";
  if (napv == 4)
    ssV[i][component] << "x " << ((BC.BadFibers >> 1) & 0x1);

  if (napv == 6)
    ssV[i][component] << ((BC.BadFibers >> 1) & 0x1) << " " << ((BC.BadFibers >> 2) & 0x1);
  ssV[i][component] << " \t " << ((BC.BadApvs) & 0x1) << " " << ((BC.BadApvs >> 1) & 0x1) << " ";
  if (napv == 4)
    ssV[i][component] << "x x " << ((BC.BadApvs >> 2) & 0x1) << " " << ((BC.BadApvs >> 3) & 0x1);
  if (napv == 6)
    ssV[i][component] << ((BC.BadApvs >> 2) & 0x1) << " " << ((BC.BadApvs >> 3) & 0x1) << " "
                      << ((BC.BadApvs >> 4) & 0x1) << " " << ((BC.BadApvs >> 5) & 0x1) << " ";

  if (BC.BadApvs) {
    NBadComponent[i][0][2] += ((BC.BadApvs >> 5) & 0x1) + ((BC.BadApvs >> 4) & 0x1) + ((BC.BadApvs >> 3) & 0x1) +
                              ((BC.BadApvs >> 2) & 0x1) + ((BC.BadApvs >> 1) & 0x1) + ((BC.BadApvs) & 0x1);
    NBadComponent[i][component][2] += ((BC.BadApvs >> 5) & 0x1) + ((BC.BadApvs >> 4) & 0x1) +
                                      ((BC.BadApvs >> 3) & 0x1) + ((BC.BadApvs >> 2) & 0x1) +
                                      ((BC.BadApvs >> 1) & 0x1) + ((BC.BadApvs) & 0x1);
  }
  if (BC.BadFibers) {
    NBadComponent[i][0][1] += ((BC.BadFibers >> 2) & 0x1) + ((BC.BadFibers >> 1) & 0x1) + ((BC.BadFibers) & 0x1);
    NBadComponent[i][component][1] +=
        ((BC.BadFibers >> 2) & 0x1) + ((BC.BadFibers >> 1) & 0x1) + ((BC.BadFibers) & 0x1);
  }
  if (BC.BadModule) {
    NBadComponent[i][0][0]++;
    NBadComponent[i][component][0]++;
  }
}

DEFINE_FWK_MODULE(SiStripHitEffFromCalibTree);
