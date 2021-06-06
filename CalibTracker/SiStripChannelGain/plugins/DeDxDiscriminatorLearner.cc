// -*- C++ -*-
//
// Package:    DeDxDiscriminatorLearner
// Class:      DeDxDiscriminatorLearner
//
/**\class DeDxDiscriminatorLearner DeDxDiscriminatorLearner.cc RecoTracker/DeDxDiscriminatorLearner/src/DeDxDiscriminatorLearner.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic Quertenmont(querten)
//         Created:  Mon October 20 10:09:02 CEST 2008
//

// system include files
#include <memory>

#include "CalibTracker/SiStripChannelGain/plugins/DeDxDiscriminatorLearner.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorLearner::DeDxDiscriminatorLearner(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<PhysicsTools::Calibration::HistogramD3D>(iConfig) {
  m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
  m_trajTrackAssociationTag =
      consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation"));
  m_tkGeomToken = esConsumes<edm::Transition::BeginRun>();

  P_Min = iConfig.getParameter<double>("P_Min");
  P_Max = iConfig.getParameter<double>("P_Max");
  P_NBins = iConfig.getParameter<int>("P_NBins");
  Path_Min = iConfig.getParameter<double>("Path_Min");
  Path_Max = iConfig.getParameter<double>("Path_Max");
  Path_NBins = iConfig.getParameter<int>("Path_NBins");
  Charge_Min = iConfig.getParameter<double>("Charge_Min");
  Charge_Max = iConfig.getParameter<double>("Charge_Max");
  Charge_NBins = iConfig.getParameter<int>("Charge_NBins");

  MinTrackMomentum = iConfig.getUntrackedParameter<double>("minTrackMomentum", 5.0);
  MaxTrackMomentum = iConfig.getUntrackedParameter<double>("maxTrackMomentum", 99999.0);
  MinTrackEta = iConfig.getUntrackedParameter<double>("minTrackEta", -5.0);
  MaxTrackEta = iConfig.getUntrackedParameter<double>("maxTrackEta", 5.0);
  MaxNrStrips = iConfig.getUntrackedParameter<unsigned>("maxNrStrips", 255);
  MinTrackHits = iConfig.getUntrackedParameter<unsigned>("MinTrackHits", 4);

  algoMode = iConfig.getUntrackedParameter<string>("AlgoMode", "SingleJob");
  HistoFile = iConfig.getUntrackedParameter<string>("HistoFile", "out.root");
  VInputFiles = iConfig.getUntrackedParameter<vector<string> >("InputFiles");

  shapetest = iConfig.getParameter<bool>("ShapeTest");
  useCalibration = iConfig.getUntrackedParameter<bool>("UseCalibration");
  m_calibrationPath = iConfig.getUntrackedParameter<string>("calibrationPath");
}

DeDxDiscriminatorLearner::~DeDxDiscriminatorLearner() {}

// ------------ method called once each job just before starting event loop  ------------

void DeDxDiscriminatorLearner::algoBeginJob(const edm::EventSetup& iSetup) {
  Charge_Vs_Path = new TH3F("Charge_Vs_Path",
                            "Charge_Vs_Path",
                            P_NBins,
                            P_Min,
                            P_Max,
                            Path_NBins,
                            Path_Min,
                            Path_Max,
                            Charge_NBins,
                            Charge_Min,
                            Charge_Max);

  if (useCalibration && calibGains.empty()) {
    const auto& tkGeom = iSetup.getData(m_tkGeomToken);

    m_off = tkGeom.offsetDU(GeomDetEnumerators::PixelBarrel);  //index start at the first pixel

    DeDxTools::makeCalibrationMap(m_calibrationPath, tkGeom, calibGains, m_off);
  }

  //Read the calibTree if in calibTree mode
  if (strcmp(algoMode.c_str(), "CalibTree") == 0)
    algoAnalyzeTheTree(iSetup);
}

// ------------ method called once each job just after ending the event loop  ------------

void DeDxDiscriminatorLearner::algoEndJob() {
  if (strcmp(algoMode.c_str(), "MultiJob") == 0) {
    TFile* Output = new TFile(HistoFile.c_str(), "RECREATE");
    Charge_Vs_Path->Write();
    Output->Write();
    Output->Close();
  } else if (strcmp(algoMode.c_str(), "WriteOnDB") == 0) {
    TFile* Input = new TFile(HistoFile.c_str());
    Charge_Vs_Path = (TH3F*)(Input->FindObjectAny("Charge_Vs_Path"))->Clone();
    Input->Close();
  } else if (strcmp(algoMode.c_str(), "CalibTree") == 0) {
    TFile* Output = new TFile(HistoFile.c_str(), "RECREATE");
    Charge_Vs_Path->Write();
    Output->Write();
    Output->Close();
    TFile* Input = new TFile(HistoFile.c_str());
    Charge_Vs_Path = (TH3F*)(Input->FindObjectAny("Charge_Vs_Path"))->Clone();
    Input->Close();
  }
}

void DeDxDiscriminatorLearner::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByToken(m_trajTrackAssociationTag, trajTrackAssociationHandle);

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag, trackCollectionHandle);

  unsigned track_index = 0;
  for (TrajTrackAssociationCollection::const_iterator it = trajTrackAssociationHandle->begin();
       it != trajTrackAssociationHandle->end();
       ++it, track_index++) {
    const Track& track = *it->val;
    const Trajectory& traj = *it->key;

    if (track.eta() < MinTrackEta || track.eta() > MaxTrackEta) {
      continue;
    }
    if (track.pt() < MinTrackMomentum || track.pt() > MaxTrackMomentum) {
      continue;
    }
    if (track.found() < MinTrackHits) {
      continue;
    }

    const vector<TrajectoryMeasurement>& measurements = traj.measurements();
    for (vector<TrajectoryMeasurement>::const_iterator it = measurements.begin(); it != measurements.end(); it++) {
      TrajectoryStateOnSurface trajState = it->updatedState();
      if (!trajState.isValid())
        continue;

      const TrackingRecHit* recHit = (*it->recHit()).hit();
      if (!recHit || !recHit->isValid())
        continue;
      LocalVector trackDirection = trajState.localDirection();
      float cosine = trackDirection.z() / trackDirection.mag();

      processHit(recHit, trajState.localMomentum().mag(), cosine, trajState);
    }
  }
}

void DeDxDiscriminatorLearner::processHit(const TrackingRecHit* recHit,
                                          float trackMomentum,
                                          float& cosine,
                                          const TrajectoryStateOnSurface& trajState) {
  auto const& thit = static_cast<BaseTrackerRecHit const&>(*recHit);
  if (!thit.isValid())
    return;

  auto const& clus = thit.firstClusterRef();
  if (!clus.isValid())
    return;

  int NSaturating = 0;
  if (clus.isPixel()) {
    return;
  } else if (clus.isStrip() && !thit.isMatched()) {
    auto& detUnit = *(recHit->detUnit());
    auto& cluster = clus.stripCluster();
    if (cluster.amplitudes().size() > MaxNrStrips) {
      return;
    }
    if (DeDxTools::IsSpanningOver2APV(cluster.firstStrip(), cluster.amplitudes().size())) {
      return;
    }
    if (!DeDxTools::IsFarFromBorder(trajState, &detUnit)) {
      return;
    }
    float pathLen = 10.0 * detUnit.surface().bounds().thickness() / fabs(cosine);
    float chargeAbs = DeDxTools::getCharge(&cluster, NSaturating, detUnit, calibGains, m_off);
    float charge = chargeAbs / pathLen;
    if (!shapetest || (shapetest && DeDxTools::shapeSelection(cluster))) {
      Charge_Vs_Path->Fill(trackMomentum, pathLen, charge);
    }
  } else if (clus.isStrip() && thit.isMatched()) {
    const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(recHit);
    if (!matchedHit)
      return;
    const GluedGeomDet* gdet = static_cast<const GluedGeomDet*>(matchedHit->det());

    auto& detUnitM = *(gdet->monoDet());
    auto& clusterM = matchedHit->monoCluster();
    if (clusterM.amplitudes().size() > MaxNrStrips) {
      return;
    }
    if (DeDxTools::IsSpanningOver2APV(clusterM.firstStrip(), clusterM.amplitudes().size())) {
      return;
    }
    if (!DeDxTools::IsFarFromBorder(trajState, &detUnitM)) {
      return;
    }
    float pathLen = 10.0 * detUnitM.surface().bounds().thickness() / fabs(cosine);
    float chargeAbs = DeDxTools::getCharge(&clusterM, NSaturating, detUnitM, calibGains, m_off);
    float charge = chargeAbs / pathLen;
    if (!shapetest || (shapetest && DeDxTools::shapeSelection(clusterM))) {
      Charge_Vs_Path->Fill(trackMomentum, pathLen, charge);
    }

    auto& detUnitS = *(gdet->stereoDet());
    auto& clusterS = matchedHit->stereoCluster();
    if (clusterS.amplitudes().size() > MaxNrStrips) {
      return;
    }
    if (DeDxTools::IsSpanningOver2APV(clusterS.firstStrip(), clusterS.amplitudes().size())) {
      return;
    }
    if (!DeDxTools::IsFarFromBorder(trajState, &detUnitS)) {
      return;
    }
    pathLen = 10.0 * detUnitS.surface().bounds().thickness() / fabs(cosine);
    chargeAbs = DeDxTools::getCharge(&clusterS, NSaturating, detUnitS, calibGains, m_off);
    charge = chargeAbs / pathLen;
    if (!shapetest || (shapetest && DeDxTools::shapeSelection(clusterS))) {
      Charge_Vs_Path->Fill(trackMomentum, pathLen, charge);
    }
  }
}

//this function is only used when we run over a calibTree instead of running over EDM files
void DeDxDiscriminatorLearner::algoAnalyzeTheTree(const edm::EventSetup& iSetup) {
  const auto& tkGeom = iSetup.getData(m_tkGeomToken);

  unsigned int NEvent = 0;
  for (unsigned int i = 0; i < VInputFiles.size(); i++) {
    printf("Openning file %3i/%3i --> %s\n", i + 1, (int)VInputFiles.size(), (char*)(VInputFiles[i].c_str()));
    fflush(stdout);
    TChain* tree = new TChain("gainCalibrationTree/tree");
    tree->Add(VInputFiles[i].c_str());

    TString EventPrefix("");
    TString EventSuffix("");

    TString TrackPrefix("track");
    TString TrackSuffix("");

    TString CalibPrefix("GainCalibration");
    TString CalibSuffix("");

    unsigned int eventnumber = 0;
    tree->SetBranchAddress(EventPrefix + "event" + EventSuffix, &eventnumber, nullptr);
    unsigned int runnumber = 0;
    tree->SetBranchAddress(EventPrefix + "run" + EventSuffix, &runnumber, nullptr);
    std::vector<bool>* TrigTech = nullptr;
    tree->SetBranchAddress(EventPrefix + "TrigTech" + EventSuffix, &TrigTech, nullptr);

    std::vector<double>* trackchi2ndof = nullptr;
    tree->SetBranchAddress(TrackPrefix + "chi2ndof" + TrackSuffix, &trackchi2ndof, nullptr);
    std::vector<float>* trackp = nullptr;
    tree->SetBranchAddress(TrackPrefix + "momentum" + TrackSuffix, &trackp, nullptr);
    std::vector<float>* trackpt = nullptr;
    tree->SetBranchAddress(TrackPrefix + "pt" + TrackSuffix, &trackpt, nullptr);
    std::vector<double>* tracketa = nullptr;
    tree->SetBranchAddress(TrackPrefix + "eta" + TrackSuffix, &tracketa, nullptr);
    std::vector<double>* trackphi = nullptr;
    tree->SetBranchAddress(TrackPrefix + "phi" + TrackSuffix, &trackphi, nullptr);
    std::vector<unsigned int>* trackhitsvalid = nullptr;
    tree->SetBranchAddress(TrackPrefix + "hitsvalid" + TrackSuffix, &trackhitsvalid, nullptr);

    std::vector<int>* trackindex = nullptr;
    tree->SetBranchAddress(CalibPrefix + "trackindex" + CalibSuffix, &trackindex, nullptr);
    std::vector<unsigned int>* rawid = nullptr;
    tree->SetBranchAddress(CalibPrefix + "rawid" + CalibSuffix, &rawid, nullptr);
    std::vector<unsigned short>* firststrip = nullptr;
    tree->SetBranchAddress(CalibPrefix + "firststrip" + CalibSuffix, &firststrip, nullptr);
    std::vector<unsigned short>* nstrips = nullptr;
    tree->SetBranchAddress(CalibPrefix + "nstrips" + CalibSuffix, &nstrips, nullptr);
    std::vector<unsigned int>* charge = nullptr;
    tree->SetBranchAddress(CalibPrefix + "charge" + CalibSuffix, &charge, nullptr);
    std::vector<float>* path = nullptr;
    tree->SetBranchAddress(CalibPrefix + "path" + CalibSuffix, &path, nullptr);
    std::vector<unsigned char>* amplitude = nullptr;
    tree->SetBranchAddress(CalibPrefix + "amplitude" + CalibSuffix, &amplitude, nullptr);
    std::vector<double>* gainused = nullptr;
    tree->SetBranchAddress(CalibPrefix + "gainused" + CalibSuffix, &gainused, nullptr);

    printf("Number of Events = %i + %i = %i\n",
           NEvent,
           (unsigned int)tree->GetEntries(),
           (unsigned int)(NEvent + tree->GetEntries()));
    NEvent += tree->GetEntries();
    printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
    printf("Looping on the Tree          :");
    int TreeStep = tree->GetEntries() / 50;
    if (TreeStep <= 1)
      TreeStep = 1;
    for (unsigned int ientry = 0; ientry < tree->GetEntries(); ientry++) {
      if (ientry % TreeStep == 0) {
        printf(".");
        fflush(stdout);
      }
      tree->GetEntry(ientry);

      int FirstAmplitude = 0;
      for (unsigned int c = 0; c < (*path).size(); c++) {
        FirstAmplitude += (*nstrips)[c];
        int t = (*trackindex)[c];
        if ((*trackpt)[t] < 5)
          continue;
        if ((*trackhitsvalid)[t] < 5)
          continue;

        int Charge = 0;
        if (useCalibration) {
          auto& gains = calibGains[tkGeom.idToDetUnit(DetId((*rawid)[c]))->index() - m_off];
          auto& gain = gains[(*firststrip)[c] / 128];
          for (unsigned int s = 0; s < (*nstrips)[c]; s++) {
            int StripCharge = (*amplitude)[FirstAmplitude - (*nstrips)[c] + s];
            if (StripCharge < 254) {
              StripCharge = (int)(StripCharge / gain);
              if (StripCharge >= 1024) {
                StripCharge = 255;
              } else if (StripCharge >= 254) {
                StripCharge = 254;
              }
            }
            Charge += StripCharge;
          }
        } else {
          Charge = (*charge)[c];
        }

        //          printf("ChargeDifference = %i Vs %i with Gain = %f\n",(*charge)[c],Charge,Gains[(*rawid)[c]]);
        double ClusterChargeOverPath = ((double)Charge) / (*path)[c];
        Charge_Vs_Path->Fill((*trackp)[t], (*path)[c], ClusterChargeOverPath);
      }
    }
    printf("\n");
  }
}

std::unique_ptr<PhysicsTools::Calibration::HistogramD3D> DeDxDiscriminatorLearner::getNewObject() {
  auto obj = std::make_unique<PhysicsTools::Calibration::HistogramD3D>(Charge_Vs_Path->GetNbinsX(),
                                                                       Charge_Vs_Path->GetXaxis()->GetXmin(),
                                                                       Charge_Vs_Path->GetXaxis()->GetXmax(),
                                                                       Charge_Vs_Path->GetNbinsY(),
                                                                       Charge_Vs_Path->GetYaxis()->GetXmin(),
                                                                       Charge_Vs_Path->GetYaxis()->GetXmax(),
                                                                       Charge_Vs_Path->GetNbinsZ(),
                                                                       Charge_Vs_Path->GetZaxis()->GetXmin(),
                                                                       Charge_Vs_Path->GetZaxis()->GetXmax());

  for (int ix = 0; ix <= Charge_Vs_Path->GetNbinsX() + 1; ix++) {
    for (int iy = 0; iy <= Charge_Vs_Path->GetNbinsY() + 1; iy++) {
      for (int iz = 0; iz <= Charge_Vs_Path->GetNbinsZ() + 1; iz++) {
        obj->setBinContent(ix, iy, iz, Charge_Vs_Path->GetBinContent(ix, iy, iz));
        //          if(Charge_Vs_Path->GetBinContent(ix,iy)!=0)printf("%i %i %i --> %f\n",ix,iy, iz, Charge_Vs_Path->GetBinContent(ix,iy,iz));
      }
    }
  }

  return obj;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorLearner);
