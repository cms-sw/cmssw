#ifndef TrackRecoDeDx_DeDxDiscriminatorLearner_H
#define TrackRecoDeDx_DeDxDiscriminatorLearner_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "RecoTracker/DeDx/interface/DeDxTools.h"

#include "TFile.h"
#include "TH3F.h"

#include <memory>

class DeDxDiscriminatorLearner : public ConditionDBWriter<PhysicsTools::Calibration::HistogramD3D> {
public:
  explicit DeDxDiscriminatorLearner(const edm::ParameterSet&);
  ~DeDxDiscriminatorLearner() override;

private:
  void algoBeginJob(const edm::EventSetup&) override;
  void algoAnalyze(const edm::Event&, const edm::EventSetup&) override;
  void algoEndJob() override;

  void processHit(const TrackingRecHit* recHit,
                  float trackMomentum,
                  float& cosine,
                  const TrajectoryStateOnSurface& trajState);
  void algoAnalyzeTheTree(const edm::EventSetup& iSetup);

  std::unique_ptr<PhysicsTools::Calibration::HistogramD3D> getNewObject() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<TrajTrackAssociationCollection> m_trajTrackAssociationTag;
  edm::EDGetTokenT<reco::TrackCollection> m_tracksTag;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_tkGeomToken;

  float MinTrackMomentum;
  float MaxTrackMomentum;
  float MinTrackEta;
  float MaxTrackEta;
  unsigned int MaxNrStrips;
  unsigned int MinTrackHits;
  float MaxTrackChiOverNdf;

  float P_Min;
  float P_Max;
  int P_NBins;
  float Path_Min;
  float Path_Max;
  int Path_NBins;
  float Charge_Min;
  float Charge_Max;
  int Charge_NBins;

  std::vector<std::string> VInputFiles;
  std::string algoMode;
  std::string HistoFile;

  TH3F* Charge_Vs_Path;

  std::string m_calibrationPath;
  bool useCalibration;
  bool shapetest;

  std::vector<std::vector<float> > calibGains;
  unsigned int m_off;
};

#endif
