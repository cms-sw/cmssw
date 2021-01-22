#ifndef CalibTracker_SiStripLorentzAngle_SiStripCalibLorentzAngle_h
#define CalibTracker_SiStripLorentzAngle_SiStripCalibLorentzAngle_h

#include <cstring>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include <TGraph.h>
#include <TProfile.h>
#include <TStyle.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TFile.h>
#include <TTree.h>
#include <TGraphErrors.h>
#include <TDirectory.h>
#include "TROOT.h"
#include "Riostream.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <memory>

class TrackerTopology;

class SiStripCalibLorentzAngle : public ConditionDBWriter<SiStripLorentzAngle> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;
  explicit SiStripCalibLorentzAngle(const edm::ParameterSet &conf);

  ~SiStripCalibLorentzAngle() override;

  std::unique_ptr<SiStripLorentzAngle> getNewObject() override;

  void algoBeginJob(const edm::EventSetup &) override;

private:
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> lorentzAngleToken_;

  const TrackerGeometry *tkGeom_ = nullptr;
  const TrackerTopology *tTopo_ = nullptr;

  typedef std::map<std::string, TProfile *> ProfileMap;
  ProfileMap Profiles;
  typedef std::map<std::string, TH1D *> TH1Dmap;
  TH1Dmap TH1Ds;
  typedef std::map<std::string, TH2D *> TH2Dmap;
  TH2Dmap TH2Ds;
  std::vector<MonitorElement *> histolist;

  TF1 *fitfunc, *fitfunc2IT, *FitFunction2IT, *FitFunction;

  float histoEntries, gR, gphi, geta, gz, globalX, globalY, globalZ, muH, theBfield, AsymmParam;
  int goodFit, goodFit1IT, badFit, TIB, TOB, Layer, MonoStereo;
  const GlobalPoint gposition;

  float mean_TIB1, mean_TIB2, mean_TIB3, mean_TIB4, mean_TOB1, mean_TOB2, mean_TOB3, mean_TOB4, mean_TOB5, mean_TOB6;

  float hallMobility, meanMobility_TIB, meanMobility_TOB;

  bool LayerDB, CalibByMC;

  TGraphErrors *TIB_graph, *TOB_graph;

  TTree *ModuleTree;
  TFile *hFile;

  TDirectory *LorentzAngle_Plots, *Rootple, *MuH, *TIB_MuH, *TOB_MuH, *MuH_vs_Phi, *TIB_Phi, *TOB_Phi, *MuH_vs_Eta,
      *TIB_Eta, *TOB_Eta;
  TDirectory *FirstIT_GoodFit_Histos, *TIB_1IT_GoodFit, *TOB_1IT_GoodFit, *SecondIT_GoodFit_Histos, *TIB_2IT_GoodFit,
      *TOB_2IT_GoodFit, *SecondIT_BadFit_Histos, *TIB_2IT_BadFit, *TOB_2IT_BadFit;

  std::map<uint32_t, float> detid_la;
  edm::ParameterSet conf_;
};

#endif
