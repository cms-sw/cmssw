#ifndef RecoLocalMuon_RPCRecHitReader_H
#define RecoLocalMuon_RPCRecHitReader_H

/** \class RPCRecHitReader
 *  Basic analyzer class which accesses based on 2D CSCRecHits
 *  and plot resolution comparing them with muon simhits
 *  From D. Fortin  - UC Riverside
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>

class TGraph;
class TFile;
class TH1F;
class TH2F;

class RPCRecHit;
class RPCGeometry;
class MuonGeometryRecord;

class RPCRecHitReader : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  /// Constructor
  RPCRecHitReader(const edm::ParameterSet& pset);

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void endJob() override;

  /// Destructor
  ~RPCRecHitReader() override;

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  unsigned int layerRecHit(RPCRecHit);

private:
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomBRToken_;
  std::string fOutputFileName;
  std::string recHitLabel1;
  std::string recHitLabel2;

  int region;
  int wheel;
  int sector;
  int station;
  int layer;
  int subsector;

  float _phi;

  float _trigger;
  float _spurious;
  float _spuriousPeak;
  float _triggerGOOD;
  float _efficiencyBAD;
  float _efficiencyGOOD;
  float _efficiencyBEST;

  TFile* fOutputFile;
  std::fstream* fout;

  TH2F* histoXY;
  TH1F* histoSlope;
  TH1F* histoChi2;
  TH1F* histoRes;
  TH1F* histoRes1;
  TH1F* histoRes2;
  TH1F* histoRes3;
  TH1F* histoPool1;
  TH1F* histoPool2;
  TH1F* histoPool3;

  TH1F* histoExpectedOcc;
  TH1F* histoRealOcc;
  TH1F* histoLocalEff;

  float yLayer;

  bool _trigRPC1;
  bool _trigRPC2;
  bool _trigRPC3;
  bool _trigRPC4;
  bool _trigRPC5;
  bool _trigRPC6;

  std::vector<bool> _trigConfig;
  std::map<int, float> _mapLayer;
  const RPCRoll* _rollEff;
};

#endif
