#ifndef CommonAlignmentMonitor_AlignmentStats_H
#define CommonAlignmentMonitor_AlignmentStats_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"

// #include <Riostream.h>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TFile.h"
#include "TTree.h"

//using namespace edm;

class AlignmentStats : public edm::one::EDAnalyzer<> {
public:
  AlignmentStats(const edm::ParameterSet &iConfig);
  ~AlignmentStats() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;
  void beginJob() override;
  void endJob() override;

private:
  // esToken
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> esTokenTTopo_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> esTokenTkGeo_;

  //////inputs from config file
  const edm::InputTag src_;
  const edm::InputTag overlapAM_;
  const bool keepTrackStats_;
  const bool keepHitPopulation_;
  const std::string statsTreeName_;
  const std::string hitsTreeName_;
  const uint32_t prescale_;

  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDGetTokenT<AliClusterValueMap> mapToken_;

  //////
  uint32_t tmpPresc_;

  //Track stats
  TFile *treefile_;
  TTree *outtree_;
  static const int MAXTRKS_ = 200;
  int run_, event_;
  unsigned int ntracks;
  float P[MAXTRKS_], Pt[MAXTRKS_], Eta[MAXTRKS_], Phi[MAXTRKS_], Chi2n[MAXTRKS_];
  int Nhits[MAXTRKS_][7];  //0=total, 1-6=Subdets

  //Hit Population
  typedef std::map<uint32_t, uint32_t> DetHitMap;
  DetHitMap hitmap_;
  DetHitMap overlapmap_;

  std::unique_ptr<TrackerTopology> trackerTopology_;
  std::unique_ptr<TrackerGeometry> trackerGeometry_;
};

#endif
