#ifndef SiPixelPhase1TrackEfficiency_h 
#define SiPixelPhase1TrackEfficiency_h 
// -*- C++ -*-
// 
// Package:     SiPixelPhase1TrackEfficiency
// Class  :     SiPixelPhase1TrackEfficiency
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class SiPixelPhase1TrackEfficiency : public SiPixelPhase1Base {
  enum {
    VALID,
    MISSING,
    EFFICIENCY
  };

  public:
  explicit SiPixelPhase1TrackEfficiency(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clustersToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trackAssociationToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
};

class SiPixelPhase1TrackEfficiencyHarvester : public SiPixelPhase1Harvester { 
  enum {
    VALID,
    MISSING,
    EFFICIENCY
  };

  public:
  explicit SiPixelPhase1TrackEfficiencyHarvester(const edm::ParameterSet& conf);

  void doHarvesting(SummationStep const& s, HistogramManager::Table& efficiency);
  
  private:
  // we use the custom arg as a tag, to not mix up different tables.
  std::map<std::string, HistogramManager::Table> valid;
  std::map<std::string, HistogramManager::Table> missing;
};

#endif
