// -*- C++ -*-
//
// Package:     SiPixelPhase1ClustersFiltered
// Class:       SiPixelPhase1ClustersFiltered
//

// Original Author: Yi-Mu "Enoch" Chen

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1FlagBase.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


class SiPixelPhase1ClustersFiltered : public SiPixelPhase1FlagBase {
  enum {
    NCLUSTERS
  };

  public:
  explicit SiPixelPhase1ClustersFiltered(const edm::ParameterSet& conf);
  void flagAnalyze(const edm::Event&, const edm::EventSetup&);

  private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > srcToken_;
};


/******************************************************************************/

SiPixelPhase1ClustersFiltered::SiPixelPhase1ClustersFiltered( const edm::ParameterSet& iConfig ) :
   SiPixelPhase1FlagBase( iConfig )
{
   srcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >( iConfig.getParameter<edm::InputTag>( "src" ) );
}

/******************************************************************************/

void SiPixelPhase1ClustersFiltered::flagAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return;

  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    auto id = DetId(it->detId());

    for(SiPixelCluster const& cluster : *it) {
      int row = cluster.x()-0.5, col = cluster.y()-0.5;
      histo[NCLUSTERS].fill(id, &iEvent, col, row);
    }
  }

  histo[NCLUSTERS].executePerEventHarvesting(&iEvent);
}

DEFINE_FWK_MODULE( SiPixelPhase1ClustersFiltered );
