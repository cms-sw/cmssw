// -*- C++ -*-
//
// Package:     SiPixelPhase1Clusters
// Class:       SiPixelPhase1Clusters
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Clusters/interface/SiPixelPhase1Clusters.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"


SiPixelPhase1Clusters::SiPixelPhase1Clusters(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig) 
{
  srcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("src"));
}

void SiPixelPhase1Clusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return;

  bool hasClusters=false;  

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  assert(tracker.isValid());
  
  edmNew::DetSetVector<SiPixelCluster>::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    auto id = DetId(it->detId());

    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(id) );
    const PixelTopology& topol = theGeomDet->specificTopology();

    for(SiPixelCluster const& cluster : *it) {
      int row = cluster.x()-0.5, col = cluster.y()-0.5;
      histo[READOUT_CHARGE].fill(double(cluster.charge()), id, &iEvent, col, row);
      histo[CHARGE].fill(double(cluster.charge()), id, &iEvent, col, row);
      histo[SIZE  ].fill(double(cluster.size()  ), id, &iEvent);
      if (cluster.size() > 1){
        histo[NCLUSTERS].fill(id, &iEvent);
        histo[NCLUSTERSINCLUSIVE].fill(id, &iEvent);
        histo[READOUT_NCLUSTERS].fill(id, &iEvent);
        hasClusters=true;
      }

      LocalPoint clustlp = topol.localPosition(MeasurementPoint(cluster.x(), cluster.y()));
      GlobalPoint clustgp = theGeomDet->surface().toGlobal(clustlp);
      histo[POSITION_B ].fill(clustgp.z(),   clustgp.phi(),   id, &iEvent);
      histo[POSITION_F ].fill(clustgp.x(),   clustgp.y(),     id, &iEvent);
      histo[POSITION_XZ].fill(clustgp.x(),   clustgp.z(),     id, &iEvent);
      histo[POSITION_YZ].fill(clustgp.y(),   clustgp.z(),     id, &iEvent);
      histo[SIZE_VS_ETA].fill(clustgp.eta(), cluster.sizeY(), id, &iEvent);

    }
  }


  if (hasClusters) histo[EVENTRATE].fill(DetId(0), &iEvent);

  histo[NCLUSTERS].executePerEventHarvesting(&iEvent);
  histo[READOUT_NCLUSTERS].executePerEventHarvesting(&iEvent);
  histo[NCLUSTERSINCLUSIVE].executePerEventHarvesting(&iEvent);

}

DEFINE_FWK_MODULE(SiPixelPhase1Clusters);

