#ifndef ClusterAnalysisFilter_H
#define  ClusterAnalysisFilter_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//needed for the geometry:
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"


namespace cms
{
 class ClusterAnalysisFilter : public edm::EDFilter {
  public:
    ClusterAnalysisFilter(const edm::ParameterSet& conf);
    ~ClusterAnalysisFilter() {}

    void beginJob(edm::EventSetup const& es);
    bool filter(edm::Event & e, edm::EventSetup const& es);

  private:

    bool TrackNumberSelector();
    bool ClusterNumberSelector();
    bool TriggerSelector();
    bool ClusterInModuleSelector ();   
    
    edm::ParameterSet conf_;

    edm::InputTag Track_src_;
    edm::InputTag ClusterInfo_src_;
    edm::InputTag Cluster_src_;    

    edm::Handle< edm::DetSetVector<SiStripClusterInfo> >  dsv_SiStripClusterInfo;
    edm::Handle< edm::DetSetVector<SiStripCluster> >  dsv_SiStripCluster;
    edm::Handle<reco::TrackCollection> trackCollection;
    edm::Handle<LTCDigiCollection> ltcdigis;

    edm::ESHandle<TrackerGeometry> tkgeom;
   };
}
#endif
