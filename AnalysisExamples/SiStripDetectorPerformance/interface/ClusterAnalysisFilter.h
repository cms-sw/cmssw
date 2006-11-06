#ifndef ClusterAnalysisFilter_H
#define  ClusterAnalysisFilter_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"

namespace cms
{
 class ClusterAnalysisFilter : public edm::EDFilter {
  public:
    ClusterAnalysisFilter(const edm::ParameterSet& conf);
    ~ClusterAnalysisFilter() {}

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
   };
}
#endif
