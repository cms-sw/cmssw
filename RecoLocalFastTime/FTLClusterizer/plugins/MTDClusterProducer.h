#ifndef RecoLocalFastTime_MTDClusterizer_MTDClusterProducer_h
#define RecoLocalFastTime_MTDClusterizer_MTDClusterProducer_h

//---------------------------------------------------------------------------
//! \class MTDClusterProducer
//!
//! \brief EDProducer to cluster FTLRecHits into FTLClusters.
//!
//---------------------------------------------------------------------------

#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"


#include "MTDClusterizerBase.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESWatcher.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"

class MTDClusterProducer : public edm::stream::EDProducer<> {
  public:
    //--- Constructor, virtual destructor (just in case)
    explicit MTDClusterProducer(const edm::ParameterSet& conf);
    ~MTDClusterProducer() override;

    //--- The top-level event method.
    void produce(edm::Event& e, const edm::EventSetup& c) override;

    //--- Execute the algorithm(s).
    template<typename T>
    void run(const T                              & input,
             FTLClusterCollection & output);

    void setupClusterizer(const edm::ParameterSet& conf);

  private:
    edm::EDGetTokenT< FTLRecHitCollection >  btlHits_;
    edm::EDGetTokenT< FTLRecHitCollection >  etlHits_;

    const std::string clusterMode_;         // user's choice of the clusterizer
    MTDClusterizerBase * clusterizer_;    // what we got (for now, one ptr to base class)
    bool readyToCluster_;                   // needed clusterizers valid => good to go!

    edm::ESWatcher<MTDDigiGeometryRecord> geomwatcher_;
    const MTDGeometry* geom_;
};

#endif
