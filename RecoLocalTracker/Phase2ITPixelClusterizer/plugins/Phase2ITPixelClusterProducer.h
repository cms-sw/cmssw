#ifndef RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelClusterProducer_h
#define RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelClusterProducer_h

//---------------------------------------------------------------------------
//! \class Phase2ITPixelClusterProducer
//!
//! \brief EDProducer to cluster PixelDigis into Phase2ITPixelClusters.
//!
//! Phase2ITPixelClusterProducer is an EDProducer subclass (i.e., a module)
//! which orchestrates clustering of PixelDigis to Phase2ITPixelClusters.
//! Consequently, the input is edm::DetSetVector<PixelDigi> and the output is
//! edmNew::DetSetVector<Phase2ITPixelCluster>.
//!
//! Phase2ITPixelClusterProducer invokes one of descendents from Phase2ITPixelClusterizerBase,
//! e.g. PixelThresholdClusterizer (which is the only available option 
//! right now).  Phase2ITPixelClusterProducer loads the PixelDigis,
//! and then iterates over DetIds, invoking Phase2ITPixelClusterizer's clusterizeDetUnit
//! to perform the clustering.  clusterizeDetUnit() returns a DetSetVector of
//! Phase2ITPixelClusters, which are then recorded in the event.
//!
//! The calibrations are not loaded at the moment (v1), although that is
//! being planned for the near future.
//!
//! \author Petar Maksimovic (JHU). 
//!         DetSetVector implementation by Vincenzo Chiochia (Uni Zurich)        
//!         Modify the local container (cache) to improve the speed. D.K. 5/07
//!
//---------------------------------------------------------------------------

#include "Phase2ITPixelClusterizerBase.h"

//#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

  class dso_hidden Phase2ITPixelClusterProducer final : public edm::stream::EDProducer<> {
  public:
    //--- Constructor, virtual destructor (just in case)
    explicit Phase2ITPixelClusterProducer(const edm::ParameterSet& conf);
    virtual ~Phase2ITPixelClusterProducer();

    void setupClusterizer();

    //--- The top-level event method.
    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

    //--- Execute the algorithm(s).
    void run(const edm::DetSetVector<PixelDigi>   & input,
	     edm::ESHandle<TrackerGeometry>       & geom,
             edmNew::DetSetVector<Phase2ITPixelCluster> & output);

  private:
    edm::ParameterSet conf_;
    edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi;
    // TO DO: maybe allow a map of pointers?
    SiPixelGainCalibrationServiceBase * theSiPixelGainCalibration_;
    std::string clusterMode_;               // user's choice of the clusterizer
    Phase2ITPixelClusterizerBase * clusterizer_;    // what we got (for now, one ptr to base class)
    bool readyToCluster_;                   // needed clusterizers valid => good to go!
    edm::InputTag src_;

    //! Optional limit on the total number of clusters
    int32_t maxTotalClusters_;
  };


#endif
