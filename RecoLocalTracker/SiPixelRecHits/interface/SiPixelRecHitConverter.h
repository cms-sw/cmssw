#ifndef RecoLocalTracker_SiPixelRecHits_SiPixelRecHitConverter_h
#define RecoLocalTracker_SiPixelRecHits_SiPixelRecHitConverter_h

//---------------------------------------------------------------------------
//! \class SiPixelRecHitConverter
//!
//! \brief EDProducer to covert SiPixelClusters into SiPixelRecHits
//!
//! SiPixelRecHitConverter is an EDProducer subclass (i.e., a module)
//! which orchestrates the conversion of SiPixelClusters into SiPixelRecHits.
//! Consequently, the input is SiPixelClusterCollection and the output is
//! SiPixelRecHitCollection.
//!
//! SiPixelRecHitConverter invokes one of descendents from 
//! ClusterParameterEstimator (templated on SiPixelCluster), e.g.
//! CPEFromDetPosition (which is the only available option 
//! right now).  SiPixelRecHitConverter loads the SiPixelClusterCollection,
//! and then iterates over DetIds, invoking the chosen CPE's methods
//! localPosition() and localError() to perform the correction (some of which
//! may be rather involved).  A RecHit is made on the spot, and appended
//! to the output collection.
//!
//! The calibrations (for the `fancy' CPE's) are not loaded at the moment, 
//! although that is being planned for the near future.
//!
//! \author Petar Maksimovic (JHU)
//!
//! \version v1, Feb 27, 2006  
//!
//---------------------------------------------------------------------------

//--- Base class for CPEs:
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

//--- Geometry + DataFormats
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollectionfwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

//--- Framework paraphernalia:
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace cms
{
  class SiPixelRecHitConverter : public edm::EDProducer
  {
  public:
    //--- Constructor, virtual destructor (just in case)
    explicit SiPixelRecHitConverter(const edm::ParameterSet& conf);
    virtual ~SiPixelRecHitConverter();

    //--- Factory method to make CPE's depending on the ParameterSet
    //--- Not sure if we need to make more than one CPE to run concurrently
    //--- on different parts of the detector (e.g., one for the barrel and the 
    //--- one for the forward).  The way the CPE's are written now, it's
    //--- likely we can use one (and they will switch internally), or
    //--- make two of the same but configure them differently.  We need a more
    //--- realistic use case...
    void setupCPE();

    //--- The top-level event method.
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

    //--- Execute the algorithm(s).
    void run(const SiPixelClusterCollection* input,
	     SiPixelRecHitCollection & output,
	     edm::ESHandle<TrackingGeometry> & geom);

  private:
    edm::ParameterSet conf_;
    // TO DO: maybe allow a map of pointers?
    std::string cpeName_;                   // what the user said s/he wanted
    PixelClusterParameterEstimator * cpe_;  // what we got (for now, one ptr to base class)
    bool ready_;                            // needed CPE's valid => good to go!
  };
}


#endif
