#ifndef RecoLocalTracker_SiPixelRecHits_SiPixelRecHitConverter_h
#define RecoLocalTracker_SiPixelRecHits_SiPixelRecHitConverter_h

//---------------------------------------------------------------------------
//! \class SiPixelRecHitConverter
//!
//! \brief EDProducer to covert SiPixelClusters into SiPixelRecHits
//!
//! SiPixelRecHitConverter is an EDProducer subclass (i.e., a module)
//! which orchestrates the conversion of SiPixelClusters into SiPixelRecHits.
//! Consequently, the input is a edm::DetSetVector<SiPixelCluster> and the output is
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
//! The calibrations are not loaded at the moment, 
//! although that is being planned for the near future.
//!
//! \author Porting from ORCA by Petar Maksimovic (JHU). Implementation of the
//!         DetSetVector by V.Chiochia (Zurich University).
//!
//! \version v2, May 30, 2006  
//! change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
//!
//---------------------------------------------------------------------------

//--- Base class for CPEs:

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

//--- Geometry + DataFormats
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"

//--- Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#define TP_OLD
#ifdef TP_OLD
#include "FWCore/Framework/interface/Handle.h"
#else
#include "DataFormats/Common/interface/Handle.h"
#endif
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class MagneticField;
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

    //--- The top-level event method.
    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

    //--- Execute the position estimator algorithm(s).
    //--- New interface with DetSetVector
    void run(const edmNew::DetSetVector<SiPixelCluster>& input,
	     SiPixelRecHitCollectionNew & output,
	     edm::ESHandle<TrackerGeometry> & geom);

    void run(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >  inputhandle,
	     SiPixelRecHitCollectionNew & output,
	     edm::ESHandle<TrackerGeometry> & geom);

  private:
    edm::ParameterSet conf_;
    // TO DO: maybe allow a map of pointers?
    std::string cpeName_="None";                   // what the user said s/he wanted
    /// const PixelClusterParameterEstimator * cpe_;  // what we got (for now, one ptr to base class)
    PixelCPEBase const * cpe_=nullptr;                    // What we got (for now, one ptr to base class)
    edm::InputTag src_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> tPixelCluster;
    bool m_newCont; // save also in emdNew::DetSetVector
  };
}


#endif
