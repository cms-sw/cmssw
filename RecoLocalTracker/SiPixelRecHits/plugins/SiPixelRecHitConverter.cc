/** SiPixelRecHitConverter.cc
 * ------------------------------------------------------
 * Description:  see SiPixelRecHitConverter.h
 * Authors:  P. Maksimovic (JHU), V.Chiochia (Uni Zurich)
 * History: Feb 27, 2006 -  initial version
 *          May 30, 2006 -  edm::DetSetVector and edm::Ref
 *          Aug 30, 2007 -  edmNew::DetSetVector
 *          Jan 31, 2008 -  change to use Lorentz angle from DB (Lotte Wilke)
 * ------------------------------------------------------
 */

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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

// Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSet2RangeMap.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

// Make heterogeneous framework happy
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"

using namespace std;

namespace cms {

  class SiPixelRecHitConverter : public edm::stream::EDProducer<> {
  public:
    //--- Constructor, virtual destructor (just in case)
    explicit SiPixelRecHitConverter(const edm::ParameterSet& conf);
    ~SiPixelRecHitConverter() override;

    //--- Factory method to make CPE's depending on the ParameterSet
    //--- Not sure if we need to make more than one CPE to run concurrently
    //--- on different parts of the detector (e.g., one for the barrel and the
    //--- one for the forward).  The way the CPE's are written now, it's
    //--- likely we can use one (and they will switch internally), or
    //--- make two of the same but configure them differently.  We need a more
    //--- realistic use case...

    //--- The top-level event method.
    void produce(edm::Event& e, const edm::EventSetup& c) override;

    //--- Execute the position estimator algorithm(s).
    void run(edm::Event& e,
             edm::Handle<SiPixelClusterCollectionNew> inputhandle,
             SiPixelRecHitCollectionNew& output,
             TrackerGeometry const& geom);

  private:
    using HMSstorage = HostProduct<uint32_t[]>;

    // TO DO: maybe allow a map of pointers?
    PixelCPEBase const* cpe_ = nullptr;  // What we got (for now, one ptr to base class)
    edm::InputTag const src_;
    std::string const cpeName_;
    edm::EDGetTokenT<SiPixelClusterCollectionNew> const tPixelCluster_;
    edm::EDPutTokenT<SiPixelRecHitCollection> const tPut_;
    edm::EDPutTokenT<HMSstorage> const tHost_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> const tTrackerGeom_;
    edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> const tCPE_;
    bool m_newCont;  // save also in emdNew::DetSetVector
  };

  //---------------------------------------------------------------------------
  //!  Constructor: set the ParameterSet and defer all thinking to setupCPE().
  //---------------------------------------------------------------------------
  SiPixelRecHitConverter::SiPixelRecHitConverter(edm::ParameterSet const& conf)
      : src_(conf.getParameter<edm::InputTag>("src")),
        cpeName_(conf.getParameter<std::string>("CPE")),
        tPixelCluster_(consumes<SiPixelClusterCollectionNew>(src_)),
        tPut_(produces<SiPixelRecHitCollection>()),
        tHost_(produces<HMSstorage>()),
        tTrackerGeom_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
        tCPE_(esConsumes<PixelClusterParameterEstimator, TkPixelCPERecord>(edm::ESInputTag("", cpeName_))) {}

  // Destructor
  SiPixelRecHitConverter::~SiPixelRecHitConverter() {}

  //---------------------------------------------------------------------------
  //! The "Event" entrypoint: gets called by framework for every event
  //---------------------------------------------------------------------------
  void SiPixelRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es) {
    // Step A.1: get input data
    edm::Handle<SiPixelClusterCollectionNew> input;
    e.getByToken(tPixelCluster_, input);

    // Step A.2: get event setup
    auto const& geom = es.getData(tTrackerGeom_);

    // Step B: create empty output collection
    SiPixelRecHitCollectionNew output;

    // Step B*: create CPE
    cpe_ = dynamic_cast<const PixelCPEBase*>(&es.getData(tCPE_));

    // Step C: Iterate over DetIds and invoke the strip CPE algorithm
    // on each DetUnit

    run(e, input, output, geom);

    output.shrink_to_fit();
    e.emplace(tPut_, std::move(output));
  }

  //---------------------------------------------------------------------------
  //!  Iterate over DetUnits, then over Clusters and invoke the CPE on each,
  //!  and make a RecHit to store the result.
  //!  New interface reading DetSetVector by V.Chiochia (May 30th, 2006)
  //---------------------------------------------------------------------------
  void SiPixelRecHitConverter::run(edm::Event& iEvent,
                                   edm::Handle<SiPixelClusterCollectionNew> inputhandle,
                                   SiPixelRecHitCollectionNew& output,
                                   TrackerGeometry const& geom) {
    if (!cpe_) {
      edm::LogError("SiPixelRecHitConverter") << " at least one CPE is not ready -- can't run!";
      // TO DO: throw an exception here?  The user may want to know...
      assert(0);
      return;  // clusterizer is invalid, bail out
    }

    int numberOfDetUnits = 0;
    int numberOfClusters = 0;

    const SiPixelClusterCollectionNew& input = *inputhandle;

    // allocate a buffer for the indices of the clusters
    auto hmsp = std::make_unique<uint32_t[]>(gpuClustering::maxNumModules + 1);
    // hitsModuleStart is a non-owning pointer to the buffer
    auto hitsModuleStart = hmsp.get();
    // fill cluster arrays
    std::array<uint32_t, gpuClustering::maxNumModules + 1> clusInModule{};
    for (auto const& dsv : input) {
      unsigned int detid = dsv.detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom.idToDetUnit(detIdObject);
      auto gind = genericDet->index();
      // FIXME to be changed to support Phase2
      if (gind >= int(gpuClustering::maxNumModules))
        continue;
      auto const nclus = dsv.size();
      assert(nclus > 0);
      clusInModule[gind] = nclus;
      numberOfClusters += nclus;
    }
    hitsModuleStart[0] = 0;
    assert(clusInModule.size() > gpuClustering::maxNumModules);
    for (int i = 1, n = clusInModule.size(); i < n; ++i)
      hitsModuleStart[i] = hitsModuleStart[i - 1] + clusInModule[i - 1];
    assert(numberOfClusters == int(hitsModuleStart[gpuClustering::maxNumModules]));

    // wrap the buffer in a HostProduct, and move it to the Event, without reallocating the buffer or affecting hitsModuleStart
    iEvent.emplace(tHost_, std::move(hmsp));

    numberOfClusters = 0;
    for (auto const& dsv : input) {
      numberOfDetUnits++;
      unsigned int detid = dsv.detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom.idToDetUnit(detIdObject);
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
      assert(pixDet);
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(output, detid);

      edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = dsv.begin(), clustEnd = dsv.end();

      for (; clustIt != clustEnd; clustIt++) {
        numberOfClusters++;
        std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple =
            cpe_->getParameters(*clustIt, *genericDet);
        LocalPoint lp(std::get<0>(tuple));
        LocalError le(std::get<1>(tuple));
        SiPixelRecHitQuality::QualWordType rqw(std::get<2>(tuple));
        // Create a persistent edm::Ref to the cluster
        SiPixelClusterRefNew cluster = edmNew::makeRefTo(inputhandle, clustIt);
        // Make a RecHit and add it to the DetSet
        SiPixelRecHit hit(lp, le, rqw, *genericDet, cluster);
        recHitsOnDetUnit.push_back(hit);

        LogDebug("SiPixelRecHitConverter") << "RecHit " << (numberOfClusters - 1)  //
                                           << " with local position " << lp << " and local error " << le;
      }  //  <-- End loop on Clusters

      LogDebug("SiPixelRecHitConverter") << "Found " << recHitsOnDetUnit.size() << " RecHits on " << detid;

    }  //    <-- End loop on DetUnits

    LogDebug("SiPixelRecHitConverter") << cpeName_ << " converted " << numberOfClusters
                                       << " SiPixelClusters into SiPixelRecHits, in " << numberOfDetUnits
                                       << " DetUnits.";
  }
}  // end of namespace cms

using cms::SiPixelRecHitConverter;

DEFINE_FWK_MODULE(SiPixelRecHitConverter);
