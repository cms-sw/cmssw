#include "PixelRecHits.h"

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

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class MagneticField;
namespace
{
  class SiPixelRecHitGPU : public edm::stream::EDProducer<>
  {
  public:
    //--- Constructor, virtual destructor (just in case)
    explicit SiPixelRecHitGPU(const edm::ParameterSet& conf);
    ~SiPixelRecHitGPU() override;

    //--- The top-level event method.
    void produce(edm::Event& e, const edm::EventSetup& c) override;

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
    std::string cpeName_="None";                   // what the user said s/he wanted
    PixelCPEBase const * cpe_=nullptr;                    // What we got (for now, one ptr to base class)

    edm::InputTag src_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> tPixelCluster;

    using GPUProd = std::vector<unsigned long long>;
    edm::InputTag gpuProd_ = edm::InputTag("siPixelDigis");
    edm::EDGetTokenT<GPUProd> tGpuProd;
    HitsOnGPU hitsOnGPU_;


    bool m_newCont; // save also in emdNew::DetSetVector
  };
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SiPixelRecHitGPU);

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

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

using namespace std;

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"


  //---------------------------------------------------------------------------
  //!  Constructor: set the ParameterSet and defer all thinking to setupCPE().
  //---------------------------------------------------------------------------
  SiPixelRecHitGPU::SiPixelRecHitGPU(edm::ParameterSet const& conf) 
    : 
    conf_(conf),
    src_( conf.getParameter<edm::InputTag>( "src" ) ),
    tPixelCluster(consumes< edmNew::DetSetVector<SiPixelCluster> >( src_)),
    tGpuProd(consumes<GPUProd>(gpuProd_)),
    hitsOnGPU_ (allocHitsOnGPU())
 {
    //--- Declare to the EDM what kind of collections we will be making.
    produces<SiPixelRecHitCollection>();

  }
  
  // Destructor
  SiPixelRecHitGPU::~SiPixelRecHitGPU() 
  { 
    // need to free hitsOnGPU
  }  
  
  //---------------------------------------------------------------------------
  //! The "Event" entrypoint: gets called by framework for every event
  //---------------------------------------------------------------------------
  void SiPixelRecHitGPU::produce(edm::Event& e, const edm::EventSetup& es)
  {

    // Step A.1: get input data
    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > input;
    e.getByToken( tPixelCluster, input);
    
    // Step A.2: get event setup
    edm::ESHandle<TrackerGeometry> geom;
    es.get<TrackerDigiGeometryRecord>().get( geom );

    // Step B: create empty output collection
    auto output = std::make_unique<SiPixelRecHitCollectionNew>();
    
    // Step B*: create CPE
    edm::ESHandle<PixelClusterParameterEstimator> hCPE;
    std::string cpeName_ = conf_.getParameter<std::string>("CPE");
    es.get<TkPixelCPERecord>().get(cpeName_,hCPE);
    cpe_ = dynamic_cast< const PixelCPEBase* >(&(*hCPE));
    
    ///  do it on GPU....

    edm::Handle<GPUProd> gh;
    e.getByToken(tGpuProd, gh);
    auto gprod = *gh;
   // invoke gpu version ......
    PixelCPEFast const * fcpe =   dynamic_cast<const PixelCPEFast *>(cpe_);
    if (!fcpe) {
      std::cout << " too bad, not a fast cpe gpu processing not possible...." << std::endl;
      assert(0);
    }
    assert(fcpe->d_paramsOnGPU);

    pixelRecHits_wrapper(* (context const *)(gprod[0]),fcpe->d_paramsOnGPU,gprod[1],gprod[2], hitsOnGPU_);



    // Step C: Iterate over DetIds and invoke the strip CPE algorithm
    // on each DetUnit

    std::cout << "Number of Clusers on CPU " << (*input).data().size() << std::endl;

    run( input, *output, geom );

    output->shrink_to_fit();
    e.put(std::move(output));

  }

  //---------------------------------------------------------------------------
  //!  Iterate over DetUnits, then over Clusters and invoke the CPE on each,
  //!  and make a RecHit to store the result.
  //!  New interface reading DetSetVector by V.Chiochia (May 30th, 2006)
  //---------------------------------------------------------------------------
  void SiPixelRecHitGPU::run(edm::Handle<edmNew::DetSetVector<SiPixelCluster> >  inputhandle,
				   SiPixelRecHitCollectionNew &output,
				   edm::ESHandle<TrackerGeometry> & geom) {
    if ( ! cpe_ ) 
      {
	edm::LogError("SiPixelRecHitGPU") << " at least one CPE is not ready -- can't run!";
	// TO DO: throw an exception here?  The user may want to know...
	assert(0);
	return;   // clusterizer is invalid, bail out
      }


    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
    
    const edmNew::DetSetVector<SiPixelCluster>& input = *inputhandle;
    
    edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin();
    
    for ( ; DSViter != input.end() ; DSViter++) {
      numberOfDetUnits++;
      unsigned int detid = DSViter->detId();
      DetId detIdObject( detid );  
      const GeomDetUnit * genericDet = geom->idToDetUnit( detIdObject );
      const PixelGeomDetUnit * pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
      assert(pixDet); 
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(output,detid);
      
      edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = DSViter->begin(), clustEnd = DSViter->end();
      
      for ( ; clustIt != clustEnd; clustIt++) {
	numberOfClusters++;
	std::tuple<LocalPoint, LocalError,SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( *clustIt, *genericDet );
	LocalPoint lp( std::get<0>(tuple) );
	LocalError le( std::get<1>(tuple) );
        SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
	// Create a persistent edm::Ref to the cluster
	edm::Ref< edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > cluster = edmNew::makeRefTo( inputhandle, clustIt);
	// Make a RecHit and add it to the DetSet
	// old : recHitsOnDetUnit.push_back( new SiPixelRecHit( lp, le, detIdObject, &*clustIt) );
	SiPixelRecHit hit( lp, le, rqw, *genericDet, cluster);
	// 
	// Now save it =================
	recHitsOnDetUnit.push_back(hit);
	// =============================

	// std::cout << "SiPixelRecHitGPUVI " << numberOfClusters << ' '<< lp << " " << le << std::endl;
      } //  <-- End loop on Clusters
	

      //  LogDebug("SiPixelRecHitGPU")
      //std::cout << "SiPixelRecHitGPUVI "
	//	<< " Found " << recHitsOnDetUnit.size() << " RecHits on " << detid //;
	//	<< std::endl;
      
      
    } //    <-- End loop on DetUnits
    
    //    LogDebug ("SiPixelRecHitGPU") 
    //  std::cout << "SiPixelRecHitGPUVI "
    //  << cpeName_ << " converted " << numberOfClusters 
    //  << " SiPixelClusters into SiPixelRecHits, in " 
    //  << numberOfDetUnits << " DetUnits." //; 
    //  << std::endl;
	
  }
