#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <memory>
#include <string>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"
#include "gpuVertexFinder.h"


class PixelVertexHeterogeneousProducer : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
public:

  using Input = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;
  using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;

  using GPUProduct = pixelVertexHeterogeneousProduct::GPUProduct;
  using CPUProduct = pixelVertexHeterogeneousProduct::CPUProduct;
  using Output = pixelVertexHeterogeneousProduct::HeterogeneousPixelVertices;


  explicit PixelVertexHeterogeneousProducer(const edm::ParameterSet&);
  ~PixelVertexHeterogeneousProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override {
    m_gpuAlgo.allocateOnGPU();
  }
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;
 private:
  // ----------member data ---------------------------

  TuplesOnCPU const * tuples_=nullptr;


  // Tracking cuts before sending tracks to vertex algo
  const float m_ptMin;


  const bool enableConversion_;
  const bool enableTransfer_;
  const edm::EDGetTokenT<HeterogeneousProduct> gpuToken_;
  edm::EDGetTokenT<reco::TrackCollection> token_Tracks_;
  edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot_;


  gpuVertexFinder::Producer m_gpuAlgo;

  bool verbose_ = false;
  
};


void PixelVertexHeterogeneousProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("minT",2);  // min number of neighbours to be "core"
  desc.add<double>("eps",0.07); // max absolute distance to cluster
  desc.add<double>("errmax",0.01); // max error to be "seed"
  desc.add<double>("chi2max",9.);   // max normalized distance to cluster

  desc.add<double>("PtMin", 0.5);
  desc.add<edm::InputTag>("TrackCollection", edm::InputTag("pixelTracks"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksHitQuadruplets"));
  desc.add<bool>("gpuEnableConversion", true);
  desc.add<bool>("gpuEnableTransfer", true);

  HeterogeneousEDProducer::fillPSetDescription(desc);
  auto label = "pixelVertexHeterogeneousProducer";
  descriptions.add(label, desc);
}


PixelVertexHeterogeneousProducer::PixelVertexHeterogeneousProducer(const edm::ParameterSet& conf) :
  HeterogeneousEDProducer(conf)
  , m_ptMin(conf.getParameter<double>("PtMin")) // 0.5 GeV
  , enableConversion_(conf.getParameter<bool>("gpuEnableConversion"))
  , enableTransfer_(enableConversion_ || conf.getParameter<bool>("gpuEnableTransfer"))
  , gpuToken_(consumes<HeterogeneousProduct>(conf.getParameter<edm::InputTag>("src")))
  , m_gpuAlgo( conf.getParameter<int>("minT")
	       ,conf.getParameter<double>("eps")
	       ,conf.getParameter<double>("errmax")
	       ,conf.getParameter<double>("chi2max")
               ,enableTransfer_
	       )
{
  produces<HeterogeneousProduct>();
  if (enableConversion_) {
    token_Tracks_ = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackCollection"));
    token_BeamSpot_ =consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot") );
    // Register my product
    produces<reco::VertexCollection>();  
  } else {   
     produces<int>();  // dummy
  }
}



void PixelVertexHeterogeneousProducer::acquireGPUCuda(
                      const edm::HeterogeneousEvent & e,
                      const edm::EventSetup & es,
                      cuda::stream_t<> &cudaStream) {

  // First fish the pixel tracks out of the event
  edm::Handle<TuplesOnCPU> gh;
  e.getByToken<Input>(gpuToken_, gh);
  auto const & gTuples = *gh;
  // std::cout << "Vertex Producers: tuples from gpu " << gTuples.nTuples << std::endl;

  tuples_ = gh.product();

  m_gpuAlgo.produce(cudaStream.id(),gTuples,m_ptMin);


}

void PixelVertexHeterogeneousProducer::produceGPUCuda(
    edm::HeterogeneousEvent & e, const edm::EventSetup & es,
    cuda::stream_t<> &cudaStream) {


  auto const & gpuProduct = m_gpuAlgo.fillResults(cudaStream.id());

  auto output = std::make_unique<GPUProduct>();
  e.put<Output>(std::move(output), heterogeneous::DisableTransfer{});

  if (!enableConversion_) return; 

  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByToken(token_Tracks_,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());
  if (verbose_)  std::cout << "PixelVertexHeterogeneousProducer" << ": Found " << tracks.size() << " tracks in TrackCollection" << "\n";

  
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(token_BeamSpot_,bsHandle);
 
  auto vertexes = std::make_unique<reco::VertexCollection>(); 


  float x0=0,y0=0,z0=0,dxdz=0,dydz=0;
  std::vector<uint32_t> itrk;
  if(!bsHandle.isValid()) {
     edm::LogWarning("PixelVertexHeterogeneousProducer") << "No beamspot found. Using returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot & bs = *bsHandle;
    x0=bs.x0();y0=bs.y0();z0=bs.z0(); dxdz=bs.dxdz();dydz=bs.dydz();
  }

  // fill legacy data format
  if (verbose_) std::cout << "found " << gpuProduct.nVertices << " vertices on GPU using " << gpuProduct.nTracks << " tracks"<< std::endl;
  if (verbose_) std::cout << "original tuple size " << (*tuples_).indToEdm.size() << std::endl;

  std::set<uint16_t> uind; // fort verifing index consistency
  for (int j=int(gpuProduct.nVertices)-1; j>=0; --j) {
    auto i = gpuProduct.sortInd[j];  // on gpu sorted in ascending order....
    assert(i>=0);
    assert(i<int(gpuProduct.nVertices));
    uind.insert(i);    
    assert(itrk.empty());
    auto z= gpuProduct.z[i];
    auto x= x0 + dxdz*z;
    auto y= y0 + dydz*z;
    z +=z0;
    reco::Vertex::Error err;
    err(2,2) = 1.f/gpuProduct.zerr[i];
    err(2,2) *= 2.;  // artifically inflate error
    //Copy also the tracks (no intention to be efficient....)
    for (auto k=0U; k<gpuProduct.nTracks; ++k) {
      if (gpuProduct.ivtx[k]==int(i)) itrk.push_back(gpuProduct.itrk[k]);
    }
    auto nt = itrk.size();
    if (nt==0) { std::cout << "vertex " << i << " with no tracks..." << std::endl; continue;}
    (*vertexes).emplace_back(reco::Vertex::Point(x,y,z), err, gpuProduct.chi2[i], nt-1, nt );
    auto & v = (*vertexes).back();
    for (auto it: itrk) {
      assert(it< (*tuples_).indToEdm.size());
      auto k = (*tuples_).indToEdm[it];
      assert(k<tracks.size());
      auto tk = reco::TrackRef(trackCollection, k);
      v.add(reco::TrackBaseRef(tk));
    }
    itrk.clear();
  }
  
  assert(uind.size()==(*vertexes).size());
  if (!uind.empty()) {
    assert(0 == *uind.begin());
    assert(uind.size()-1 == *uind.rbegin());  
  }
  

  if (verbose_) {
    edm::LogInfo("PixelVertexHeterogeneousProducer") << ": Found " << vertexes->size() << " vertexes\n";
    for (unsigned int i=0; i<vertexes->size(); ++i) {
      edm::LogInfo("PixelVertexHeterogeneousProducer") << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize() << " tracks with a position of " << (*vertexes)[i].z() << " +- " << std::sqrt( (*vertexes)[i].covariance(2,2) );
    }
    
    std::cout << ": Found " << vertexes->size() << " vertexes\n";
    for (unsigned int i=0; i<vertexes->size(); ++i) {
      std::cout << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize() << " tracks with a position of " << (*vertexes)[i].z() << " +- " << std::sqrt( (*vertexes)[i].covariance(2,2) )
		<< " chi2 " << (*vertexes)[i].normalizedChi2() << std::endl;
    }
  }
  
  if(vertexes->empty() && bsHandle.isValid()){
    
    const reco::BeamSpot & bs = *bsHandle;
    
    GlobalError bse(bs.rotatedCovariance3D());
    if ( (bse.cxx() <= 0.) ||
	 (bse.cyy() <= 0.) ||
	 (bse.czz() <= 0.) ) {
      AlgebraicSymMatrix33 we;
      we(0,0)=10000;
      we(1,1)=10000;
      we(2,2)=10000;
      vertexes->push_back(reco::Vertex(bs.position(), we,0.,0.,0));
      
      edm::LogInfo("PixelVertexHeterogeneousProducer") << "No vertices found. Beamspot with invalid errors " << bse.matrix()
						       << "\nWill put Vertex derived from dummy-fake BeamSpot into Event.\n"
						       << (*vertexes)[0].x() << "\n"
						       << (*vertexes)[0].y() << "\n"
						       << (*vertexes)[0].z() << "\n";
    } else {
      vertexes->push_back(reco::Vertex(bs.position(),
				       bs.rotatedCovariance3D(),0.,0.,0));
      
      edm::LogInfo("PixelVertexHeterogeneousProducer") << "No vertices found. Will put Vertex derived from BeamSpot into Event:\n"
						       << (*vertexes)[0].x() << "\n"
						       << (*vertexes)[0].y() << "\n"
						       << (*vertexes)[0].z() << "\n";
    }
  }
  
  else if(vertexes->empty() && !bsHandle.isValid())
    {
      edm::LogWarning("PixelVertexHeterogeneousProducer") << "No beamspot and no vertex found. No vertex returned.";
    }
  
  e.put(std::move(vertexes));
}


void PixelVertexHeterogeneousProducer::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup)
{
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelVertexHeterogeneousProducer);
