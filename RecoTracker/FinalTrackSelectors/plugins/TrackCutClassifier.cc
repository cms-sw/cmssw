#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"


#include "DataFormats/TrackReco/interface/Track.h"

#include <cassert>

#include "getBestVertex.h"


namespace {

  
  void fillArrayF(float * x,const edm::ParameterSet & cfg, const char * name) {
    auto v = cfg.getParameter< std::vector<double> >(name);
    assert(v.size()==3);
    std::copy(std::begin(v),std::end(v),x);
  }
  
  void fillArrayI(int * x,const edm::ParameterSet & cfg, const char * name) {
    auto v = cfg.getParameter< std::vector<int> >(name);
    assert(v.size()==3);
    std::copy(std::begin(v),std::end(v),x);
  }

  // fake mva value to return for loose,tight,hp
  constexpr float mvaVal[3] = {-.5,.5,1.};

  template<typename T,typename Comp>
  inline float cut(T val, const T * cuts, Comp comp) {
    for (int i=2; i>=0; --i) {
      if ( comp(val,cuts[i]) ) return mvaVal[i];
    }
    return -1.f; 
  }
    
  inline float chi2(reco::Track const & tk) { return tk.normalizedChi2();}
  
  inline int lostLayers(reco::Track const & tk) {
      return tk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    }

  inline int n3DLayers(reco::Track const & tk) {
    return tk.hitPattern().pixelLayersWithMeasurement() +
      tk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  }
  
  inline int nPixelHits(reco::Track const & tk) {
    return tk.hitPattern().numberOfValidPixelHits();
  }
      
  inline float dz(reco::Track const & trk, Point const & bestVertex) {
      return std::abs(trk.dz(bestVertex));
  }
  inline float dr(reco::Track const & trk, Point const & bestVertex) {
      return std::abs(trk.dxy(bestVertex));
  }

  struct Cuts {
    
    Cuts(const edm::ParameterSet & cfg) {
      fillArrayF(maxChi2,cfg,"maxChi2");
      fillArrayF(maxChi2n,cfg,"maxChi2n");
      fillArrayI(minPixelHits,cfg,"minPixelHits");
      fillArrayI(min3DLayers,cfg,"min3DLayers");
      fillArrayI(minLayers,cfg,"minLayers");
      fillArrayI(maxLostLayers,cfg,"maxLostLayers");
      fillArrayF(maxDz,cfg,"maxDz");
      fillArrayF(maxDr,cfg,"maxDr");

    }
    
    
    
    float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices,
		   GBRForest const *) const {

     
      
      float ret = 1.f;
      auto  nLayers = trk.hitPattern().trackerLayersWithMeasurement();
      ret = std::min(ret,cut(nLayers,minLayers,std::greater_equal<int>()));
      if (ret==-1.f) return ret;

      ret = std::min(ret,cut(chi2(trk)/float(nLayers),maxChi2n,std::less_equal<float>()));
      if (ret==-1.f) return ret;
      ret = std::min(ret,cut(chi2(trk),maxChi2,std::less_equal<float>()));
      if (ret==-1.f) return ret;
     
      ret = std::min(ret,cut(n3DLayers(trk),min3DLayers,std::greater_equal<int>()));
      if (ret==-1.f) return ret;
      ret = std::min(ret,cut(nPixelHits(trk),minPixelHits,std::greater_equal<int>()));
      if (ret==-1.f) return ret;
      ret = std::min(ret,cut(lostLayers(trk),maxLostLayers,std::less_equal<int>()));
     
      if (maxDz[2]<std::numeric_limits<float>::max() || maxDr[2]<std::numeric_limits<float>::max()) {
        if (ret==-1.f) return ret;
        Point bestVertex = getBestVertex(trk,vertices);
        ret = std::min(ret,cut(dz(trk,bestVertex), maxDz,std::less_equal<float>()));
        ret = std::min(ret,cut(dr(trk,bestVertex), maxDr,std::less_equal<float>()));
      }

      

      return ret;
      
    }



    static const char * name() { return "TrackCutClassifier";}

    static void fillDescriptions(edm::ParameterSetDescription & desc) {
      desc.add<std::vector<int>>("minPixelHits",{0,0,1});
      desc.add<std::vector<int>>("minLayers",{3,4,5});
      desc.add<std::vector<int>>("min3DLayers",{1,2,3});
      desc.add<std::vector<int>>("maxLostLayers",{99,3,3});
      desc.add<std::vector<double>>("maxChi2",{9999.,25.,16.});
      desc.add<std::vector<double>>("maxChi2n",{9999.,1.0,0.4});
      desc.add<std::vector<double>>("maxDz",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()});
      desc.add<std::vector<double>>("maxDr",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()});

    }


    float maxChi2[3];
    float maxChi2n[3];
    int minLayers[3];
    int min3DLayers[3];
    int minPixelHits[3];
    int maxLostLayers[3];
    float maxDz[3];
    float maxDr[3];

  };


  using TrackCutClassifier = TrackMVAClassifier<Cuts>;
  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackCutClassifier);

