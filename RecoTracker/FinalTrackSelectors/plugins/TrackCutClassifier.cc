#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"


#include "DataFormats/TrackReco/interface/Track.h"

#include <cassert>

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
    for (int i=2; i>=0; --i) 
      if ( comp(val,cuts[i]) ) return mvaVal[i];
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
  
  
  struct Cuts {
    
    Cuts(const edm::ParameterSet & cfg) {
      fillArrayF(maxChi2,cfg,"maxChi2");
      fillArrayI(min3DLayers,cfg,"min3DLayers");
      fillArrayI(maxLostLayers,cfg,"maxLostLayers");
		
    }
    
    
    
    float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices,
		   GBRForest const *) const {

     
      
      float ret = -1.f;
      ret = cut(chi2(trk),maxChi2,std::less_equal<float>());
      if (ret==-1.f) return ret;
      ret = std::min(ret,cut(n3DLayers(trk),min3DLayers,std::greater_equal<int>()));
      if (ret==-1.f) return ret;
      ret = std::min(ret,cut(lostLayers(trk),maxLostLayers,std::less_equal<int>()));

      return ret;
      
    }



    static const char * name() { return "TrackCutClassifier";}

    static void fillDescriptions(edm::ParameterSetDescription & desc) {
      desc.add<std::vector<int>>("min3DLayers",{1,2,3});
      desc.add<std::vector<int>>("maxLostLayers",{99,3,3});
      desc.add<std::vector<double>>("maxChi2",{9999.,25.,16.});
  }


    float maxChi2[3];
    int min3DLayers[3];
    int maxLostLayers[3];

  };


  using TrackCutClassifier = TrackMVAClassifier<Cuts>;
  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackCutClassifier);

