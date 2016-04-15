#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"


#include "DataFormats/TrackReco/interface/Track.h"

#include <cassert>

#include "getBestVertex.h"
#include "powN.h"


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
    
  inline float chi2n(reco::Track const & tk) { return tk.normalizedChi2();}

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

  inline void dzCut_par1(reco::Track const & trk, int & nLayers, const float * par, const int * exp, float dzCut[]) {
    float dzE =  trk.dzError();
    for (int i=2; i>=0; --i) {
      dzCut[i] = powN(par[i]*nLayers,exp[i])*dzE;
    }
  }
  inline void drCut_par1(reco::Track const & trk, int & nLayers, const float * par, const int * exp, float drCut[]) {
    float drE =  trk.d0Error();
    for (int i=2; i>=0; --i) {
      drCut[i] = powN(par[i]*nLayers,exp[i])*drE;
    }
  }
  
  inline void dzCut_par2(reco::Track const & trk, int & nLayers, const float * par, const int * exp, const float * d0err, const float * d0err_par, float dzCut[]) {
    float pt = float(trk.pt());
    float p  = float(trk.p());

    for (int i=2; i>=0; --i) {
      // parametrized d0 resolution for the track pt
      float nomd0E = sqrt(d0err[i]*d0err[i]+(d0err_par[i]/pt)*(d0err_par[i]/pt));
      // parametrized z0 resolution for the track pt and eta
      float nomdzE = nomd0E*(p/pt); // cosh(eta):=abs(p)/pt
      
      dzCut[i] = powN(par[i]*nLayers,exp[i])*nomdzE;
    }
  }
  inline void drCut_par2(reco::Track const & trk, int & nLayers, const float* par, const int * exp, const float * d0err, const float * d0err_par, float drCut[]) {
    float pt = trk.pt();

    for (int i=2; i>=0; --i) {
      // parametrized d0 resolution for the track pt
      float nomd0E = sqrt(d0err[i]*d0err[i]+(d0err_par[i]/pt)*(d0err_par[i]/pt));

      drCut[i] = powN(par[i]*nLayers,exp[i])*nomd0E;
    }
  }
  
  inline void dzCut_wPVerror_par(reco::Track const & trk, int & nLayers, const float * par, const int * exp, Point const & bestVertexError, float dzCut[]) {
    float dzE = trk.dzError();
    float zPVerr = bestVertexError.z();

    float dzErrPV = std::sqrt(dzE*dzE+zPVerr*zPVerr);
    for (int i=2; i>=0; --i) {
      dzCut[i] = par[i]*dzErrPV; 	
      if (exp[i] != 0)
	dzCut[i] *= pow(nLayers,exp[i]);
    }
  }

  inline void drCut_wPVerror_par(reco::Track const & trk, int & nLayers, const float* par, const int * exp, Point const & bestVertexError, float drCut[]) {
    float drE = trk.d0Error();
    float rPVerr = sqrt(bestVertexError.x()*bestVertexError.y()); // shouldn't it be bestVertex.xError()*bestVertex.xError()+bestVertex.yError()*bestVertex.yError() ?!?!?

    float drErrPV = std::sqrt(drE*drE+rPVerr*rPVerr);
    for (int i=2; i>=0; --i) {
      drCut[i] = par[i]*drErrPV;
      if (exp[i] != 0)
	drCut[i] *= pow(nLayers,exp[i]);
    }

  }
  
  struct Cuts {
    
    Cuts(const edm::ParameterSet & cfg) {
      fillArrayF(minNdof,      cfg,"minNdof");
      fillArrayF(maxChi2,      cfg,"maxChi2");
      fillArrayF(maxChi2n,     cfg,"maxChi2n");
      fillArrayI(minPixelHits, cfg,"minPixelHits");
      fillArrayI(min3DLayers,  cfg,"min3DLayers");
      fillArrayI(minLayers,    cfg,"minLayers");
      fillArrayI(maxLostLayers,cfg,"maxLostLayers");
      minNVtxTrk = cfg.getParameter<int>("minNVtxTrk");
      fillArrayF(maxDz,        cfg,"maxDz");
      fillArrayF(maxDzWrtBS,   cfg,"maxDzWrtBS");
      fillArrayF(maxDr,        cfg,"maxDr");
      edm::ParameterSet dz_par = cfg.getParameter<edm::ParameterSet>("dz_par");
      fillArrayI(dz_exp,       dz_par,"dz_exp");
      fillArrayF(dz_par1,      dz_par,"dz_par1");
      fillArrayF(dz_par2,      dz_par,"dz_par2");
      fillArrayF(dzWPVerr_par, dz_par,"dzWPVerr_par");
      edm::ParameterSet dr_par = cfg.getParameter<edm::ParameterSet>("dr_par");
      fillArrayI(dr_exp,       dr_par,"dr_exp");
      fillArrayF(dr_par1,      dr_par,"dr_par1");
      fillArrayF(dr_par2,      dr_par,"dr_par2");
      fillArrayF(d0err,        dr_par,"d0err");
      fillArrayF(d0err_par,    dr_par,"d0err_par");
      fillArrayF(drWPVerr_par, dr_par,"drWPVerr_par");
    }
    
    
    
    float operator()(reco::Track const & trk,
		     reco::BeamSpot const & beamSpot,
		     reco::VertexCollection const & vertices,
		     GBRForest const *) const {
      
      float ret = 1.f;
      ret = std::min(ret,cut(float(trk.ndof()),minNdof,std::greater_equal<float>()) );
      if (ret==-1.f) return ret;

      auto  nLayers = trk.hitPattern().trackerLayersWithMeasurement();
      ret = std::min(ret,cut(nLayers,minLayers,std::greater_equal<int>()));
      if (ret==-1.f) return ret;

      ret = std::min(ret,cut(chi2n(trk)/float(nLayers),maxChi2n,std::less_equal<float>()));
      if (ret==-1.f) return ret;

      ret = std::min(ret,cut(chi2n(trk),maxChi2,std::less_equal<float>()));
      if (ret==-1.f) return ret;
     
      ret = std::min(ret,cut(n3DLayers(trk),min3DLayers,std::greater_equal<int>()));
      if (ret==-1.f) return ret;

      ret = std::min(ret,cut(nPixelHits(trk),minPixelHits,std::greater_equal<int>()));
      if (ret==-1.f) return ret;

      ret = std::min(ret,cut(lostLayers(trk),maxLostLayers,std::less_equal<int>()));
      if (ret==-1.f) return ret;
      
      // original dz and dr cut
      if (maxDz[2]<std::numeric_limits<float>::max() || maxDr[2]<std::numeric_limits<float>::max()) {

	// if not primaryVertices are reconstructed, check compatibility w.r.t. beam spot
        Point bestVertex = getBestVertex(trk,vertices,minNVtxTrk); // min number of tracks [2 (=default) for offline, 3 for HLT]
	float maxDzcut[3];
	std::copy(std::begin(maxDz),std::end(maxDz),std::begin(maxDzcut));
	if (bestVertex.z() < -99998.) {
	  bestVertex = beamSpot.position();
	  std::copy(std::begin(maxDzWrtBS),std::end(maxDzWrtBS),std::begin(maxDzcut));
	}
	ret = std::min(ret,cut(dr(trk,bestVertex), maxDr,std::less<float>()));
	if (ret==-1.f) return ret;

	ret = std::min(ret,cut(dz(trk,bestVertex), maxDzcut,std::less<float>()));
	if (ret==-1.f) return ret;

      }


      // parametrized dz and dr cut by using PV error
      if (dzWPVerr_par[2]<std::numeric_limits<float>::max() || drWPVerr_par[2]<std::numeric_limits<float>::max()) {
	
	Point bestVertexError(-1.,-1.,-1.);
	Point bestVertex = getBestVertex_withError(trk,vertices,bestVertexError,minNVtxTrk); // min number of tracks [2 (=default) for offline, 3 for HLT]
	
	float maxDz_par[3];
	float maxDr_par[3];
	dzCut_wPVerror_par(trk,nLayers,dzWPVerr_par,dz_exp,bestVertexError, maxDz_par);
	drCut_wPVerror_par(trk,nLayers,drWPVerr_par,dr_exp,bestVertexError, maxDr_par);
	
	ret = std::min(ret,cut(dr(trk,bestVertex), maxDr_par,std::less<float>()));
	if (ret==-1.f) return ret;

	ret = std::min(ret,cut(dz(trk,bestVertex), maxDr_par,std::less<float>()));
	if (ret==-1.f) return ret;

      }

      // parametrized dz and dr cut by using their error
      if (dz_par1[2]<std::numeric_limits<float>::max() || dr_par1[2]<std::numeric_limits<float>::max()) {

	float maxDz_par1[3];
	float maxDr_par1[3];
	dzCut_par1(trk,nLayers,dz_par1,dz_exp, maxDz_par1);
	drCut_par1(trk,nLayers,dr_par1,dr_exp, maxDr_par1);

	float maxDz_par[3];
	float maxDr_par[3];
	std::copy(std::begin(maxDz_par1),std::end(maxDz_par1),std::begin(maxDz_par));
	std::copy(std::begin(maxDr_par1),std::end(maxDr_par1),std::begin(maxDr_par));

	// parametrized dz and dr cut by using d0 and z0 resolution
	if (dz_par2[2]<std::numeric_limits<float>::max() || dr_par2[2]<std::numeric_limits<float>::max()) {      

	  float maxDz_par2[3];
	  float maxDr_par2[3];
	  dzCut_par2(trk,nLayers,dz_par2,dz_exp,d0err,d0err_par, maxDz_par2);
	  drCut_par2(trk,nLayers,dr_par2,dr_exp,d0err,d0err_par, maxDr_par2);

	  
	  for (int i=2; i>=0; --i) {
	    if (maxDr_par2[i]<maxDr_par[i]) maxDr_par[i] = maxDr_par2[i];
	    if (maxDz_par2[i]<maxDz_par[i]) maxDz_par[i] = maxDz_par2[i];
	  }
	}

        Point bestVertex = getBestVertex(trk,vertices,minNVtxTrk); // min number of tracks 3 @HLT
	if (bestVertex.z() < -99998.) {
	  bestVertex = beamSpot.position();
	}

        ret = std::min(ret,cut(dz(trk,bestVertex), maxDz_par,std::less<float>()));
	if (ret==-1.f) return ret;
        ret = std::min(ret,cut(dr(trk,bestVertex), maxDr_par,std::less<float>()));	
	if (ret==-1.f) return ret;


      }
      if (ret==-1.f) return ret;

      return ret;
      
    }



    static const char * name() { return "TrackCutClassifier";}

    static void fillDescriptions(edm::ParameterSetDescription & desc) {
      desc.add<std::vector<int>>("minPixelHits", { 0, 0, 1});
      desc.add<std::vector<int>>("minLayers",    { 3, 4, 5});
      desc.add<std::vector<int>>("min3DLayers",  { 1, 2, 3});
      desc.add<std::vector<int>>("maxLostLayers",{99, 3, 3});
      desc.add<std::vector<double>>("minNdof",      {-1.,  -1., -1.});
      desc.add<std::vector<double>>("maxChi2",      {9999.,25., 16. });
      desc.add<std::vector<double>>("maxChi2n",     {9999., 1.0, 0.4});

      desc.add<int>("minNVtxTrk", 2);

      desc.add<std::vector<double>>("maxDz",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()});
      desc.add<std::vector<double>>("maxDzWrtBS",{std::numeric_limits<float>::max(),24.,15.});
      desc.add<std::vector<double>>("maxDr",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()});

      edm::ParameterSetDescription dz_par;
      dz_par.add<std::vector<int>>   ("dz_exp",      {std::numeric_limits<int>::max(),  std::numeric_limits<int>::max(),  std::numeric_limits<int>::max()}  ); // par = 4
      dz_par.add<std::vector<double>>("dz_par1",     {std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}); // par = 0.4
      dz_par.add<std::vector<double>>("dz_par2",     {std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}); // par = 0.35
      dz_par.add<std::vector<double>>("dzWPVerr_par",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}); // par = 3.
      desc.add<edm::ParameterSetDescription>("dz_par", dz_par);

      edm::ParameterSetDescription dr_par;
      dr_par.add<std::vector<int>>   ("dr_exp", {std::numeric_limits<int>::max(),  std::numeric_limits<int>::max(),  std::numeric_limits<int>::max()}  ); // par = 4
      dr_par.add<std::vector<double>>("dr_par1",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}); // par = 0.4
      dr_par.add<std::vector<double>>("dr_par2",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}); // par = 0.3
      dr_par.add<std::vector<double>>("d0err",     {0.003, 0.003, 0.003});
      dr_par.add<std::vector<double>>("d0err_par", {0.001, 0.001, 0.001});
      dr_par.add<std::vector<double>>("drWPVerr_par",{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}); // par = 3.
      desc.add<edm::ParameterSetDescription>("dr_par", dr_par);

    }

    float minNdof[3];
    float maxChi2[3];
    float maxChi2n[3];
    int minLayers[3];
    int min3DLayers[3];
    int minPixelHits[3];
    int maxLostLayers[3];
    int minNVtxTrk;
    float maxDz[3];
    float maxDzWrtBS[3];
    float maxDr[3];
    int   dz_exp[3];
    float dz_par1[3];
    float dz_par2[3];
    float dzWPVerr_par[3];
    int   dr_exp[3];
    float dr_par1[3];
    float dr_par2[3];
    float d0err[3];
    float d0err_par[3];
    float drWPVerr_par[3];
  };


  using TrackCutClassifier = TrackMVAClassifier<Cuts>;
  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackCutClassifier);

